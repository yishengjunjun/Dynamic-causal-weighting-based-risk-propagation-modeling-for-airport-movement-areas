"""训练折内建网、预测评价及成对动态干预实验。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

import numpy as np
import torch

from .data import split_indices
from .core import grey_relational, initial_load, nri, sri, ndi
from .baselines import StaticPropagationModel, VARBaseline, BinaryDBNBaseline
from .models import CCRLModel, TemporalGraphBaseline
from .reporting import write_json, write_csv, plot_results, markdown_report, clean_json
from .dynamic import stress_attacks, stress_controls


def evaluate_prediction(pred, target, capacities, *, proba=None, event_ids=None):
    """在指定容量单位下评价节点预测，并按事件合并多窗口失效标记。"""
    pred, target, capacities = map(np.asarray, (pred, target, capacities))
    if pred.ndim != 2 or not pred.size or pred.shape != target.shape:
        raise ValueError("predictions and target must be nonempty [samples,nodes] arrays")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("predictions and target must be finite")
    if (
        capacities.shape != (pred.shape[1],)
        or not np.isfinite(capacities).all()
        or np.any(capacities <= 0)
    ):
        raise ValueError("capacities must be finite positive node-aligned values")
    if proba is not None:
        proba = np.asarray(proba)
        if (
            proba.shape != target.shape
            or not np.isfinite(proba).all()
            or np.any((proba < 0) | (proba > 1))
        ):
            raise ValueError("state probabilities must align with target and lie in [0,1]")
    norm_pred, norm_true = pred / capacities, target / capacities
    label = norm_true >= 1
    scores = np.clip(norm_pred, 0, 1)
    pred_event, true_event = (norm_pred >= 1).any(1), label.any(1)
    high_event = scores.max(1) > 0.85
    event_mse = np.mean((norm_pred - norm_true) ** 2, axis=1)
    if event_ids is not None:
        event_ids = np.asarray(event_ids)
        if event_ids.shape != (len(pred),):
            raise ValueError("event_ids must align with prediction samples")
        groups = [np.flatnonzero(event_ids == event) for event in dict.fromkeys(event_ids.tolist())]
        pred_event = np.array([pred_event[group].any() for group in groups])
        true_event = np.array([true_event[group].any() for group in groups])
        high_event = np.array([high_event[group].any() for group in groups])
        event_mse = np.array([event_mse[group].mean() for group in groups])
    return {
        "mse_normalized": float(np.mean((norm_pred - norm_true) ** 2)),
        "mae_normalized": float(np.mean(np.abs(norm_pred - norm_true))),
        "event_mean_mse_normalized": float(event_mse.mean()),
        "node_failure_accuracy": float(np.mean((norm_pred >= 1) == label)),
        "event_failure_accuracy": float(np.mean(pred_event == true_event)),
        "prediction_risk_score_median": float(np.median(scores)),
        "risk_bin_lt_0_5_pct": float(100 * np.mean(scores < 0.5)),
        "risk_bin_0_5_to_0_7_pct": float(100 * np.mean((scores >= 0.5) & (scores <= 0.7))),
        "risk_bin_0_7_to_0_85_pct": float(100 * np.mean((scores > 0.7) & (scores <= 0.85))),
        "risk_bin_gt_0_85_pct": float(100 * np.mean(scores > 0.85)),
        "high_risk_threshold": 0.85,
        "high_risk_event_flag_rate": float(np.mean(high_event)),
        "positive_event_recall_at_0_85": (
            float(high_event[true_event].mean()) if true_event.any() else None
        ),
        "true_positive_event_fraction": float(true_event.mean()),
        "evaluated_samples": len(pred),
        "evaluated_events": len(true_event),
        "evaluated_nodes": pred.shape[1],
        "risk_score_binning_scope": "pooled sample-node normalized load scores",
        "event_definition": "any selected node at capacity in any window of the event",
        "high_risk_flag_definition": "event maximum normalized predicted load >0.85",
        "dbn_state_brier": float(np.mean((proba - label) ** 2)) if proba is not None else None,
    }


def _make_model(name, data, adj, seed, config):
    common = (adj, data.capacities, data.impedance)
    from .relations import remap_relations

    relations = config.get("relations") or data.metadata.get("relations", {})
    graph_options = {
        "node_ids": data.node_ids,
        "relations": remap_relations(relations, data.node_ids),
    }
    params = {
        key: config[key]
        for key in (
            "hidden",
            "layers",
            "kernel_size",
            "lr",
            "gamma",
            "action_scale",
            "max_weight",
            "refinement_steps",
            "epsilon",
            "actor_lr",
            "critic_lr",
            "optimizer",
        )
        if key in config
    }
    params["include_load_channel"] = not config.get("strict_state_input", True)
    if name == "DWPM":
        return CCRLModel(*common, seed=seed, **graph_options, **params)
    if name == "NoRL":
        return CCRLModel(*common, seed=seed, use_rl=False, **graph_options, **params)
    if name == "NoRL_NoAttention":
        return CCRLModel(
            *common, seed=seed, use_rl=False, use_attention=False, **graph_options, **params
        )
    if name == "NDWPM":
        return StaticPropagationModel(*common, **graph_options)
    if name == "VAR":
        return VARBaseline(lag=config.get("var_lag", 1), ridge=config.get("var_ridge", 0.01))
    if name == "DBN":
        return BinaryDBNBaseline(
            adj,
            data.capacities,
            max_parents=config.get("dbn_max_parents", 2),
            smoothing=config.get("dbn_smoothing", 1),
        )
    if name == "T-GNN":
        return TemporalGraphBaseline(
            *common,
            hidden=config["hidden"],
            layers=config.get("layers", 2),
            kernel_size=config.get("kernel_size", 3),
            lr=config.get("lr", 0.001),
            seed=seed,
            **graph_options,
        )
    raise ValueError(f"unknown model: {name}")


def _fit_score(
    name,
    data,
    train,
    test,
    adj,
    seed,
    config,
    *,
    validation=None,
    checkpoint_path=None,
    resume_from=None,
):
    _validate_fold_indices(
        train, test, validation, len(data.target), event_ids=getattr(data, "event_ids", None)
    )
    model = _make_model(name, data, adj, seed, config)
    fit_options = {"epochs": config["epochs"], "batch_size": config.get("batch_size", 64)}
    if name in ("DWPM", "NoRL", "NoRL_NoAttention", "T-GNN"):
        future = getattr(data, "future_loads", None)
        mode = config.get("training_mode", "refinement")
        if name != "T-GNN" and mode == "transition" and future is None:
            raise ValueError("transition training requires observed future_loads")
        fit_options.update(
            training_mode=mode, future_loads=None if future is None else future[train]
        )
        if name != "T-GNN":
            fit_options["warmup_epochs"] = config.get("warmup_epochs", 0)
        if checkpoint_path is not None:
            fit_options.update(
                checkpoint_path=checkpoint_path,
                resume_from=resume_from,
                data_fingerprint=data.fingerprint(),
                config_fingerprint=hashlib.sha256(
                    json.dumps(
                        {
                            key: value
                            for key, value in config.items()
                            if key not in ("epochs", "resume_checkpoints")
                        },
                        sort_keys=True,
                        allow_nan=False,
                    ).encode()
                ).hexdigest(),
            )
    started = time.perf_counter()
    trace = model.fit(data.history[train], data.target[train], **fit_options)
    training_state = getattr(model, "_training_state", None)
    if training_state is not None:
        trace = list(training_state["curve"])
    train_seconds = time.perf_counter() - started
    pred = model.predict(data.history[test])
    proba = model.predict_proba(data.history[test]) if name == "DBN" else None
    ids = getattr(data, "event_ids", None)
    metrics = evaluate_prediction(
        pred,
        data.target[test],
        data.capacities,
        proba=proba,
        event_ids=ids[test] if ids is not None else None,
    )
    metrics.update(
        model=name,
        seed=seed,
        train_samples=len(train),
        test_samples=len(test),
        train_events=len(np.unique(ids[train])) if ids is not None else len(train),
        test_events=len(np.unique(ids[test])) if ids is not None else len(test),
        train_seconds=train_seconds,
        nodes=data.history.shape[2],
        edges=int(np.asarray(adj).sum()),
    )
    if training_state is not None:
        metrics.update(
            cumulative_training_epochs=training_state["epochs_done"],
            cumulative_warmup_epochs=training_state["warmup_done"],
        )
    if validation is not None and len(validation):
        metrics["validation_mse_normalized"] = evaluate_prediction(
            model.predict(data.history[validation]), data.target[validation], data.capacities
        )["mse_normalized"]
    return model, metrics, trace


def _ndi_batch(load, cap, bc):
    """式26的批量形式，阈值相等时采用激活侧单侧极限。"""
    return (bc[None, :] * load * np.where(load >= cap[None, :], 2.0, 0.0)).sum(1)


class _FrozenPhysics:
    """将固定权重对照绑定到数据中声明的节点关系。"""

    def __init__(self, data):
        from .relations import compile_relations

        self.data = data
        self.plan = compile_relations(
            data.node_ids, getattr(data, "metadata", {}).get("relations", {})
        )

    def propagate_numpy(self, loads, weights, capacities=None):
        from .relations import propagate_with_relations

        return propagate_with_relations(
            loads,
            weights,
            self.data.capacities if capacities is None else capacities,
            self.data.impedance,
            plan=self.plan,
        )


def attack_analysis(data, weights, *, steps=12, eta=0.7, perturbation_levels=(0.2, 0.4, 0.6)):
    """拓扑初始负载下的固定权重节点压力对照。"""
    base = initial_load(data.adjacency, data.capacities, eta=eta)
    nr = {
        str(float(level) * 100): nri(data.adjacency, base * (1 + level), data.capacities).tolist()
        for level in perturbation_levels
    }
    n = len(base)
    model = _FrozenPhysics(data)
    impact, sri_curves, ndi_curves, cache = {}, [], [], {}
    for level in perturbation_levels:
        attacks = np.repeat(base[None, :], n, axis=0)
        attacks[np.arange(n), np.arange(n)] += level * base
        for _ in range(steps):
            attacks = model.propagate_numpy(attacks, weights)
        impact[str(float(level) * 100)] = ((attacks.sum(1) - base.sum()) / base.sum()).tolist()
    for node in range(n):
        load = base.copy()
        load[node] = data.capacities[node] * (1 + 1e-6)
        ss, nn = [], []
        for _ in range(steps + 1):
            failed = load >= data.capacities
            key = failed.tobytes()
            if key not in cache:
                cache[key] = sri(data.adjacency, failed)
            ss.append(cache[key])
            nn.append(ndi(data.adjacency, load, data.capacities))
            load = model.propagate_numpy(load, weights)
        sri_curves.append(ss)
        ndi_curves.append(nn)
    return {
        "nri": nr,
        "perturbation_global_load_change": impact,
        "sri": sri_curves,
        "ndi": ndi_curves,
        "weight_mode": "frozen",
        "initial_state": "topology",
        "steps": steps,
    }


def control_analysis(
    data,
    weights,
    *,
    steps=12,
    random_repetitions=1000,
    seed=0,
    eta=0.7,
    capacity_scale=1,
    control_fraction=0.15,
    control_reduction=0.5,
):
    """以拓扑初始状态与固定权重执行成对压力实验。"""
    n = len(data.capacities)
    base = initial_load(data.adjacency, data.capacities, eta=eta)
    result = stress_controls(
        data,
        _FrozenPhysics(data),
        np.tile(base, (1, 2, 1)),
        steps=steps,
        mode="frozen",
        frozen_weights=weights,
        random_repetitions=random_repetitions,
        seed=seed,
        control_fraction=control_fraction,
        control_reduction=control_reduction,
        capacity_scale=capacity_scale,
        initial_state="topology",
        eta=eta,
    )
    return result


def _validate_fold_indices(train, test, validation, event_count, *, event_ids=None):
    """同时检查样本互斥和事件组互斥，防止多窗口跨折。"""
    folds, groups = [], []
    for label, values in (("train", train), ("test", test), ("validation", validation)):
        values = (
            np.asarray([] if values is None else values, dtype=int)
            if values is None
            else np.asarray(values)
        )
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{label} indices must be a one-dimensional integer array")
        if (
            np.any(values < 0)
            or np.any(values >= event_count)
            or len(np.unique(values)) != len(values)
        ):
            raise ValueError(f"{label} indices contain duplicates or out-of-range values")
        if label != "validation" and not len(values):
            raise ValueError(f"{label} indices cannot be empty")
        folds.append(set(values.tolist()))
        groups.append(
            set(np.asarray(event_ids)[values].tolist())
            if event_ids is not None
            else set(values.tolist())
        )
    if any(folds[i] & folds[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("train, validation and test indices must be disjoint")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("train, validation and test event groups must be disjoint")


def _train_mean_weights(model, history, train, batch_size):
    """对完整训练折按样本求平均动态权重。"""
    total, count = None, 0
    for start in range(0, len(train), batch_size):
        batch = history[train[start : start + batch_size]]
        current = model.weights(batch).sum(axis=0, dtype=np.float64)
        total = current if total is None else total + current
        count += len(batch)
    return total / count


def experiment_defaults(profile="full"):
    """返回预先声明的实验、训练、建网及干预参数。"""
    if profile not in ("full", "quick"):
        raise ValueError("profile must be full or quick")
    return {
        "profile": profile,
        "seeds": [0, 1, 2] if profile == "full" else [0],
        "epochs": 40 if profile == "full" else 2,
        "hidden": 32,
        "layers": 3,
        "kernel_size": 3,
        "lr": 0.001,
        "epsilon": 1e-8,
        "actor_lr": 0.001,
        "critic_lr": 0.001,
        "warmup_epochs": 5 if profile == "full" else 0,
        "optimizer": "adam",
        "gamma": 0.9,
        "action_scale": 0.02,
        "max_weight": 1.0,
        "refinement_steps": 3,
        "training_mode": "transition",
        "propagation_steps": 12,
        "random_control_repetitions": 1000 if profile == "full" else 5,
        "strict_state_input": True,
        "airport_holdout": None,
        "eta": 0.7,
        "control_fraction": 0.15,
        "control_reduction": 0.5,
        "stratified_train_fraction": 0.7,
        "rolling_test_years": [],
        "additional_train_fractions": [],
        "printed_train_count": None,
        "batch_size": 64,
        "var_lag": 1,
        "var_ridge": 0.01,
        "dbn_max_parents": 2,
        "dbn_smoothing": 1.0,
        "grey_thresholds": [0.1, 0.2, 0.3, 0.4, 0.5],
        "grey_quantile_diagnostics": False,
        "capacity_scales": [0.9, 1.0, 1.1],
        "eta_scales": [0.9, 1.0, 1.1],
        "perturbation_levels": [0.2, 0.4, 0.6],
        "inference_repetitions": 30,
        "cpu_threads": 2,
        "enabled_protocols": ["primary"],
        "enabled_experiments": [
            "benchmark",
            "ablation",
            "grey",
            "capacity",
            "eta",
            "attacks",
            "controls",
        ],
        "model_names": ["DWPM", "NDWPM", "DBN", "VAR", "T-GNN"],
        "control_weight_modes": ["dynamic", "frozen"],
        "control_rank_mode": "adaptive",
        "stress_history_count": 4,
        "stress_batch_size": 256,
        "network_mode": "train_fitted",
        "grey_threshold": 0.3,
        "min_factor_prevalence": 0.01,
        "exclude_constant_factors": True,
        "min_edge_support": 2,
        "permutations": 999 if profile == "full" else 19,
        "fdr_alpha": 0.05,
        "seed": 2026,
        "capacity_omega_be": 0.5,
        "capacity_omega_pr": 0.5,
        "capacity_scale": 1.0,
        "capacity_normalization": "unit_mean",
        "capacity_floor": 1e-6,
        "capacity_mode": "grey_topology",
        "train_capacity_quantile": 0.95,
        "expert_edges": [],
        "impedance_recovery_fraction": 0.1,
        "impedance_delta_tolerance": 1e-9,
        "relations": {},
        "resume_checkpoints": {},
    }


def validate_experiment_config(config):
    """校验完整配置，不根据输入结果自动开启实验或调整参数。"""
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object with known experiment keys")
    defaults = experiment_defaults(config.get("profile", "full"))
    if not isinstance(config, dict) or set(config) - set(defaults):
        raise ValueError("config must be a JSON object with known experiment keys")
    config = {**defaults, **config}
    for key in (
        "epochs",
        "hidden",
        "layers",
        "kernel_size",
        "refinement_steps",
        "propagation_steps",
        "random_control_repetitions",
        "batch_size",
        "var_lag",
        "dbn_max_parents",
        "inference_repetitions",
        "cpu_threads",
        "stress_history_count",
        "stress_batch_size",
        "min_edge_support",
        "permutations",
    ):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (
        isinstance(config["warmup_epochs"], bool)
        or not isinstance(config["warmup_epochs"], int)
        or config["warmup_epochs"] < 0
    ):
        raise ValueError("warmup_epochs must be a nonnegative integer")
    seeds = config["seeds"]
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds must be distinct nonnegative integers")
    selections = {
        "enabled_protocols": {"primary", "additional_fraction", "rolling", "airport", "count"},
        "enabled_experiments": {
            "benchmark",
            "ablation",
            "grey",
            "capacity",
            "eta",
            "attacks",
            "controls",
        },
        "model_names": {"DWPM", "NDWPM", "DBN", "VAR", "T-GNN"},
        "control_weight_modes": {"dynamic", "frozen"},
    }
    for key, allowed in selections.items():
        values = config[key]
        if (
            not isinstance(values, list)
            or not values
            or len(set(values)) != len(values)
            or set(values) - allowed
        ):
            raise ValueError(f"{key} must contain distinct supported choices")
    if "primary" not in config["enabled_protocols"]:
        raise ValueError("primary protocol is required")
    if "DWPM" not in config["model_names"]:
        raise ValueError("model_names must include DWPM")
    if "airport" in config["enabled_protocols"] and (
        not isinstance(config["airport_holdout"], str) or not config["airport_holdout"].strip()
    ):
        raise ValueError("airport_holdout must identify the enabled held-out airport")
    for key in (
        "lr",
        "actor_lr",
        "critic_lr",
        "epsilon",
        "action_scale",
        "max_weight",
        "dbn_smoothing",
        "eta",
        "capacity_scale",
        "capacity_floor",
    ):
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], (int, float))
            or not np.isfinite(config[key])
            or config[key] <= 0
        ):
            raise ValueError(f"{key} must be finite and positive")
    for key in ("var_ridge", "capacity_omega_be", "capacity_omega_pr", "impedance_delta_tolerance"):
        if not np.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["capacity_omega_be"] + config["capacity_omega_pr"] <= 0:
        raise ValueError("capacity weights cannot both be zero")
    if not np.isfinite(config["gamma"]) or not 0 <= config["gamma"] < 1:
        raise ValueError("gamma must lie in [0,1)")
    for key in (
        "control_reduction",
        "grey_threshold",
        "min_factor_prevalence",
        "impedance_recovery_fraction",
    ):
        if not np.isfinite(config[key]) or not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must lie in [0,1]")
    for key in (
        "control_fraction",
        "stratified_train_fraction",
        "fdr_alpha",
        "train_capacity_quantile",
    ):
        upper = (
            config[key] <= 1
            if key in ("control_fraction", "train_capacity_quantile")
            else config[key] < 1
        )
        if not np.isfinite(config[key]) or not config[key] > 0 or not upper:
            raise ValueError(f"{key} outside permitted range")
    for key, allowed in {
        "optimizer": ("adam", "sgd"),
        "training_mode": ("refinement", "transition"),
        "control_rank_mode": ("fixed", "adaptive"),
        "network_mode": ("train_fitted", "external_fixed"),
        "capacity_mode": ("grey_topology", "train_quantile"),
        "capacity_normalization": ("unit_mean", "none"),
    }.items():
        if config[key] not in allowed:
            raise ValueError(f"unsupported {key}")
    for key in ("strict_state_input", "grey_quantile_diagnostics", "exclude_constant_factors"):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in (
        "additional_train_fractions",
        "grey_thresholds",
        "capacity_scales",
        "eta_scales",
        "perturbation_levels",
    ):
        values = config[key]
        if not isinstance(values, list) or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
            for v in values
        ):
            raise ValueError(f"{key} must be a finite numeric list")
        if key == "additional_train_fractions" and not all(0 < v < 1 for v in values):
            raise ValueError("invalid additional_train_fractions")
        if key == "grey_thresholds" and not all(0 <= v <= 1 for v in values):
            raise ValueError("invalid grey_thresholds")
        if key in ("capacity_scales", "eta_scales") and not all(v > 0 for v in values):
            raise ValueError(f"invalid {key}")
        if key == "perturbation_levels" and not all(v >= 0 for v in values):
            raise ValueError("invalid perturbation_levels")
    for protocol, key in (
        ("additional_fraction", "additional_train_fractions"),
        ("rolling", "rolling_test_years"),
    ):
        if protocol in config["enabled_protocols"] and not config[key]:
            raise ValueError(f"{key} required for enabled protocol")
    if not isinstance(config["rolling_test_years"], list) or any(
        isinstance(y, bool) or not isinstance(y, int) for y in config["rolling_test_years"]
    ):
        raise ValueError("rolling_test_years must be an integer list")
    count = config["printed_train_count"]
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
        raise ValueError("printed_train_count must be positive or null")
    if "count" in config["enabled_protocols"] and count is None:
        raise ValueError("printed_train_count required for count protocol")
    if not isinstance(config["relations"], dict) or not isinstance(
        config["resume_checkpoints"], dict
    ):
        raise ValueError("relations and resume_checkpoints must be objects")
    json.dumps(config, allow_nan=False)
    return config


def _experiment_config(profile, override):
    if override is not None and not isinstance(override, dict):
        raise ValueError("config must be a JSON object")
    if override and override.get("profile", profile) != profile:
        raise ValueError("config profile differs from selected profile")
    return validate_experiment_config({**experiment_defaults(profile), **(override or {})})


def _protocol_splits(data, config):
    seed = config["seeds"][0]
    result = []
    if "additional_fraction" in config["enabled_protocols"]:
        result += [
            (f"stratified_{fraction:g}", {"protocol": "stratified", "train_fraction": fraction})
            for fraction in config["additional_train_fractions"]
        ]
    if "rolling" in config["enabled_protocols"]:
        result += [
            (f"rolling_{year}", {"protocol": "rolling", "test_year": year})
            for year in config["rolling_test_years"]
        ]
    if "airport" in config["enabled_protocols"]:
        result.append(
            ("airport_holdout", {"protocol": "airport", "holdout": config["airport_holdout"]})
        )
    splits = [(tag, kw, split_indices(data, seed=seed, **kw)) for tag, kw in result]
    if "count" in config["enabled_protocols"]:
        groups = np.unique(data.event_ids)
        count = config["printed_train_count"]
        if count >= len(groups):
            raise ValueError("printed_train_count must leave at least one test event")
        selected = np.random.default_rng(seed).permutation(groups)
        train = np.flatnonzero(np.isin(data.event_ids, selected[:count]))
        test = np.flatnonzero(np.isin(data.event_ids, selected[count:]))
        splits.append(
            (
                f"counts_{count}_{len(groups)-count}",
                {"train_event_count": count},
                (train, np.array([], dtype=int), test),
            )
        )
    return splits


def _prepare_fold_data(data, train, config, grey_threshold=None):
    from .network import prepare_fold_data

    return prepare_fold_data(data, train, config, grey_threshold=grey_threshold)


def _prediction_rows(data, indices, pred, capacities, **labels):
    rows = []
    for event in dict.fromkeys(data.event_ids[indices].tolist()):
        mask = data.event_ids[indices] == event
        p = pred[mask] / capacities
        y = data.target[indices[mask]] / capacities
        rows.append(
            {
                **labels,
                "event_id": str(event),
                "windows": int(mask.sum()),
                "nodes": pred.shape[1],
                "mse_normalized": float(np.square(p - y).mean()),
                "max_predicted_load_ratio": float(p.max()),
                "max_observed_load_ratio": float(y.max()),
                "predicted_failure": bool((p >= 1).any()),
                "observed_failure": bool((y >= 1).any()),
            }
        )
    return rows


def run_experiments(data, outdir, *, profile="full", progress=print, config_override=None):
    """按固定配置执行全部实验，在每次训练前独立拟合网络与容量。"""
    config = _experiment_config(profile, config_override)
    data.validate()
    from .relations import compile_relations

    compile_relations(data.node_ids, config.get("relations") or data.metadata.get("relations", {}))
    torch.set_num_threads(config["cpu_threads"])
    seed = config["seeds"][0]
    train, validation, test = split_indices(
        data, seed=seed, train_fraction=config["stratified_train_fraction"]
    )
    extra_splits = _protocol_splits(data, config)
    for _, _, indices in extra_splits:
        _validate_fold_indices(
            indices[0], indices[2], indices[1], len(data.target), event_ids=data.event_ids
        )
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    from .artifacts import export_data_views

    export_data_views(data, out / "input_summary")
    report = {
        "provenance": data.metadata["provenance"],
        "data_fingerprint": data.fingerprint(),
        "config": config,
        "data_summary": {
            "samples": len(data.target),
            "events": len(np.unique(data.event_ids)),
            "input_nodes": len(data.node_ids),
            "history_steps": data.history.shape[1],
            "candidate_factors": data.factor_values.shape[1],
        },
        "benchmarks": [],
        "training_traces": {},
        "network_folds": {},
        "primary_protocol": "primary_stratified",
        "primary_seed": seed,
        "evaluation_policy": "all configured results retained; hyperparameters fixed before evaluation",
        "method": {
            "network_mode": config["network_mode"],
            "capacity_mode": config["capacity_mode"],
            "training_mode": config["training_mode"],
            "control_weight_modes": config["control_weight_modes"],
            "control_ranking": config["control_rank_mode"],
            "state_input": (
                "edge weights"
                if config["strict_state_input"]
                else "edge weights and observed loads"
            ),
            "supervised_warmup_epochs": config["warmup_epochs"],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu_threads": torch.get_num_threads(),
        },
    }
    event_rows = []
    main_model = main_data = None
    runs = [
        (
            "primary_stratified",
            {"train_fraction": config["stratified_train_fraction"]},
            current_seed,
            split_indices(
                data, seed=current_seed, train_fraction=config["stratified_train_fraction"]
            ),
        )
        for current_seed in config["seeds"]
    ]
    runs += [(tag, kw, seed, indices) for tag, kw, indices in extra_splits]
    for tag, settings, current_seed, (fold_train, fold_val, fold_test) in runs:
        fold, audit = _prepare_fold_data(data, fold_train, config)
        key = f"{tag}_seed{current_seed}"
        report["network_folds"][key] = audit
        write_json(out / "networks" / f"{key}.json", audit)
        write_json(
            out / "splits" / f"{key}.json",
            {
                "settings": settings,
                "train_event_ids": np.unique(data.event_ids[fold_train]),
                "validation_event_ids": np.unique(data.event_ids[fold_val]),
                "test_event_ids": np.unique(data.event_ids[fold_test]),
                "train_sample_indices": fold_train,
                "validation_sample_indices": fold_val,
                "test_sample_indices": fold_test,
            },
        )
        for name in (
            config["model_names"] if "benchmark" in config["enabled_experiments"] else ["DWPM"]
        ):
            progress(f"{tag} seed={current_seed}: {name} ({len(fold.node_ids)} nodes)")
            checkpoint = (
                out / "checkpoints" / f"{key}_{name}_training.pt"
                if name in ("DWPM", "T-GNN")
                else None
            )
            model, metrics, trace = _fit_score(
                name,
                fold,
                fold_train,
                fold_test,
                fold.adjacency,
                current_seed,
                config,
                validation=fold_val,
                checkpoint_path=checkpoint,
                resume_from=config["resume_checkpoints"].get(f"{key}_{name}"),
            )
            metrics.update({**settings, "protocol": tag})
            report["benchmarks"].append(metrics)
            report["training_traces"][f"{key}_{name}"] = trace
            event_rows += _prediction_rows(
                fold,
                fold_test,
                model.predict(fold.history[fold_test]),
                fold.capacities,
                protocol=tag,
                model=name,
                seed=current_seed,
            )
            if tag == "primary_stratified" and current_seed == seed and name == "DWPM":
                main_model, main_data = model, fold
                model.save(out / "checkpoints" / "dwpm.pt")
    export_data_views(main_data, out / "primary_network")
    report["ablations"] = []
    if "ablation" in config["enabled_experiments"]:
        for name in ("DWPM", "NoRL", "NoRL_NoAttention", "NDWPM"):
            progress("ablation: " + name)
            _, metrics, _ = _fit_score(
                name, main_data, train, test, main_data.adjacency, seed, config
            )
            report["ablations"].append(metrics)
    sensitivity = {
        "grey_threshold": [],
        "grey_diagnostic_quantiles": [],
        "capacity_scale": [],
        "eta_scale": [],
        "capacity_prediction": [],
    }
    report["sensitivity"] = sensitivity
    grey_predictions = []
    thresholds = (
        [("threshold", value) for value in config["grey_thresholds"]]
        if "grey" in config["enabled_experiments"]
        else []
    )
    if config["grey_quantile_diagnostics"] and "grey" in config["enabled_experiments"]:
        grades, _ = grey_relational(data.factor_values[train], data.factor_reference[train])
        thresholds += [("quantile", float(np.quantile(grades, q)), q) for q in (0.25, 0.5, 0.75)]
    for item in thresholds:
        kind, threshold = item[:2]
        rows = sensitivity["grey_threshold" if kind == "threshold" else "grey_diagnostic_quantiles"]
        progress(f"grey {kind}={threshold:g}")
        try:
            changed, audit = _prepare_fold_data(data, train, config, grey_threshold=threshold)
        except ValueError as error:
            rows.append({"threshold": threshold, "status": "not_trainable", "reason": str(error)})
            continue
        model, metrics, _ = _fit_score(
            "DWPM", changed, train, test, changed.adjacency, seed, config
        )
        metrics.update(
            threshold=threshold,
            status="completed",
            retained_candidates=len(changed.node_ids),
            retained_modeled_nodes=len(changed.node_ids),
            retained_edges=int(changed.adjacency.sum()),
            graph_fingerprint=audit["graph_fingerprint"],
            node_ids=changed.node_ids.tolist(),
            capacity_definition=config["capacity_mode"],
        )
        metrics["graph_structure_fingerprint"] = hashlib.sha256(
            np.ascontiguousarray(changed.adjacency).tobytes()
            + json.dumps(changed.node_ids.tolist()).encode()
        ).hexdigest()
        if kind == "quantile":
            metrics["train_grade_quantile"] = item[2]
        rows.append(metrics)
        pred = model.predict(changed.history[test])
        grey_predictions.append((changed, pred, metrics))
        write_json(out / "networks" / f"grey_{kind}_{threshold:.8g}.json", audit)
        event_rows += _prediction_rows(
            changed,
            test,
            pred,
            changed.capacities,
            protocol=f"grey_{kind}_{threshold:.8g}",
            model="DWPM",
            seed=seed,
        )
    completed = [row for row in sensitivity["grey_threshold"] if row.get("status") == "completed"]
    sensitivity["grey_sweep_degenerate"] = (
        len({row["graph_structure_fingerprint"] for row in completed}) <= 1
    )
    shared = set(main_data.node_ids.tolist())
    for changed, _, _ in grey_predictions:
        shared &= set(changed.node_ids.tolist())
    shared_ids = [str(node) for node in main_data.node_ids if node in shared]
    sensitivity["grey_shared_node_ids"] = shared_ids
    for changed, pred, metrics in grey_predictions:
        if shared_ids:
            columns = [list(changed.node_ids).index(node) for node in shared_ids]
            original = [list(main_data.node_ids).index(node) for node in shared_ids]
            shared_metrics = evaluate_prediction(
                pred[:, columns],
                changed.target[test][:, columns],
                main_data.capacities[original],
                event_ids=data.event_ids[test],
            )
            metrics["shared_node_metrics"] = shared_metrics
            metrics["shared_node_capacity_source"] = "primary training fold capacities"
        else:
            metrics["shared_node_metrics"] = None
    weights = _train_mean_weights(main_model, main_data.history, train, config["batch_size"])
    # 压力历史按已保存的测试索引顺序取样，不按预测优劣挑选。
    stress_indices = test[: config["stress_history_count"]]
    histories = main_data.history[stress_indices]
    write_json(
        out / "stress_selection.json",
        {
            "sample_indices": stress_indices,
            "event_ids": data.event_ids[stress_indices],
            "selection": "first saved primary test indices",
            "history_only": True,
        },
    )
    common = {
        "steps": config["propagation_steps"],
        "seed": seed,
        "control_fraction": config["control_fraction"],
        "control_reduction": config["control_reduction"],
        "ranking": config["control_rank_mode"],
        "batch_size": config["stress_batch_size"],
        "random_repetitions": config["random_control_repetitions"],
    }
    report["control_comparison"] = {}
    report["attack_comparison"] = {}
    for mode in config["control_weight_modes"]:
        if "controls" in config["enabled_experiments"]:
            progress(f"{mode}: paired controls")
            report["control_comparison"][mode] = stress_controls(
                main_data, main_model, histories, mode=mode, frozen_weights=weights, **common
            )
        if "attacks" in config["enabled_experiments"]:
            progress(f"{mode}: node attacks")
            report["attack_comparison"][mode] = stress_attacks(
                main_data,
                main_model,
                histories,
                mode=mode,
                frozen_weights=weights,
                steps=config["propagation_steps"],
                perturbation_levels=config["perturbation_levels"],
                batch_size=config["stress_batch_size"],
            )
    primary_mode = (
        "dynamic"
        if "dynamic" in config["control_weight_modes"]
        else config["control_weight_modes"][0]
    )
    report["controls"] = report["control_comparison"].get(primary_mode, {})
    report["node_attacks"] = report["attack_comparison"].get(primary_mode, {})
    if "capacity" in config["enabled_experiments"]:
        for scale in config["capacity_scales"]:
            progress(f"capacity scale={scale:g}")
            changed = replace(main_data, capacities=main_data.capacities * scale)
            model, _, _ = _fit_score("DWPM", changed, train, test, changed.adjacency, seed, config)
            metrics = evaluate_prediction(
                model.predict(changed.history[test]),
                main_data.target[test],
                main_data.capacities,
                event_ids=data.event_ids[test],
            )
            metrics.update(
                scale=scale,
                evaluation_units="primary fold nominal capacity; identical labels across scales",
            )
            sensitivity["capacity_prediction"].append(metrics)
            result = stress_controls(
                main_data,
                main_model,
                histories,
                mode=primary_mode,
                frozen_weights=weights,
                capacity_scale=scale,
                **{**common, "random_repetitions": 1},
            )
            sensitivity["capacity_scale"].append(
                {
                    "scale": scale,
                    "status": result["status"],
                    "mean_final_ndi": (
                        result["curves"]["none"][-1] if result["status"] == "completed" else None
                    ),
                    "quantity": "propagation NDI; forecast model weights unchanged; propagation capacity perturbed",
                }
            )
    if "eta" in config["enabled_experiments"]:
        for scale in config["eta_scales"]:
            result = stress_controls(
                main_data,
                main_model,
                histories,
                mode=primary_mode,
                frozen_weights=weights,
                initial_state="topology",
                eta=config["eta"] * scale,
                **{**common, "random_repetitions": 1},
            )
            sensitivity["eta_scale"].append(
                {
                    "scale": scale,
                    "status": result["status"],
                    "mean_final_ndi": (
                        result["curves"]["none"][-1] if result["status"] == "completed" else None
                    ),
                    "quantity": "topology-initialized stress NDI; eta sets initial load and is not a forecasting input",
                }
            )
    sensitivity["eta_prediction"] = (
        "eta is an initial stress-load coefficient; observed-history forecasting does not use eta"
    )
    example = main_data.history[test[:1]]
    rollout = main_model.rollout(example, steps=config["propagation_steps"])
    future = getattr(main_data, "future_loads", None)
    future = main_data.target[:, None] if future is None else future
    observed_steps = min(future.shape[1], config["propagation_steps"])
    trajectory_metrics = [
        {
            "horizon": horizon + 1,
            **evaluate_prediction(
                rollout["loads"][:, horizon + 1], future[test[:1], horizon], main_data.capacities
            ),
        }
        for horizon in range(observed_steps)
    ]
    dynamic_rollout = {
        "event_id": str(data.event_ids[test[0]]),
        "steps": config["propagation_steps"],
        "selection": "first saved primary test sample",
        "load_trajectory": rollout["loads"][0],
        "dynamic_weight_trajectory": rollout["weights"][0],
        "first_step_observed_target": main_data.target[test[0]],
        "first_step_evaluation": trajectory_metrics[0],
        "observed_horizon_metrics": trajectory_metrics,
        "observed_horizons": observed_steps,
        "future_accuracy_computed": observed_steps > 1,
        "scope": "closed loop; predicted loads appended to history; only available observed horizons evaluated",
    }
    write_json(out / "dynamic_rollout.json", dynamic_rollout)
    report["dynamic_rollout"] = {
        key: value
        for key, value in dynamic_rollout.items()
        if key not in ("load_trajectory", "dynamic_weight_trajectory", "first_step_observed_target")
    }
    report["dynamic_rollout"]["artifact"] = "dynamic_rollout.json"
    for _ in range(3):
        main_model.predict(example)
    times = []
    for _ in range(config["inference_repetitions"]):
        started = time.perf_counter()
        main_model.predict(example)
        times.append(time.perf_counter() - started)
    report["inference_timing"] = {
        "median_ms": 1000 * float(np.median(times)),
        "p95_ms": 1000 * float(np.quantile(times, 0.95)),
        "repetitions": config["inference_repetitions"],
        "scope": "one sample CPU prediction including weight refinements",
    }
    write_json(out / "metrics.json", report)
    write_csv(out / "tables" / "benchmark_all_protocols.csv", report["benchmarks"])
    write_csv(out / "tables" / "event_predictions.csv", event_rows)
    write_csv(out / "tables" / "ablations.csv", report["ablations"])
    for key in (
        "grey_threshold",
        "grey_diagnostic_quantiles",
        "capacity_prediction",
        "capacity_scale",
        "eta_scale",
    ):
        write_csv(out / "tables" / f"{key}_sensitivity.csv", sensitivity[key])
    plot_results(main_data, clean_json(report), out / "figures")
    (out / "REPORT.md").write_text(markdown_report(report), encoding="utf-8")
    progress(f"complete: {out}")
    return report
