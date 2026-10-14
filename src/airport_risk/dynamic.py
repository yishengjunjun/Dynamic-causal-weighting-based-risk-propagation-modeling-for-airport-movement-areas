"""基于观测历史的成对节点扰动与动态干预实验。"""

from __future__ import annotations

import hashlib
import numpy as np
import networkx as nx

from .core import initial_load, nri, sri


def _validate_inputs(data, histories, steps, mode, frozen_weights, batch_size):
    histories = np.asarray(histories, dtype=float)
    n = len(data.capacities)
    if (
        histories.ndim != 3
        or histories.shape[0] == 0
        or histories.shape[1] < 2
        or histories.shape[2] != n
    ):
        raise ValueError("stress histories must be nonempty [samples,H>=2,N]")
    if not np.isfinite(histories).all() or np.any(histories < 0):
        raise ValueError("stress histories must contain finite nonnegative observed loads")
    for label, value in (("steps", steps), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if mode not in ("dynamic", "frozen"):
        raise ValueError("weight mode must be dynamic or frozen")
    if mode == "frozen":
        weight = np.asarray(frozen_weights)
        if weight.shape != (n, n) or not np.isfinite(weight).all():
            raise ValueError("frozen_weights must be a finite node-aligned matrix")
    return histories


def _betweenness(adjacency):
    graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    return np.array(list(nx.betweenness_centrality(graph, normalized=True).values()))


def _ndi(loads, capacities, betweenness):
    return (loads * (loads >= capacities) * 2 * betweenness).sum(axis=1)


def _advance(model, loads, weights, capacities, impedance):
    if not hasattr(model, "propagate_numpy"):
        raise TypeError("stress models must expose relation-aware propagate_numpy")
    return model.propagate_numpy(loads, weights, capacities=capacities)


def _select_nri(adjacency, loads, capacities, k):
    """优先选择 NRI 有定义的节点；不足预算时按负载容量比补齐。"""
    masks = np.zeros_like(loads, dtype=bool)
    fallback_count = 0
    fallback_nodes = np.zeros(loads.shape[1], dtype=int)
    for row, load in enumerate(loads):
        scores = nri(adjacency, load, capacities)
        finite = np.flatnonzero(np.isfinite(scores))
        order = finite[np.argsort(-scores[finite], kind="stable")]
        selected = order[:k].tolist()
        missing = k - len(selected)
        if missing:
            candidates = np.setdiff1d(np.arange(len(load)), finite, assume_unique=True)
            extra = candidates[np.argsort(-(load / capacities)[candidates], kind="stable")][
                :missing
            ]
            selected.extend(extra.tolist())
            fallback_count += missing
            fallback_nodes[extra] += 1
        masks[row, selected] = True
    return masks, fallback_count, fallback_nodes


def _initial_histories(data, histories, *, initial_state="observed", eta=0.7, capacities=None):
    result = histories.copy()
    if initial_state == "topology":
        result[:, -1] = initial_load(
            data.adjacency, data.capacities if capacities is None else capacities, eta=eta
        )
    elif initial_state != "observed":
        raise ValueError("initial_state must be observed or topology")
    return result


def _rollout_batch(
    data,
    model,
    histories,
    *,
    steps,
    mode,
    frozen_weights,
    capacities,
    strategy="none",
    selection=None,
    ranking="fixed",
    control_reduction=0.5,
    k=1,
    retain_paths=False,
):
    """在闭环中更新历史；权重和排序均只依赖当前及过去负载。"""
    history = histories.copy()
    loads = history[:, -1].copy()
    bc = _betweenness(data.adjacency)
    values = [_ndi(loads, capacities, bc)]
    paths = [loads.copy()] if retain_paths else None
    weight_paths = []
    fallback = 0
    fallback_nodes = np.zeros(loads.shape[1], dtype=int)
    if strategy == "NRI" and ranking == "fixed":
        selection, count, nodes = _select_nri(data.adjacency, loads, capacities, k)
        fallback += count
        fallback_nodes += nodes
    for _ in range(steps):
        weights = model.weights(history) if mode == "dynamic" else frozen_weights
        if strategy == "NRI" and ranking == "adaptive":
            selection, count, nodes = _select_nri(data.adjacency, loads, capacities, k)
            fallback += count
            fallback_nodes += nodes
        next_load = _advance(model, loads, weights, capacities, data.impedance)
        if strategy != "none":
            # 干预抑制所选目标节点的正向新增负载，保留阻抗产生的负向恢复量。
            delta = next_load - loads
            next_load = loads + delta - control_reduction * selection * np.maximum(delta, 0)
        if not np.isfinite(next_load).all() or np.any(next_load < 0):
            raise FloatingPointError("stress propagation produced invalid loads")
        loads = next_load
        history = np.concatenate((history[:, 1:], loads[:, None]), axis=1)
        values.append(_ndi(loads, capacities, bc))
        if retain_paths:
            paths.append(loads.copy())
            weight_paths.append(
                np.broadcast_to(weights, (len(loads),) + data.adjacency.shape).copy()
            )
    result = {
        "ndi": np.stack(values, axis=1),
        "final_loads": loads,
        "nri_fallback_selections": fallback,
        "nri_fallback_node_counts": fallback_nodes,
    }
    if retain_paths:
        result.update(loads=np.stack(paths, axis=1), weights=np.stack(weight_paths, axis=1))
    return result


def stress_controls(
    data,
    model,
    histories,
    *,
    steps=12,
    mode="dynamic",
    frozen_weights=None,
    random_repetitions=1000,
    seed=0,
    control_fraction=0.15,
    control_reduction=0.5,
    ranking="fixed",
    batch_size=256,
    capacity_scale=1.0,
    initial_state="observed",
    eta=0.7,
):
    """在相同观测历史和逐节点攻击下比较等预算、等强度干预。"""
    histories = _validate_inputs(data, histories, steps, mode, frozen_weights, batch_size)
    if not 0 < control_fraction <= 1 or not 0 <= control_reduction <= 1:
        raise ValueError("invalid control budget or intensity")
    if ranking not in ("fixed", "adaptive"):
        raise ValueError("ranking must be fixed or adaptive")
    if (
        isinstance(random_repetitions, bool)
        or not isinstance(random_repetitions, int)
        or random_repetitions < 1
    ):
        raise ValueError("random_repetitions must be a positive integer")
    if not np.isfinite(capacity_scale) or capacity_scale <= 0:
        raise ValueError("capacity_scale must be finite and positive")
    if not np.asarray(data.adjacency).any():
        return {
            "status": "not_applicable",
            "reason": "network has no propagation edges",
            "weight_mode": mode,
            "ranking": ranking,
            "curves": {},
            "controlled_nodes": 0,
            "final_ndi_reduction": {},
            "observed_histories": len(histories),
        }
    n = len(data.capacities)
    cap = data.capacities * capacity_scale
    k = max(1, int(np.ceil(control_fraction * n)))
    anchor = _initial_histories(
        data, histories, initial_state=initial_state, eta=eta, capacities=cap
    )
    attacked = np.repeat(anchor, n, axis=0)
    attacked_nodes = np.tile(np.arange(n), len(anchor))
    attacked[np.arange(len(attacked)), -1, attacked_nodes] = np.maximum(
        attacked[np.arange(len(attacked)), -1, attacked_nodes], cap[attacked_nodes] * (1 + 1e-6)
    )
    degrees = data.adjacency.sum(0) + data.adjacency.sum(1)
    degree_ids = np.argsort(-degrees, kind="stable")[:k]
    rng = np.random.default_rng(seed)
    random_nodes = np.array([rng.choice(n, k, replace=False) for _ in range(random_repetitions)])
    curves, fallbacks, fallback_nodes_by_strategy = {}, {}, {}
    for strategy in ("none", "degree", "NRI", "random"):
        total = np.zeros(steps + 1, dtype=float)
        count = 0
        fallback = 0
        fallback_nodes = np.zeros(n, dtype=int)
        repetitions = random_repetitions if strategy == "random" else 1
        # 每个随机节点集合应用于相同的完整攻击集合，避免策略间攻击分布差异。
        for repetition in range(repetitions):
            for start in range(0, len(attacked), batch_size):
                batch = attacked[start : start + batch_size]
                selection = np.zeros((len(batch), n), dtype=bool)
                if strategy in ("degree", "random"):
                    selection[
                        :, degree_ids if strategy == "degree" else random_nodes[repetition]
                    ] = True
                result = _rollout_batch(
                    data,
                    model,
                    batch,
                    steps=steps,
                    mode=mode,
                    frozen_weights=frozen_weights,
                    capacities=cap,
                    strategy=strategy,
                    selection=selection,
                    ranking=ranking,
                    control_reduction=control_reduction,
                    k=k,
                )
                total += result["ndi"].sum(0)
                count += len(batch)
                fallback += result["nri_fallback_selections"]
                fallback_nodes += result["nri_fallback_node_counts"]
        label = "NRI_load_ratio" if strategy == "NRI" else strategy
        curves[label] = (total / count).tolist()
        fallbacks[label] = fallback
        fallback_nodes_by_strategy[label] = {
            str(data.node_ids[i]): int(value) for i, value in enumerate(fallback_nodes) if value
        }
    baseline = curves["none"][-1]
    return {
        "status": "completed",
        "weight_mode": mode,
        "ranking": ranking,
        "initial_state": initial_state,
        "curves": curves,
        "final_ndi_reduction": {
            key: float(1 - curve[-1] / baseline) if baseline else None
            for key, curve in curves.items()
            if key != "none"
        },
        "observed_histories": len(histories),
        "attacks_per_strategy": len(attacked),
        "random_repetitions": random_repetitions,
        "random_seed": seed,
        "random_selection_sha256": hashlib.sha256(random_nodes.astype("<i8").tobytes()).hexdigest(),
        "controlled_nodes": k,
        "control_fraction": k / n,
        "requested_control_fraction": control_fraction,
        "control_intensity": control_reduction,
        "rounding": "ceil(control_fraction*N)",
        "capacity_scale": capacity_scale,
        "degree_ids": data.node_ids[degree_ids].tolist(),
        "control_operator": "reduce positive incoming load increments; retain negative recovery increments",
        "nri_undefined_policy": "rank finite NRI first; fill remaining budget by descending load/capacity ratio, stable node order",
        "nri_fallback_selections": fallbacks,
        "nri_fallback_nodes": fallback_nodes_by_strategy,
        "nri_strategy_label": "NRI-priority and load/capacity-ratio completion composite ranking",
        "scope": "paired closed-loop stress scenarios from observed histories; no future targets are read",
    }


def stress_attacks(
    data,
    model,
    histories,
    *,
    steps=12,
    mode="dynamic",
    frozen_weights=None,
    perturbation_levels=(0.2, 0.4, 0.6),
    batch_size=256,
):
    """对每个观测历史分别攻击全部节点，统计动态传播影响。"""
    histories = _validate_inputs(data, histories, steps, mode, frozen_weights, batch_size)
    if not np.asarray(data.adjacency).any():
        return {
            "status": "not_applicable",
            "reason": "network has no propagation edges",
            "weight_mode": mode,
            "observed_histories": len(histories),
            "steps": steps,
        }
    n, count = len(data.capacities), len(histories)
    cap = data.capacities
    base = histories[:, -1]
    nri_levels, nri_counts, global_impact, excess_impact = {}, {}, {}, {}
    baseline_final = []
    for start in range(0, count, batch_size):
        result = _rollout_batch(
            data,
            model,
            histories[start : start + batch_size],
            steps=steps,
            mode=mode,
            frozen_weights=frozen_weights,
            capacities=cap,
        )
        baseline_final.extend(result["final_loads"].sum(1))
    baseline_final = np.asarray(baseline_final)
    for level in perturbation_levels:
        if not np.isfinite(level) or level < 0:
            raise ValueError("perturbation levels must be finite and nonnegative")
        label = str(float(level) * 100)
        local = np.array([nri(data.adjacency, row * (1 + level), cap) for row in base])
        valid = np.isfinite(local)
        nri_counts[label] = valid.sum(0).tolist()
        nri_levels[label] = np.divide(
            np.where(valid, local, 0).sum(0),
            valid.sum(0),
            out=np.full(n, np.nan),
            where=valid.sum(0) > 0,
        ).tolist()
        attack = np.repeat(histories, n, axis=0)
        nodes = np.tile(np.arange(n), count)
        attack[np.arange(len(attack)), -1, nodes] *= 1 + level
        final = []
        for start in range(0, len(attack), batch_size):
            result = _rollout_batch(
                data,
                model,
                attack[start : start + batch_size],
                steps=steps,
                mode=mode,
                frozen_weights=frozen_weights,
                capacities=cap,
            )
            final.append(result["final_loads"].sum(1))
        final = np.concatenate(final).reshape(count, n)
        denominators = base.sum(1)
        relative = np.divide(
            final - denominators[:, None],
            denominators[:, None],
            out=np.full_like(final, np.nan),
            where=denominators[:, None] > 0,
        )
        global_impact[label] = (
            np.nanmean(relative, axis=0).tolist() if np.isfinite(relative).any() else [None] * n
        )
        excess = np.divide(
            final - baseline_final[:, None],
            denominators[:, None],
            out=np.full_like(final, np.nan),
            where=denominators[:, None] > 0,
        )
        excess_impact[label] = (
            np.nanmean(excess, axis=0).tolist() if np.isfinite(excess).any() else [None] * n
        )
    attacked = np.repeat(histories, n, axis=0)
    nodes = np.tile(np.arange(n), count)
    attacked[np.arange(len(attacked)), -1, nodes] = np.maximum(
        attacked[np.arange(len(attacked)), -1, nodes], cap[nodes] * (1 + 1e-6)
    )
    all_sri, all_ndi = [], []
    cache = {}
    for start in range(0, len(attacked), batch_size):
        result = _rollout_batch(
            data,
            model,
            attacked[start : start + batch_size],
            steps=steps,
            mode=mode,
            frozen_weights=frozen_weights,
            capacities=cap,
            retain_paths=True,
        )
        all_ndi.append(result["ndi"])
        sri_values = []
        for trajectory in result["loads"]:
            curve = []
            for loads in trajectory:
                failed = loads >= cap
                key = failed.tobytes()
                if key not in cache:
                    cache[key] = sri(data.adjacency, failed)
                curve.append(cache[key])
            sri_values.append(curve)
        all_sri.extend(sri_values)
    sri_array = np.asarray(all_sri).reshape(count, n, steps + 1)
    return {
        "status": "completed",
        "weight_mode": mode,
        "observed_histories": count,
        "steps": steps,
        "nri": nri_levels,
        "perturbation_global_load_change": global_impact,
        "nri_finite_history_counts": nri_counts,
        "perturbation_excess_over_unperturbed_rollout": excess_impact,
        "sri": sri_array.mean(0).tolist(),
        "sri_nonfinite_history_counts": (~np.isfinite(sri_array)).sum(0).tolist(),
        "ndi": np.concatenate(all_ndi).reshape(count, n, steps + 1).mean(0).tolist(),
        "attack": "each source node independently raised to max(observed load,capacity*(1+1e-6))",
        "global_load_change_definition": "(perturbed final total-initial total)/initial total",
        "excess_load_change_definition": "(perturbed final total-unperturbed final total)/initial total",
        "scope": "observed-history closed-loop rollout; weights recomputed each step in dynamic mode",
    }
