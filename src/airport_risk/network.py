"""训练域因子筛选、因果链频率检验与容量网络构建。"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import hashlib
import json

import networkx as nx
import numpy as np

from .core import capacities as graph_capacities, grey_relational
from .nlp import approved_chains
from .text_mining import motif_summary

DEFAULTS = {
    "grey_threshold": 0.3,
    "min_factor_prevalence": 0.01,
    "exclude_constant_factors": True,
    "min_edge_support": 2,
    "permutations": 999,
    "fdr_alpha": 0.05,
    "seed": 2026,
    "capacity_mode": "grey_topology",
    "capacity_omega_be": 0.5,
    "capacity_omega_pr": 0.5,
    "capacity_scale": 1.0,
    "capacity_normalization": "unit_mean",
    "capacity_floor": 1e-6,
    "train_capacity_quantile": 0.95,
    "impedance_recovery_fraction": 0.1,
    "impedance_delta_tolerance": 1e-9,
    "expert_edges": [],
    "network_mode": "train_fitted",
}


def _settings(config):
    if config is not None and not isinstance(config, dict):
        raise ValueError("网络配置必须为字典")
    result = {
        **DEFAULTS,
        **{key: value for key, value in (config or {}).items() if key in DEFAULTS},
    }
    result["expert_edges"] = (
        [dict(edge) if isinstance(edge, dict) else edge for edge in result["expert_edges"]]
        if isinstance(result["expert_edges"], list)
        else result["expert_edges"]
    )
    for key in (
        "grey_threshold",
        "min_factor_prevalence",
        "fdr_alpha",
        "train_capacity_quantile",
        "impedance_recovery_fraction",
    ):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{key} 必须在 [0,1] 内")
    if result["fdr_alpha"] == 0:
        raise ValueError("fdr_alpha 必须大于零")
    for key in ("min_edge_support", "permutations"):
        if type(result[key]) is not int or result[key] < 1:
            raise ValueError(f"{key} 必须为正整数")
    if type(result["seed"]) is not int or result["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    for key in ("capacity_scale", "capacity_floor"):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{key} 必须为有限正值")
    for key in ("capacity_omega_be", "capacity_omega_pr", "impedance_delta_tolerance"):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{key} 必须为有限非负值")
    if result["capacity_omega_be"] + result["capacity_omega_pr"] <= 0:
        raise ValueError("容量系数之和必须为正")
    if not isinstance(result["capacity_mode"], str) or result["capacity_mode"] not in {
        "grey_topology",
        "train_quantile",
    }:
        raise ValueError("capacity_mode 必须为 grey_topology 或 train_quantile")
    if not isinstance(result["capacity_normalization"], str) or result[
        "capacity_normalization"
    ] not in {"unit_mean", "none"}:
        raise ValueError("capacity_normalization 必须为 unit_mean 或 none")
    if type(result["exclude_constant_factors"]) is not bool:
        raise ValueError("exclude_constant_factors 必须为布尔值")
    if not isinstance(result["network_mode"], str) or result["network_mode"] not in {
        "train_fitted",
        "external_fixed",
    }:
        raise ValueError("network_mode 必须为 train_fitted 或 external_fixed")
    if not isinstance(result["expert_edges"], list):
        raise ValueError("expert_edges 必须为审核边列表")
    seen = set()
    for edge in result["expert_edges"]:
        if not isinstance(edge, dict) or set(edge) != {
            "source",
            "target",
            "status",
            "reviewer",
            "basis",
        }:
            raise ValueError("专家边必须仅包含 source、target、status、reviewer、basis")
        if (
            any(
                not isinstance(edge[key], str) or not edge[key].strip()
                for key in ("source", "target", "reviewer", "basis")
            )
            or not isinstance(edge["status"], str)
            or edge["status"] not in {"approved", "rejected"}
        ):
            raise ValueError("专家边字段必须为有效审核记录")
        key = (edge["source"], edge["target"])
        if key[0] == key[1] or key in seen:
            raise ValueError("专家边不能为自环或重复审核边")
        seen.add(key)
    return result


def validate_network_config(config=None):
    """严格校验网络专用配置并返回全部默认值。"""
    if config is not None and (not isinstance(config, dict) or set(config) - set(DEFAULTS)):
        raise ValueError("网络专用配置存在未知字段")
    return _settings(config)


def benjamini_hochberg(pvalues):
    """返回与输入顺序对应的 Benjamini–Hochberg 校正 q 值。"""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p 值必须为 [0,1] 内的有限向量")
    if not len(values):
        return values.copy()
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    output = np.empty_like(values)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _event_chains(annotated_chains, index):
    """统一事件级链列表，审核关系只接受已批准的肯定边。"""
    if isinstance(annotated_chains, dict):
        events = list(annotated_chains.values())
    else:
        events = list(annotated_chains or [])
    output = []
    for item in events:
        if isinstance(item, dict):
            if "relations" in item:
                chains = approved_chains(item["relations"])
            elif "chain" in item:
                if (
                    item.get("review_status") != "approved"
                    or item.get("polarity", "affirmative") != "affirmative"
                    or item.get("certainty", "asserted") == "uncertain"
                ):
                    continue
                if not isinstance(item.get("reviewer"), str) or not item["reviewer"].strip():
                    raise ValueError("批准的因果链必须包含审核人")
                chains = [item["chain"]]
            else:
                raise ValueError("因果链对象需要 chain 或 relations 字段")
        elif isinstance(item, (list, tuple)):
            chains = [item] if not item or isinstance(item[0], str) else item
        else:
            raise ValueError("因果链必须为序列或审核对象")
        normalized = []
        for chain in chains:
            if isinstance(chain, str) or not all(isinstance(node, str) for node in chain):
                raise ValueError("因果链必须是因子 ID 序列")
            if any(node not in index for node in chain):
                raise ValueError("因果链含目录外因子")
            if len(chain) > 1:
                normalized.append([index[node] for node in chain])
        if normalized:
            output.append(normalized)
    return output


def _edge_counts(events, n, rng=None):
    counts = np.zeros((n, n), dtype=np.int64)
    for chains in events:
        edges = set()
        for chain in chains:
            ordered = chain if rng is None else rng.permutation(chain)
            edges.update((int(a), int(b)) for a, b in zip(ordered, ordered[1:]) if a != b)
        for source, target in edges:
            counts[source, target] += 1
    return counts


def permutation_edge_test(events, n, *, permutations=999, seed=2026):
    """在各事件链内置换节点顺序，检验有向邻接频率是否高于顺序零假设。"""
    if type(n) is not int or n < 1 or type(permutations) is not int or permutations < 1:
        raise ValueError("节点数与置换次数必须为正整数")
    observed = _edge_counts(events, n)
    exceedances = np.zeros_like(observed)
    rng = np.random.default_rng(seed)
    for _ in range(permutations):
        exceedances += _edge_counts(events, n, rng) >= observed
    pvalues = (exceedances + 1.0) / (permutations + 1.0)
    mask = ~np.eye(n, dtype=bool)
    qvalues = np.ones((n, n))
    qvalues[mask] = benjamini_hochberg(pvalues[mask])
    np.fill_diagonal(pvalues, 1.0)
    return observed, pvalues, qvalues


def graph_statistics(adjacency, node_ids):
    """统计有向三元模体与无向投影的确定性模块划分。"""
    graph = nx.from_numpy_array(np.asarray(adjacency) > 0, create_using=nx.DiGraph)
    projection = graph.to_undirected()
    if projection.number_of_edges():
        groups = list(nx.community.greedy_modularity_communities(projection))
        modularity = float(nx.community.modularity(projection, groups))
    else:
        groups = [{node} for node in graph.nodes]
        modularity = 0.0
    groups.sort(key=lambda group: min(group))
    census = (
        nx.triadic_census(graph)
        if len(graph) >= 3
        else {
            key: 0
            for key in (
                "003",
                "012",
                "102",
                "021D",
                "021U",
                "021C",
                "111D",
                "111U",
                "030T",
                "030C",
                "201",
                "120D",
                "120U",
                "120C",
                "210",
                "300",
            )
        }
    )
    return {
        "node_count": len(graph),
        "edge_count": graph.number_of_edges(),
        "density": float(nx.density(graph)),
        "weak_components": nx.number_weakly_connected_components(graph),
        "communities": [[str(node_ids[index]) for index in sorted(group)] for group in groups],
        "modularity_undirected_projection": modularity,
        "triadic_census": {key: int(value) for key, value in census.items()},
        "degree_motifs": motif_summary(adjacency),
    }


@dataclass
class NetworkState:
    active_indices: np.ndarray
    adjacency: np.ndarray
    grades: np.ndarray
    capacities: np.ndarray
    impedance: np.ndarray
    node_ids: np.ndarray
    edge_counts: np.ndarray
    pvalues: np.ndarray
    qvalues: np.ndarray
    statistics: dict
    selection: dict

    @property
    def node_indices(self):
        return self.active_indices

    def to_dict(self):
        return {
            field.name: (value.tolist() if isinstance(value, np.ndarray) else value)
            for field in fields(self)
            for value in [getattr(self, field.name)]
        }


def _impedance(history, config):
    differences = np.diff(history, axis=1)
    drops = differences < -config["impedance_delta_tolerance"]
    evidence = drops.mean(axis=(0, 1))
    return ((evidence >= config["impedance_recovery_fraction"]) & drops.any(axis=(0, 1))).astype(
        np.int64
    )


class NetworkBuilder:
    """在训练因子与审核链上构建筛选后的有向风险网络。"""

    def __init__(self, config=None):
        self.config = _settings(config)

    def fit(
        self,
        factor_values_train,
        factor_reference_train,
        annotated_chains_train,
        node_ids,
        config=None,
        *,
        history_train=None,
    ):
        settings = _settings({**self.config, **(config or {})})
        values = np.asarray(factor_values_train, dtype=float)
        reference = np.asarray(factor_reference_train, dtype=float)
        ids = np.asarray(node_ids).astype("U")
        if (
            values.ndim != 2
            or not len(values)
            or values.shape[1] != len(ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError("训练因子矩阵必须与唯一候选 ID 对齐")
        if (
            reference.shape != (len(values),)
            or not np.isfinite(values).all()
            or not np.isfinite(reference).all()
            or np.any(values < 0)
        ):
            raise ValueError("训练灰关联输入必须有限且因子值非负")
        grades, grey_mask = grey_relational(values, reference, threshold=settings["grey_threshold"])
        prevalence = (values > 0).mean(axis=0)
        variable = np.ptp(values, axis=0) > 1e-12
        mask = grey_mask & (prevalence >= settings["min_factor_prevalence"]) & (prevalence > 0)
        if settings["exclude_constant_factors"]:
            mask &= variable
        active = np.flatnonzero(mask)
        if not len(active):
            raise ValueError("训练域筛选后没有活动因子")
        index = {str(node): i for i, node in enumerate(ids)}
        events = _event_chains(annotated_chains_train, index)
        counts, pvalues, qvalues = permutation_edge_test(
            events, len(ids), permutations=settings["permutations"], seed=settings["seed"]
        )
        adjacency = (
            (counts >= settings["min_edge_support"]) & (qvalues <= settings["fdr_alpha"])
        ).astype(np.int64)
        overrides = []
        for edge in settings["expert_edges"]:
            if (
                not isinstance(edge, dict)
                or not edge.get("reviewer")
                or edge.get("status") not in {"approved", "rejected"}
                or not edge.get("basis")
            ):
                raise ValueError("专家边需包含 source、target、status、reviewer、basis")
            if (
                edge.get("source") not in index
                or edge.get("target") not in index
                or edge["source"] == edge["target"]
            ):
                raise ValueError("专家边必须连接目录内两个不同因子")
            adjacency[index[edge["source"]], index[edge["target"]]] = int(
                edge["status"] == "approved"
            )
            overrides.append(dict(edge))
        adjacency = adjacency[np.ix_(active, active)]
        selected_grades = grades[active]
        history = None if history_train is None else np.asarray(history_train, dtype=float)
        if history is not None:
            if (
                history.ndim != 3
                or history.shape[0] != len(values)
                or history.shape[2] != len(ids)
                or history.shape[1] < 2
                or not np.isfinite(history).all()
                or np.any(history < 0)
            ):
                raise ValueError("训练历史必须为有限非负 [训练样本,时间>=2,候选因子]")
            history = history[..., active]
        if settings["capacity_mode"] == "train_quantile":
            if history is None:
                raise ValueError("train_quantile 容量需要训练负载历史")
            capacity = np.quantile(
                history.reshape(-1, len(active)), settings["train_capacity_quantile"], axis=0
            )
            capacity = np.maximum(capacity, settings["capacity_floor"]) * settings["capacity_scale"]
        else:
            capacity = graph_capacities(
                adjacency,
                selected_grades,
                omega_be=settings["capacity_omega_be"],
                omega_pr=settings["capacity_omega_pr"],
            )
            capacity = np.maximum(capacity, settings["capacity_floor"])
            if settings["capacity_normalization"] == "unit_mean":
                capacity = capacity / capacity.mean()
            capacity *= settings["capacity_scale"]
        impedance = (
            np.zeros(len(active), dtype=np.int64)
            if history is None
            else _impedance(history, settings)
        )
        selection = {
            "candidate_ids": ids.tolist(),
            "grey_grades": grades.tolist(),
            "prevalence": prevalence.tolist(),
            "variable": variable.tolist(),
            "selected": mask.tolist(),
            "config": settings,
            "expert_edges": overrides,
            "causal_event_count": len(events),
            "training_rows": len(values),
        }
        self.state_ = NetworkState(
            active,
            adjacency,
            selected_grades,
            capacity,
            impedance,
            ids[active],
            counts[np.ix_(active, active)],
            pvalues[np.ix_(active, active)],
            qvalues[np.ix_(active, active)],
            graph_statistics(adjacency, ids[active]),
            selection,
        )
        return self.state_

    def transform(self, values):
        if not hasattr(self, "state_"):
            raise RuntimeError("NetworkBuilder 必须先 fit")
        values = np.asarray(values)
        if values.ndim < 1 or values.shape[-1] != len(self.state_.selection["candidate_ids"]):
            raise ValueError("变换末轴必须与训练候选因子顺序一致")
        return np.take(values, self.state_.active_indices, axis=-1)


def _chains_for_train(metadata, event_ids):
    mapping = metadata.get("annotated_chains", metadata.get("causal_chains", {}))
    if not isinstance(mapping, dict):
        raise ValueError("metadata.annotated_chains 必须按事件 ID 组织")
    return {event: mapping[event] for event in sorted(set(map(str, event_ids))) if event in mapping}


def prepare_fold_data(data, train, config, grey_threshold=None):
    """独立构建训练折网络，再将同一节点选择应用于全部样本。"""
    data.validate()
    indices = np.asarray(train)
    if (
        indices.ndim != 1
        or not len(indices)
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= len(data.target))
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("训练索引必须为非空、唯一、有效整数向量")
    settings = _settings(config)
    if grey_threshold is not None:
        settings = _settings({**settings, "grey_threshold": grey_threshold})
    mapping = np.asarray(
        data.metadata.get("node_factor_indices", np.arange(len(data.node_ids))), dtype=int
    )
    if mapping.shape != (len(data.node_ids),):
        raise ValueError("候选节点与因子列映射不一致")
    builder = NetworkBuilder(settings)
    state = builder.fit(
        data.factor_values[indices][:, mapping],
        data.factor_reference[indices],
        _chains_for_train(data.metadata, data.event_ids[indices]),
        data.node_ids,
        history_train=data.history[indices],
    )
    active = state.active_indices
    if len(active) < 2:
        raise ValueError("训练域筛选后少于两个活动节点，无法建立预测网络")
    external = settings["network_mode"] == "external_fixed"
    if external:
        source = data.metadata.get("network_source")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("外部先验网络必须提供 metadata.network_source")
        state.adjacency = data.adjacency[np.ix_(active, active)].copy()
        state.capacities = data.capacities[active].copy()
        state.impedance = data.impedance[active].copy()
        state.statistics = graph_statistics(state.adjacency, state.node_ids)
    digest = hashlib.sha256()
    for array in (state.node_ids, state.adjacency, state.grades, state.capacities, state.impedance):
        digest.update(np.ascontiguousarray(array).tobytes())
    audit = state.to_dict()
    audit.update(
        node_indices=active.tolist(),
        original_node_indices=active.tolist(),
        graph_fingerprint=digest.hexdigest(),
        train_event_ids=sorted(set(map(str, data.event_ids[indices]))),
        train_sample_count=len(indices),
        train_only=not external,
        source="external_prior" if external else "training_fold",
        status="completed",
    )
    original = np.asarray(
        data.metadata.get("original_node_indices", np.arange(len(data.node_ids))), dtype=int
    )
    if original.shape != (len(data.node_ids),):
        raise ValueError("original_node_indices 必须与输入节点对齐")
    audit["original_node_indices"] = original[active].tolist()
    metadata = {
        **data.metadata,
        "node_factor_indices": mapping[active].tolist(),
        "network_policy": "fitted_fold",
        "network_fold_audit": audit,
        "original_node_indices": original[active].tolist(),
        "original_node_ids": data.metadata.get("original_node_ids", data.node_ids.tolist()),
    }
    names = data.metadata.get("node_names")
    if names is not None:
        if not isinstance(names, list) or len(names) != len(data.node_ids):
            raise ValueError("metadata.node_names 必须与输入节点对齐")
        metadata["node_names"] = [names[index] for index in active]
    arguments = {
        "history": data.history[..., active].copy(),
        "target": data.target[..., active].copy(),
        "adjacency": state.adjacency,
        "grades": state.grades,
        "capacities": state.capacities,
        "impedance": state.impedance,
        "node_ids": state.node_ids,
        "metadata": metadata,
    }
    if hasattr(data, "future_loads") and data.future_loads is not None:
        arguments["future_loads"] = data.future_loads[..., active].copy()
    fold = replace(data, **arguments).validate()
    return fold, audit
