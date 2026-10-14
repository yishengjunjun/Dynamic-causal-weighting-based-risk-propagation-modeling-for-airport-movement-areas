"""可配置的节点关系、运行上下文与同步风险传播。"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import torch
from .core import impedance_theta


@dataclass(frozen=True)
class RelationPlan:
    """与节点顺序绑定的关系配置。"""

    node_ids: tuple[str, ...]
    mutex_groups: tuple[tuple[int, ...], ...] = ()
    cooccurrence_groups: tuple[tuple[int, ...], ...] = ()
    cumulative_groups: tuple[tuple[int, ...], ...] = ()
    context_theta: tuple[float, ...] = ()


def _relation_groups(groups, size):
    validated = []
    for group in groups:
        members = tuple(group)
        if (
            len(members) < 2
            or any(not isinstance(i, Integral) or isinstance(i, (bool, np.bool_)) for i in members)
            or len(set(members)) != len(members)
            or any(i < 0 or i >= size for i in members)
        ):
            raise ValueError("relation groups require at least two distinct integer node indices")
        validated.append(members)
    return validated


def table2_context(years_service, workload, working_hours):
    """表 2 分段系数；左闭右开，末段包含右端点，区间空隙返回 NaN。"""

    def bins(values, intervals):
        value = np.asarray(values, dtype=float)
        output = np.full(value.shape, np.nan)
        for lower, upper, theta, last in intervals:
            mask = (value >= lower) & ((value <= upper) if last else (value < upper))
            output[mask] = theta
        return output

    return {
        "years_service_theta": bins(
            years_service, [(0, 6, 0.15, False), (6, 15, 0.25, False), (15, 30, 0.4, True)]
        ),
        "workload_theta": bins(
            workload, [(0, 0.3, 0.25, False), (0.3, 0.7, 0, False), (0.7, 1, -0.2, True)]
        ),
        "working_hours_theta": bins(
            working_hours, [(1, 2, 0.25, True), (3, 4, 0, True), (5, 6, -0.1, True)]
        ),
    }


def compile_relations(node_ids, config=None):
    """验证关系声明并将节点 ID 解析为索引；关系组不得重叠。"""
    ids = tuple(str(v) for v in node_ids)
    if not ids or len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("node_ids must be nonempty and unique")
    if isinstance(config, RelationPlan):
        if config.node_ids != ids:
            raise ValueError("relation plan node order differs from graph")
        return config
    config = {} if config is None else dict(config)
    allowed = {
        "mutex_groups",
        "cooccurrence_groups",
        "cumulative_groups",
        "context",
        "context_weights",
    }
    if set(config) - allowed:
        raise ValueError(f"unknown relation settings: {sorted(set(config)-allowed)}")
    mapping = {name: i for i, name in enumerate(ids)}
    occupied = set()
    parsed = {}
    for key in ("mutex_groups", "cooccurrence_groups", "cumulative_groups"):
        groups = []
        for members in config.get(key, []):
            names = tuple(str(v) for v in members)
            if (
                len(names) < 2
                or len(set(names)) != len(names)
                or any(name not in mapping for name in names)
            ):
                raise ValueError("relation groups require at least two distinct known node IDs")
            if occupied.intersection(names):
                raise ValueError("relation groups overlap or conflict")
            occupied.update(names)
            # 索引排序使互斥平局规则与配置排列无关。
            groups.append(tuple(sorted(mapping[name] for name in names)))
        parsed[key] = tuple(groups)
    coefficients = np.asarray(config.get("context_weights", [1 / 3, 1 / 3, 1 / 3]), dtype=float)
    if (
        coefficients.shape != (3,)
        or not np.isfinite(coefficients).all()
        or np.any(coefficients < 0)
        or not np.isclose(coefficients.sum(), 1)
    ):
        raise ValueError("context_weights must contain three nonnegative values summing to one")
    theta = np.zeros(len(ids))
    for name, values in config.get("context", {}).items():
        if str(name) not in mapping or set(values) != {
            "years_service",
            "workload",
            "working_hours",
        }:
            raise ValueError("context requires a known node and all three Table 2 fields")
        looked = table2_context(
            values["years_service"], values["workload"], values["working_hours"]
        )
        terms = np.array(
            [looked[k] for k in ("years_service_theta", "workload_theta", "working_hours_theta")],
            dtype=float,
        )
        if terms.shape != (3,) or not np.isfinite(terms).all():
            raise ValueError("context value falls outside Table 2 intervals")
        theta[mapping[str(name)]] = float(terms @ coefficients)
    return RelationPlan(ids, context_theta=tuple(theta), **parsed)


def remap_relations(config, selected_ids):
    """截取子图关系；部分保留的组仅在至少保留两个成员时继续生效。"""
    ids = set(str(v) for v in selected_ids)
    source = {} if config is None else dict(config)
    result = {}
    for key in ("mutex_groups", "cooccurrence_groups", "cumulative_groups"):
        result[key] = [
            retained
            for group in source.get(key, [])
            if len(retained := [str(v) for v in group if str(v) in ids]) >= 2
        ]
    result["context"] = {
        str(k): dict(v) for k, v in source.get("context", {}).items() if str(k) in ids
    }
    if "context_weights" in source:
        result["context_weights"] = list(source["context_weights"])
    return result


def _node_vectors(load, capacities):
    loads, cap = np.asarray(load, dtype=float), np.asarray(capacities, dtype=float)
    if (
        loads.ndim != 1
        or not len(loads)
        or loads.shape != cap.shape
        or not np.isfinite(loads).all()
        or not np.isfinite(cap).all()
        or np.any(loads < 0)
        or np.any(cap <= 0)
    ):
        raise ValueError(
            "load and capacities must be aligned finite node vectors; load must be nonnegative and capacities positive"
        )
    return loads, cap


def _gates_numpy(loads, cap, plan):
    active = (loads >= cap).astype(float)
    for group in plan.mutex_groups:
        index = np.array(group)
        eligible = np.where(active[..., index] > 0, loads[..., index] / cap[index], -np.inf)
        winners = np.argmax(eligible, axis=-1)
        active[..., index] *= np.eye(len(group))[winners]
    for group in plan.cooccurrence_groups:
        index = np.array(group)
        active[..., index] *= np.all(active[..., index] > 0, axis=-1, keepdims=True)
    for group in plan.cumulative_groups:
        index = np.array(group)
        dose = np.maximum(loads[..., index] / cap[index] - 1, 0).mean(axis=-1, keepdims=True)
        active[..., index] = dose
    return active


def gate_sources(
    load, capacities, *, mutex_groups=(), cooccurrence_groups=(), cumulative_groups=()
):
    """互斥按相对负载择优；共现要求全激活；累积采用组平均超载量。"""
    loads, cap = _node_vectors(load, capacities)
    groups = [
        _relation_groups(g, len(cap))
        for g in (mutex_groups, cooccurrence_groups, cumulative_groups)
    ]
    config = {
        key: [[str(i) for i in group] for group in values]
        for key, values in zip(("mutex_groups", "cooccurrence_groups", "cumulative_groups"), groups)
    }
    plan = compile_relations([str(i) for i in range(len(cap))], config)
    return _gates_numpy(loads, cap, plan)


def propagate_with_relations(
    load,
    weights,
    capacities,
    impedance,
    *,
    plan=None,
    mutex_groups=(),
    cooccurrence_groups=(),
    cumulative_groups=(),
):
    """公式 9–12 的关系门控形式，支持批次和广播权重。"""
    loads, cap = np.asarray(load, dtype=float), np.asarray(capacities, dtype=float)
    w, imp = np.asarray(weights, dtype=float), np.asarray(impedance, dtype=float)
    n = len(cap)
    if (
        cap.shape != (n,)
        or n == 0
        or np.any(cap <= 0)
        or not np.isfinite(cap).all()
        or loads.ndim < 1
        or loads.shape[-1] != n
        or not np.isfinite(loads).all()
        or np.any(loads < 0)
    ):
        raise ValueError("loads and capacities must be finite nonnegative aligned node arrays")
    if w.shape[-2:] != (n, n) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("weights must be a finite nonnegative source→destination matrix")
    if imp.shape != cap.shape or not np.isin(imp, [0, 1]).all():
        raise ValueError("impedance must be a binary node vector")
    if plan is None:
        groups = [
            _relation_groups(g, n) for g in (mutex_groups, cooccurrence_groups, cumulative_groups)
        ]
        config = {
            key: [[str(i) for i in group] for group in values]
            for key, values in zip(
                ("mutex_groups", "cooccurrence_groups", "cumulative_groups"), groups
            )
        }
        plan = compile_relations([str(i) for i in range(n)], config)
    if len(plan.node_ids) != n:
        raise ValueError("relation plan node count differs from loads")
    gate = _gates_numpy(loads, cap, plan)
    theta = impedance_theta(loads, cap)
    if any(plan.context_theta):
        theta = np.clip(theta + np.asarray(plan.context_theta), -1, 1)
    result = loads + (1 - imp * theta) * (gate[..., :, None] * w).sum(axis=-2) * cap
    if not np.isfinite(result).all():
        raise FloatingPointError("propagation diverged to nonfinite loads")
    return result


def propagate_relations_torch(loads, weights, capacities, impedance, plan=None):
    """与 NumPy 传播逐项一致；累积强度和阻抗项保留负载梯度。"""
    active = (loads >= capacities).to(loads.dtype)
    if plan is not None:
        for group in plan.mutex_groups:
            index = list(group)
            eligible = torch.where(
                active[..., index] > 0, loads[..., index] / capacities[index], -torch.inf
            )
            winner = eligible.argmax(dim=-1)
            mask = torch.nn.functional.one_hot(winner, len(group)).to(loads.dtype)
            active = active.clone()
            active[..., index] = active[..., index] * mask
        for group in plan.cooccurrence_groups:
            index = list(group)
            gate = (active[..., index] > 0).all(dim=-1, keepdim=True)
            active = active.clone()
            active[..., index] = active[..., index] * gate
        for group in plan.cumulative_groups:
            index = list(group)
            dose = (
                (loads[..., index] / capacities[index] - 1).clamp_min(0).mean(dim=-1, keepdim=True)
            )
            active = active.clone()
            active[..., index] = dose.expand_as(active[..., index])
    theta = torch.sigmoid(-loads) - torch.exp(-capacities) * torch.sigmoid(loads)
    if plan is not None and any(plan.context_theta):
        context = loads.new_tensor(plan.context_theta)
        theta = (theta + context).clamp(-1, 1)
    return (
        loads + (1 - impedance * theta) * (active.unsqueeze(-1) * weights).sum(dim=-2) * capacities
    )
