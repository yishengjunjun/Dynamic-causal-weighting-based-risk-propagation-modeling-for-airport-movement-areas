"""公式 1–12、24–26 的 NumPy 数值计算。

邻接矩阵与边权矩阵的轴顺序为 [源节点, 目标节点]。
"""

from collections import deque
from functools import lru_cache

import numpy as np


def _finite(value, name):
    result = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _graph(adjacency):
    graph = _finite(adjacency, "adjacency")
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1] or not len(graph):
        raise ValueError("adjacency must be a nonempty square matrix")
    if np.any(graph < 0):
        raise ValueError("adjacency cannot contain negative entries")
    return graph > 0


def _node_inputs(load, capacities, impedance=None):
    cap = _finite(capacities, "capacities")
    loads = _finite(load, "load")
    if cap.ndim != 1 or not len(cap) or np.any(cap < 0):
        raise ValueError("capacities must be a nonempty nonnegative vector")
    if loads.ndim < 1 or loads.shape[-1] != len(cap):
        raise ValueError("load's last dimension must match capacities")
    if np.any(loads < 0):
        raise ValueError("risk loads cannot be negative")
    if impedance is None:
        return loads, cap
    imp = _finite(impedance, "impedance")
    if imp.shape != cap.shape or np.any((imp != 0) & (imp != 1)):
        raise ValueError("impedance must be a binary vector matching capacities")
    return loads, cap, imp


def grey_relational(values, reference, threshold=0.3):
    """公式 1–4：以 c 表示样本，以 k 表示因子。

    公式 2 计算相对于 1 的均方根偏差，而不是样本标准差。
    公式 3 的差值极值在整个样本／因子矩阵上计算。
    若输入与参考序列完全一致，则关联系数为 1。
    """
    data = _finite(values, "values")
    ref = _finite(reference, "reference")
    if data.ndim != 2 or 0 in data.shape:
        raise ValueError("values must be [nonempty samples, nonempty factors]")
    if ref.shape != (data.shape[0],):
        raise ValueError("reference must have one value per sample")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    rms_from_one = np.sqrt(np.mean((data - 1.0) ** 2, axis=0))
    rho = 1.0 / (1.0 + np.mean(rms_from_one))
    differences = np.abs(data - ref[:, None])
    minimum, maximum = float(differences.min()), float(differences.max())
    if maximum == 0:
        coefficients = np.ones_like(differences)
    else:
        coefficients = (minimum + rho * maximum) / (differences + rho * maximum)
    grades = coefficients.mean(axis=0)
    return grades, grades > threshold


def _shortest_paths(graph):
    """计算有序节点对之间的距离，以及精确的整数最短路径条数。"""
    n = len(graph)
    neighbors = [np.flatnonzero(graph[i]).tolist() for i in range(n)]
    distances = np.full((n, n), -1, dtype=int)
    counts = [[0] * n for _ in range(n)]
    for source in range(n):
        distances[source, source] = 0
        counts[source][source] = 1
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for target in neighbors[node]:
                if distances[source, target] < 0:
                    distances[source, target] = distances[source, node] + 1
                    queue.append(target)
                if distances[source, target] == distances[source, node] + 1:
                    counts[source][target] += counts[source][node]
    return distances, counts


@lru_cache(maxsize=128)
def _cached_graph_statistics(n, encoded):
    graph = np.frombuffer(encoded, dtype=np.bool_).reshape(n, n)
    distances, counts = _shortest_paths(graph)
    total = sum(counts[q][w] for q in range(n) for w in range(n) if q != w)
    through = [0] * n
    conventional = np.zeros(n, dtype=float)
    for source in range(n):
        for target in range(n):
            denominator = counts[source][target]
            if source == target or denominator == 0:
                continue
            for node in range(n):
                if node == source or node == target:
                    continue
                if (
                    distances[source, node] >= 0
                    and distances[node, target] >= 0
                    and distances[source, node] + distances[node, target]
                    == distances[source, target]
                ):
                    multiplicity = counts[source][node] * counts[node][target]
                    through[node] += multiplicity
                    conventional[node] += multiplicity / denominator
    capacity = np.array([v / total for v in through]) if total else np.zeros(n)
    if n > 2:
        conventional /= (n - 1) * (n - 2)
    else:
        conventional[:] = 0
    capacity.setflags(write=False)
    conventional.setflags(write=False)
    distances.setflags(write=False)
    return capacity, conventional, total, distances


def _statistics(graph):
    return _cached_graph_statistics(len(graph), graph.tobytes())


@lru_cache(maxsize=1024)
def _cached_path_total(n, encoded):
    graph = np.frombuffer(encoded, dtype=np.bool_).reshape(n, n)
    _, counts = _shortest_paths(graph)
    return sum(counts[q][w] for q in range(n) for w in range(n) if q != w)


def topological_capacity(adjacency):
    """公式 6：最短路径条数之和的比值，统计时排除路径端点。

    网络为有向无权图，不可达节点对贡献为零。
    拓扑容量按最短路径总条数归一化，无边图的拓扑容量为零。
    """
    return _statistics(_graph(adjacency))[0].copy()


def capacities(adjacency, grades, *, omega_be=0.5, omega_pr=0.5, scale=1.0):
    """公式 5–7。容量系数及整体倍率须由实验配置确定。"""
    graph = _graph(adjacency)
    grade = _finite(grades, "grades")
    if grade.shape != (len(graph),) or np.any(grade < 0) or grade.sum() <= 0:
        raise ValueError("grades must be nonnegative, node-aligned and sum to >0")
    be = _finite(omega_be, "omega_be")
    pr = _finite(omega_pr, "omega_pr")
    for name, coefficient in (("omega_be", be), ("omega_pr", pr)):
        if coefficient.ndim > 1 or (coefficient.ndim == 1 and coefficient.shape != grade.shape):
            raise ValueError(f"{name} must be scalar or node-aligned")
        if np.any(coefficient < 0):
            raise ValueError(f"{name} cannot be negative")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return scale * (be * _statistics(graph)[0] + pr * grade / grade.sum())


def initial_load(adjacency, capacities, eta=0.7):
    """公式 8：有向图的总度数定义为入度与出度之和。"""
    graph = _graph(adjacency)
    _, cap = _node_inputs(np.zeros(len(graph)), capacities)
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    incoming, outgoing = graph.sum(axis=0), graph.sum(axis=1)
    return eta * cap * (incoming + 1.0) / (incoming + outgoing + 2.0)


def _sigmoid(value):
    return np.exp(-np.logaddexp(0.0, -value))


def impedance_theta(load, capacity):
    """稳定计算公式 12，并保留过载时可能出现的负值，不做截断。"""
    loads = _finite(load, "load")
    cap = _finite(capacity, "capacity")
    if np.any(cap < 0):
        raise ValueError("capacity cannot be negative")
    return _sigmoid(-loads) - np.exp(-cap) * _sigmoid(loads)


def propagate_step(load, weights, capacities, impedance, *, theta_mode="literal"):
    """同步执行公式 9–12，支持 load[...,N] 和 weights[...,N,N]。

    调用方须预先按网络结构将不存在的边权置零。
    有限的有符号权重保持原值。``clipped`` 将阻抗系数截断到 [0,1]；
    ``literal`` 保留公式 12 的计算值，节点过载时可能放大负载增量。
    """
    loads, cap, imp = _node_inputs(load, capacities, impedance)
    weight = _finite(weights, "weights")
    if weight.ndim < 2 or weight.shape[-2:] != (len(cap), len(cap)):
        raise ValueError("weights must end in [source N, destination N]")
    theta = impedance_theta(loads, cap)
    if theta_mode == "clipped":
        theta = np.clip(theta, 0.0, 1.0)
    elif theta_mode != "literal":
        raise ValueError("theta_mode must be literal or clipped")
    active = (loads >= cap).astype(float)
    incoming = np.einsum("...i,...ij->...j", active, weight) * cap
    result = loads + (1.0 - imp * theta) * incoming
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("propagation diverged to nonfinite loads")
    return result


def simulate(load, weights, cap, imp, steps=10, controls=None, *, theta_mode="literal"):
    """在指定传播步数内计算形状为 [steps+1,N] 的负载序列。

    权重可以为 [N,N]，或时间长度至少为 steps 的 [时间,N,N]。
    控制量将目标节点的入边负载增量乘以 (1-controls)。
    """
    loads, cap, imp = _node_inputs(load, cap, imp)
    if loads.ndim != 1:
        raise ValueError("simulate expects one node-aligned load vector")
    if not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    weight = _finite(weights, "weights")
    if weight.ndim not in (2, 3) or weight.shape[-2:] != (len(cap), len(cap)):
        raise ValueError("weights must be [N,N] or [time,N,N]")
    if weight.ndim == 3 and len(weight) < steps:
        raise ValueError("weight time series is shorter than simulation horizon")
    control = np.zeros(len(cap)) if controls is None else _finite(controls, "controls")
    if control.shape != cap.shape or np.any((control < 0) | (control > 1)):
        raise ValueError("controls must be a node vector in [0,1]")
    trajectory = [loads.copy()]
    for time in range(steps):
        current_weight = weight if weight.ndim == 2 else weight[time]
        predicted = propagate_step(loads, current_weight, cap, imp, theta_mode=theta_mode)
        loads = loads + (predicted - loads) * (1.0 - control)
        trajectory.append(loads.copy())
    return np.stack(trajectory)


def nri(adjacency, load, cap):
    """公式 24：邻域为直接后继节点，路径量为最短路径距离。

    邻居求和项对每个后继节点累加 (out_degree_i+1)。
    负载不小于容量或邻域为空时，指标未定义，返回 NaN。
    """
    graph = _graph(adjacency)
    loads, capacity = _node_inputs(load, cap)
    if loads.ndim != 1 or len(graph) != len(capacity):
        raise ValueError("nri requires one graph-aligned load vector")
    n = len(graph)
    distances = _statistics(graph)[3]
    output = np.full(n, np.nan)
    for node in range(n):
        successors = np.flatnonzero(graph[node])
        outgoing = len(successors)
        if loads[node] >= capacity[node] or not outgoing:
            continue
        path_sum = distances[node, successors].sum()
        neighbor_degree_sum = outgoing * (outgoing + 1)
        denominator = (capacity[node] - loads[node]) * neighbor_degree_sum * path_sum
        if denominator > 0:
            output[node] = capacity[node] * outgoing * n * (n - 1) / denominator
    return output


def sri(adjacency, failed):
    """公式 25：失效节点数与存活节点数之比乘以剩余路径比例。

    路径统计采用有向无权图上的最短路径条数。移除所有连接失效节点的边，
    但 N 仍保留为原始节点总数。全部失效时为 +inf；
    若原始网络没有边，则 SRI 未定义，返回 NaN。
    """
    graph = _graph(adjacency)
    failure = np.asarray(failed)
    if failure.shape != (len(graph),):
        raise ValueError("failed must be a node-aligned vector")
    if not np.all((failure == 0) | (failure == 1)):
        raise ValueError("failed must contain boolean/binary indicators")
    failure = failure.astype(bool)
    n, triggered = len(graph), int(failure.sum())
    before = _cached_path_total(n, graph.tobytes())
    if before == 0:
        return float("nan")
    if triggered == n:
        return float("inf")
    remaining = graph.copy()
    remaining[failure, :] = False
    remaining[:, failure] = False
    after = _cached_path_total(n, remaining.tobytes())
    return float(triggered / (n - triggered) * (after / before))


def ndi(adjacency, load, cap):
    """公式 26：以有向归一化介数中心性加权激活节点负载。

    对负载不小于容量的节点计算 2*sum(b_i*L_i)，激活条件为 L_i >= C_i。
    """
    graph = _graph(adjacency)
    loads, capacity = _node_inputs(load, cap)
    if loads.ndim != 1 or len(graph) != len(capacity):
        raise ValueError("ndi requires one graph-aligned load vector")
    betweenness = _statistics(graph)[1]
    return float(2.0 * np.sum(betweenness * loads * (loads >= capacity)))


def global_metrics(adjacency, load, cap):
    """计算传播指标；未定义的浮点值仍保留为未定义状态。"""
    loads, capacity = _node_inputs(load, cap)
    if loads.ndim != 1:
        raise ValueError("global_metrics expects one node-aligned load vector")
    failed = loads >= capacity
    return {
        "sri": sri(adjacency, failed),
        "ndi": ndi(adjacency, loads, capacity),
        "failed": int(failed.sum()),
        "failed_fraction": float(failed.mean()),
    }
