"""静态传播、岭回归 VAR 与双时间片 DBN 对照模型。

参数估计仅使用 ``fit`` 输入；邻接方向为 [源节点,目标节点]。
算法、参数及传播口径见 ``docs/MODEL.md``。
"""

from __future__ import annotations

import numpy as np


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _history(value, *, nodes=None, min_steps=1):
    h = _array(value)
    if h.ndim != 3 or h.shape[0] == 0 or h.shape[1] < min_steps or h.shape[2] == 0:
        raise ValueError(f"history must have shape [B>0,H>={min_steps},N>0]")
    if nodes is not None and h.shape[2] != nodes:
        raise ValueError("history node count differs from fitted model")
    if not np.isfinite(h).all():
        raise ValueError("history contains NaN or Inf")
    return h


def _training(history, target, *, nodes=None, min_steps=1):
    h = _history(history, nodes=nodes, min_steps=min_steps)
    y = _array(target)
    if y.shape != (h.shape[0], h.shape[2]) or not np.isfinite(y).all():
        raise ValueError("target must be finite with shape [B,N]")
    return h, y


def _graph(adjacency, capacities):
    a, c = _array(adjacency), _array(capacities)
    if c.ndim != 1 or not len(c) or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("capacities must be finite, strictly positive [N]")
    if a.shape != (len(c), len(c)) or not np.isin(a, [0, 1]).all():
        raise ValueError("adjacency must be binary [N,N]")
    if np.any(np.diag(a)):
        raise ValueError("spatial adjacency must have zero diagonal")
    return a.astype(bool), c


def _log(prediction, target):
    return [{"epoch": 0, "train_mse": float(np.mean((prediction - target) ** 2))}]


class StaticPropagationModel:
    """公式 17 的训练历史均值与出边均匀注意力构成的静态传播对照。"""

    def __init__(
        self, adjacency, capacities, impedance, *, epsilon=1e-6, node_ids=None, relations=None
    ):
        self.adjacency, self.capacities = _graph(adjacency, capacities)
        impedance = _array(impedance)
        if impedance.shape != self.capacities.shape or not np.isin(impedance, [0, 1]).all():
            raise ValueError("impedance must be binary with shape [N]")
        self.impedance = impedance.astype(bool)
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be strictly positive")
        self.epsilon = float(epsilon)
        self.weights_ = None
        from .relations import compile_relations

        self.node_ids = tuple(
            str(v) for v in (range(len(self.capacities)) if node_ids is None else node_ids)
        )
        if len(self.node_ids) != len(self.capacities):
            raise ValueError("node_ids must have one entry per node")
        self.relation_plan = compile_relations(self.node_ids, relations)

    def fit(self, history, target, *, epochs=None, **kwargs):
        h, y = _training(history, target, nodes=len(self.capacities), min_steps=2)
        if np.any(h < 0) or np.any(y < 0):
            raise ValueError("physical risk loads must be nonnegative")
        prev = h[:, :-1, :]
        increment = np.maximum(h[:, 1:, :] - prev, 0)
        degree = self.adjacency.sum(axis=1, keepdims=True)
        attention = self.adjacency / np.maximum(degree, 1)
        # 仅计算边张量 [B,T,E]；分母对应源节点，增量对应目标节点。
        sources, destinations = np.nonzero(self.adjacency)
        estimates = increment[..., destinations] / (prev[..., sources] + self.epsilon)
        self.weights_ = np.zeros(self.adjacency.shape, dtype=float)
        self.weights_[sources, destinations] = (
            estimates.mean(axis=(0, 1)) * attention[sources, destinations]
        )
        return _log(self.predict(h), y)

    def predict(self, history):
        if self.weights_ is None:
            raise RuntimeError("fit must be called before predict")
        h = _history(history, nodes=len(self.capacities))
        if np.any(h < 0):
            raise ValueError("physical risk loads must be nonnegative")
        return self.propagate_numpy(h[:, -1, :], self.weights_)

    def propagate_numpy(self, loads, weights, capacities=None):
        """复用静态模型的关系门控和上下文阻抗。"""
        from .relations import propagate_with_relations

        return propagate_with_relations(
            loads,
            weights,
            self.capacities if capacities is None else capacities,
            self.impedance,
            plan=self.relation_plan,
        )

    def weights(self, history):
        if self.weights_ is None:
            raise RuntimeError("fit must be called before weights")
        h = _history(history, nodes=len(self.capacities))
        return np.broadcast_to(self.weights_, (len(h), *self.weights_.shape)).copy()

    def rollout(self, history, *, steps):
        """固定权重下逐周期传播，返回负载与边权轨迹。"""
        if (
            isinstance(steps, (bool, np.bool_))
            or not isinstance(steps, (int, np.integer))
            or steps < 1
        ):
            raise ValueError("steps must be a positive integer")
        h = _history(history, nodes=len(self.capacities))
        weights = self.weights(h)
        loads = [h[:, -1].copy()]
        for _ in range(steps):
            loads.append(self.propagate_numpy(loads[-1], weights))
        return {
            "loads": np.stack(loads, axis=1),
            "weights": np.repeat(weights[:, None], steps, axis=1),
        }


class VARBaseline:
    """事件历史窗口上的定阶岭回归 VAR；标准化仅使用训练集统计量。"""

    def __init__(self, *, lag=1, ridge=0.01):
        if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)) or lag < 1:
            raise ValueError("lag must be a positive integer")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        self.lag, self.ridge = int(lag), float(ridge)
        self.coef_ = None

    def fit(self, history, target, *, epochs=None, **kwargs):
        h, y = _training(history, target, min_steps=self.lag)
        self.n_nodes_ = h.shape[2]
        x = h[:, -self.lag :, :].reshape(len(h), -1)
        self.x_mean_, self.y_mean_ = x.mean(axis=0), y.mean(axis=0)
        self.x_scale_, self.y_scale_ = x.std(axis=0), y.std(axis=0)
        self.x_scale_ = np.where(self.x_scale_ > 1e-12, self.x_scale_, 1.0)
        self.y_scale_ = np.where(self.y_scale_ > 1e-12, self.y_scale_, 1.0)
        z = np.column_stack([np.ones(len(x)), (x - self.x_mean_) / self.x_scale_])
        yz = (y - self.y_mean_) / self.y_scale_
        if self.ridge == 0:
            self.coef_ = np.linalg.lstsq(z, yz, rcond=None)[0]
        else:
            penalty = np.eye(z.shape[1]) * self.ridge
            penalty[0, 0] = 0
            self.coef_ = np.linalg.solve(z.T @ z + penalty, z.T @ yz)
        return _log(self.predict(h), y)

    def predict(self, history):
        if self.coef_ is None:
            raise RuntimeError("fit must be called before predict")
        h = _history(history, nodes=self.n_nodes_, min_steps=self.lag)
        x = h[:, -self.lag :, :].reshape(len(h), -1)
        z = np.column_stack([np.ones(len(x)), (x - self.x_mean_) / self.x_scale_])
        return (z @ self.coef_) * self.y_scale_ + self.y_mean_


class BinaryDBNBaseline:
    """含自回归父节点与受限空间父节点的双时间片二元条件概率表 DBN。"""

    def __init__(self, adjacency, capacities, *, max_parents=2, smoothing=1.0):
        self.adjacency, self.capacities = _graph(adjacency, capacities)
        if (
            isinstance(max_parents, (bool, np.bool_))
            or not isinstance(max_parents, (int, np.integer))
            or not 0 <= max_parents <= 12
        ):
            raise ValueError("max_parents must be an integer from 0 to 12")
        if not np.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing must be finite and strictly positive")
        self.max_parents, self.smoothing = int(max_parents), float(smoothing)
        self.parents_ = [
            np.array([j, *np.flatnonzero(self.adjacency[:, j])[: self.max_parents]], dtype=int)
            for j in range(len(self.capacities))
        ]
        self.cpts_ = None

    @staticmethod
    def _configuration(states, parents):
        # 第一位是节点自身的上一时刻状态，后续各位是父节点状态。
        return states[:, parents].astype(np.int64) @ (1 << np.arange(len(parents)))

    def fit(self, history, target, *, epochs=None, **kwargs):
        h, y = _training(history, target, nodes=len(self.capacities))
        if np.any(h < 0) or np.any(y < 0):
            raise ValueError("physical risk loads must be nonnegative")
        previous = h[:, -1, :] >= self.capacities
        next_state = y >= self.capacities
        self.cpts_, self.counts_ = [], []
        self.state_means_ = np.empty((len(self.capacities), 2), dtype=float)
        for j, parents in enumerate(self.parents_):
            config = self._configuration(previous, parents)
            counts = np.zeros((1 << len(parents), 2), dtype=np.int64)
            np.add.at(counts, (config, next_state[:, j].astype(int)), 1)
            cpt = (counts + self.smoothing) / (
                counts.sum(axis=1, keepdims=True) + 2 * self.smoothing
            )
            self.counts_.append(counts)
            self.cpts_.append(cpt)
            for state in (0, 1):
                rows = next_state[:, j] == bool(state)
                # 训练集中未观测状态的条件均值回退至该节点的训练集均值。
                self.state_means_[j, state] = y[rows, j].mean() if rows.any() else y[:, j].mean()
        return _log(self.predict(h), y)

    def predict_proba(self, history):
        """返回 P(下一时刻节点失效 | 已观测的上一时刻状态)，形状为 [B,N]。"""
        if self.cpts_ is None:
            raise RuntimeError("fit must be called before predict")
        h = _history(history, nodes=len(self.capacities))
        if np.any(h < 0):
            raise ValueError("physical risk loads must be nonnegative")
        states = h[:, -1, :] >= self.capacities
        return np.column_stack(
            [
                cpt[self._configuration(states, parents), 1]
                for parents, cpt in zip(self.parents_, self.cpts_)
            ]
        )

    def predict(self, history):
        """返回以训练集条件均值混合得到的预测，单位为原始物理负载。"""
        p = self.predict_proba(history)
        return (1 - p) * self.state_means_[:, 0] + p * self.state_means_[:, 1]
