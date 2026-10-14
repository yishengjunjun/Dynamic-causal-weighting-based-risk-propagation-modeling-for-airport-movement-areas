"""因果时序编码、动态因果权重与连续动作 actor–critic。

网络、传播和训练定义见 ``docs/MODEL.md``。
"""

from __future__ import annotations

import math
import hashlib
import json
import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .relations import compile_relations, propagate_with_relations, propagate_relations_torch


def _integer(value: int, name: str, *, minimum: int = 1) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


class CausalConv1d(nn.Module):
    """公式 13：h[t] = sum_k W[k] x[t-d*k] + b，W[0] 对应当前时刻。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        kernel_size = _integer(kernel_size, "kernel_size")
        dilation = _integer(dilation, "dilation")
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: Tensor) -> Tensor:
        # Conv1d 使用互相关；翻转参数轴后，参数下标与公式中的滞后 k 一致。
        return F.conv1d(
            F.pad(x, (self.left_padding, 0)),
            self.conv.weight.flip(-1),
            self.conv.bias,
            dilation=self.conv.dilation,
        )


class CausalTemporalEncoder(nn.Module):
    """对各节点共享参数的因果编码器；每一层都保留时间序列长度。"""

    def __init__(
        self, hidden: int = 16, layers: int = 2, kernel_size: int = 3, in_channels: int = 1
    ):
        super().__init__()
        hidden = _integer(hidden, "hidden")
        layers = _integer(layers, "layers")
        self.layers = nn.ModuleList(
            CausalConv1d(in_channels if i == 0 else hidden, hidden, kernel_size, 2**i)
            for i in range(layers)
        )

    def forward(self, signals: Tensor) -> Tensor:
        # 输入 [B,H,N,C]；节点共享卷积参数。
        b, h, n, channels = signals.shape
        x = signals.permute(0, 2, 3, 1).reshape(b * n, channels, h)
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x.reshape(b, n, -1, h).permute(0, 3, 1, 2)


def masked_row_softmax(logits: Tensor, adjacency: Tensor) -> Tensor:
    """计算源节点到目标节点的注意力；没有后继节点时，该行全部为零。"""
    mask = adjacency.bool().unsqueeze(0)
    masked = logits.masked_fill(~mask, -torch.inf)
    # 空邻接行在归一化前置零，避免 softmax(-inf, ...) 产生 NaN。
    masked = torch.where(mask.any(dim=-1, keepdim=True), masked, torch.zeros_like(masked))
    return torch.softmax(masked, dim=-1) * mask.to(logits.dtype)


def propagate_torch(
    loads: Tensor, weights: Tensor, capacities: Tensor, impedance: Tensor, relations=None
) -> Tensor:
    """公式 9–12 的同步传播；配置关系时应用相应源门控和上下文阻抗。"""
    return propagate_relations_torch(loads, weights, capacities, impedance, relations)


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _training_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        if array is None:
            digest.update(b"none")
        else:
            value = np.ascontiguousarray(array, dtype=np.float32)
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def _rng_snapshot():
    state = np.random.get_state()
    return {
        "torch_cpu": torch.random.get_rng_state(),
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _validate_graph(
    adjacency: np.ndarray,
    capacities: np.ndarray,
    impedance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(adjacency)
    c = np.asarray(capacities, dtype=np.float32)
    z = np.asarray(impedance, dtype=np.float32)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] == 0:
        raise ValueError("adjacency must be a nonempty square source→destination matrix")
    if not np.isin(a, [0, 1]).all():
        raise ValueError("adjacency must be binary")
    if np.any(np.diag(a)):
        raise ValueError("adjacency must have zero diagonal")
    if c.shape != (a.shape[0],) or not np.isfinite(c).all() or np.any(c <= 0):
        raise ValueError("capacities must be finite, positive and have one entry per node")
    if z.shape != c.shape or not np.isfinite(z).all() or np.any((z != 0) & (z != 1)):
        raise ValueError("impedance must be a binary vector, one entry per node")
    return a.astype(bool), c, z


class _HistoryModel(nn.Module):
    def _history(self, history: np.ndarray) -> Tensor:
        value = np.asarray(history, dtype=np.float32)
        if value.ndim != 3 or value.shape[2] != self.capacities.numel() or value.shape[1] < 2:
            raise ValueError("history must have shape [batch, observed_time>=2, nodes]")
        if value.shape[0] == 0 or not np.isfinite(value).all() or np.any(value < 0):
            raise ValueError("history must contain finite nonnegative loads and a nonempty batch")
        return torch.as_tensor(value, dtype=torch.float32, device=self.capacities.device)

    def _target(self, target: np.ndarray, batch: int) -> Tensor:
        value = np.asarray(target, dtype=np.float32)
        if (
            value.shape != (batch, self.capacities.numel())
            or not np.isfinite(value).all()
            or np.any(value < 0)
        ):
            raise ValueError("target must be finite nonnegative loads of shape [batch, nodes]")
        return torch.as_tensor(value, dtype=torch.float32, device=self.capacities.device)

    def _signals(self, history: Tensor) -> Tensor:
        binary_state = (history >= self.capacities).to(history.dtype)
        if self.include_load_channel:
            # 连续负载按节点容量归一化后与二元状态共同编码。
            return torch.stack((binary_state, history / self.capacities), dim=-1)
        return binary_state.unsqueeze(-1)

    def encode(self, history: np.ndarray) -> np.ndarray:
        """返回全部已观测位置的时序特征，形状为 [B,H,N,P]。"""
        self.eval()
        with torch.no_grad():
            return self.encoder(self._signals(self._history(history))).cpu().numpy()

    def propagate_numpy(self, loads, weights, capacities=None):
        """用模型节点关系和运行上下文执行一次物理传播。"""
        return propagate_with_relations(
            loads,
            weights,
            self.capacities.detach().cpu().numpy() if capacities is None else capacities,
            self.impedance.detach().cpu().numpy(),
            plan=self.relation_plan,
        )

    def _propagate(self, loads, weights):
        return propagate_torch(loads, weights, self.capacities, self.impedance, self.relation_plan)

    def _checkpoint(self, path: str | Path, config: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": 3, "config": config, "state_dict": self.state_dict()}, path)

    def save_training(self, path: str | Path) -> None:
        """保存参数、优化器、训练阶段、数据指纹与随机状态。"""
        if not hasattr(self, "_training_state"):
            raise RuntimeError("fit must complete at least one training stage before saving")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 3,
            "kind": type(self).__name__,
            "config": self._config,
            "state_dict": self.state_dict(),
            "training_state": self._training_state,
            "model_config_digest": _digest(self._config),
            "rng": _rng_snapshot(),
        }
        temporary = path.with_name(path.name + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load_training(cls, path: str | Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("format_version") != 3
            or checkpoint.get("kind") != cls.__name__
            or "training_state" not in checkpoint
        ):
            raise ValueError("unsupported training checkpoint version or model kind")
        if checkpoint.get("model_config_digest") != _digest(checkpoint["config"]):
            raise ValueError("training checkpoint model configuration digest mismatch")
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model._training_state = checkpoint["training_state"]
        model.eval()
        return model

    def reset_training(self):
        """清除续训状态，保留当前参数用于新的独立训练任务。"""
        if hasattr(self, "_training_state"):
            del self._training_state

    def _start_training(
        self, x, y, future, protocol, resume_from, data_fingerprint, config_fingerprint
    ):
        for name, value in (
            ("data_fingerprint", data_fingerprint),
            ("config_fingerprint", config_fingerprint),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string or None")
        signature = {
            "arrays": _training_digest(
                x.detach().cpu().numpy(),
                y.detach().cpu().numpy(),
                None if future is None else future.detach().cpu().numpy(),
            ),
            "data_fingerprint": data_fingerprint,
            "config_fingerprint": config_fingerprint,
            "model_config_digest": _digest(self._config),
            "protocol": protocol,
        }
        if resume_from is not None:
            restored = type(self).load_training(resume_from)
            if _digest(restored._config) != _digest(self._config):
                raise ValueError("resume model configuration mismatch")
            if restored._training_state["signature"] != signature:
                raise ValueError("resume training data or configuration fingerprint mismatch")
            self.load_state_dict(restored.state_dict(), strict=True)
            self._training_state = restored._training_state
        if hasattr(self, "_training_state"):
            if self._training_state["signature"] != signature:
                raise ValueError("resume training data or configuration fingerprint mismatch")
            state = self._training_state
        else:
            state = {
                "signature": signature,
                "warmup_done": 0,
                "epochs_done": 0,
                "curve": [],
                "optimizers": {},
            }
        generator = torch.Generator(device=x.device).manual_seed(self.seed)
        if "generator" in state:
            generator.set_state(state["generator"].cpu())
        return state, generator

    def _finish_training(self, state, generator, optimizers, checkpoint_path):
        state["generator"] = generator.get_state()
        state["optimizers"] = {
            name: optimizer.state_dict() for name, optimizer in optimizers.items()
        }
        self._training_state = copy.deepcopy(state)
        self.eval()
        if checkpoint_path is not None:
            self.save_training(checkpoint_path)

    def _future(self, future_loads, target, batch):
        value = np.asarray(future_loads, dtype=np.float32)
        if (
            value.ndim != 3
            or value.shape[0] != batch
            or value.shape[1] < 1
            or value.shape[2] != self.capacities.numel()
            or not np.isfinite(value).all()
            or np.any(value < 0)
        ):
            raise ValueError("future_loads must be finite nonnegative [batch,future_time>=1,nodes]")
        if not np.array_equal(value[:, 0], target.detach().cpu().numpy()):
            raise ValueError("target must equal the first observed future_loads step")
        return torch.as_tensor(value, dtype=torch.float32, device=self.capacities.device)


class CCRLModel(_HistoryModel):
    """公式 13–23 的因果权重模型，支持离线细化与物理转移训练。

    ``fit`` 的目标是下一时刻的原始负载，仅用于损失和奖励计算。
    ``weights`` 和 ``predict`` 只接收历史观测。
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        capacities: np.ndarray,
        impedance: np.ndarray,
        *,
        hidden: int = 16,
        layers: int = 2,
        kernel_size: int = 3,
        lr: float = 0.001,
        gamma: float = 0.9,
        action_scale: float = 0.02,
        max_weight: float = 1.0,
        use_rl: bool = True,
        use_attention: bool = True,
        seed: int = 0,
        include_load_channel: bool = False,
        refinement_steps: int = 3,
        epsilon: float = 1e-8,
        actor_lr: float | None = None,
        critic_lr: float | None = None,
        optimizer: str = "adam",
        training_mode: str = "refinement",
        node_ids=None,
        relations=None,
    ):
        super().__init__()
        a, c, z = _validate_graph(adjacency, capacities, impedance)
        if (
            not np.isfinite([lr, action_scale, max_weight, gamma]).all()
            or min(lr, action_scale, max_weight) <= 0
            or not 0 <= gamma < 1
        ):
            raise ValueError("lr/action_scale/max_weight must be positive and gamma in [0,1)")
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and strictly positive")
        actor_lr = lr if actor_lr is None else actor_lr
        critic_lr = lr if critic_lr is None else critic_lr
        if not np.isfinite([actor_lr, critic_lr]).all() or actor_lr <= 0 or critic_lr <= 0:
            raise ValueError("actor_lr and critic_lr must be finite and strictly positive")
        if not isinstance(optimizer, str) or optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        if training_mode not in {"refinement", "transition"}:
            raise ValueError("training_mode must be 'refinement' or 'transition'")
        self.training_mode = training_mode
        refinement_steps = _integer(refinement_steps, "refinement_steps")
        seed = _integer(seed, "seed", minimum=0)
        self.node_ids = tuple(str(v) for v in (range(len(c)) if node_ids is None else node_ids))
        if len(self.node_ids) != len(c):
            raise ValueError("node_ids must have one entry per node")
        self.relation_plan = compile_relations(self.node_ids, relations)
        self.relations = {} if relations is None else copy.deepcopy(relations)
        self.register_buffer("adjacency", torch.as_tensor(a, dtype=torch.bool))
        self.register_buffer("capacities", torch.as_tensor(c))
        self.register_buffer("impedance", torch.as_tensor(z))
        edge_index = np.argwhere(a)
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))
        state_dim = max(len(edge_index), 1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = CausalTemporalEncoder(
                hidden,
                layers,
                kernel_size,
                2 if include_load_channel else 1,
            )
            self.query = nn.Linear(hidden, hidden, bias=False)
            self.key = nn.Linear(hidden, hidden, bias=False)
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, state_dim),
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
            # 初始策略均值为零，保留公式 17 的初始权重。
            nn.init.zeros_(self.actor[-1].weight)
            nn.init.zeros_(self.actor[-1].bias)
        self.actor_log_std = nn.Parameter(torch.full((state_dim,), -0.7))
        self.hidden = hidden
        self.lr = float(lr)
        self.actor_lr = float(actor_lr)
        self.critic_lr = float(critic_lr)
        self.optimizer = optimizer
        self.gamma = float(gamma)
        self.action_scale = float(action_scale)
        self.max_weight = float(max_weight)
        self.use_rl = bool(use_rl)
        self.use_attention = bool(use_attention)
        self.seed = int(seed)
        self.include_load_channel = bool(include_load_channel)
        self.refinement_steps = int(refinement_steps)
        self.epsilon = float(epsilon)
        self._config = dict(
            adjacency=a.astype(int).tolist(),
            capacities=c.tolist(),
            impedance=z.tolist(),
            hidden=hidden,
            layers=layers,
            kernel_size=kernel_size,
            lr=lr,
            gamma=gamma,
            action_scale=action_scale,
            max_weight=max_weight,
            use_rl=use_rl,
            use_attention=use_attention,
            seed=seed,
            include_load_channel=include_load_channel,
            refinement_steps=refinement_steps,
            epsilon=epsilon,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            optimizer=optimizer,
            training_mode=training_mode,
            node_ids=list(self.node_ids),
            relations=self.relations,
        )

    def _attention(self, history: Tensor) -> Tensor:
        if self.use_attention:
            embedding = self.encoder(self._signals(history))[:, -1]
            q, k = self.query(embedding), self.key(embedding)
            logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.hidden)
        else:
            b, _, n = history.shape
            logits = history.new_zeros((b, n, n))
        return masked_row_softmax(logits, self.adjacency)

    def _base_weights(self, history: Tensor) -> Tensor:
        # 公式 17 使用最后两个历史观测，预测目标不参与初始边权计算。
        observed_delta = (history[:, -1] - history[:, -2]).clamp_min(0.0)
        previous_source = history[:, -2].unsqueeze(-1)
        weights = (
            self._attention(history)
            * observed_delta.unsqueeze(-2)
            / (previous_source + self.epsilon)
        )
        return weights.clamp(0.0, self.max_weight) * self.adjacency

    def _edge_state(self, dense: Tensor) -> Tensor:
        if self.edge_index.shape[0] == 0:
            return dense.new_zeros((dense.shape[0], 1))
        return dense[:, self.edge_index[:, 0], self.edge_index[:, 1]]

    def _dense_state(self, edges: Tensor) -> Tensor:
        b = edges.shape[0]
        n = self.capacities.numel()
        dense = edges.new_zeros((b, n, n))
        if self.edge_index.shape[0]:
            dense[:, self.edge_index[:, 0], self.edge_index[:, 1]] = edges
        return dense

    def _policy(self, state: Tensor) -> torch.distributions.Normal:
        mean = self.action_scale * torch.tanh(self.actor(state))
        std = self.action_scale * self.actor_log_std.clamp(-5.0, 1.0).exp()
        return torch.distributions.Normal(mean, std.expand_as(mean))

    def _inference_weights(self, history: Tensor) -> Tensor:
        base = self._base_weights(history)
        if self.use_rl and self.edge_index.shape[0]:
            state = self._edge_state(base)
            for _ in range(self.refinement_steps if self.training_mode == "refinement" else 1):
                state = (state + self._policy(state).mean).clamp(0.0, self.max_weight)
            return self._dense_state(state)
        return base

    def weights(self, history: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self._inference_weights(self._history(history)).cpu().numpy()

    def predict(self, history: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            x = self._history(history)
            return self._propagate(x[:, -1], self._inference_weights(x)).cpu().numpy()

    def rollout(self, history: np.ndarray, *, steps: int) -> dict[str, np.ndarray]:
        """闭环传播，按周期重算权重并将预测负载追加到固定长度历史窗口。

        返回 ``loads[B,steps+1,N]``（含初始负载）和
        ``weights[B,steps,N,N]``。追加值为模型预测，不是新观测。
        """
        steps = _integer(steps, "steps")
        self.eval()
        with torch.no_grad():
            window = self._history(history)
            loads = [window[:, -1]]
            weights = []
            for _ in range(steps):
                current_weights = self._inference_weights(window)
                next_load = self._propagate(window[:, -1], current_weights)
                weights.append(current_weights)
                loads.append(next_load)
                window = torch.cat((window[:, 1:], next_load.unsqueeze(1)), dim=1)
            return {
                "loads": torch.stack(loads, dim=1).cpu().numpy(),
                "weights": torch.stack(weights, dim=1).cpu().numpy(),
            }

    def fit(
        self,
        history: np.ndarray,
        target: np.ndarray,
        *,
        epochs: int = 4,
        batch_size: int = 64,
        warmup_epochs: int | None = None,
        training_mode: str = "refinement",
        future_loads=None,
        checkpoint_path=None,
        resume_from=None,
        data_fingerprint=None,
        config_fingerprint=None,
    ) -> list[dict]:
        """执行预热及新增训练轮次；转移模式逐时更新预测历史并对照观测轨迹。"""
        epochs = _integer(epochs, "epochs")
        batch_size = _integer(batch_size, "batch_size")
        warmup_epochs = (
            epochs if warmup_epochs is None else _integer(warmup_epochs, "warmup_epochs", minimum=0)
        )
        if training_mode not in {"refinement", "transition"}:
            raise ValueError("training_mode must be 'refinement' or 'transition'")
        self.training_mode = training_mode
        self._config["training_mode"] = training_mode
        x = self._history(history)
        y = self._target(target, len(x))
        if training_mode == "transition" and future_loads is None:
            raise ValueError("transition training requires observed future_loads")
        future = None if future_loads is None else self._future(future_loads, y, len(x))
        protocol = {
            "batch_size": batch_size,
            "warmup_epochs": warmup_epochs,
            "training_mode": training_mode,
        }
        state, generator = self._start_training(
            x, y, future, protocol, resume_from, data_fingerprint, config_fingerprint
        )
        optimizer_class = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[self.optimizer]
        optimizers = {
            "features": optimizer_class(
                [*self.encoder.parameters(), *self.query.parameters(), *self.key.parameters()],
                lr=self.lr,
            ),
            "actor": optimizer_class(
                [*self.actor.parameters(), self.actor_log_std], lr=self.actor_lr
            ),
            "critic": optimizer_class(self.critic.parameters(), lr=self.critic_lr),
        }
        for name, saved in state["optimizers"].items():
            optimizers[name].load_state_dict(saved)
        self.train()
        curve = []
        for epoch in range(state["warmup_done"], warmup_epochs):
            losses = []
            for indices in torch.randperm(len(x), generator=generator, device=x.device).split(
                batch_size
            ):
                batch = x[indices]
                pred = self._propagate(batch[:, -1], self._base_weights(batch))
                loss = ((pred - y[indices]) / self.capacities).square().mean()
                if loss.requires_grad:
                    optimizers["features"].zero_grad(set_to_none=True)
                    loss.backward()
                    optimizers["features"].step()
                losses.append((float(loss.detach()), len(indices)))
            row = {
                "phase": "supervised_warmup",
                "epoch": epoch + 1,
                "loss": sum(v * n for v, n in losses) / len(x),
            }
            curve.append(row)
            state["warmup_done"] = epoch + 1
        if self.use_rl and self.edge_index.shape[0]:
            for epoch in range(state["epochs_done"], state["epochs_done"] + epochs):
                rewards, actor_losses, critic_losses = [], [], []
                horizon = (
                    self.refinement_steps if training_mode == "refinement" else future.shape[1]
                )
                for indices in torch.randperm(len(x), generator=generator, device=x.device).split(
                    batch_size
                ):
                    window = x[indices]
                    with torch.no_grad():
                        edge_state = self._edge_state(self._base_weights(window))
                    for step in range(horizon):
                        policy = self._policy(edge_state)
                        action = torch.normal(
                            policy.loc.detach(), policy.scale.detach(), generator=generator
                        )
                        next_edges = (edge_state + action).clamp(0, self.max_weight)
                        with torch.no_grad():
                            prediction = self._propagate(
                                window[:, -1], self._dense_state(next_edges)
                            )
                            observed = (
                                y[indices]
                                if training_mode == "refinement"
                                else future[indices, step]
                            )
                            reward = -(prediction - observed).square().sum(-1)
                            if training_mode == "transition":
                                next_window = torch.cat(
                                    (window[:, 1:], prediction.unsqueeze(1)), dim=1
                                )
                                following_state = self._edge_state(self._base_weights(next_window))
                            else:
                                next_window, following_state = window, next_edges
                            next_value = (
                                self.critic(following_state).squeeze(-1)
                                if step + 1 < horizon
                                else torch.zeros_like(reward)
                            )
                        value = self.critic(edge_state).squeeze(-1)
                        td_error = reward + self.gamma * next_value - value
                        critic_loss = 0.5 * td_error.square().mean()
                        actor_loss = -(policy.log_prob(action).sum(-1) * td_error.detach()).mean()
                        optimizers["critic"].zero_grad(set_to_none=True)
                        critic_loss.backward()
                        optimizers["critic"].step()
                        optimizers["actor"].zero_grad(set_to_none=True)
                        actor_loss.backward()
                        optimizers["actor"].step()
                        rewards.append((float(reward.mean()), len(indices)))
                        actor_losses.append((float(actor_loss.detach()), len(indices)))
                        critic_losses.append((float(critic_loss.detach()), len(indices)))
                        window, edge_state = next_window.detach(), following_state.detach()

                def average(values):
                    return sum(v * n for v, n in values) / (len(x) * horizon)

                row = {
                    "phase": (
                        "actor_critic"
                        if training_mode == "refinement"
                        else "actor_critic_transition"
                    ),
                    "epoch": epoch + 1,
                    "reward": average(rewards),
                    "actor_loss": average(actor_losses),
                    "critic_loss": average(critic_losses),
                }
                curve.append(row)
                state["epochs_done"] = epoch + 1
        state["curve"].extend(curve)
        self._finish_training(state, generator, optimizers, checkpoint_path)
        return curve

    def save(self, path: str | Path) -> None:
        self._checkpoint(path, self._config)

    @classmethod
    def load(cls, path: str | Path) -> "CCRLModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 3:
            raise ValueError("unsupported checkpoint version; use a version 3 checkpoint")
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        return model


class TemporalGraphBaseline(_HistoryModel):
    """因果 TCN、入邻居均值聚合及节点边强度组成的时序图传播对照。"""

    def __init__(
        self,
        adjacency: np.ndarray,
        capacities: np.ndarray,
        impedance: np.ndarray | None = None,
        *,
        hidden: int = 16,
        layers: int = 2,
        kernel_size: int = 3,
        lr: float = 0.001,
        seed: int = 0,
        include_load_channel: bool = True,
        node_ids=None,
        relations=None,
    ):
        super().__init__()
        if impedance is None:
            impedance = np.zeros(len(capacities), dtype=np.float32)
        a, c, z = _validate_graph(adjacency, capacities, impedance)
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be positive")
        seed = _integer(seed, "seed", minimum=0)
        self.node_ids = tuple(str(v) for v in (range(len(c)) if node_ids is None else node_ids))
        if len(self.node_ids) != len(c):
            raise ValueError("node_ids must have one entry per node")
        self.relation_plan = compile_relations(self.node_ids, relations)
        self.relations = {} if relations is None else copy.deepcopy(relations)
        self.register_buffer("adjacency", torch.as_tensor(a, dtype=torch.bool))
        self.register_buffer("capacities", torch.as_tensor(c))
        self.register_buffer("impedance", torch.as_tensor(z))
        self.include_load_channel = bool(include_load_channel)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = CausalTemporalEncoder(
                hidden,
                layers,
                kernel_size,
                2 if include_load_channel else 1,
            )
            self.regressor = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )
        self.lr = float(lr)
        self.seed = int(seed)
        self._config = dict(
            adjacency=a.astype(int).tolist(),
            capacities=c.tolist(),
            impedance=z.tolist(),
            hidden=hidden,
            layers=layers,
            kernel_size=kernel_size,
            lr=lr,
            seed=seed,
            include_load_channel=include_load_channel,
            node_ids=list(self.node_ids),
            relations=self.relations,
        )

    def _weights(self, history: Tensor) -> Tensor:
        h = self.encoder(self._signals(history))[:, -1]
        incoming = self.adjacency.T.to(h.dtype)
        incoming = incoming / incoming.sum(dim=-1, keepdim=True).clamp_min(1)
        neighbors = torch.einsum("ji,bip->bjp", incoming, h)
        destination_strength = torch.sigmoid(
            self.regressor(torch.cat((h, neighbors), dim=-1)).squeeze(-1)
        )
        degree = self.adjacency.sum(-1, keepdim=True).clamp_min(1)
        return self.adjacency.to(h.dtype).unsqueeze(0) * destination_strength.unsqueeze(-2) / degree

    def _forward(self, history: Tensor) -> Tensor:
        return self._propagate(history[:, -1], self._weights(history))

    def weights(self, history: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self._weights(self._history(history)).cpu().numpy()

    def rollout(self, history: np.ndarray, *, steps: int) -> dict[str, np.ndarray]:
        steps = _integer(steps, "steps")
        self.eval()
        with torch.no_grad():
            window = self._history(history)
            loads, weights = [window[:, -1]], []
            for _ in range(steps):
                w = self._weights(window)
                prediction = self._propagate(window[:, -1], w)
                weights.append(w)
                loads.append(prediction)
                window = torch.cat((window[:, 1:], prediction.unsqueeze(1)), dim=1)
            return {
                "loads": torch.stack(loads, dim=1).cpu().numpy(),
                "weights": torch.stack(weights, dim=1).cpu().numpy(),
            }

    def fit(
        self,
        history: np.ndarray,
        target: np.ndarray,
        *,
        epochs: int = 4,
        batch_size: int = 64,
        training_mode: str = "refinement",
        future_loads=None,
        checkpoint_path=None,
        resume_from=None,
        data_fingerprint=None,
        config_fingerprint=None,
    ) -> list[dict]:
        """监督学习单步或闭环轨迹负载，优化器和随机顺序支持精确续训。"""
        epochs, batch_size = _integer(epochs, "epochs"), _integer(batch_size, "batch_size")
        if training_mode not in {"refinement", "transition"}:
            raise ValueError("training_mode must be 'refinement' or 'transition'")
        x, y = self._history(history), self._target(target, len(history))
        if training_mode == "transition" and future_loads is None:
            raise ValueError("transition training requires observed future_loads")
        future = None if future_loads is None else self._future(future_loads, y, len(x))
        protocol = {"batch_size": batch_size, "training_mode": training_mode}
        state, generator = self._start_training(
            x, y, future, protocol, resume_from, data_fingerprint, config_fingerprint
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if state["optimizers"]:
            optimizer.load_state_dict(state["optimizers"]["supervised"])
        self.train()
        curve = []
        for epoch in range(state["epochs_done"], state["epochs_done"] + epochs):
            losses = []
            for indices in torch.randperm(len(x), generator=generator, device=x.device).split(
                batch_size
            ):
                window = x[indices]
                horizon = 1 if training_mode == "refinement" else future.shape[1]
                loss = window.new_zeros(())
                for step in range(horizon):
                    prediction = self._forward(window)
                    observed = (
                        y[indices] if training_mode == "refinement" else future[indices, step]
                    )
                    loss = (
                        loss + ((prediction - observed) / self.capacities).square().mean() / horizon
                    )
                    window = torch.cat((window[:, 1:], prediction.unsqueeze(1)), dim=1)
                if loss.requires_grad:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                losses.append((float(loss.detach()), len(indices)))
            curve.append(
                {
                    "phase": "supervised",
                    "epoch": epoch + 1,
                    "loss": sum(v * n for v, n in losses) / len(x),
                }
            )
            state["epochs_done"] = epoch + 1
        state["curve"].extend(curve)
        self._finish_training(state, generator, {"supervised": optimizer}, checkpoint_path)
        return curve

    def predict(self, history: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            return self._forward(self._history(history)).cpu().numpy()

    def save(self, path: str | Path) -> None:
        self._checkpoint(path, self._config)

    @classmethod
    def load(cls, path: str | Path) -> "TemporalGraphBaseline":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 3:
            raise ValueError("unsupported checkpoint version; use a version 3 checkpoint")
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        return model
