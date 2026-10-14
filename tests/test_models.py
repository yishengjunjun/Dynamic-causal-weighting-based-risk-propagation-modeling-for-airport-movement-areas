"""CCRL 的数值计算、因果性和训练行为测试。"""

import inspect

import numpy as np
import pytest
import torch

from airport_risk.models import (
    CCRLModel,
    CausalConv1d,
    TemporalGraphBaseline,
    masked_row_softmax,
    propagate_torch,
)


@pytest.fixture
def example():
    a = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    c = np.array([1.0, 0.8, 1.1, 0.7], dtype=np.float32)
    z = np.array([0, 0, 1, 1], dtype=np.float32)
    rng = np.random.default_rng(18)
    history = rng.uniform(0.05, 1.2, size=(24, 7, 4)).astype(np.float32) * c
    # 一个失效源节点与后继节点的不同时间模式共同产生实际损失。
    history[:, -1, 0] = 1.15
    fixed = np.where(a, 0.15, 0.0).astype(np.float32)
    with torch.no_grad():
        target = propagate_torch(
            torch.tensor(history[:, -1]), torch.tensor(fixed), torch.tensor(c), torch.tensor(z)
        ).numpy()
    return a, c, z, history, target


def test_causal_convolution_matches_hand_sum():
    conv = CausalConv1d(1, 1, kernel_size=3, dilation=2)
    with torch.no_grad():
        conv.conv.weight.fill_(1.0)
        conv.conv.bias.zero_()
    x = torch.arange(1.0, 8.0).reshape(1, 1, 7)
    result = conv(x).detach().numpy().ravel()
    # 手算 x[t] + x[t-2] + x[t-4]，不存在的历史位置补零。
    np.testing.assert_array_equal(result, [1, 2, 4, 6, 9, 12, 15])


def test_equation13_asymmetric_kernel_indexes_observed_lags():
    conv = CausalConv1d(1, 1, kernel_size=3, dilation=2)
    with torch.no_grad():
        conv.conv.weight.copy_(torch.tensor([[[1.0, 10.0, 100.0]]]))
        conv.conv.bias.zero_()
    x = torch.arange(1.0, 8.0).reshape(1, 1, 7)
    result = conv(x).detach().numpy().ravel()
    np.testing.assert_array_equal(result, [1, 2, 13, 24, 135, 246, 357])


def test_encoder_early_features_ignore_future_observations(example):
    a, c, z, history, _ = example
    model = CCRLModel(a, c, z, seed=8)
    changed = history.copy()
    changed[:, 4:] = 5 * c
    original_features = model.encode(history)
    changed_features = model.encode(changed)
    # CPU 卷积后端在不同完整输入上可能相差一个 float32 最小精度单位。
    np.testing.assert_allclose(
        original_features[:, :4], changed_features[:, :4], atol=2e-7, rtol=2e-6
    )
    assert not np.array_equal(original_features[:, 4:], changed_features[:, 4:])
    signals = torch.rand((2, 7, 4, 1), requires_grad=True)
    model.encoder(signals)[:, :4].sum().backward()
    assert torch.count_nonzero(signals.grad[:, 4:]).item() == 0


def test_masked_softmax_has_zero_rows_and_only_existing_edges():
    a = torch.tensor([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=torch.bool)
    logits = torch.tensor([[[100.0, 0.0, 0.0], [5.0, 8.0, -2.0], [9.0, 9.0, 9.0]]])
    attention = masked_row_softmax(logits, a).numpy()[0]
    np.testing.assert_array_equal(attention, [[0, 0.5, 0.5], [0, 0, 1], [0, 0, 0]])
    assert np.isfinite(attention).all()


def test_equation17_uses_observed_increment_and_previous_source_only():
    a = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    model = CCRLModel(a, np.ones(3), np.zeros(3), use_rl=False, use_attention=False)
    history = np.array([[[1.0, 0.1, 0.2], [1.2, 0.3, 0.4]]], dtype=np.float32)
    expected = np.array([[[0, 0.1, 0.1], [0, 0, 0], [0, 0, 0]]])
    np.testing.assert_allclose(model.weights(history), expected, atol=2e-8)
    # 只有源节点 0 失效，各目标节点增量为 W[0,dst] * C_dst。
    np.testing.assert_allclose(model.predict(history), [[1.2, 0.4, 0.5]], atol=5e-8)


def test_equation12_stable_form_matches_direct_expression():
    loads = torch.tensor([[0.2, 0.8, 1.5]], dtype=torch.float64)
    c = torch.tensor([0.3, 0.5, 1.0], dtype=torch.float64)
    z = torch.tensor([0, 1, 1], dtype=torch.float64)
    w = torch.tensor([[[0, 0.2, 0], [0, 0, 0.4], [0.1, 0, 0]]], dtype=torch.float64)
    theta = (1 - torch.exp(loads - c)) / (1 + torch.exp(loads))
    delta = ((loads >= c).unsqueeze(-1) * w).sum(-2) * c
    expected = loads + (1 - z * theta) * delta
    torch.testing.assert_close(propagate_torch(loads, w, c, z), expected)
    extreme = propagate_torch(loads * 1e4, w, c, z)
    assert torch.isfinite(extreme).all()


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_actor_critic_train_and_encoder_receives_supervised_gradient(example, optimizer):
    a, c, z, history, target = example
    model = CCRLModel(a, c, z, hidden=8, seed=3, optimizer=optimizer)
    before_actor = [p.detach().clone() for p in model.actor.parameters()]
    before_critic = [p.detach().clone() for p in model.critic.parameters()]
    before_features = [p.detach().clone() for p in model.encoder.parameters()]
    log = model.fit(history, target, epochs=2, batch_size=8)
    assert len(log) == 4
    assert {row["phase"] for row in log} == {"supervised_warmup", "actor_critic"}
    assert any(
        not torch.equal(old, new) for old, new in zip(before_actor, model.actor.parameters())
    )
    assert any(
        not torch.equal(old, new) for old, new in zip(before_critic, model.critic.parameters())
    )
    assert any(
        not torch.equal(old, new) for old, new in zip(before_features, model.encoder.parameters())
    )
    weights = model.weights(history)
    assert np.all(weights[:, ~a.astype(bool)] == 0)
    assert np.isfinite(weights).all()
    assert (weights >= 0).all() and (weights <= model.max_weight).all()


def test_inference_never_accepts_target_or_depends_on_external_target(example):
    a, c, z, history, target = example
    model = CCRLModel(a, c, z, seed=4)
    model.fit(history, target, epochs=1, batch_size=24)
    prediction = model.predict(history)
    target[:] = 12345
    np.testing.assert_array_equal(model.predict(history), prediction)
    assert list(inspect.signature(model.predict).parameters) == ["history"]
    assert list(inspect.signature(model.weights).parameters) == ["history"]
    np.testing.assert_array_equal(model.predict(history), model.predict(history))


def test_multistep_policy_repeated_mean_updates_match_manual(example):
    a, c, z, history, _ = example
    model = CCRLModel(a, c, z, refinement_steps=3, seed=7)
    with torch.no_grad():
        model.actor[-1].bias.fill_(0.3)
        x = model._history(history)
        state = model._edge_state(model._base_weights(x))
        step_delta = model.action_scale * np.tanh(0.3)
        expected = model._dense_state((state + 3 * step_delta).clamp(0, model.max_weight)).numpy()
    np.testing.assert_allclose(model.weights(history), expected, atol=2e-8)


def test_checkpoint_roundtrip_is_exact_and_safe(example, tmp_path):
    a, c, z, history, target = example
    model = CCRLModel(
        a,
        c,
        z,
        hidden=8,
        include_load_channel=True,
        refinement_steps=2,
        epsilon=1e-5,
        actor_lr=0.002,
        critic_lr=0.003,
        optimizer="sgd",
        seed=9,
    )
    model.fit(history, target, epochs=1)
    destination = tmp_path / "model.pt"
    model.save(destination)
    checkpoint = torch.load(destination, weights_only=True)
    assert set(checkpoint) == {"format_version", "config", "state_dict"}
    assert checkpoint["format_version"] == 3
    restored = CCRLModel.load(destination)
    assert restored.include_load_channel and restored.refinement_steps == 2
    assert restored.epsilon == 1e-5
    assert restored.actor_lr == 0.002 and restored.critic_lr == 0.003
    assert restored.optimizer == "sgd"
    np.testing.assert_array_equal(restored.predict(history), model.predict(history))
    np.testing.assert_array_equal(restored.weights(history), model.weights(history))


def test_training_seed_isolated_from_ambient_random_state_and_gamma_matters(example):
    a, c, z, history, target = example
    first = CCRLModel(a, c, z, hidden=8, gamma=0.9, seed=11)
    second = CCRLModel(a, c, z, hidden=8, gamma=0.9, seed=11)
    no_bootstrap = CCRLModel(a, c, z, hidden=8, gamma=0.0, seed=11)
    torch.manual_seed(11111)
    first_log = first.fit(history, target, epochs=2, batch_size=8)
    torch.manual_seed(22222)
    second_log = second.fit(history, target, epochs=2, batch_size=8)
    assert first_log == second_log
    np.testing.assert_array_equal(first.predict(history), second.predict(history))
    no_bootstrap.fit(history, target, epochs=2, batch_size=8)
    assert any(
        not torch.equal(p, q)
        for p, q in zip(first.critic.parameters(), no_bootstrap.critic.parameters())
    )


def test_no_edges_is_finite_and_exact_persistence():
    history = np.array([[[0.3, 1.2], [0.5, 1.3]]], dtype=np.float32)
    model = CCRLModel(np.zeros((2, 2)), np.ones(2), np.zeros(2))
    model.fit(history, np.array([[0.6, 1.4]]), epochs=1)
    np.testing.assert_array_equal(model.weights(history), np.zeros((1, 2, 2)))
    np.testing.assert_array_equal(model.predict(history), history[:, -1])


def test_explicit_zero_warmup_freezes_temporal_parameters(example):
    a, c, z, history, target = example
    model = CCRLModel(a, c, z, hidden=8, seed=13)
    before = [p.detach().clone() for p in model.encoder.parameters()]
    log = model.fit(history, target, epochs=1, warmup_epochs=0)
    assert [row["phase"] for row in log] == ["actor_critic"]
    assert all(torch.equal(old, new) for old, new in zip(before, model.encoder.parameters()))


def test_explicit_epsilon_controls_equation17_denominator():
    a = np.array([[0, 1], [0, 0]])
    model = CCRLModel(
        a,
        np.ones(2),
        np.zeros(2),
        use_attention=False,
        use_rl=False,
        epsilon=0.125,
    )
    history = np.array([[[0.5, 0.1], [1.2, 0.35]]], dtype=np.float32)
    np.testing.assert_allclose(model.weights(history), [[[0, 0.4], [0, 0]]])


def test_closed_loop_rollout_recomputes_weights_and_rolls_observed_window(monkeypatch):
    a = np.array([[0, 1], [0, 0]])
    model = CCRLModel(
        a,
        np.ones(2),
        np.zeros(2),
        use_attention=False,
        use_rl=False,
        epsilon=0.1,
        max_weight=10,
    )
    history = np.array([[[0.2, 0.05], [0.5, 0.1], [1.5, 0.6]]], dtype=np.float32)
    original = history.copy()
    first_prediction = model.predict(history)
    observed_windows = []
    attention = model._attention

    def capture(window):
        observed_windows.append(window.detach().cpu().numpy().copy())
        return attention(window)

    monkeypatch.setattr(model, "_attention", capture)
    result = model.rollout(history, steps=2)
    assert result["loads"].shape == (1, 3, 2)
    assert result["weights"].shape == (1, 2, 2, 2)
    np.testing.assert_array_equal(result["loads"][:, 0], original[:, -1])
    np.testing.assert_array_equal(result["loads"][:, 1], first_prediction)
    np.testing.assert_allclose(result["weights"][0, :, 0, 1], [5 / 6, 25 / 48])
    np.testing.assert_allclose(result["loads"][0, :, 1], [0.6, 0.6 + 5 / 6, 0.6 + 5 / 6 + 25 / 48])
    np.testing.assert_array_equal(observed_windows[0], original)
    expected_second_window = np.concatenate((original[:, 1:], first_prediction[:, None]), axis=1)
    np.testing.assert_array_equal(observed_windows[1], expected_second_window)
    np.testing.assert_array_equal(history, original)
    assert list(inspect.signature(model.rollout).parameters) == ["history", "steps"]


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_rollout_requires_positive_integer_steps(example, steps):
    a, c, z, history, _ = example
    model = CCRLModel(a, c, z)
    with pytest.raises(ValueError, match="steps"):
        model.rollout(history, steps=steps)


@pytest.mark.parametrize("model_class", [CCRLModel, TemporalGraphBaseline])
def test_model_construction_preserves_ambient_cpu_rng_state(model_class, example):
    a, c, z, _, _ = example
    torch.manual_seed(2048)
    state = torch.random.get_rng_state().clone()
    model_class(a, c, z, seed=7)
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("model_class", [CCRLModel, TemporalGraphBaseline])
@pytest.mark.parametrize("adjacency", [np.array([[0, 0.5], [0, 0]]), np.eye(2)])
def test_graph_mask_rejects_weights_and_self_loops(model_class, adjacency):
    with pytest.raises(ValueError, match="binary|zero diagonal"):
        model_class(adjacency, np.ones(2), np.zeros(2))


def test_temporal_graph_baseline_can_fit_and_reload(example, tmp_path):
    a, c, z, history, target = example
    model = TemporalGraphBaseline(a, c, z, hidden=8, seed=19)
    before = model.predict(history)
    log = model.fit(history, target, epochs=2, batch_size=8)
    after = model.predict(history)
    assert len(log) == 2
    assert not np.array_equal(before, after)
    assert np.isfinite(after).all() and (after >= 0).all()
    model.save(tmp_path / "tgn.pt")
    restored = TemporalGraphBaseline.load(tmp_path / "tgn.pt")
    np.testing.assert_array_equal(restored.predict(history), after)


@pytest.mark.parametrize("model_class", [CCRLModel, TemporalGraphBaseline])
def test_legacy_checkpoint_requires_explicit_kernel_migration(model_class, example, tmp_path):
    a, c, z, _, _ = example
    model = model_class(a, c, z)
    destination = tmp_path / "legacy.pt"
    model.save(destination)
    checkpoint = torch.load(destination, weights_only=True)
    checkpoint["format_version"] = 1
    torch.save(checkpoint, destination)
    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        model_class.load(destination)


def test_invalid_inputs_fail_explicitly(example):
    a, c, z, history, target = example
    with pytest.raises(ValueError, match="positive"):
        CCRLModel(a, np.zeros_like(c), z)
    model = CCRLModel(a, c, z)
    with pytest.raises(ValueError, match="observed_time"):
        model.predict(history[:, :1])
    with pytest.raises(ValueError, match="nonnegative"):
        model.predict(-history)
    with pytest.raises(ValueError, match="target"):
        model.fit(history, target[:, :1])
    with pytest.raises(ValueError, match="warmup_epochs"):
        model.fit(history, target, warmup_epochs=-1)
    with pytest.raises(ValueError, match="epochs"):
        model.fit(history, target, epochs=1.5)
    with pytest.raises(ValueError, match="refinement_steps"):
        CCRLModel(a, c, z, refinement_steps=1.5)
    with pytest.raises(ValueError, match="epsilon"):
        CCRLModel(a, c, z, epsilon=0)
    with pytest.raises(ValueError, match="actor_lr"):
        CCRLModel(a, c, z, actor_lr=0)
    with pytest.raises(ValueError, match="optimizer"):
        CCRLModel(a, c, z, optimizer="unverified")


@pytest.mark.parametrize("mode", ["refinement", "transition"])
def test_exact_actor_critic_resume_restores_optimizer_and_rng(example, tmp_path, mode):
    a, c, z, history, target = example
    future = np.stack((target, target + 0.05), axis=1) if mode == "transition" else None
    full = CCRLModel(a, c, z, hidden=8, seed=29, include_load_channel=True)
    split = CCRLModel(a, c, z, hidden=8, seed=29, include_load_channel=True)
    kwargs = dict(
        batch_size=8,
        warmup_epochs=2,
        training_mode=mode,
        future_loads=future,
        data_fingerprint="dataset-one",
        config_fingerprint="config-one",
    )
    full_curve = full.fit(history, target, epochs=4, **kwargs)
    first_curve = split.fit(
        history, target, epochs=2, checkpoint_path=tmp_path / "train.pt", **kwargs
    )
    stored = torch.load(tmp_path / "train.pt", weights_only=True)
    assert stored["training_state"]["epochs_done"] == 2
    assert stored["training_state"]["optimizers"]["actor"]["state"]
    torch.manual_seed(83222)
    resumed = CCRLModel.load_training(tmp_path / "train.pt")
    last_curve = resumed.fit(history, target, epochs=2, **kwargs)
    assert full_curve == first_curve + last_curve
    for key, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key
    np.testing.assert_array_equal(full.predict(history), resumed.predict(history))
    resumed.save(tmp_path / "infer.pt")
    np.testing.assert_array_equal(
        CCRLModel.load(tmp_path / "infer.pt").predict(history), resumed.predict(history)
    )


def test_resume_rejects_changed_data_configuration_and_external_fingerprint(example, tmp_path):
    a, c, z, history, target = example
    model = CCRLModel(a, c, z, hidden=8, seed=30)
    model.fit(
        history,
        target,
        epochs=1,
        warmup_epochs=1,
        batch_size=8,
        checkpoint_path=tmp_path / "resume.pt",
        data_fingerprint="one",
    )
    for changes in ({"history": history + 0.001}, {"data_fingerprint": "two"}, {"batch_size": 12}):
        loaded = CCRLModel.load_training(tmp_path / "resume.pt")
        kwargs = dict(
            history=history,
            target=target,
            epochs=1,
            warmup_epochs=1,
            batch_size=8,
            data_fingerprint="one",
        )
        kwargs.update(changes)
        with pytest.raises(ValueError, match="fingerprint"):
            loaded.fit(**kwargs)
    changed_model = CCRLModel(a, c, z, hidden=8, seed=31)
    with pytest.raises(ValueError, match="configuration"):
        changed_model.fit(
            history,
            target,
            epochs=1,
            warmup_epochs=1,
            batch_size=8,
            resume_from=tmp_path / "resume.pt",
            data_fingerprint="one",
        )


def test_transition_updates_physical_history_and_uses_trajectory_labels(example, monkeypatch):
    a, c, z, history, target = example
    model = CCRLModel(a, c, z, hidden=8, seed=35)
    seen = []
    original = model._base_weights

    def capture(window):
        seen.append(window.detach().numpy().copy())
        return original(window)

    monkeypatch.setattr(model, "_base_weights", capture)
    future = np.stack((target, target + 0.2), axis=1)
    rows = model.fit(
        history,
        target,
        epochs=1,
        batch_size=len(history),
        warmup_epochs=0,
        training_mode="transition",
        future_loads=future,
    )
    assert rows[0]["phase"] == "actor_critic_transition"
    # 第二次策略状态由上一步物理预测历史计算，历史窗口不保持不变。
    np.testing.assert_array_equal(seen[1][:, :-1], seen[0][:, 1:])
    assert not np.array_equal(seen[1][:, -1], seen[0][:, -1])
    assert not np.array_equal(seen[1][:, -1], future[:, 0])
    with pytest.raises(ValueError, match="future_loads"):
        CCRLModel(a, c, z).fit(history, target, training_mode="transition")
    invalid = future.copy()
    invalid[:, 0] += 0.1
    with pytest.raises(ValueError, match="target must equal"):
        CCRLModel(a, c, z).fit(history, target, training_mode="transition", future_loads=invalid)


def test_temporal_resume_is_bitwise_exact(example, tmp_path):
    a, c, z, history, target = example
    full = TemporalGraphBaseline(a, c, z, hidden=8, seed=36)
    split = TemporalGraphBaseline(a, c, z, hidden=8, seed=36)
    first = full.fit(history, target, epochs=4, batch_size=8)
    left = split.fit(
        history, target, epochs=2, batch_size=8, checkpoint_path=tmp_path / "tgn-train.pt"
    )
    resumed = TemporalGraphBaseline.load_training(tmp_path / "tgn-train.pt")
    right = resumed.fit(history, target, epochs=2, batch_size=8)
    assert first == left + right
    for key, value in full.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key]), key


@pytest.mark.parametrize("model_class", [CCRLModel, TemporalGraphBaseline])
def test_model_predict_and_rollout_use_same_relation_operator(model_class):
    a = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    c, z = np.ones(3), np.array([0, 0, 1])
    relations = {
        "mutex_groups": [["A", "B"]],
        "context": {"C": {"years_service": 10, "workload": 0.8, "working_hours": 5}},
    }
    model = model_class(a, c, z, node_ids=["A", "B", "C"], relations=relations)
    h = np.array([[[1.1, 1.1, 0.1], [1.5, 1.2, 0.2]]], dtype=np.float32)
    expected = model.propagate_numpy(h[:, -1], model.weights(h))
    np.testing.assert_allclose(model.predict(h), expected, atol=1e-7)
    np.testing.assert_allclose(model.rollout(h, steps=1)["loads"][:, 1], expected, atol=1e-7)
    override = model.propagate_numpy(h[:, -1], model.weights(h), capacities=c * 0.9)
    assert not np.array_equal(override, expected)


def test_temporal_zero_edge_graph_has_finite_observed_loss_and_persistence():
    history = np.array([[[0.2, 1.1], [0.4, 1.2]], [[0.3, 0.1], [0.5, 0.3]]], dtype=np.float32)
    target = history[:, -1] + 0.1
    model = TemporalGraphBaseline(np.zeros((2, 2)), np.ones(2), np.zeros(2))
    curve = model.fit(history, target, epochs=2, batch_size=1)
    assert len(curve) == 2 and all(np.isfinite(row["loss"]) and row["loss"] > 0 for row in curve)
    np.testing.assert_array_equal(model.predict(history), history[:, -1])
    np.testing.assert_array_equal(
        model.rollout(history, steps=2)["loads"][:, 1:], np.repeat(history[:, -1:, :], 2, axis=1)
    )


def test_fit_resume_from_matches_loaded_resume(example, tmp_path):
    a, c, z, history, target = example
    model = CCRLModel(a, c, z, hidden=8, seed=51)
    model.fit(
        history,
        target,
        epochs=1,
        warmup_epochs=1,
        batch_size=8,
        checkpoint_path=tmp_path / "resume.pt",
    )
    loaded = CCRLModel.load_training(tmp_path / "resume.pt")
    fresh = CCRLModel(a, c, z, hidden=8, seed=51)
    left = loaded.fit(history, target, epochs=1, warmup_epochs=1, batch_size=8)
    right = fresh.fit(
        history, target, epochs=1, warmup_epochs=1, batch_size=8, resume_from=tmp_path / "resume.pt"
    )
    assert left == right
    for key, value in fresh.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key]), key
