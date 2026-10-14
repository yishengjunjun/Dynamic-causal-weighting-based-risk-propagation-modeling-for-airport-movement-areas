import numpy as np
import pytest

from airport_risk.baselines import BinaryDBNBaseline, StaticPropagationModel, VARBaseline


def test_dbn_cpt_laplace_counts_by_hand():
    adjacency = np.array([[0, 1], [0, 0]])
    model = BinaryDBNBaseline(adjacency, [1, 1], max_parents=1, smoothing=1)
    previous = np.array([[0, 0], [0, 0], [2, 0], [2, 0]], dtype=float)
    history = np.repeat(previous[:, None, :], 2, axis=1)
    target = np.array([[0.2, 0.2], [0.3, 1.5], [2.0, 0.3], [2.2, 0.4]])
    model.fit(history, target)

    # 节点 1 的父变量为［自身上一状态，节点 0 上一状态］。
    # 配置 00：一次正常、一次失效，平滑概率 p=2/4。
    # 配置 10（索引 2）：两次正常，平滑概率 p=1/4。
    assert model.parents_[1].tolist() == [1, 0]
    np.testing.assert_array_equal(model.counts_[1], [[1, 1], [0, 0], [2, 0], [0, 0]])
    np.testing.assert_allclose(model.cpts_[1][:, 1], [0.5, 0.5, 0.25, 0.5])

    query = np.array([[[0, 0]], [[2, 0]], [[0, 2]], [[2, 2]]], dtype=float)
    np.testing.assert_allclose(model.predict_proba(query)[:, 1], [0.5, 0.25, 0.5, 0.5])
    # 状态均值：正常为 .3，失效为 1.5；连续输出按概率混合训练均值。
    np.testing.assert_allclose(model.state_means_[1], [0.3, 1.5])
    np.testing.assert_allclose(model.predict(query)[:, 1], [0.9, 0.6, 0.9, 0.9])


def test_dbn_threshold_equality_and_absent_state_fallback():
    model = BinaryDBNBaseline(np.zeros((2, 2)), [1, 1], max_parents=0)
    history = np.array([[[1, 1]], [[1, 1]]], dtype=float)
    target = np.array([[1, 0.2], [1, 0.4]])
    model.fit(history, target)
    # 等于容量也算失效；节点 0 的目标全失效，因此 p=(2+1)/(2+2)。
    np.testing.assert_allclose(model.predict_proba(history), [[0.75, 0.25], [0.75, 0.25]])
    np.testing.assert_allclose(model.state_means_[0], [1, 1])
    np.testing.assert_allclose(model.state_means_[1], [0.3, 0.3])
    assert np.isfinite(model.predict(history)).all()


def test_dbn_cross_slice_graph_allows_spatial_cycles_and_limits_parents():
    graph = np.ones((4, 4)) - np.eye(4)
    model = BinaryDBNBaseline(graph, np.ones(4), max_parents=2)
    assert all(len(parents) == 3 for parents in model.parents_)
    assert model.parents_[3].tolist() == [3, 0, 1]
    h = np.ones((3, 2, 4)) * 0.5
    model.fit(h, np.ones((3, 4)) * 0.8)
    assert model.predict_proba(h).shape == (3, 4)


def test_var_recovers_known_multivariate_lag_one_mapping():
    rng = np.random.default_rng(20)
    x = rng.uniform(0.1, 2.0, (80, 2))
    a = np.array([[0.4, 0.2], [0.1, 0.7]])
    bias = np.array([0.2, 0.5])
    history = np.stack([rng.uniform(0, 1, x.shape), x], axis=1)
    target = x @ a + bias
    model = VARBaseline(lag=1, ridge=0)
    model.fit(history, target)
    query_x = np.array([[3.0, 1.2], [0.2, 4.0]])
    query = np.stack([np.zeros_like(query_x), query_x], axis=1)
    # 拟合和预测均不接收查询样本的目标值。
    np.testing.assert_allclose(model.predict(query), query_x @ a + bias, atol=1e-11)
    np.testing.assert_allclose(model.x_mean_, x.mean(axis=0))
    frozen_mean = model.x_mean_.copy()
    model.predict(query * 100)
    np.testing.assert_array_equal(model.x_mean_, frozen_mean)


def test_var_lag_two_preserves_chronological_feature_order():
    rng = np.random.default_rng(9)
    h = rng.normal(size=(50, 3, 2))
    coefficients = np.array([[0.4, 0.1], [-0.3, 0.7], [0.8, -0.2], [0.1, 0.5]])
    y = h[:, -2:, :].reshape(50, 4) @ coefficients + [0.3, -0.4]
    model = VARBaseline(lag=2, ridge=0)
    model.fit(h, y)
    query = rng.normal(size=(5, 3, 2))
    expected = query[:, -2:, :].reshape(5, 4) @ coefficients + [0.3, -0.4]
    np.testing.assert_allclose(model.predict(query), expected, atol=1e-11)


def test_var_ridge_handles_constant_and_collinear_features():
    history = np.ones((5, 2, 3))
    target = np.array([[0.2, 2, 4]] * 5)
    model = VARBaseline(ridge=0.01)
    model.fit(history, target)
    np.testing.assert_allclose(model.predict(history), target)
    assert np.isfinite(model.coef_).all()


def test_static_weights_use_training_history_and_source_destination_orientation():
    graph = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    h = np.array([[[1, 1, 1], [2, 3, 5], [4, 6, 4]]], dtype=float)
    model = StaticPropagationModel(graph, [1, 1, 1], [False] * 3, epsilon=0.01)
    model.fit(h, np.array([[5, 7, 6]], dtype=float))
    expected = np.zeros((3, 3))
    expected[0, 1] = 0.5 * (2 / 1.01 + 3 / 2.01) / 2
    expected[0, 2] = 0.5 * (4 / 1.01 + 0 / 2.01) / 2
    expected[1, 2] = (4 / 1.01 + 0 / 3.01) / 2
    np.testing.assert_allclose(model.weights_, expected)
    # 静态权重仅依赖历史 h，修改拟合目标不得改变这些权重。
    model.fit(h, np.array([[100, 10, 1]], dtype=float))
    np.testing.assert_allclose(model.weights_, expected)
    assert model.predict(h).shape == (1, 3)


def test_predict_requires_fit_and_invalid_inputs_raise():
    h = np.ones((2, 2, 2))
    for model in (
        VARBaseline(),
        BinaryDBNBaseline(np.zeros((2, 2)), [1, 1]),
        StaticPropagationModel(np.zeros((2, 2)), [1, 1], [False, False]),
    ):
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(h)
    with pytest.raises(ValueError, match="lag"):
        VARBaseline(lag=0)
    with pytest.raises(ValueError, match="smoothing"):
        BinaryDBNBaseline(np.zeros((2, 2)), [1, 1], smoothing=0)
    with pytest.raises(ValueError, match="NaN"):
        VARBaseline().fit(h * np.nan, np.ones((2, 2)))
    with pytest.raises(ValueError, match="target"):
        VARBaseline().fit(h, np.ones((2, 3)))


@pytest.mark.parametrize("impedance", [[0, 2], [0, np.nan], [0, -1]])
def test_static_impedance_rejects_nonbinary_values(impedance):
    with pytest.raises(ValueError, match="impedance.*binary"):
        StaticPropagationModel(np.zeros((2, 2)), np.ones(2), impedance)


def test_boolean_hyperparameters_do_not_silently_become_integer_orders():
    with pytest.raises(ValueError, match="lag"):
        VARBaseline(lag=True)
    with pytest.raises(ValueError, match="max_parents"):
        BinaryDBNBaseline(np.zeros((2, 2)), np.ones(2), max_parents=True)


def test_static_relation_plan_and_rollout_are_used_in_main_predictions():
    graph = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    history = np.array([[[1, 1, 0.1], [1.4, 1.2, 0.2]], [[1, 1, 0.2], [1.3, 1.5, 0.4]]])
    model = StaticPropagationModel(
        graph,
        np.ones(3),
        [0, 0, 1],
        node_ids=["A", "B", "C"],
        relations={"mutex_groups": [["A", "B"]]},
    )
    model.fit(history, history[:, -1])
    weights = model.weights(history)
    expected = model.propagate_numpy(history[:, -1], weights)
    np.testing.assert_array_equal(model.predict(history), expected)
    np.testing.assert_array_equal(model.rollout(history, steps=1)["loads"][:, 1], expected)
    # 每个样本仅保留相对负载较高的互斥源。
    assert expected[0, 2] < history[0, -1, 2] + weights[0, :, 2].sum()
