"""使用独立手算结果检查论文的数值核心。"""

import numpy as np
import pytest

from airport_risk.core import (
    capacities,
    global_metrics,
    grey_relational,
    impedance_theta,
    initial_load,
    ndi,
    nri,
    propagate_step,
    simulate,
    sri,
    topological_capacity,
)


def chain():
    return np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)


def diamond_tail():
    graph = np.zeros((5, 5))
    for source, target in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]:
        graph[source, target] = 1
    return graph


def test_grey_rms_is_centered_at_one_and_threshold_is_strict():
    values = np.array([[1, 2], [1, 3]], dtype=float)
    rho = 1 / (1 + np.sqrt(2.5) / 2)
    expected = np.array([1, (2 * rho / (1 + 2 * rho) + 2 * rho / (2 + 2 * rho)) / 2])
    grades, keep = grey_relational(values, np.ones(2), threshold=1)
    np.testing.assert_allclose(grades, expected)
    assert not keep.any()


def test_grey_identical_reference_has_one_coefficients():
    grades, keep = grey_relational(np.ones((3, 2)), np.ones(3))
    np.testing.assert_array_equal(grades, [1, 1])
    assert keep.all()


@pytest.mark.parametrize("values,reference", [([], []), ([[1, 2]], [1, 2]), ([[np.nan]], [1])])
def test_grey_rejects_missing_or_misaligned_input(values, reference):
    with pytest.raises(ValueError):
        grey_relational(values, reference)


def test_path_count_capacity_is_not_conventional_betweenness():
    # 所有可达有序节点对共有 11 条最短路径，其中
    # 0→3 和 0→4 各有两条；节点 1、2 各经过两条，节点 3 经过四条。
    result = topological_capacity(diamond_tail())
    np.testing.assert_allclose(result, [0, 2 / 11, 2 / 11, 4 / 11, 0])
    assert not np.isclose(result[1], 1 / 12)  # 节点 1 的常规归一化介数中心性


def test_capacity_uses_directed_topology_and_ratio_of_sums():
    np.testing.assert_allclose(topological_capacity(chain()), [0, 1 / 3, 0])
    np.testing.assert_array_equal(topological_capacity(np.zeros((3, 3))), [0, 0, 0])
    scaled_edges = chain() * 9  # 邻接矩阵只定义拓扑，9 不表示路径距离
    np.testing.assert_allclose(topological_capacity(scaled_edges), [0, 1 / 3, 0])


def test_capacity_combines_normalized_grades_and_node_coefficients():
    expected = 2 * (
        np.array([0.2, 0.3, 0.4]) * np.array([0, 1 / 3, 0]) + 0.5 * np.array([1, 2, 1]) / 4
    )
    actual = capacities(chain(), [1, 2, 1], omega_be=[0.2, 0.3, 0.4], scale=2)
    np.testing.assert_allclose(actual, expected)
    with pytest.raises(ValueError):
        capacities(chain(), [0, 0, 0])


def test_initial_load_uses_in_and_total_degree():
    np.testing.assert_allclose(initial_load(chain(), [1, 2, 3], eta=0.6), [0.2, 0.6, 1.2])
    np.testing.assert_allclose(initial_load(np.zeros((1, 1)), [2], eta=0.6), [0.6])


def test_literal_impedance_formula_preserves_negative_overload_and_is_stable():
    loads = np.array([0.5, 2, 3])
    expected = (1 - np.exp(loads - 2)) / (1 + np.exp(loads))
    np.testing.assert_allclose(impedance_theta(loads, 2), expected, atol=1e-15)
    assert impedance_theta(3, 2) < 0
    assert impedance_theta(2, 2) == pytest.approx(0, abs=1e-15)
    assert impedance_theta(1000, 2) == pytest.approx(-np.exp(-2))


def test_propagation_is_synchronous_and_uses_source_target_direction():
    weights = chain() * 0.4
    # 节点 1 在本步超过容量，但要到下一步才能向节点 2 传播；
    # 若误将权重矩阵转置，结果将发生变化。
    result = propagate_step([1, 0.8, 0], weights, np.ones(3), np.zeros(3))
    np.testing.assert_allclose(result, [1, 1.2, 0])
    next_result = propagate_step(result, weights, np.ones(3), np.zeros(3))
    np.testing.assert_allclose(next_result, [1, 1.6, 0.4])


def test_equality_activates_source_and_target_capacity_scales_increment():
    graph = np.array([[0, 0.5], [0, 0]])
    np.testing.assert_allclose(propagate_step([1, 0], graph, [1, 3], [0, 0]), [1, 1.5])
    np.testing.assert_allclose(propagate_step([0.999, 0], graph, [1, 3], [0, 0]), [0.999, 0])


def test_numpy_batch_broadcasting():
    loads = np.array([[1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(
        propagate_step(loads, chain() * 0.4, np.ones(3), np.zeros(3)), [[1, 0.4, 0], [0, 1, 0.4]]
    )
    batched_weights = np.stack([chain() * 0.4, chain() * 0.2])
    np.testing.assert_allclose(
        propagate_step(loads, batched_weights, np.ones(3), np.zeros(3)), [[1, 0.4, 0], [0, 1, 0.2]]
    )


def test_literal_impedance_amplifies_overload_clipped_variant_is_explicit():
    weights = np.array([[0, 0.5], [0, 0]])
    literal = propagate_step([1, 3], weights, [1, 2], [0, 1])
    clipped = propagate_step([1, 3], weights, [1, 2], [0, 1], theta_mode="clipped")
    assert literal[1] > clipped[1]
    assert clipped[1] == pytest.approx(4)


def test_simulation_controls_modify_only_new_incoming_increment():
    output = simulate(
        [1, 0, 0], chain() * 0.4, np.ones(3), np.zeros(3), steps=3, controls=[0, 1, 0]
    )
    np.testing.assert_allclose(output, np.tile([1, 0, 0], (4, 1)))
    assert simulate([1, 0, 0], chain(), np.ones(3), np.zeros(3), steps=0).shape == (1, 3)


def test_simulation_accepts_declared_time_variant_weights():
    output = simulate(
        [1, 0, 0], np.stack([chain() * 0.4, chain() * 0.2]), np.ones(3), np.zeros(3), steps=2
    )
    np.testing.assert_allclose(output, [[1, 0, 0], [1, 0.4, 0], [1, 0.6, 0]])
    with pytest.raises(ValueError):
        simulate([1, 0, 0], np.ones((1, 3, 3)), np.ones(3), np.zeros(3), steps=2)


def test_nri_literal_neighbor_sum_and_undefined_domain():
    # 按公式计算：C*Dout*N(N-1) / ((C-L)*sum(Dout_i+1)*sum(distance)) = 6/2。
    actual = nri(chain(), np.zeros(3), np.ones(3))
    np.testing.assert_allclose(actual[:2], [3, 3])
    assert np.isnan(actual[2])
    assert np.isnan(nri(chain(), [1, 0, 0], np.ones(3))[0])


def test_sri_path_count_product_and_complete_failure_boundary():
    assert sri(chain(), [False, False, False]) == 0
    # 初始共有三条路径；删除节点 0 后剩一条，T/(N-T)=1/2。
    assert sri(chain(), [True, False, False]) == pytest.approx(1 / 6)
    # 删除中心节点会破坏所有路径，路径比乘积为 0。
    assert sri(chain(), [False, True, False]) == 0
    assert np.isinf(sri(chain(), [True, True, True]))
    assert np.isnan(sri(np.zeros((3, 3)), [False, False, False]))


def test_ndi_equality_extension_and_normalized_directed_betweenness():
    # 有向链中心节点的介数为 1/((3-1)*(3-2))=1/2，因此 2*b*L=1。
    assert ndi(chain(), [0, 1, 0], np.ones(3)) == pytest.approx(1)
    assert ndi(chain(), [0, 0.999, 0], np.ones(3)) == 0
    assert ndi(chain(), [0, 2, 0], np.ones(3)) == pytest.approx(2)
    metrics = global_metrics(chain(), [0, 1, 0], np.ones(3))
    assert metrics["failed"] == 1
    assert metrics["failed_fraction"] == pytest.approx(1 / 3)


def test_invalid_shapes_values_and_modes_are_rejected():
    with pytest.raises(ValueError):
        topological_capacity([[1, 2]])
    with pytest.raises(ValueError):
        propagate_step([1, 0, 0], chain(), np.ones(3), [0, 0.5, 0])
    with pytest.raises(ValueError):
        propagate_step([1, 0, 0], chain(), np.ones(3), np.zeros(3), theta_mode="unknown")
    with pytest.raises(ValueError):
        simulate([1, 0, 0], chain(), np.ones(3), np.zeros(3), controls=[0, 2, 0])
