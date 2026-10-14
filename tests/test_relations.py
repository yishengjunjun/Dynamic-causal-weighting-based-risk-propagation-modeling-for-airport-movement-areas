"""关系算子的传播一致性、确定性及输入契约。"""

import numpy as np
import pytest

from airport_risk.core import propagate_step
from airport_risk.relations import gate_sources, propagate_with_relations, table2_context


def test_empty_relation_config_preserves_paper_step():
    load = np.array([1.3, 0.2, 1.1])
    cap = np.ones(3)
    imp = np.array([0, 1, 0])
    weight = np.array([[0, 0.2, 0], [0, 0, 0.1], [0, 0.3, 0]])
    np.testing.assert_allclose(
        propagate_with_relations(load, weight, cap, imp), propagate_step(load, weight, cap, imp)
    )


def test_mutex_uses_only_highest_overload_and_joint_gate_requires_every_member():
    np.testing.assert_array_equal(
        gate_sources([1.5, 1.2, 0.9], [1, 1, 1], mutex_groups=[(0, 1)]), [1, 0, 0]
    )
    np.testing.assert_array_equal(
        gate_sources([1.5, 0.9, 1.2], [1, 1, 1], cooccurrence_groups=[(0, 1)]), [0, 0, 1]
    )
    np.testing.assert_array_equal(
        gate_sources([2, 4], [1, 2], mutex_groups=[(np.int64(0), np.int64(1))]), [1, 0]
    )


@pytest.mark.parametrize(
    "load,capacity",
    [
        ([1, 2], [1, np.nan]),
        ([1, 2], [1, np.inf]),
        ([[1, 2]], [[1, 1]]),
        ([-1, 2], [1, 1]),
        ([], []),
    ],
)
def test_invalid_node_vectors_are_rejected(load, capacity):
    with pytest.raises(ValueError):
        gate_sources(load, capacity)


@pytest.mark.parametrize("group", [(0, 1.5), (0, True), (0, 0), (-1, 1), (0, 2), (0,)])
def test_relation_groups_require_distinct_integer_node_indices(group):
    with pytest.raises(ValueError):
        gate_sources([1, 1], [1, 1], mutex_groups=[group])


def test_table2_keeps_negative_coefficients_and_unspecified_duration_gap():
    values = table2_context([5, 10, 20], [0.2, 0.5, 0.8], [1.5, 2.5, 5.5])
    np.testing.assert_allclose(values["years_service_theta"], [0.15, 0.25, 0.4])
    np.testing.assert_allclose(values["workload_theta"], [0.25, 0, -0.2])
    assert np.isnan(values["working_hours_theta"][1])
    assert values["working_hours_theta"][2] == -0.1


def test_node_id_relations_remap_and_conflicts():
    from airport_risk.relations import compile_relations, remap_relations

    source = {
        "mutex_groups": [["A", "B", "C"]],
        "context": {"C": {"years_service": 10, "workload": 0.8, "working_hours": 5}},
    }
    selected = remap_relations(source, ["B", "C"])
    plan = compile_relations(["C", "B"], selected)
    assert plan.mutex_groups == ((0, 1),)
    np.testing.assert_allclose(plan.context_theta, [-1 / 60, 0])
    assert remap_relations(source, ["A"])["mutex_groups"] == []
    with pytest.raises(ValueError, match="overlap|conflict"):
        compile_relations(
            ["A", "B", "C"], {"mutex_groups": [["A", "B"]], "cooccurrence_groups": [["B", "C"]]}
        )
    with pytest.raises(ValueError, match="known node"):
        compile_relations(["A", "B"], {"mutex_groups": [["A", "X"]]})
    with pytest.raises(ValueError, match="outside"):
        compile_relations(
            ["A"], {"context": {"A": {"years_service": 31, "workload": 0.5, "working_hours": 3}}}
        )


def test_relations_numpy_torch_match_with_context_cumulative_and_gradients():
    import torch
    from airport_risk.relations import compile_relations, propagate_relations_torch

    ids = list("ABCDEFG")
    config = {
        "mutex_groups": [["A", "B"]],
        "cooccurrence_groups": [["C", "D"]],
        "cumulative_groups": [["E", "F"]],
        "context": {"G": {"years_service": 10, "workload": 0.8, "working_hours": 5}},
    }
    plan = compile_relations(ids, config)
    cap = np.array([1, 2, 1, 1, 1, 1, 1.5], dtype=float)
    load = np.array([[1.5, 3, 1.2, 0.7, 1.6, 0.9, 0.3], [1.2, 3.2, 1.2, 1.3, 1.5, 1.7, 0.4]])
    weight = (np.ones((7, 7)) - np.eye(7)) * 0.1
    imp = np.array([0, 0, 0, 0, 0, 0, 1])
    value = torch.tensor(load, dtype=torch.float64, requires_grad=True)
    edges = torch.tensor(weight, dtype=torch.float64, requires_grad=True)
    actual = propagate_relations_torch(value, edges, torch.tensor(cap), torch.tensor(imp), plan)
    expected = propagate_with_relations(load, weight, cap, imp, plan=plan)
    np.testing.assert_allclose(actual.detach(), expected, atol=1e-13)
    actual.sum().backward()
    assert torch.isfinite(value.grad).all() and torch.isfinite(edges.grad).all()
    assert value.grad[:, 4:6].min() >= 1
    assert value.grad[:, 4].min() > 1
    # 第一条记录中共现组有一个未激活节点，两个成员都不传播。
    np.testing.assert_array_equal(edges.grad[2:4] > 0, np.ones((2, 7), dtype=bool))


def test_cumulative_sources_use_mean_positive_overload():
    actual = gate_sources([1.6, 0.9, 1.1], [1, 1, 1], cumulative_groups=[(0, 1)])
    np.testing.assert_allclose(actual, [0.3, 0.3, 1])
