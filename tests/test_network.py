"""训练域建网测试使用数值和审核协议夹具。"""

import copy

import numpy as np
import pytest

from airport_risk.data import RiskData
from airport_risk.network import (
    NetworkBuilder,
    benjamini_hochberg,
    graph_statistics,
    permutation_edge_test,
    prepare_fold_data,
)


def test_bh_adjustment_matches_hand_calculation_and_permutation_has_add_one():
    np.testing.assert_allclose(
        benjamini_hochberg([0.01, 0.04, 0.03, 0.2]), [0.04, 0.0533333333333, 0.0533333333333, 0.2]
    )
    events = [[[0, 1]] for _ in range(15)]
    counts, p, q = permutation_edge_test(events, 2, permutations=999, seed=1)
    assert counts[0, 1] == 15 and counts[1, 0] == 0
    assert p[0, 1] >= 1 / 1000
    assert q[0, 1] < 0.05 and p[1, 0] == 1
    np.testing.assert_array_equal(permutation_edge_test(events, 2, permutations=999, seed=1)[1], p)
    with pytest.raises(ValueError):
        benjamini_hochberg([np.nan])


def test_builder_discards_inactive_and_constant_factors_not_fixed_63():
    values = np.array([[0, 1, 0, 1], [1, 0, 0, 1]] * 10, dtype=float)
    history = np.stack([values, values + 0.2], axis=1)
    config = {"grey_threshold": 0, "permutations": 99, "fdr_alpha": 1, "min_edge_support": 2}
    chains = [[["a", "b"], ["a", "b"]] for _ in range(20)]
    builder = NetworkBuilder(config)
    state = builder.fit(values, values.mean(1), chains, ["a", "b", "c", "d"], history_train=history)
    assert state.active_indices.tolist() == [0, 1]
    assert state.adjacency.tolist() == [[0, 1], [0, 0]]
    assert state.edge_counts[0, 1] == 20
    assert state.capacities.mean() == pytest.approx(1)
    assert np.all(state.capacities > 0)
    assert state.impedance.tolist() == [0, 0]
    np.testing.assert_array_equal(builder.transform(values), values[:, :2])
    assert state.statistics["node_count"] == 2
    assert state.to_dict()["active_indices"] == [0, 1]
    with pytest.raises(ValueError, match="没有活动"):
        NetworkBuilder(config).fit(np.zeros((4, 2)), np.zeros(4), [], ["a", "b"])


def test_rejected_pending_uncertain_chains_do_not_create_edges_expert_is_audited():
    values = np.array([[0, 1], [1, 0]] * 4, dtype=float)
    chain = {"chain": ["a", "b"], "review_status": "pending"}
    expert = {
        "source": "b",
        "target": "a",
        "status": "approved",
        "reviewer": "domain-review",
        "basis": "审核协议夹具",
    }
    state = NetworkBuilder({"grey_threshold": 0, "permutations": 9, "expert_edges": [expert]}).fit(
        values, values.mean(1), [chain], ["a", "b"]
    )
    assert state.adjacency.tolist() == [[0, 0], [1, 0]]
    assert state.edge_counts.sum() == 0
    assert state.selection["expert_edges"] == [expert]
    with pytest.raises(ValueError, match="专家边"):
        NetworkBuilder(
            {
                "grey_threshold": 0,
                "permutations": 9,
                "expert_edges": [{"source": "a", "target": "b"}],
            }
        ).fit(values, values.mean(1), [], ["a", "b"])


def _dataset():
    values = np.array(
        [[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2]], dtype=float
    )
    history = np.stack([values * 0.5, values], axis=1)
    return RiskData(
        history=history,
        target=values + 1,
        adjacency=np.zeros((3, 3), dtype=int),
        capacities=np.ones(3),
        impedance=np.zeros(3, dtype=int),
        grades=np.ones(3),
        years=np.full(6, 2020),
        airports=np.array(["fixture-airport"] * 6),
        event_ids=np.array([f"fixture-{i}" for i in range(6)]),
        node_ids=np.array(["a", "b", "c"]),
        factor_values=values,
        factor_reference=values.mean(1),
        metadata={
            "provenance": "authorized_real",
            "node_factor_indices": [0, 1, 2],
            "network_policy": "train_fold",
            "annotated_chains": {f"fixture-{i}": [["a", "b"]] for i in range(6)},
        },
    ).validate()


def test_fold_state_is_independent_of_test_values_targets_and_chains():
    data = _dataset()
    config = {"grey_threshold": 0, "permutations": 29, "fdr_alpha": 1, "min_edge_support": 1}
    fold, audit = prepare_fold_data(data, np.arange(4), config)
    changed = copy.deepcopy(data)
    changed.factor_values[4:] = 99999
    changed.factor_reference[4:] = 99999
    changed.target[4:] = 99999
    changed.history[4:] = 99999
    changed.metadata["annotated_chains"]["fixture-4"] = [["b", "a"]] * 100
    two, second = prepare_fold_data(changed, np.arange(4), config)
    assert audit["graph_fingerprint"] == second["graph_fingerprint"]
    assert audit["train_event_ids"] == [f"fixture-{i}" for i in range(4)]
    assert audit["node_indices"] == [0, 1]
    assert fold.history.shape[-1] == two.history.shape[-1] == 2
    assert fold.factor_values.shape == data.factor_values.shape
    assert fold.metadata["node_factor_indices"] == [0, 1]
    assert audit["train_only"] is True
    np.testing.assert_array_equal(fold.adjacency, [[0, 1], [0, 0]])
    np.testing.assert_array_equal(data.adjacency, np.zeros((3, 3)))


def test_training_quantile_capacity_and_recovery_only_use_history():
    values = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=float)
    history = np.array([[[0, 1], [2, 0]]] * 4, dtype=float)
    state = NetworkBuilder(
        {
            "grey_threshold": 0,
            "permutations": 9,
            "capacity_mode": "train_quantile",
            "train_capacity_quantile": 1,
        }
    ).fit(values, values.mean(1), [], ["a", "b"], history_train=history)
    np.testing.assert_allclose(state.capacities, [2, 1])
    assert state.impedance.tolist() == [0, 1]
    with pytest.raises(ValueError, match="训练负载历史"):
        NetworkBuilder(
            {"grey_threshold": 0, "permutations": 9, "capacity_mode": "train_quantile"}
        ).fit(values, values.mean(1), [], ["a", "b"])


def test_external_prior_requires_source_and_preserves_capacity():
    data = _dataset()
    config = {"grey_threshold": 0, "permutations": 9, "network_mode": "external_fixed"}
    with pytest.raises(ValueError, match="network_source"):
        prepare_fold_data(data, np.arange(4), config)
    data.metadata["network_source"] = "独立外部网络协议夹具"
    data.adjacency[1, 0] = 1
    data.capacities = np.array([2, 3, 4], dtype=float)
    fold, audit = prepare_fold_data(data, np.arange(4), config)
    assert fold.capacities.tolist() == [2, 3]
    assert fold.adjacency.tolist() == [[0, 0], [1, 0]]
    assert audit["train_only"] is False


def test_graph_full_triads_and_communities_count_directed_triangle():
    statistics = graph_statistics(np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]]), ["a", "b", "c"])
    assert statistics["triadic_census"]["030T"] == 1
    assert statistics["weak_components"] == 1
    assert sorted(node for group in statistics["communities"] for node in group) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("grey_threshold", True),
        ("capacity_scale", True),
        ("capacity_omega_be", False),
        ("impedance_delta_tolerance", True),
        ("impedance_recovery_fraction", False),
        ("fdr_alpha", 0),
        ("permutations", True),
    ],
)
def test_network_numeric_config_rejects_booleans_and_invalid_alpha(key, value):
    with pytest.raises(ValueError):
        NetworkBuilder({key: value})


def test_expert_config_rejects_unknown_duplicate_and_conflicting_overrides():
    from airport_risk.network import validate_network_config

    edge = {
        "source": "a",
        "target": "b",
        "status": "approved",
        "reviewer": "reviewer",
        "basis": "protocol fixture",
    }
    with pytest.raises(ValueError, match="未知字段"):
        validate_network_config({"threshold_typo": 0.3})
    with pytest.raises(ValueError, match="仅包含"):
        NetworkBuilder({"expert_edges": [{**edge, "hidden_override": True}]})
    with pytest.raises(ValueError, match="重复"):
        NetworkBuilder({"expert_edges": [edge, {**edge, "status": "rejected"}]})
    first, second = validate_network_config({}), validate_network_config({})
    first["expert_edges"].append(edge)
    assert second["expert_edges"] == []


def test_impedance_fraction_boundary_requires_actual_recovery():
    values = np.array([[0, 1], [1, 0]], dtype=float)
    history = np.array([[[1, 1], [0, 1]], [[1, 1], [1, 1]]], dtype=float)
    config = {"grey_threshold": 0, "permutations": 1, "impedance_recovery_fraction": 0.5}
    state = NetworkBuilder(config).fit(
        values, values.mean(1), [], ["a", "b"], history_train=history
    )
    assert state.impedance.tolist() == [1, 0]
    state = NetworkBuilder({**config, "impedance_recovery_fraction": 0}).fit(
        values, values.mean(1), [], ["a", "b"], history_train=history
    )
    assert state.impedance.tolist() == [1, 0]


def test_external_metadata_does_not_override_explicit_train_fitted_mode():
    data = _dataset()
    data.metadata["network_policy"] = "external_prior"
    data.metadata["network_source"] = "独立先验协议夹具"
    data.adjacency[1, 0] = 1
    data.capacities = np.array([2, 3, 4], dtype=float)
    config = {
        "grey_threshold": 0,
        "permutations": 9,
        "fdr_alpha": 1,
        "min_edge_support": 1,
        "network_mode": "train_fitted",
    }
    fold, audit = prepare_fold_data(data, np.arange(4), config)
    assert audit["source"] == "training_fold"
    assert audit["train_only"] is True
    assert audit["selection"]["config"]["network_mode"] == "train_fitted"
    assert fold.adjacency.tolist() == [[0, 1], [0, 0]]
    assert fold.capacities.mean() == pytest.approx(1)
