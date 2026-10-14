import numpy as np
from airport_risk.experiments import evaluate_prediction, _ndi_batch
from airport_risk.core import ndi
from airport_risk.reporting import clean_json


def test_risk_bins_cover_unreported_interval_and_all_data():
    pred = np.array([[0.1, 0.5, 0.7, 0.8, 0.85, 0.9, 1.2]])
    result = evaluate_prediction(pred, pred, np.ones(7))
    assert abs(sum(result[k] for k in result if k.startswith("risk_bin_")) - 100) < 1e-10
    assert result["risk_bin_0_7_to_0_85_pct"] > 0
    assert result["node_failure_accuracy"] == 1
    assert result["mse_normalized"] == 0


def test_scores_are_not_mislabelled_accuracy():
    pred = np.array([[0.9, 0.9], [0.9, 0.9]])
    target = np.array([[2, 2], [2, 2]])
    result = evaluate_prediction(pred, target, np.ones(2))
    assert result["high_risk_event_flag_rate"] == 1
    assert result["node_failure_accuracy"] == 0


def test_json_nonfinite_reported_null_not_fake_zero():
    assert clean_json({"x": float("inf"), "y": np.nan, "z": np.array([1.0, np.nan])}) == {
        "x": None,
        "y": None,
        "z": [1.0, None],
    }


def test_batched_ndi_same_as_scalar_including_equality():
    import networkx as nx

    a = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    bc = np.array(
        list(
            nx.betweenness_centrality(
                nx.from_numpy_array(a, create_using=nx.DiGraph), normalized=True
            ).values()
        )
    )
    loads = np.array([[0.5, 0.5, 0.5], [1, 1, 1], [1.1, 1.2, 1.3]])
    cap = np.ones(3)
    np.testing.assert_allclose(_ndi_batch(loads, cap, bc), [ndi(a, l, cap) for l in loads])


def test_prediction_evaluation_rejects_invalid_capacity_and_probabilities():
    import pytest

    for capacity in (np.array([0.0, 1.0]), np.array([1.0]), np.array([np.inf, 1.0])):
        with pytest.raises(ValueError, match="capacities"):
            evaluate_prediction(np.ones((2, 2)), np.ones((2, 2)), capacity)
    with pytest.raises(ValueError, match="probabilities"):
        evaluate_prediction(
            np.ones((2, 2)), np.ones((2, 2)), np.ones(2), proba=np.ones((2, 2)) * 1.1
        )
    with pytest.raises(ValueError, match="finite"):
        evaluate_prediction(np.ones((2, 2)), np.array([[1.0, np.nan], [1.0, 1.0]]), np.ones(2))


def test_multiwindow_event_accuracy_aggregates_by_event():
    pred = np.array([[1.1, 0.2], [0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
    target = np.array([[0.1, 0.2], [1.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
    result = evaluate_prediction(pred, target, np.ones(2), event_ids=np.array(["A", "A", "B", "B"]))
    assert result["event_failure_accuracy"] == 1
    assert result["evaluated_events"] == 2
    assert result["node_failure_accuracy"] == 0.75


def test_only_enabled_protocols_require_airport_and_year_settings():
    import pytest
    from airport_risk.experiments import _experiment_config

    config = _experiment_config("quick", None)
    assert config["enabled_protocols"] == ["primary"]
    assert config["airport_holdout"] is None
    with pytest.raises(ValueError, match="airport_holdout"):
        _experiment_config("quick", {"enabled_protocols": ["primary", "airport"]})
    with pytest.raises(ValueError, match="rolling_test_years"):
        _experiment_config("quick", {"enabled_protocols": ["primary", "rolling"]})


def test_group_overlap_is_rejected_even_with_distinct_sample_indices():
    import pytest
    from airport_risk.experiments import _validate_fold_indices

    with pytest.raises(ValueError, match="event groups"):
        _validate_fold_indices(
            np.array([0]), np.array([1]), None, 3, event_ids=np.array(["A", "A", "B"])
        )


def test_transition_requires_actual_future_observations_before_training(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from airport_risk import experiments

    data = SimpleNamespace(
        history=np.ones((4, 2, 2)), target=np.ones((4, 2)), capacities=np.ones(2)
    )

    class Estimator:
        def fit(self, *args, **kwargs):
            pytest.fail("missing transition labels reached training")

    monkeypatch.setattr(experiments, "_make_model", lambda *args: Estimator())
    with pytest.raises(ValueError, match="observed future_loads"):
        experiments._fit_score(
            "DWPM",
            data,
            np.array([0, 1]),
            np.array([2, 3]),
            np.zeros((2, 2)),
            0,
            {"epochs": 1, "training_mode": "transition"},
        )
