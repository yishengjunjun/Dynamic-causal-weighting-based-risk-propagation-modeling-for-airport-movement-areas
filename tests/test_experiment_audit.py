"""独立核验干预结果平均值和训练折边界。"""

from types import SimpleNamespace

import numpy as np

from airport_risk import experiments
from airport_risk.core import initial_load, ndi, simulate


def test_random_control_chunk_average_matches_independent_scalar_simulation():
    # 53 次重复特意跨越每批 50 行的边界。
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
    capacities = np.ones(3)
    weights = adjacency * 0.8
    impedance = np.zeros(3)
    data = SimpleNamespace(
        adjacency=adjacency,
        capacities=capacities,
        impedance=impedance,
        node_ids=np.array(["a", "b", "c"]),
    )
    result = experiments.control_analysis(data, weights, steps=3, random_repetitions=53, seed=17)
    base = initial_load(adjacency, capacities, eta=0.7)
    rng = np.random.default_rng(17)
    independent = []
    for _ in range(53):
        controls = np.zeros(3)
        controls[rng.choice(3, 1, replace=False)] = 0.5
        for attacked in range(3):
            loads = base.copy()
            loads[attacked] = capacities[attacked] * (1 + 1e-6)
            trajectory = simulate(loads, weights, capacities, impedance, steps=3, controls=controls)
            independent.append([ndi(adjacency, row, capacities) for row in trajectory])
    np.testing.assert_allclose(
        result["curves"]["random"], np.mean(independent, axis=0), rtol=1e-12, atol=1e-12
    )
    baseline = np.array(result["curves"]["none"])
    for name in ("degree", "NRI_load_ratio", "random"):
        # 在本例中，非负传播和同等衰减不会增加负载。
        assert np.all(np.array(result["curves"][name]) <= baseline + 1e-12)


def test_fit_score_supplies_only_training_rows_to_estimator(monkeypatch):
    history = np.arange(8 * 2 * 3, dtype=float).reshape(8, 2, 3) / 100
    target = np.arange(8 * 3, dtype=float).reshape(8, 3) / 100
    data = SimpleNamespace(history=history, target=target, capacities=np.ones(3))
    train, validation, test = np.array([0, 2, 4, 6]), np.array([1, 3]), np.array([5, 7])

    class SpyEstimator:
        def fit(self, fit_history, fit_target, **kwargs):
            self.training_history = fit_history.copy()
            self.training_target = fit_target.copy()
            self.mean = fit_target.mean(axis=0)
            return []

        def predict(self, pred_history):
            return np.broadcast_to(self.mean, (len(pred_history), 3)).copy()

    model = SpyEstimator()
    monkeypatch.setattr(experiments, "_make_model", lambda *args, **kwargs: model)
    experiments._fit_score(
        "DWPM", data, train, test, np.zeros((3, 3)), 0, {"epochs": 1}, validation=validation
    )
    np.testing.assert_array_equal(model.training_history, history[train])
    np.testing.assert_array_equal(model.training_target, target[train])
    for row in np.r_[validation, test]:
        assert not np.any(np.all(model.training_history == history[row], axis=(1, 2)))


def test_failure_accuracy_uses_truth_not_high_score_frequency():
    # 所有预测都超过 .85，但只有半数样本实际失效。
    pred = np.array([[0.9, 0.9], [0.9, 0.9]])
    truth = np.array([[0.2, 0.3], [1.2, 1.3]])
    result = experiments.evaluate_prediction(pred, truth, np.ones(2))
    assert result["event_failure_accuracy"] == 0.5
    assert result["node_failure_accuracy"] == 0.5
    # 该例说明高风险预测频率不能当作准确率。
    assert result["high_risk_event_flag_rate"] == 1.0


def test_overlapping_folds_are_rejected_before_estimator_construction(monkeypatch):
    import pytest

    data = SimpleNamespace(
        history=np.ones((4, 2, 2)), target=np.ones((4, 2)), capacities=np.ones(2)
    )
    monkeypatch.setattr(
        experiments, "_make_model", lambda *args: pytest.fail("overlapping fold reached estimator")
    )
    with pytest.raises(ValueError, match="disjoint"):
        experiments._fit_score(
            "DWPM", data, np.array([0, 1]), np.array([1, 2]), np.zeros((2, 2)), 1, {"epochs": 1}
        )
    with pytest.raises(ValueError, match="duplicates"):
        experiments._fit_score(
            "DWPM", data, np.array([0, 0]), np.array([2, 3]), np.zeros((2, 2)), 1, {"epochs": 1}
        )


def test_train_weight_average_covers_all_training_events_only():
    history = np.arange(12, dtype=float).reshape(6, 2, 1)
    train = np.array([0, 2, 3, 5])
    observed = []

    class WeightEstimator:
        def weights(self, batch):
            observed.extend(batch[:, 0, 0].tolist())
            return batch[:, 0, :].reshape(-1, 1, 1)

    result = experiments._train_mean_weights(WeightEstimator(), history, train, batch_size=3)
    assert observed == history[train, 0, 0].tolist()
    np.testing.assert_allclose(result, history[train, 0, 0].mean())


def test_pipeline_respects_configured_protocols_and_train_only_selection(monkeypatch, tmp_path):
    import json
    from pathlib import Path
    from airport_risk.data import RiskData, split_indices

    years = np.repeat(np.arange(2008, 2022), 4)
    count = len(years)
    data = RiskData(
        history=np.arange(count * 3 * 2, dtype=float).reshape(count, 3, 2) / 1000,
        target=np.arange(count * 2, dtype=float).reshape(count, 2) / 100,
        adjacency=np.array([[0, 1], [0, 0]]),
        capacities=np.ones(2),
        impedance=np.zeros(2),
        grades=np.array([0.5, 0.5]),
        years=years,
        airports=np.array(["A", "Z"] * (count // 2)),
        event_ids=np.array([f"E{i}" for i in range(count)]),
        node_ids=np.array(["a1", "b1"]),
        factor_values=np.arange(count * 2, dtype=float).reshape(count, 2) + 1,
        factor_reference=np.arange(count, dtype=float) + 1,
        metadata={"provenance": "synthetic", "node_factor_indices": [0, 1]},
        future_loads=np.arange(count * 2, dtype=float).reshape(count, 1, 2) / 100,
    ).validate()
    calls, selectors = [], []

    class SpyModel:
        def __init__(self, name):
            self.name = name

        def fit(self, history, target, **kwargs):
            calls.append((self.name, history.copy(), target.copy(), kwargs))
            return []

        def predict(self, history):
            return np.zeros((len(history), 2))

        def predict_proba(self, history):
            return np.zeros((len(history), 2))

        def weights(self, history):
            return np.broadcast_to(data.adjacency * 0.2, (len(history), 2, 2)).copy()

        def propagate_numpy(self, loads, weights, capacities=None):
            from airport_risk.core import propagate_step

            return propagate_step(
                loads,
                weights,
                data.capacities if capacities is None else capacities,
                data.impedance,
            )

        def rollout(self, history, *, steps):
            return {
                "loads": np.concatenate(
                    [
                        history[
                            :,
                            -1:,
                        ],
                        np.zeros((len(history), steps, 2)),
                    ],
                    axis=1,
                ),
                "weights": np.broadcast_to(
                    data.adjacency * 0.2, (len(history), steps, 2, 2)
                ).copy(),
            }

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"test fixture checkpoint")

    monkeypatch.setattr(experiments, "_make_model", lambda name, *args: SpyModel(name))

    def selector(values, reference, **kwargs):
        selectors.append((values.copy(), reference.copy()))
        return np.array([0.4, 0.6]), np.array([True, True])

    def prepare_fold(value, train_rows, settings, grey_threshold=None):
        if grey_threshold is not None:
            selector(value.factor_values[train_rows], value.factor_reference[train_rows])
        return value, {
            "graph_fingerprint": "test_graph",
            "node_ids": value.node_ids.tolist(),
            "train_event_ids": value.event_ids[train_rows].tolist(),
            "node_indices": [0, 1],
        }

    monkeypatch.setattr(experiments, "_prepare_fold_data", prepare_fold)
    monkeypatch.setattr(experiments, "plot_results", lambda *args: None)
    config = {
        "seeds": [7],
        "enabled_protocols": ["primary", "additional_fraction", "rolling", "airport", "count"],
        "epochs": 1,
        "airport_holdout": "Z",
        "stratified_train_fraction": 0.6,
        "additional_train_fractions": [0.65],
        "rolling_test_years": [2020],
        "printed_train_count": 9,
        "batch_size": 7,
        "grey_thresholds": [0.2, 0.6],
        "grey_quantile_diagnostics": False,
        "capacity_scales": [1.0],
        "eta_scales": [1.0],
        "perturbation_levels": [0.1, 0.3],
        "inference_repetitions": 3,
        "random_control_repetitions": 2,
        "propagation_steps": 1,
        "cpu_threads": 1,
        "warmup_epochs": 0,
        "control_fraction": 0.5,
        "control_reduction": 0.3,
    }
    original_capacities = data.capacities.copy()
    report = experiments.run_experiments(
        data, tmp_path, config_override=config, progress=lambda _: None
    )
    np.testing.assert_array_equal(data.capacities, original_capacities)
    assert report["primary_seed"] == 7
    assert "paper_reference_numbers" not in report
    assert report["method"]["training_mode"] == "transition"
    assert {r["protocol"] for r in report["benchmarks"]} == {
        "primary_stratified",
        "stratified_0.65",
        "rolling_2020",
        "airport_holdout",
        "counts_9_47",
    }
    holdout = json.loads((tmp_path / "splits/airport_holdout_seed7.json").read_text())
    assert holdout["settings"]["holdout"] == "Z"
    assert all(data.airports[int(event_id[1:])] == "Z" for event_id in holdout["test_event_ids"])
    train, _, test = split_indices(data, seed=7, train_fraction=0.6)
    for values, reference in selectors:
        np.testing.assert_array_equal(values, data.factor_values[train])
        np.testing.assert_array_equal(reference, data.factor_reference[train])
    for name, history, target, kwargs in calls[:5]:
        np.testing.assert_array_equal(history, data.history[train])
        np.testing.assert_array_equal(target, data.target[train])
        assert kwargs["batch_size"] == 7
        if name == "DWPM":
            assert kwargs["warmup_epochs"] == 0
    assert report["controls"]["requested_control_fraction"] == 0.5
    assert report["controls"]["control_intensity"] == 0.3
    assert report["inference_timing"]["repetitions"] == 3
    assert set(report["node_attacks"]["nri"]) == {"10.0", "30.0"}
    assert report["sensitivity"]["grey_sweep_degenerate"] is True
    assert report["sensitivity"]["grey_diagnostic_quantiles"] == []
    dynamic = json.loads((tmp_path / "dynamic_rollout.json").read_text())
    assert dynamic["event_id"] == str(data.event_ids[test[0]])
    assert dynamic["future_accuracy_computed"] is False
    np.testing.assert_array_equal(dynamic["load_trajectory"][0], data.history[test[0], -1])
    np.testing.assert_array_equal(dynamic["first_step_observed_target"], data.target[test[0]])
    assert (tmp_path / "REPORT.md").read_text().find("60.0%") >= 0
    assert not (tmp_path / "tables/table2_context_reference.csv").exists()


def test_undefined_nri_nodes_cannot_be_silently_ranked_as_valid_controls():
    import pytest

    data = SimpleNamespace(
        adjacency=np.zeros((3, 3)),
        capacities=np.ones(3),
        impedance=np.zeros(3),
        node_ids=np.array(["a", "b", "c"]),
    )
    result = experiments.control_analysis(data, np.zeros((3, 3)), random_repetitions=1)
    assert result["status"] == "not_applicable"
    assert result["curves"] == {}
    assert result["reason"] == "network has no propagation edges"


def test_unknown_airport_cannot_fall_back_to_the_first_input_airport():
    import pytest

    with pytest.raises(ValueError, match="airport_holdout"):
        experiments._experiment_config("full", {"enabled_protocols": ["primary", "airport"]})


def test_grey_sweep_rebuilds_node_axis_and_reports_shared_capacity_metrics(monkeypatch, tmp_path):
    from pathlib import Path
    import airport_risk.network as network
    from airport_risk.data import RiskData

    count = 10
    history = np.arange(count * 3 * 3).reshape(count, 3, 3) / 100 + 0.5
    target = history[:, -1] + 0.1
    data = RiskData(
        history=history,
        target=target,
        adjacency=np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        capacities=np.ones(3),
        impedance=np.zeros(3),
        grades=np.ones(3),
        years=np.repeat(2019, count),
        airports=np.repeat("A", count),
        event_ids=np.array([f"E{i}" for i in range(count)]),
        node_ids=np.array(["a", "b", "c"]),
        factor_values=history.mean(1),
        factor_reference=history.mean((1, 2)),
        metadata={
            "provenance": "synthetic",
            "node_factor_indices": [0, 1, 2],
            "network_source": "independent fixture topology",
        },
        future_loads=target[:, None],
    ).validate()
    seen = []

    def grades(values, reference, threshold=0.3):
        return np.array([0.4, 0.7, 0.9]), np.array([0.4, 0.7, 0.9]) > threshold

    monkeypatch.setattr(network, "grey_relational", grades)

    class Estimator:
        def __init__(self, fold):
            self.fold = fold

        def fit(self, history, target, **kwargs):
            seen.append((history.shape[-1], target.shape[-1], kwargs["future_loads"].shape[-1]))
            return []

        def predict(self, history):
            return history[:, -1].copy()

        def weights(self, history):
            return np.broadcast_to(
                self.fold.adjacency * 0.2, (len(history),) + self.fold.adjacency.shape
            ).copy()

        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"fixture")

        def rollout(self, history, steps):
            return {
                "loads": np.repeat(
                    history[
                        :,
                        -1:,
                    ],
                    steps + 1,
                    axis=1,
                ),
                "weights": np.zeros(
                    (len(history), steps, len(self.fold.node_ids), len(self.fold.node_ids))
                ),
            }

    monkeypatch.setattr(experiments, "_make_model", lambda name, fold, *args: Estimator(fold))
    monkeypatch.setattr(experiments, "plot_results", lambda *args: None)
    report = experiments.run_experiments(
        data,
        tmp_path,
        profile="quick",
        progress=lambda *_: None,
        config_override={
            "epochs": 1,
            "model_names": ["DWPM"],
            "enabled_experiments": ["benchmark", "grey"],
            "grey_thresholds": [0.2, 0.5, 0.8],
            "permutations": 3,
            "network_mode": "external_fixed",
            "capacity_mode": "train_quantile",
            "inference_repetitions": 1,
            "propagation_steps": 1,
        },
    )
    rows = report["sensitivity"]["grey_threshold"]
    assert [(row["status"], row.get("retained_modeled_nodes")) for row in rows] == [
        ("completed", 3),
        ("completed", 2),
        ("not_trainable", None),
    ]
    assert (3, 3, 3) in seen and (2, 2, 2) in seen
    assert report["sensitivity"]["grey_shared_node_ids"] == ["b", "c"]
    assert (
        rows[0]["shared_node_metrics"]["evaluated_nodes"]
        == rows[1]["shared_node_metrics"]["evaluated_nodes"]
        == 2
    )
    assert report["sensitivity"]["grey_sweep_degenerate"] is False
    assert (tmp_path / "networks/grey_threshold_0.5.json").exists()


def test_experiment_resume_preserves_complete_trace_and_exact_training_sequence(tmp_path):
    from airport_risk.data import RiskData

    count, n = 8, 3
    history = 0.5 + np.arange(count * 3 * n).reshape(count, 3, n) % 11 / 10
    future = np.stack((history[:, -1] + 0.07, history[:, -1] + 0.14), axis=1)
    data = RiskData(
        history=history,
        target=future[:, 0],
        adjacency=np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        capacities=np.ones(n),
        impedance=np.zeros(n),
        grades=np.ones(n),
        years=np.repeat(2019, count),
        airports=np.repeat("A", count),
        event_ids=np.array([f"E{i}" for i in range(count)]),
        node_ids=np.array(["a", "b", "c"]),
        factor_values=history.mean(1),
        factor_reference=history.mean((1, 2)),
        metadata={"provenance": "synthetic", "node_factor_indices": [0, 1, 2]},
        future_loads=future,
    ).validate()
    train, test = np.arange(6), np.arange(6, 8)
    config = {
        **experiments.experiment_defaults("quick"),
        "epochs": 1,
        "warmup_epochs": 1,
        "hidden": 4,
        "layers": 1,
        "batch_size": 3,
        "cpu_threads": 1,
    }
    checkpoint = tmp_path / "first.pt"
    experiments._fit_score(
        "DWPM", data, train, test, data.adjacency, 19, config, checkpoint_path=checkpoint
    )
    resumed, metrics, trace = experiments._fit_score(
        "DWPM",
        data,
        train,
        test,
        data.adjacency,
        19,
        {**config, "epochs": 2, "resume_checkpoints": {"primary": "first.pt"}},
        checkpoint_path=tmp_path / "resumed.pt",
        resume_from=checkpoint,
    )
    whole, _, whole_trace = experiments._fit_score(
        "DWPM",
        data,
        train,
        test,
        data.adjacency,
        19,
        {**config, "epochs": 3},
        checkpoint_path=tmp_path / "whole.pt",
    )
    assert metrics["cumulative_training_epochs"] == 3
    assert metrics["cumulative_warmup_epochs"] == 1
    assert trace == whole_trace
    assert len(trace) == 4
    np.testing.assert_array_equal(
        resumed.predict(data.history[test]), whole.predict(data.history[test])
    )
