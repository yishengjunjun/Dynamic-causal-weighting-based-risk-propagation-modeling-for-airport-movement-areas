"""独立验证动态闭环、成对控制和未来观测隔离。"""

from types import SimpleNamespace
import numpy as np

from airport_risk.core import propagate_step
from airport_risk.dynamic import _rollout_batch, stress_controls, stress_attacks


def scenario():
    return SimpleNamespace(
        adjacency=np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]),
        capacities=np.ones(3),
        impedance=np.zeros(3),
        node_ids=np.array(["a", "b", "c"]),
    )


class HistoryWeights:
    def __init__(self, data):
        self.data = data
        self.observed = []

    def weights(self, history):
        self.observed.append(history.copy())
        strength = np.minimum(history[:, -1].mean(1), 2) / 2
        return strength[:, None, None] * self.data.adjacency

    def propagate_numpy(self, loads, weights, capacities=None):
        return propagate_step(
            loads,
            weights,
            self.data.capacities if capacities is None else capacities,
            self.data.impedance,
        )


def test_dynamic_weights_are_recomputed_after_each_prediction_and_match_scalar_equations():
    data = scenario()
    model = HistoryWeights(data)
    histories = np.array([[[0.3, 0.1, 0.2], [0.7, 0.2, 0.3], [1.2, 0.6, 0.4]]])
    result = _rollout_batch(
        data,
        model,
        histories,
        steps=3,
        mode="dynamic",
        frozen_weights=None,
        capacities=data.capacities,
        retain_paths=True,
    )
    independent = histories.copy()
    for step in range(3):
        np.testing.assert_array_equal(model.observed[step], independent)
        strength = min(independent[0, -1].mean(), 2) / 2
        expected = propagate_step(
            independent[0, -1], data.adjacency * strength, data.capacities, data.impedance
        )
        np.testing.assert_allclose(result["loads"][0, step + 1], expected)
        independent = np.concatenate((independent[:, 1:], expected[None, None]), axis=1)
    assert not np.array_equal(result["weights"][0, 0], result["weights"][0, -1])


def test_frozen_comparison_never_recomputes_weights():
    data = scenario()
    model = HistoryWeights(data)
    histories = np.array([[[0.2, 0.3, 0.4], [1.2, 0.6, 0.4]]])
    result = _rollout_batch(
        data,
        model,
        histories,
        steps=3,
        mode="frozen",
        frozen_weights=data.adjacency * 0.2,
        capacities=data.capacities,
        retain_paths=True,
    )
    assert model.observed == []
    np.testing.assert_array_equal(result["weights"][0, 0], result["weights"][0, -1])


def test_control_modes_share_attacks_budgets_random_draws_and_do_not_read_future():
    data = scenario()
    histories = np.array([[[0.2, 0.3, 0.4], [0.9, 0.8, 0.7]], [[0.3, 0.2, 0.4], [1.1, 0.9, 0.8]]])
    settings = dict(
        steps=2,
        random_repetitions=3,
        seed=37,
        control_fraction=0.5,
        control_reduction=0.3,
        batch_size=2,
    )
    dynamic = stress_controls(data, HistoryWeights(data), histories, mode="dynamic", **settings)
    frozen = stress_controls(
        data,
        HistoryWeights(data),
        histories,
        mode="frozen",
        frozen_weights=data.adjacency * 0.2,
        **settings
    )
    assert dynamic["random_selection_sha256"] == frozen["random_selection_sha256"]
    assert dynamic["attacks_per_strategy"] == frozen["attacks_per_strategy"] == 6
    assert dynamic["controlled_nodes"] == frozen["controlled_nodes"] == 2
    assert dynamic["control_intensity"] == frozen["control_intensity"] == 0.3
    data.target = np.full((2, 3), 9999.0)
    again = stress_controls(data, HistoryWeights(data), histories, mode="dynamic", **settings)
    assert again == dynamic
    assert np.any(np.abs(np.array(dynamic["curves"]["none"]) - frozen["curves"]["none"]) > 1e-9)


def test_controls_preserve_negative_recovery_increments():
    data = scenario()
    histories = np.array([[[1.0, 1.0, 1.0], [1.5, 1.6, 1.7]]])

    class RecoveringModel:
        def propagate_numpy(self, loads, weights, capacities=None):
            return loads - 0.1

    result = _rollout_batch(
        data,
        RecoveringModel(),
        histories,
        steps=2,
        mode="frozen",
        frozen_weights=data.adjacency,
        capacities=data.capacities,
        strategy="degree",
        selection=np.ones((1, 3), dtype=bool),
        control_reduction=1,
        retain_paths=True,
    )
    np.testing.assert_allclose(result["final_loads"], histories[:, -1] - 0.2)


def test_capacity_sensitivity_passes_physical_capacity_override():
    data = scenario()
    model = HistoryWeights(data)
    history = np.array([[[0.2, 0.3, 0.4], [1.2, 0.6, 0.4]]])
    result = _rollout_batch(
        data,
        model,
        history,
        steps=1,
        mode="frozen",
        frozen_weights=data.adjacency * 0.2,
        capacities=np.ones(3) * 2,
        retain_paths=True,
    )
    # 提高容量后全部节点均不激活，负载保持原值。
    np.testing.assert_array_equal(result["final_loads"], history[:, -1])


def test_attack_results_are_invariant_to_batch_partition():
    data = scenario()
    histories = np.array([[[0.2, 0.3, 0.4], [0.9, 0.8, 0.7]], [[0.3, 0.2, 0.4], [1.1, 0.9, 0.8]]])
    small = stress_attacks(
        data, HistoryWeights(data), histories, steps=2, perturbation_levels=[0.1], batch_size=1
    )
    large = stress_attacks(
        data, HistoryWeights(data), histories, steps=2, perturbation_levels=[0.1], batch_size=8
    )
    for key in ("nri", "perturbation_global_load_change"):
        np.testing.assert_allclose(small[key]["10.0"], large[key]["10.0"], equal_nan=True)
    np.testing.assert_allclose(small["ndi"], large["ndi"])
