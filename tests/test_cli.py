"""命令入口、训练检查点与预测产物的集成验证。"""

import json

import numpy as np

from airport_risk import cli
from airport_risk.data import RiskData
from airport_risk.experiments import experiment_defaults
from airport_risk.models import CCRLModel


def test_train_resume_infer_rollout_share_node_contract(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)
    history = rng.uniform(0.2, 0.8, (10, 3, 3))
    target = history[:, -1] + 0.1
    data = RiskData(
        history,
        target,
        np.zeros((3, 3)),
        np.ones(3),
        np.zeros(3),
        np.ones(3),
        np.full(10, 2019),
        np.full(10, "FIXTURE"),
        np.array([f"F{i}" for i in range(10)]),
        np.array(["n0", "n1", "n2"]),
        history.mean(1),
        history.mean((1, 2)),
        {
            "provenance": "synthetic",
            "source_id": "software integration fixture",
            "measurement_type": "observed_load",
            "load_units": "dimensionless_normalized",
            "node_factor_indices": [0, 1, 2],
            "annotated_chains": {},
        },
        future_loads=target[:, None],
    ).validate()
    data_path = tmp_path / "data.npz"
    data.save(data_path)
    settings = experiment_defaults("quick")
    settings.update(
        epochs=1,
        hidden=4,
        layers=1,
        kernel_size=2,
        cpu_threads=1,
        grey_threshold=0.0,
        capacity_mode="train_quantile",
        warmup_epochs=0,
        expert_edges=[
            {
                "source": "n0",
                "target": "n1",
                "status": "approved",
                "reviewer": "fixture",
                "basis": "software integration graph",
            },
            {
                "source": "n1",
                "target": "n2",
                "status": "approved",
                "reviewer": "fixture",
                "basis": "software integration graph",
            },
        ],
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema_version": 2, "experiment": settings}))
    original = cli.preflight

    def fixture_preflight(*args):
        result = original(*args)
        assert result["issues"] == ["实验入口需有来源声明的数据；测试夹具不可用于实证评价"]
        return {**result, "ready": True, "issues": [], "scope": "software_integration_fixture"}

    monkeypatch.setattr(cli, "preflight", fixture_preflight)
    common = ["--config", str(config), "--data", str(data_path)]
    first, second = tmp_path / "first", tmp_path / "second"
    assert cli.main(["train", *common, "--out", str(first)]) == 0
    assert (
        cli.main(["train", *common, "--resume", str(first / "training.pt"), "--out", str(second)])
        == 0
    )
    model = CCRLModel.load_training(second / "training.pt")
    assert model._training_state["epochs_done"] == 2
    assert tuple(model.node_ids) == ("n0", "n1", "n2")
    saved_history = tmp_path / "history.npy"
    np.save(saved_history, history[:2])
    prediction, trajectory = tmp_path / "prediction.npy", tmp_path / "rollout.npz"
    assert (
        cli.main(
            [
                "infer",
                "--checkpoint",
                str(second / "model.pt"),
                "--history",
                str(saved_history),
                "--out",
                str(prediction),
            ]
        )
        == 0
    )
    assert (
        cli.main(
            [
                "rollout",
                "--checkpoint",
                str(second / "model.pt"),
                "--history",
                str(saved_history),
                "--steps",
                "2",
                "--out",
                str(trajectory),
            ]
        )
        == 0
    )
    np.testing.assert_array_equal(np.load(prediction), model.predict(history[:2]))
    with np.load(trajectory, allow_pickle=False) as arrays:
        assert arrays["loads"].shape == (2, 3, 3)
        assert arrays["weights"].shape == (2, 2, 3, 3)
    manifest = json.loads((first / "run_manifest.json").read_text())
    assert manifest["preflight"]["scope"] == "software_integration_fixture"
