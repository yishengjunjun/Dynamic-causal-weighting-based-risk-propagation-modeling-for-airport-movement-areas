"""配置与命令入口的边界测试。"""

from copy import deepcopy
import json

import numpy as np
import pytest

from airport_risk import cli
from airport_risk.data import RiskData
from airport_risk.experiments import experiment_defaults
from airport_risk.reconstruction import preflight, validate_config, resolve_config, source_manifest


def explicit_config():
    experiment = experiment_defaults("quick")
    experiment.update(capacity_mode="train_quantile", enabled_experiments=["benchmark"])
    return {"schema_version": 2, "experiment": experiment}


def input_fixture():
    count = 12
    return RiskData(
        history=np.full((count, 3, 2), 0.4),
        target=np.full((count, 2), 0.5),
        adjacency=np.array([[0, 1], [0, 0]]),
        capacities=np.ones(2),
        impedance=np.zeros(2, dtype=bool),
        grades=np.array([0.4, 0.6]),
        years=np.full(count, 2019),
        airports=np.full(count, "FIXTURE"),
        event_ids=np.array([f"FIXTURE_{i}" for i in range(count)]),
        node_ids=np.array(["a1", "b1"]),
        factor_values=np.ones((count, 2)),
        factor_reference=np.ones(count),
        metadata={
            "provenance": "synthetic",
            "node_factor_indices": [0, 1],
            "measurement_type": "observed_load",
            "source_id": "software fixture",
            "load_units": "source_declared",
        },
        future_loads=np.full((count, 1, 2), 0.5),
    ).validate()


def write_inputs(tmp_path, config, data=None):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False))
    data_path = tmp_path / "fixture.npz" if data is not None else None
    if data is not None:
        data.save(data_path)
    return config_path, data_path


def test_filled_default_and_paper_configs_are_valid():
    from pathlib import Path

    for path in (Path(__file__).parents[1] / "configs").glob("*.json"):
        if path.name == "factors.json":
            continue
        assert validate_config(json.loads(path.read_text())) == []
    config = resolve_config(explicit_config())
    assert config["experiment"]["enabled_protocols"] == ["primary"]
    assert config["preparation"]["history_steps"] >= 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("epochs", True),
        ("lr", float("inf")),
        ("gamma", 1),
        ("var_ridge", -1),
        ("eta", 0),
        ("eta_scales", [2]),
        ("stratified_train_fraction", 1),
        ("control_fraction", 0),
        ("control_reduction", 2),
        ("strict_state_input", 1),
        ("seeds", [1, 1]),
        ("seeds", [2**64]),
        ("optimizer", "unknown"),
        ("warmup_epochs", -1),
    ],
)
def test_invalid_parameters_rejected(key, value):
    config = explicit_config()
    config["experiment"][key] = value
    assert validate_config(config)


def test_unknown_schema_and_preparation_keys_rejected():
    config = explicit_config()
    config["schema_version"] = True
    assert validate_config(config)
    config = explicit_config()
    config["preparation"] = {"history_steps": 1}
    assert validate_config(config)
    config = explicit_config()
    config["invented"] = 1
    assert validate_config(config)


def test_software_fixture_cannot_enter_empirical_cli(tmp_path):
    cp, dp = write_inputs(tmp_path, explicit_config(), input_fixture())
    result = preflight(cp, dp)
    assert not result["ready"]
    assert any("测试夹具不可" in x for x in result["issues"])
    assert result["data_fingerprint"]


def test_only_enabled_protocols_require_matching_coverage(tmp_path):
    data = input_fixture()
    data.metadata["provenance"] = "authorized_real"
    config = explicit_config()
    cp, dp = write_inputs(tmp_path, config, data)
    assert preflight(cp, dp)["ready"]
    config["experiment"].update(
        enabled_protocols=["primary", "rolling", "airport"],
        rolling_test_years=[2025],
        airport_holdout="UNKNOWN",
    )
    cp, dp = write_inputs(tmp_path, config, data)
    issues = preflight(cp, dp)["issues"]
    assert sum("划分" in issue for issue in issues) == 2


def test_transition_needs_future_and_units_match_capacity(tmp_path):
    data = input_fixture()
    data.metadata["provenance"] = "authorized_real"
    data.future_loads = None
    config = explicit_config()
    config["experiment"]["capacity_mode"] = "grey_topology"
    cp, dp = write_inputs(tmp_path, config, data)
    issues = preflight(cp, dp)["issues"]
    assert any("future_loads" in issue for issue in issues)
    assert any("有量纲" in issue for issue in issues)


def test_invalid_input_never_launches_training(tmp_path, monkeypatch):
    cp, _ = write_inputs(tmp_path, explicit_config())
    invoked = []
    monkeypatch.setattr(cli, "run_experiments", lambda *a, **k: invoked.append(True))
    assert (
        cli.main(
            [
                "run",
                "--config",
                str(cp),
                "--data",
                str(tmp_path / "missing.npz"),
                "--out",
                str(tmp_path / "run"),
            ]
        )
        == 2
    )
    assert not invoked and not (tmp_path / "run").exists()


def test_cli_has_no_event_fabrication_command():
    with pytest.raises(SystemExit):
        cli.main(["generate"])


def test_file_changed_after_preflight_cannot_launch(tmp_path, monkeypatch):
    cp, dp = write_inputs(tmp_path, explicit_config(), input_fixture())
    monkeypatch.setattr(
        cli,
        "preflight",
        lambda *a: {
            "ready": True,
            "config_sha256": "outdated",
            "data_fingerprint": "outdated",
            "issues": [],
        },
    )
    invoked = []
    monkeypatch.setattr(cli, "run_experiments", lambda *a, **k: invoked.append(True))
    with pytest.raises(ValueError, match="输入文件发生变化"):
        cli.main(["run", "--config", str(cp), "--data", str(dp), "--out", str(tmp_path / "run")])
    assert not invoked and not (tmp_path / "run").exists()


def test_installed_source_manifest_is_nonempty(tmp_path):
    manifest = source_manifest(tmp_path / "site-packages-parent")
    assert "installed/airport_risk/cli.py" in manifest
    assert all(len(value) == 64 for value in manifest.values())
