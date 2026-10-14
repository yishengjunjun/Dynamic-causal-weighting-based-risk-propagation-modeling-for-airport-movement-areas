"""样本构建使用软件夹具，验证来源与时间边界。"""

import csv
import json

import numpy as np
import pytest

from airport_risk.data import RiskData, split_indices
from airport_risk.preparation import prepare_annotations, prepare_observations


def observations(tmp_path, *, missing=False):
    path = tmp_path / "observations.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["event_id", "airport", "timestamp", "node_id", "load"])
        for event in range(4):
            for step in range(6):
                for node in range(2):
                    if missing and step == 2 and node == 1:
                        continue
                    writer.writerow(
                        [
                            f"fixture-{event}",
                            "FIXTURE",
                            f"2019-01-01T00:0{step}:00+08:00",
                            f"n{node}",
                            event + step + node,
                        ]
                    )
    return path


def test_windows_are_grouped_by_event_and_future_never_enters_features(tmp_path):
    data = prepare_observations(
        observations(tmp_path),
        ["n0", "n1"],
        source_id="software fixture",
        history_steps=2,
        future_steps=2,
        provenance="synthetic",
    )
    assert data.history.shape == (12, 2, 2)
    assert data.future_loads.shape == (12, 2, 2)
    assert len(set(data.sample_ids)) == 12
    assert len(set(data.event_ids)) == 4
    np.testing.assert_array_equal(data.target, data.future_loads[:, 0])
    np.testing.assert_allclose(data.factor_values, data.history.mean(1))
    train, _, test = split_indices(data, seed=4)
    assert not set(data.event_ids[train]) & set(data.event_ids[test])
    path = tmp_path / "windows.npz"
    data.save(path)
    assert RiskData.load(path).fingerprint() == data.fingerprint()
    changed = RiskData.load(path)
    changed.future_loads[:, 1] += 10
    assert changed.fingerprint() != data.fingerprint()


def test_missing_load_requires_explicit_past_only_policy(tmp_path):
    path = observations(tmp_path, missing=True)
    with pytest.raises(ValueError, match="缺少节点"):
        prepare_observations(path, ["n0", "n1"], source_id="fixture", history_steps=2)
    data = prepare_observations(
        path,
        ["n0", "n1"],
        source_id="fixture",
        history_steps=2,
        missing="forward_fill",
        provenance="synthetic",
    )
    assert data.history[1, 1, 1] == data.history[1, 0, 1]


def test_date_only_and_irregular_times_are_rejected(tmp_path):
    path = observations(tmp_path)
    text = path.read_text()
    path.write_text(text.replace("2019-01-01T00:00:00+08:00", "2019-01-01"))
    with pytest.raises(ValueError, match="完整日期及时间"):
        prepare_observations(path, ["n0", "n1"], source_id="fixture")
    path.write_text(text.replace("00:02:00", "00:02:30"))
    with pytest.raises(ValueError, match="间隔不规则"):
        prepare_observations(path, ["n0", "n1"], source_id="fixture")


def annotations(tmp_path):
    path = tmp_path / "annotations.json"
    doc = {
        "schema_version": 1,
        "events": [
            {
                "event_id": f"f{i}",
                "airport": "FIXTURE",
                "year": 2019,
                "reviewed": True,
                "reviewer": "test",
                "causal_chains": [["n0", "n1", "n2"]],
            }
            for i in range(4)
        ],
    }
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path, doc


def test_reviewed_chain_encoding_is_explicit_causal_stage(tmp_path):
    path, doc = annotations(tmp_path)
    data = prepare_annotations(
        path, ["n0", "n1", "n2"], source_id="fixture", provenance="synthetic"
    )
    np.testing.assert_array_equal(data.history[0], [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(data.target[0], [1, 1, 0])
    assert data.metadata["time_axis"] == "causal_stage"
    assert data.metadata["measurement_type"] == "coded_annotations"
    doc["events"][0]["reviewed"] = False
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="尚未审核"):
        prepare_annotations(path, ["n0", "n1", "n2"], source_id="fixture")


def test_group_metadata_and_future_contract(tmp_path):
    data = prepare_observations(
        observations(tmp_path),
        ["n0", "n1"],
        source_id="fixture",
        history_steps=2,
        provenance="synthetic",
    )
    data.years[0] = 2020
    with pytest.raises(ValueError, match="share its year"):
        data.validate()
    data.years[0] = 2019
    data.future_loads[0, 0, 0] += 1
    with pytest.raises(ValueError, match="target must equal"):
        data.validate()


def test_shared_node_chains_keep_all_precedence_and_cycles_require_explicit_stages(tmp_path):
    path, doc = annotations(tmp_path)
    for event in doc["events"]:
        event["causal_chains"] = [["n0", "n1", "n2"], ["n3", "n0"]]
    path.write_text(json.dumps(doc))
    data = prepare_annotations(
        path, ["n0", "n1", "n2", "n3"], source_id="fixture", provenance="synthetic"
    )
    np.testing.assert_array_equal(data.history[0, 1], [0, 0, 0, 1])
    np.testing.assert_array_equal(data.target[0], [1, 0, 0, 1])
    doc["events"][0]["causal_chains"] = [["n0", "n1"], ["n1", "n0"]]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="循环因果链"):
        prepare_annotations(path, ["n0", "n1", "n2", "n3"], source_id="fixture")


def test_synthetic_source_markers_survive_preparation(tmp_path):
    path, doc = annotations(tmp_path)
    doc["source"] = {"contains_synthetic_markers": True}
    path.write_text(json.dumps(doc))
    data = prepare_annotations(path, ["n0", "n1", "n2"], source_id="fixture")
    assert data.metadata["provenance"] == "synthetic"


def test_cross_year_event_assigned_to_latest_observation_year(tmp_path):
    path = observations(tmp_path)
    text = path.read_text()
    text = text.replace("2019-01-01T00:00:00+08:00", "2018-12-31T23:59:00+08:00")
    for step in range(1, 6):
        text = text.replace(f"2019-01-01T00:0{step}:00+08:00", f"2019-01-01T00:0{step-1}:00+08:00")
    path.write_text(text)
    data = prepare_observations(
        path,
        ["n0", "n1"],
        source_id="fixture",
        history_steps=2,
        load_units="fixture_units",
        provenance="synthetic",
    )
    assert (data.years == 2019).all()
    assert data.metadata["event_year_rule"] == "latest_observation_local_year"
