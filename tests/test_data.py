from copy import deepcopy

import numpy as np
import pytest

from airport_risk.data import RiskData, node_catalog, split_indices


def make_data(years=None):
    years = np.array([2008, 2008, 2009, 2009, 2010, 2010] if years is None else years)
    b = len(years)
    return RiskData(
        history=np.ones((b, 3, 2)) * 0.4,
        target=np.ones((b, 2)) * 0.5,
        adjacency=np.array([[0, 1], [0, 0]]),
        capacities=np.ones(2),
        impedance=np.array([False, True]),
        grades=np.array([0.4, 0.6]),
        years=years,
        airports=np.array(["A" if i % 2 else "B" for i in range(b)]),
        event_ids=np.array([f"E{i}" for i in range(b)]),
        node_ids=np.array(["a1", "b1"]),
        factor_values=np.ones((b, 3)),
        factor_reference=np.ones(b),
        metadata={"provenance": "synthetic", "seed": 1},
    ).validate()


def test_save_load_without_pickle_and_complete_fingerprint(tmp_path):
    data = make_data()
    data.event_ids = data.event_ids.astype(object)
    data.save(tmp_path / "data.npz")
    restored = RiskData.load(tmp_path / "data.npz")
    assert restored.fingerprint() == data.fingerprint()
    assert restored.event_ids.dtype.kind == "U"
    for field, value in [
        ("grades", np.array([0.5, 0.6])),
        ("node_ids", np.array(["a2", "b1"])),
        ("metadata", {"provenance": "synthetic", "seed": 2}),
    ]:
        changed = deepcopy(data)
        setattr(changed, field, value)
        assert changed.fingerprint() != data.fingerprint()
    reordered = deepcopy(data)
    reordered.metadata = {"seed": 1, "provenance": "synthetic"}
    assert reordered.fingerprint() == data.fingerprint()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("impedance", np.array([0, 2]), "binary"),
        ("impedance", np.array([0, np.nan]), "binary"),
        ("node_ids", np.array([["a1", "b1"]]), "node IDs"),
        ("node_ids", np.array(["a1", "a1"]), "node IDs"),
        ("factor_values", np.ones(6), "factor_values"),
        ("factor_reference", np.ones((6, 1)), "factor/reference"),
        ("years", np.array([2008, 2008, 2009, 2009, 2010, 2010.5]), "integer"),
        ("grades", np.array([-0.1, 0.6]), "grades"),
        ("metadata", {"provenance": "synthetic", "bad": float("nan")}, "JSON"),
    ],
)
def test_dataset_rejects_malformed_attributes(field, value, match):
    data = make_data()
    setattr(data, field, value)
    with pytest.raises(ValueError, match=match):
        data.validate()


def test_dataset_rejects_empty_events():
    data = make_data()
    data.history = data.history[:0]
    data.target = data.target[:0]
    with pytest.raises(ValueError, match="history"):
        data.validate()


def test_stratified_split_keeps_whole_events_disjoint_and_deterministic():
    data = make_data()
    train, validation, test = split_indices(data, protocol="stratified", seed=5)
    assert len(validation) == 0
    assert not set(train) & set(test)
    np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(6))
    np.testing.assert_array_equal(data.years[train], [2008, 2009, 2010])
    np.testing.assert_array_equal(data.years[test], [2008, 2009, 2010])
    again = split_indices(data, protocol="stratified", seed=5)
    np.testing.assert_array_equal(train, again[0])
    np.testing.assert_array_equal(test, again[2])


def test_singleton_year_cannot_silently_produce_empty_training_stratum():
    data = make_data([2008, 2009, 2010])
    with pytest.raises(ValueError, match="each year"):
        split_indices(data, protocol="stratified")


def test_rolling_and_airport_splits_respect_external_boundaries():
    data = make_data(np.repeat(np.arange(2008, 2022), 2))
    train, validation, test = split_indices(data, protocol="rolling", test_year=2020)
    assert set(data.years[train]) == set(range(2008, 2017))
    assert set(data.years[validation]) == {2017, 2018, 2019}
    assert set(data.years[test]) == {2020}
    assert not set(train) & set(validation) and not set(train) & set(test)
    train, validation, test = split_indices(data, protocol="airport", holdout="B")
    assert set(data.airports[train]) == {"A"}
    assert set(data.airports[test]) == {"B"}
    assert len(validation) == 0


def test_authorized_real_requires_explicit_factor_mapping():
    data = make_data()
    data.metadata["provenance"] = "authorized_real"
    with pytest.raises(ValueError, match="node_factor_indices"):
        data.validate()


def test_airport_holdout_cannot_be_selected_implicitly():
    with pytest.raises(ValueError, match="explicit holdout"):
        split_indices(make_data(), protocol="airport")


def test_node_catalog_preserves_published_ids_and_distinct_duplicate_labels():
    ids, names = node_catalog()
    assert len(ids) == len(names) == 63
    assert len(set(ids)) == 63
    assert names[list(ids).index("c4")] == names[list(ids).index("c10")]
