"""文本接口测试使用独立语言夹具，不包含机场调查记录。"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from airport_risk.nlp import (
    FactorCatalog,
    HanLPSRLBackend,
    ImportedSRLBackend,
    approved_chains,
    relations_from_srl,
    review_relation,
    rule_relations,
    validate_srl_frames,
)


def _catalog():
    return FactorCatalog(
        [
            {"id": "a", "name": "条件甲", "definition": "输入条件甲", "aliases": ["条件甲"]},
            {"id": "b", "name": "结果乙", "definition": "输出结果乙", "aliases": ["结果乙"]},
        ]
    )


def test_factor_registry_has_98_unique_definitions_and_table_nodes():
    catalog = FactorCatalog.load(Path(__file__).parents[1] / "configs/factors.json")
    assert len(catalog.ids) == len(set(catalog.ids)) == 98
    assert sum(item["catalog_group"] == "paper_table_1" for item in catalog.factors) == 63
    assert {"a36", "b8", "c13", "d6", "x35"} <= set(catalog.ids)
    assert all(item["definition"] and item["aliases"] for item in catalog.factors)


def test_rule_direction_and_review_retains_negation_uncertainty():
    affirmative = rule_relations("由于条件甲导致结果乙。")
    assert affirmative[0]["cause_phrase"] == "条件甲"
    assert affirmative[0]["effect_phrase"] == "结果乙"
    assert affirmative[0]["polarity"] == "affirmative"
    assert rule_relations("条件甲并未导致结果乙。")[0]["polarity"] == "negated"
    assert rule_relations("未发现条件甲导致结果乙的证据。")[0]["polarity"] == "negated"
    assert rule_relations("疑似条件甲导致结果乙。")[0]["certainty"] == "uncertain"
    backwards = rule_relations("Result B happened because of condition A.")[0]
    assert backwards["cause_phrase"] == "condition A"
    assert backwards["effect_phrase"] == "Result B happened"
    assert (
        rule_relations("条件甲导致结果乙，条件丙引发结果丁。")
        and len(rule_relations("条件甲导致结果乙，条件丙引发结果丁。")) == 2
    )
    assert rule_relations("同时观察到条件甲和结果乙。") == []


def test_semantic_frames_validate_offsets_and_only_causal_predicate_edges():
    tokens = ["条件甲", "导致", "结果乙"]
    frames = [[["条件甲", "ARG0", 0, 1], ["导致", "PRED", 1, 2], ["结果乙", "ARG1", 2, 3]]]
    relations = relations_from_srl(tokens, frames)
    assert relations[0]["cause_phrase"] == "条件甲"
    assert relations[0]["method"] == "imported_srl"
    assert relations[0]["review_status"] == "pending"
    noncausal = [[["条件甲", "ARG0", 0, 1], ["观察", "PRED", 1, 2], ["结果乙", "ARG1", 2, 3]]]
    assert relations_from_srl(["条件甲", "观察", "结果乙"], noncausal) == []
    with pytest.raises(ValueError, match="区间不一致"):
        validate_srl_frames(tokens, [[["错误文本", "PRED", 0, 1]]])
    with pytest.raises(ValueError, match="一个 PRED"):
        validate_srl_frames(tokens, [[["条件甲", "ARG0", 0, 1]]])


def test_imported_srl_rejects_changed_source_and_missing_event(tmp_path):
    item = {
        "text": "条件甲导致结果乙",
        "tokens": ["条件甲", "导致", "结果乙"],
        "frames": [[["条件甲", "ARG0", 0, 1], ["导致", "PRED", 1, 2], ["结果乙", "ARG1", 2, 3]]],
    }
    path = tmp_path / "srl.json"
    path.write_text(json.dumps({"fixture": item}, ensure_ascii=False))
    backend = ImportedSRLBackend.load(path)
    assert backend.extract(item["text"], event_id="fixture")[0]["effect_phrase"] == "结果乙"
    with pytest.raises(ValueError, match="不一致"):
        backend.extract("改变后的文本", event_id="fixture")
    with pytest.raises(KeyError):
        backend.extract("条件甲", event_id="absent")


def test_hanlp_local_adapter_obeys_official_document_api(tmp_path, monkeypatch):
    model = tmp_path / "local-model"
    model.mkdir()
    calls = []

    def component(text, *, tasks):
        calls.append((text, tasks))
        return {
            "tok/fine": ["条件甲", "导致", "结果乙"],
            "srl": [[["条件甲", "ARG0", 0, 1], ["导致", "PRED", 1, 2], ["结果乙", "ARG1", 2, 3]]],
        }

    monkeypatch.setitem(sys.modules, "hanlp", SimpleNamespace(load=lambda path: component))
    result = HanLPSRLBackend(model).extract("条件甲导致结果乙")
    assert calls == [("条件甲导致结果乙", "srl")]
    assert result[0]["method"] == "hanlp_neural_srl"
    with pytest.raises(FileNotFoundError):
        HanLPSRLBackend(tmp_path / "absent")


def test_review_state_controls_edges_and_rejects_false_approval():
    catalog = _catalog()
    relation = catalog.annotation_template(
        rule_relations("条件甲导致结果乙")[0], event_id="fixture"
    )
    assert relation["cause_factor_candidates"] == ["a"]
    assert approved_chains([relation]) == []
    approved = review_relation(
        relation,
        status="approved",
        reviewer="reviewer",
        cause_factor_ids=["a"],
        effect_factor_ids=["b"],
        catalog=catalog,
    )
    assert approved_chains([approved]) == [["a", "b"]]
    with pytest.raises(ValueError, match="目录外"):
        review_relation(
            relation,
            status="approved",
            reviewer="reviewer",
            cause_factor_ids=["absent"],
            effect_factor_ids=["b"],
            catalog=catalog,
        )
    denied = rule_relations("条件甲未导致结果乙")[0]
    with pytest.raises(ValueError, match="否定"):
        review_relation(
            denied,
            status="approved",
            reviewer="reviewer",
            cause_factor_ids=["a"],
            effect_factor_ids=["b"],
        )
    uncertain = {**approved, "certainty": "uncertain"}
    assert approved_chains([uncertain]) == []
    assert approved_chains([uncertain], include_uncertain=True) == [["a", "b"]]


def test_srl_and_rules_do_not_approve_negated_predicate_or_conflate_cause_argument():
    assert rule_relations("Condition A did not cause result B.")[0]["polarity"] == "negated"
    tokens = ["条件甲", "并未", "导致", "结果乙"]
    frames = [[["条件甲", "ARG0", 0, 1], ["导致", "PRED", 2, 3], ["结果乙", "ARG1", 3, 4]]]
    assert relations_from_srl(tokens, frames)[0]["polarity"] == "negated"
    tokens = ["由于条件甲", "发生", "结果乙"]
    frames = [[["由于条件甲", "ARGM-CAU", 0, 1], ["发生", "PRED", 1, 2], ["结果乙", "ARG1", 2, 3]]]
    relation = relations_from_srl(tokens, frames)[0]
    assert "条件甲" not in relation["effect_phrase"]
    assert "结果乙" in relation["effect_phrase"]
