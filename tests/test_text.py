import numpy as np
import pytest

from airport_risk.text_mining import (
    extract_keywords,
    frequency_graph,
    heuristic_roles,
    motif_summary,
    thematic_clusters,
)


def test_keyword_boundaries_and_unsegmented_chinese():
    taxonomy = {"fog": ["fog"], "wet": ["湿滑"], "atc": ["ATC"]}
    result = extract_keywords(
        ["Fogginess was reported.", "FOG caused trouble.", "机场跑道湿滑导致事故，ATC未及时发现。"],
        taxonomy,
    )
    assert result == [[], ["fog"], ["wet", "atc"]]


def test_empty_or_string_taxonomy_terms_are_rejected():
    with pytest.raises(ValueError, match="empty"):
        extract_keywords(["fog"], {"x": [""]})
    with pytest.raises(ValueError, match="lists"):
        extract_keywords(["fog"], {"x": "fog"})


def test_first_causal_marker_controls_role_direction():
    result = heuristic_roles("Fog caused a collision because of delayed handling.")
    assert result[0]["cause_phrase"] == "Fog"
    assert result[0]["effect_phrase"] == "a collision because of delayed handling"
    backwards = heuristic_roles("A collision happened because of fog.")
    assert backwards[0]["cause_phrase"] == "fog"
    assert backwards[0]["effect_phrase"] == "A collision happened"
    assert backwards[0]["requires_expert_review"] is True
    assert heuristic_roles("Weather conditions were recorded.") == []
    assert heuristic_roles("because of fog") == []


def test_frequency_graph_uses_only_explicit_chain_order():
    graph = frequency_graph([["a", "b", "c"], ["a", "c"]], ["a", "b", "c"])
    np.testing.assert_array_equal(graph, [[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    with pytest.raises(ValueError, match="unique"):
        frequency_graph([], ["a", "a"])
    with pytest.raises(ValueError, match="unknown"):
        frequency_graph([["a", "x"]], ["a", "b"])
    with pytest.raises(ValueError, match="sequences"):
        frequency_graph(["ab"], ["a", "b"])


def test_motif_summary_direction_and_validation():
    graph = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    summary = motif_summary(graph)
    assert summary["branching_nodes"] == 1
    assert summary["converging_nodes"] == 1
    assert summary["chain_nodes"] == 1
    assert summary["hybrid_nodes"] == 0
    with pytest.raises(ValueError, match="square"):
        motif_summary(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="self"):
        motif_summary(np.eye(2))


def test_clustering_method_and_seed_are_repeatable():
    docs = [
        "fog reduces visibility",
        "dense fog and visibility",
        "radio equipment failure",
        "radio communication equipment",
    ]
    one = thematic_clusters(docs, n_clusters=2, seed=10)
    two = thematic_clusters(docs, n_clusters=2, seed=10)
    assert one == two
    assert len(one["labels"]) == 4
    assert one["method"] == "TF-IDF + KMeans"
    with pytest.raises(ValueError, match="n_clusters"):
        thematic_clusters(docs, n_clusters=5)


@pytest.mark.parametrize(
    "narrative,cause,effect",
    [
        ("由于跑道湿滑导致航空器偏出跑道。后续开展调查。", "跑道湿滑", "航空器偏出跑道"),
        ("调查指出，因通信中断造成指令传递延迟。", "通信中断", "指令传递延迟"),
        ("由于未按规定检查，导致设备故障。", "未按规定检查", "设备故障"),
    ],
)
def test_chinese_explicit_causal_candidates_require_review(narrative, cause, effect):
    result = heuristic_roles(narrative)
    assert result == [
        {
            "cause_phrase": cause,
            "effect_phrase": effect,
            "method": "rule_based_candidates",
            "requires_expert_review": True,
        }
    ]


@pytest.mark.parametrize(
    "narrative",
    [
        "未发现因果关系。",
        "未发现由于跑道湿滑导致事故的证据。",
        "没有证据表明因通信中断造成事故。",
        "事故并非由于跑道湿滑导致。",
        "由于天气变化并未导致延误。",
        "因通信中断未造成事故。",
        "研究因果关系造成了额外工作量。",
    ],
)
def test_chinese_denied_causality_and_noncausal_words_are_not_relations(narrative):
    assert heuristic_roles(narrative) == []


def test_chinese_topic_clusters_use_character_ngrams_and_auto_detection():
    docs = [
        "低能见度和跑道湿滑影响运行",
        "浓雾导致低能见度与跑道湿滑",
        "通信设备故障影响指令传递",
        "无线电设备发生通信故障",
    ]
    automatic = thematic_clusters(docs, n_clusters=2, seed=10)
    chinese = thematic_clusters(docs, n_clusters=2, seed=10, language="zh")
    assert automatic == chinese
    assert len(chinese["labels"]) == len(docs)
    terms = [term for group in chinese["top_terms"] for term in group]
    assert terms
    assert all(2 <= len(term) <= 4 for term in terms)
    assert any("通信" in term or "跑道" in term for term in terms)
    assert not any(doc in terms for doc in docs)


def test_explicit_english_topics_preserve_original_settings_and_validate_language():
    docs = [
        "fog reduces visibility",
        "dense fog and visibility",
        "radio equipment failure",
        "radio communication equipment",
    ]
    assert thematic_clusters(docs, n_clusters=2, seed=10) == thematic_clusters(
        docs, n_clusters=2, seed=10, language="en"
    )
    with pytest.raises(ValueError, match="language"):
        thematic_clusters(docs, n_clusters=2, language="unsupported")


@pytest.mark.parametrize(
    "narrative",
    [
        "Fog did not cause a collision.",
        "There is no evidence that fog caused a collision.",
        "A collision was never caused by fog.",
    ],
)
def test_english_denied_causality_is_not_an_affirmative_candidate(narrative):
    assert heuristic_roles(narrative) == []


def test_text_features_lock_vocabulary_and_idf_to_training():
    from airport_risk.text_mining import TextFeatureModel

    train = ["条件甲导致结果乙", "条件甲影响步骤丁", "步骤丁和结果乙"]
    model = TextFeatureModel(language="zh", n_clusters=2, seed=1).fit(train)
    vocabulary = model.vocabulary()
    before = model.vectorizer_.idf_.copy()
    features = model.transform(["新词戊新词己", "条件甲导致结果乙"])
    assert features.shape == (2, len(vocabulary))
    assert "新词" not in model.vocabulary()
    assert features[0].nnz == 0
    np.testing.assert_array_equal(model.vectorizer_.idf_, before)
    assert model.predict([train[0]]).shape == (1,)
    with pytest.raises(RuntimeError, match="fit"):
        TextFeatureModel().transform(train)
