"""事件关键词、显式因果候选、训练域文本特征与有向链统计。"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


def extract_keywords(narratives, taxonomy):
    """词面匹配采用 ASCII 单词边界和中文连续子串。

    该匹配仅作为关键词基线，不处理否定或因果归因。
    分类词表由输入配置指定；基于语料构建时仅使用训练集。
    """
    patterns = {}
    for factor, terms in taxonomy.items():
        if isinstance(terms, str) or not terms:
            raise ValueError("taxonomy surface forms must be nonempty lists of strings")
        compiled = []
        for term in terms:
            if not isinstance(term, str) or not term.strip():
                raise ValueError("taxonomy surface forms cannot be empty")
            term = term.strip().casefold()
            # Unicode 的 \w 也包含汉字，英文边界断言会漏掉
            # 普通未分词中文句子内部的关键词。
            left = (
                r"(?<![a-z0-9_])"
                if term[0].isascii() and (term[0].isalnum() or term[0] == "_")
                else ""
            )
            right = (
                r"(?![a-z0-9_])"
                if term[-1].isascii() and (term[-1].isalnum() or term[-1] == "_")
                else ""
            )
            compiled.append(re.compile(left + re.escape(term) + right))
        patterns[factor] = compiled
    labels = []
    for text in narratives:
        if not isinstance(text, str):
            raise ValueError("narratives must be strings")
        text = text.casefold()
        labels.append(
            [
                factor
                for factor, terms in patterns.items()
                if any(pattern.search(text) for pattern in terms)
            ]
        )
    return labels


def heuristic_roles(narrative):
    """按显式句式提取待审核的中英文因果候选关系。

    中文仅识别明确的“由于……导致……”和“因……造成……”句式，
    并过滤显式否定表达；候选关系须经因果审核。
    英文关系方向由首个明确因果连接词决定。
    """
    if not isinstance(narrative, str):
        raise ValueError("narrative must be a string")
    chinese_pattern = re.compile(
        r"(?:由于|(?<![原诱基根])因(?![果素])(?:为)?)(?P<cause>[^。！？!?；;\n]+?)"
        r"(?P<connector>导致|造成)(?P<effect>[^。！？!?；;\n]+)"
    )
    denial_before = re.compile(
        r"未发现|未证明|未证实|未能证实|尚未证实|没有证据|无证据|"
        r"无法证实|不能认定|不能认为|并非|并不是|不是|否认"
    )
    denial_connector = re.compile(
        r"(?:未|没有|并未|不|并不|不会|未能|尚未)(?:直接|必然|实际|明确)?$"
    )
    for match in chinese_pattern.finditer(narrative):
        marker = narrative[match.start() :]
        connector = match.group("connector")
        # 仅接受这两组明确句式，不将其他词面组合视为因果标注。
        if (marker.startswith("由于") and connector != "导致") or (
            not marker.startswith("由于") and connector != "造成"
        ):
            continue
        prefix = re.split(r"[。！？!?；;，,\n]", narrative[: match.start()])[-1]
        cause = match.group("cause").strip(" ，,：:")
        effect = match.group("effect").strip(" ，,：:")
        if (
            not cause
            or not effect
            or denial_before.search(prefix)
            or denial_connector.search(cause)
        ):
            continue
        return [
            {
                "cause_phrase": cause,
                "effect_phrase": effect,
                "method": "rule_based_candidates",
                "requires_expert_review": True,
            }
        ]
    match = re.search(r"\b(caused|led to|resulted in|because of)\b", narrative, flags=re.I)
    if match is None:
        return []
    prefix = narrative[: match.start()]
    if re.search(
        r"\b(?:not|never|no evidence|no proof|not proven|not established|cannot establish)\b[^.!?;]*$",
        prefix,
        flags=re.I,
    ):
        return []
    left, right = [s.strip(" .") for s in (narrative[: match.start()], narrative[match.end() :])]
    if not left or not right:
        return []
    if match.group(1).casefold() == "because of":
        left, right = right, left
    return [
        {
            "cause_phrase": left,
            "effect_phrase": right,
            "method": "rule_based_candidates",
            "requires_expert_review": True,
        }
    ]


def thematic_clusters(narratives, *, n_clusters=4, seed=0, language="auto"):
    """探索性的中英文 TF-IDF 聚类，不用于预测验证。

    auto 在出现汉字时选择中文；中文采用字符 2–4 元组，避免把整句
    当成单个词。英文使用停用词过滤与单词 1–2 元组。
    若将聚类结果作为模型特征，应仅在训练事件叙述上拟合。
    预测任务使用 TextFeatureModel 的独立 fit/transform 接口。
    """
    if len(narratives) < 2:
        raise ValueError("need at least two narratives")
    if not isinstance(n_clusters, (int, np.integer)) or not 1 <= n_clusters <= len(narratives):
        raise ValueError("n_clusters must be in [1,number of documents]")
    if language not in ("auto", "zh", "en"):
        raise ValueError("language must be auto, zh or en")
    if not all(isinstance(text, str) for text in narratives):
        raise ValueError("narratives must be strings")
    selected = language
    if selected == "auto":
        selected = (
            "zh"
            if any(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", text) for text in narratives)
            else "en"
        )
    vec = (
        TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        if selected == "zh"
        else TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    )
    features = vec.fit_transform(narratives)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(features)
    return {
        "labels": km.labels_.tolist(),
        "top_terms": [
            [vec.get_feature_names_out()[j] for j in np.argsort(c)[-8:][::-1]]
            for c in km.cluster_centers_
        ],
        "method": "TF-IDF + KMeans",
    }


def frequency_graph(annotated_chains, node_ids):
    """因果链顺序只能来自显式标注，不能由关键词共现推断。"""
    if len(set(node_ids)) != len(node_ids):
        raise ValueError("node IDs must be unique")
    index = {node: i for i, node in enumerate(node_ids)}
    counts = np.zeros((len(index), len(index)), dtype=int)
    for chain in annotated_chains:
        if isinstance(chain, str):
            raise ValueError("causal chains must be sequences of factor IDs")
        if not all(node in index for node in chain):
            raise ValueError("unknown factor in causal chain")
        for a, b in zip(chain, chain[1:]):
            if a != b:
                counts[index[a], index[b]] += 1
    return counts


def motif_summary(adjacency):
    values = np.asarray(adjacency)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not len(values):
        raise ValueError("adjacency must be a nonempty square matrix")
    if not np.isfinite(values).all() or np.any(values < 0) or np.diag(values).any():
        raise ValueError("adjacency must be finite, nonnegative and have no self-edges")
    a = values > 0
    return {
        "branching_nodes": int(np.sum(a.sum(1) > 1)),
        "converging_nodes": int(np.sum(a.sum(0) > 1)),
        "chain_nodes": int(np.sum((a.sum(0) == 1) & (a.sum(1) == 1))),
        "hybrid_nodes": int(np.sum((a.sum(0) > 1) & (a.sum(1) > 1))),
        "note": "directed degree motifs",
    }


class TextFeatureModel:
    """在训练叙述上确定词表、IDF 和聚类中心，留出叙述仅变换。"""

    def __init__(self, *, language="auto", n_clusters=4, seed=2026, min_df=1):
        self.language = language
        self.n_clusters = n_clusters
        self.seed = seed
        self.min_df = min_df

    def fit(self, narratives):
        narratives = list(narratives)
        if not narratives or not all(isinstance(s, str) for s in narratives):
            raise ValueError("训练叙述必须是非空字符串列表")
        if self.language not in ("auto", "zh", "en"):
            raise ValueError("language 必须为 auto、zh 或 en")
        if type(self.n_clusters) is not int or not 1 <= self.n_clusters <= len(narratives):
            raise ValueError("n_clusters 必须在训练样本数范围内")
        selected = self.language
        if selected == "auto":
            selected = "zh" if any(re.search(r"[\u3400-\u9fff]", s) for s in narratives) else "en"
        self.language_ = selected
        self.vectorizer_ = (
            TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=self.min_df)
            if selected == "zh"
            else TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=self.min_df)
        )
        features = self.vectorizer_.fit_transform(narratives)
        self.clusterer_ = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10).fit(
            features
        )
        self.train_document_count_ = len(narratives)
        return self

    def transform(self, narratives):
        if not hasattr(self, "vectorizer_"):
            raise RuntimeError("文本特征模型必须先在训练叙述上 fit")
        narratives = list(narratives)
        if not all(isinstance(s, str) for s in narratives):
            raise ValueError("叙述必须是字符串")
        return self.vectorizer_.transform(narratives)

    def predict(self, narratives):
        return self.clusterer_.predict(self.transform(narratives))

    def vocabulary(self):
        if not hasattr(self, "vectorizer_"):
            raise RuntimeError("文本特征模型尚未训练")
        return self.vectorizer_.get_feature_names_out().tolist()
