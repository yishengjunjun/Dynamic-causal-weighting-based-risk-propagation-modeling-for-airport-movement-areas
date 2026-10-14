"""机场事件因子目录、语义角色标注接口与因果关系审核。"""

from __future__ import annotations

import json
from pathlib import Path
import re

from .text_mining import extract_keywords

CAUSAL_PREDICATES = {
    "导致",
    "造成",
    "引发",
    "引起",
    "致使",
    "caused",
    "cause",
    "causes",
    "led to",
    "resulted in",
}
NEGATION = re.compile(
    r"未发现|未证实|未证明|无证据|没有证据|并非|并不是|否认|尚未证实|不能认定|cannot establish|no evidence|not proven",
    re.I,
)
NEGATED_CONNECTOR = re.compile(
    r"(?:未|没有|并未|并不|不|不会|未能|尚未)(?:直接|必然|实际|明确)?$|\b(?:not|never)\b[^.!?;]*$",
    re.I,
)
UNCERTAINTY = re.compile(
    r"疑似|怀疑|可能|尚待|推测|无法确定|不排除|suspect|possible|possibly|may have|might|uncertain",
    re.I,
)


class FactorCatalog:
    """具有稳定 ID、别名和操作定义的候选因素目录。"""

    def __init__(self, factors, *, catalog_id="custom"):
        self.factors = [dict(factor) for factor in factors]
        self.catalog_id = catalog_id
        identifiers = [factor.get("id") for factor in self.factors]
        if (
            not identifiers
            or any(not isinstance(i, str) or not i for i in identifiers)
            or len(set(identifiers)) != len(identifiers)
        ):
            raise ValueError("因子目录需包含非空、唯一的字符串 ID")
        for factor in self.factors:
            if not factor.get("name") or not factor.get("definition"):
                raise ValueError("每项因子必须具备名称和操作定义")
            aliases = factor.get("aliases")
            if (
                not isinstance(aliases, list)
                or not aliases
                or any(not isinstance(s, str) or not s.strip() for s in aliases)
            ):
                raise ValueError("因子别名必须为非空字符串列表")
        self.ids = identifiers
        self.by_id = {factor["id"]: factor for factor in self.factors}
        self.taxonomy = {factor["id"]: factor["aliases"] for factor in self.factors}

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("因子目录 schema_version 必须为 1")
        return cls(payload["factors"], catalog_id=payload.get("catalog_id", "custom"))

    def match(self, phrase):
        return extract_keywords([phrase], self.taxonomy)[0]

    def annotation_template(self, relation, *, event_id):
        """将词面映射候选与审批字段一并导出。"""
        result = dict(relation)
        result.update(
            event_id=str(event_id),
            cause_factor_candidates=self.match(relation["cause_phrase"]),
            effect_factor_candidates=self.match(relation["effect_phrase"]),
            cause_factor_ids=[],
            effect_factor_ids=[],
            review_status="pending",
            reviewer=None,
        )
        return result


def _relation(cause, effect, *, method, context, negated=False, frame=None):
    uncertain = bool(UNCERTAINTY.search(context))
    return {
        "cause_phrase": cause.strip(" ，,：:。."),
        "effect_phrase": effect.strip(" ，,：:。."),
        "method": method,
        "polarity": "negated" if negated else "affirmative",
        "certainty": "uncertain" if uncertain else "asserted",
        "review_status": "pending",
        "requires_expert_review": True,
        "semantic_frame": frame,
    }


def rule_relations(narrative):
    """按显式因果连接词提取方向，并保留否定及不确定性。"""
    if not isinstance(narrative, str):
        raise ValueError("叙述必须是字符串")
    results = []
    for sentence in re.split(r"[。！？!?；;\n]+", narrative):
        sentence = sentence.strip()
        if not sentence:
            continue
        # 逗号连接的完整因果分句分别处理；引导短语后的逗号仍属于同一分句。
        clauses = re.split(r"[，,](?=[^，,]*(?:导致|造成|引发|引起|致使))", sentence)
        if len(clauses) > 1 and not re.search(r"导致|造成|引发|引起|致使", clauses[0]):
            clauses = [sentence]
        for clause in clauses:
            cn = re.search(r"(?P<left>.+?)(?P<verb>导致|造成|引发|引起|致使)(?P<right>.+)", clause)
            if cn:
                left, right = cn.group("left").strip(), cn.group("right").strip()
                negated = bool(NEGATION.search(left) or NEGATED_CONNECTOR.search(left))
                left = re.sub(
                    r"^(?:调查(?:认为|指出|发现)[，,：:]?|由于|因为|因(?![果素]))", "", left
                ).strip(" ，,：:")
                left = re.sub(
                    r"(?:并未|没有|未能|未|不)(?:直接|必然|实际|明确)?$", "", left
                ).strip()
                if left and right:
                    results.append(
                        _relation(
                            left,
                            right,
                            method="explicit_causal_rules",
                            context=clause,
                            negated=negated,
                        )
                    )
                continue
            en = re.search(
                r"\b(caused by|because of|caused|causes|cause|led to|resulted in)\b", clause, re.I
            )
            if en:
                left, right = clause[: en.start()].strip(), clause[en.end() :].strip()
                negated = bool(NEGATION.search(left) or NEGATED_CONNECTOR.search(left))
                if en.group().casefold() in ("caused by", "because of"):
                    left, right = right, left
                if left and right:
                    results.append(
                        _relation(
                            left,
                            right,
                            method="explicit_causal_rules",
                            context=clause,
                            negated=negated,
                        )
                    )
    return results


def validate_srl_frames(tokens, frames):
    """校验 HanLP 四元组的词元区间；区间采用左闭右开约定。"""
    if not isinstance(tokens, list) or any(not isinstance(t, str) or not t for t in tokens):
        raise ValueError("tokens 必须为非空词元组成的列表")
    normalized = []
    for frame in frames:
        arguments = []
        for item in frame:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                raise ValueError("SRL 标注必须为四元组")
            phrase, role, start, end = item
            if (
                not isinstance(phrase, str)
                or not isinstance(role, str)
                or type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(tokens)
            ):
                raise ValueError("SRL 词元区间或语义角色无效")
            if phrase.replace(" ", "") != "".join(tokens[start:end]).replace(" ", ""):
                raise ValueError("SRL 短语与词元区间不一致")
            arguments.append([phrase, role, start, end])
        if sum(item[1] == "PRED" for item in arguments) != 1:
            raise ValueError("每个 SRL 语义帧必须有一个 PRED")
        normalized.append(arguments)
    return normalized


def relations_from_srl(tokens, frames, *, narrative=None, method="imported_srl"):
    """只将因果谓词或明确的原因状语转换为因果候选关系。"""
    frames = validate_srl_frames(tokens, frames)
    context = narrative if narrative is not None else "".join(tokens)
    output = []
    for frame in frames:
        roles = {}
        for phrase, role, start, end in frame:
            roles.setdefault(role, []).append(phrase)
        predicate = roles["PRED"][0].casefold()
        causes, effects = [], []
        if predicate in CAUSAL_PREDICATES:
            causes, effects = roles.get("ARG0", []), roles.get("ARG1", [])
        elif roles.get("ARGM-CAU"):
            causes = roles["ARGM-CAU"]
            effects = ["".join(roles.get("ARG0", []) + roles["PRED"] + roles.get("ARG1", []))]
        predicate_start = next(item[2] for item in frame if item[1] == "PRED")
        preceding = "".join(tokens[:predicate_start])
        for cause in causes:
            for effect in effects:
                output.append(
                    _relation(
                        cause,
                        effect,
                        method=method,
                        context=context,
                        negated=bool(
                            roles.get("ARGM-NEG")
                            or NEGATION.search(context)
                            or NEGATED_CONNECTOR.search(preceding)
                        ),
                        frame=frame,
                    )
                )
    return output


class ImportedSRLBackend:
    """读取本地已标注语义帧，按事件 ID 查询并校验。"""

    def __init__(self, annotations):
        self.annotations = dict(annotations)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def extract(self, narrative, *, event_id):
        if str(event_id) not in self.annotations:
            raise KeyError(f"事件 {event_id} 没有 SRL 标注")
        item = self.annotations[str(event_id)]
        if item.get("text", narrative) != narrative:
            raise ValueError("SRL 标注对应文本与当前叙述不一致")
        return relations_from_srl(item["tokens"], item["frames"], narrative=narrative)


class HanLPSRLBackend:
    """使用本地 HanLP 多任务模型执行中文分词与神经 SRL。"""

    def __init__(self, model_path):
        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError("HanLP 模型路径不存在")
        try:
            import hanlp
        except ImportError as error:
            raise ImportError("神经 SRL 需要安装 hanlp 并提供本地多任务模型") from error
        self.model = hanlp.load(str(path))
        self.model_id = path.name

    def extract(self, narrative, *, event_id=None):
        document = self.model(narrative, tasks="srl")
        token_keys = [key for key in document if key.startswith("tok")]
        if not token_keys or "srl" not in document:
            raise ValueError("HanLP 模型必须同时返回分词与 srl 字段")
        tokens, frames = document[token_keys[0]], document["srl"]
        # 单句输入与句子列表输入均使用同一词元区间定义。
        if tokens and isinstance(tokens[0], list):
            output = []
            for sentence_tokens, sentence_frames in zip(tokens, frames, strict=True):
                output.extend(
                    relations_from_srl(
                        sentence_tokens,
                        sentence_frames,
                        narrative="".join(sentence_tokens),
                        method="hanlp_neural_srl",
                    )
                )
        else:
            output = relations_from_srl(
                tokens, frames, narrative=narrative, method="hanlp_neural_srl"
            )
        for relation in output:
            relation["model_id"] = self.model_id
        return output


def review_relation(
    relation,
    *,
    status,
    reviewer,
    cause_factor_ids=None,
    effect_factor_ids=None,
    catalog=None,
    note="",
):
    """记录审核决定；否定关系不能作为肯定边批准。"""
    if (
        status not in {"approved", "rejected", "pending"}
        or not isinstance(reviewer, str)
        or not reviewer.strip()
    ):
        raise ValueError("审核状态及审核人无效")
    result = dict(relation)
    causes = list(
        cause_factor_ids if cause_factor_ids is not None else result.get("cause_factor_ids", [])
    )
    effects = list(
        effect_factor_ids if effect_factor_ids is not None else result.get("effect_factor_ids", [])
    )
    if status == "approved":
        if result.get("polarity", "affirmative") != "affirmative":
            raise ValueError("否定因果关系不能批准为肯定因果边")
        if not causes or not effects:
            raise ValueError("批准关系必须明确源因子和目标因子")
        if catalog is not None and any(factor not in catalog.by_id for factor in causes + effects):
            raise ValueError("审核关系含目录外因子")
    result.update(
        review_status=status,
        reviewer=reviewer.strip(),
        review_note=str(note),
        cause_factor_ids=causes,
        effect_factor_ids=effects,
        requires_expert_review=status == "pending",
    )
    return result


def approved_chains(relations, *, include_uncertain=False):
    """将批准且方向明确的关系转为有向链；不以词面共现创建边。"""
    output = []
    for relation in relations:
        if (
            relation.get("review_status") != "approved"
            or relation.get("polarity", "affirmative") != "affirmative"
        ):
            continue
        if not include_uncertain and relation.get("certainty", "asserted") == "uncertain":
            continue
        if not isinstance(relation.get("reviewer"), str) or not relation["reviewer"].strip():
            raise ValueError("批准关系必须包含审核人")
        for source in relation.get("cause_factor_ids", []):
            for target in relation.get("effect_factor_ids", []):
                if source != target:
                    output.append([source, target])
    return output
