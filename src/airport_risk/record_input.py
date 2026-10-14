"""事件记录的只读导入、来源追踪、日期解析与文本审计。

保留原始业务字段、缺失值及行列位置，并将日期与文本异常记录为审计警告。
"""

from __future__ import annotations

from collections import Counter
import csv
from datetime import date, datetime, time, timedelta
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any

TITLE_FIELDS = ("标题", "事件名称", "事件标题")
NARRATIVE_FIELDS = ("事件经过", "简要经过", "鸟击信息", "事件描述", "事件叙述", "经过")
CAUSAL_FIELDS = ("致因抽取数据库", "致因文本", "原始致因文本", "原因描述", "鸟击原因", "事件主因")
DATE_FIELDS = ("事发时间", "时间", "事发日期", "事件日期", "日期", "发生时间")
UNCERTAINTY_PATTERN = re.compile(r"来源不明|原因不明|无法确定|尚未确定|未确定|未证实|疑似|不详")
TEXT_DATE_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?:年|[-/.]\d{1,2}[-/.]\d{1,2})")


def _header(value: Any) -> str:
    """表头去除空白字符；原始列名另外保存在来源映射中。"""
    return re.sub(r"\s+", "", str(value)) if value is not None else ""


def _is_header(values: list[Any] | tuple[Any, ...]) -> bool:
    names = {_header(value) for value in values}
    has_identity = bool(names.intersection(("序号", *TITLE_FIELDS)))
    return has_identity and bool(names.intersection((*NARRATIVE_FIELDS, *CAUSAL_FIELDS)))


def _json_cell(value: Any) -> Any:
    """仅做 JSON 表达转换，不清洗或改写单元格中的业务内容。"""
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _columns(values: list[Any] | tuple[Any, ...]) -> list[dict]:
    """为重复列创建唯一键，鸟击第二个时间列保留为时分字段。"""
    used: set[str] = set()
    columns = []
    for index, original in enumerate(values, 1):
        normalized = _header(original) or f"未命名列_{index}"
        key = normalized
        if key in used and normalized == "时间":
            key = "时间（时分）"
        suffix = 2
        while key in used:
            key = f"{normalized}__{suffix}"
            suffix += 1
        used.add(key)
        columns.append({"column": index, "original_header": _json_cell(original), "field_key": key})
    return columns


def _choose(fields: dict, candidates: tuple[str, ...]) -> tuple[str | None, str]:
    for key in candidates:
        value = fields.get(key)
        if value is not None and str(value).strip():
            return key, str(value)
    return None, ""


def _date_value(value: Any, *, epoch: str | None) -> tuple[str | None, list[str]]:
    """解析日期字段；缺失、无效或缺少日期系统的信息返回审计标记。"""
    warnings: list[str] = []
    if value is None or value == "":
        return None, ["missing_date"]
    if isinstance(value, datetime):
        return value.date().isoformat(), warnings
    if isinstance(value, date):
        return value.isoformat(), warnings
    if isinstance(value, bool):
        return None, ["date_parse_failed"]
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            return None, ["date_parse_failed"]
        if int(value) == value and 1900 <= value <= 2100:
            return None, ["year_without_full_date"]
        digits = str(int(value)) if int(value) == value else ""
        if re.fullmatch(r"(?:19|20)\d{6}", digits):
            try:
                return datetime.strptime(digits, "%Y%m%d").date().isoformat(), warnings
            except ValueError:
                return None, ["date_parse_failed"]
        # 仅在 XLSX 明确提供日期系统时解释数值日期；CSV 数字仍需人工确认。
        if epoch is None:
            return None, ["numeric_date_without_excel_epoch"]
        if not 1 <= value <= 2958465:
            return None, ["date_parse_failed"]
        if epoch == "1900" and int(value) == 60:
            return None, ["excel_serial_60_invalid_calendar_date"]
        offset = value + 1 if epoch == "1900" and value < 60 else value
        base = datetime(1899, 12, 30) if epoch == "1900" else datetime(1904, 1, 1)
        try:
            return (base + timedelta(days=offset)).date().isoformat(), ["excel_numeric_date_parsed"]
        except (ValueError, OverflowError):
            return None, ["date_parse_failed"]
    text = str(value).strip()
    if re.fullmatch(r"(?:19|20)\d{2}|2100", text):
        return None, ["year_without_full_date"]
    if re.fullmatch(r"(?:19|20)\d{6}", text):
        try:
            return datetime.strptime(text, "%Y%m%d").date().isoformat(), warnings
        except ValueError:
            return None, ["date_parse_failed"]
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat(), warnings
    except ValueError:
        match = re.fullmatch(r"(\d{4})[年/.-](\d{1,2})[月/.-](\d{1,2})(?:日)?(?:[ T].*)?", text)
        if match:
            try:
                return date(*(int(part) for part in match.groups())).isoformat(), warnings
            except ValueError:
                pass
    return None, ["date_parse_failed"]


def _synthetic_markers(fields: dict, title: str) -> list[str]:
    """提取输入字段、标题及标识符中的既有合成来源标记。"""
    markers = []
    if str(fields.get("数据属性", "")).strip().casefold() in ("合成", "synthetic"):
        markers.append("数据属性")
    if "【合成】" in title:
        markers.append("合成标题")
    for key, value in fields.items():
        if isinstance(value, str) and value.startswith(("SYN-RAW-", "SYN_EVT_", "SYN-EVT-")):
            markers.append(f"合成标识符:{key}")
    if str(fields.get("provenance", "")).casefold() == "synthetic":
        markers.append("provenance")
    return markers


def _read_tables(payload: bytes, suffix: str) -> tuple[list[dict], dict]:
    """CSV 使用标准库；XLSX 的 openpyxl 只在该路径内按需导入。"""
    if suffix == ".csv":
        encoding = "utf-8-sig"
        try:
            text = payload.decode(encoding)
        except UnicodeDecodeError:
            encoding = "gb18030"
            text = payload.decode(encoding)
        rows = list(csv.reader(io.StringIO(text, newline="")))
        return [{"name": "CSV", "rows": rows}], {
            "format": "csv",
            "encoding": encoding,
            "excel_date_system": None,
        }
    try:
        import openpyxl
    except ImportError as error:
        raise ImportError("Reading XLSX requires the optional openpyxl dependency") from error
    workbook = openpyxl.load_workbook(io.BytesIO(payload), read_only=True, data_only=False)
    try:
        tables = [
            {"name": sheet.title, "rows": list(sheet.iter_rows(values_only=True))}
            for sheet in workbook
        ]
        epoch = "1904" if workbook.epoch.year == 1904 else "1900"
    finally:
        workbook.close()
    return tables, {
        "format": "xlsx",
        "excel_date_system": epoch,
        "formula_policy": "preserve_uncomputed",
    }


def read_event_records(path: str | Path) -> dict:
    """只读导入中文事件表，保留源行号、原字段及来源校验状态。"""
    path = Path(path)
    suffix = path.suffix.casefold()
    if suffix not in (".xlsx", ".csv"):
        raise ValueError("Only .xlsx and .csv input files are supported")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    tables, format_info = _read_tables(payload, suffix)
    records: list[dict] = []
    import_warnings: list[dict] = []
    sheet_metadata = []
    for sheet_index, table in enumerate(tables, 1):
        rows = table["rows"]
        header_index = next((i for i, row in enumerate(rows[:10]) if _is_header(row)), None)
        if header_index is None:
            import_warnings.append(
                {"source_sheet": table["name"], "code": "event_header_not_found_in_first_10_rows"}
            )
            continue
        columns = _columns(rows[header_index])
        sheet_info = {
            "source_sheet": table["name"],
            "header_row": header_index + 1,
            "columns": columns,
            "imported_rows": 0,
            "skipped_placeholder_rows": 0,
        }
        for row_index, values in enumerate(rows[header_index + 1 :], header_index + 2):
            if _is_header(values):
                import_warnings.append(
                    {
                        "source_sheet": table["name"],
                        "source_row": row_index,
                        "code": "repeated_header_skipped",
                    }
                )
                continue
            fields = {
                column["field_key"]: _json_cell(values[i]) if i < len(values) else None
                for i, column in enumerate(columns)
            }
            for i in range(len(columns), len(values)):
                fields[f"额外列_{i + 1}"] = _json_cell(values[i])
            title_field, title = _choose(fields, TITLE_FIELDS)
            narrative_field, narrative = _choose(fields, NARRATIVE_FIELDS)
            causal_field, causal_text = _choose(fields, CAUSAL_FIELDS)
            has_row_identity = fields.get("序号") is not None and bool(
                str(fields.get("序号")).strip()
            )
            # 分类说明可能只填“事件主因”等列；没有事件身份或叙述时不算记录。
            # 同时保留带原始序号与致因文本的记录，并保持标题、叙述的缺失值。
            if not any(text.strip() for text in (title, narrative)) and not (
                has_row_identity and causal_text.strip()
            ):
                sheet_info["skipped_placeholder_rows"] += 1
                continue
            warnings = []
            if len(values) > len(columns):
                warnings.append("extra_source_columns_preserved")
            if not narrative.strip():
                warnings.append("missing_narrative")
            if not title.strip():
                warnings.append("missing_title")
            date_field = next((field for field in DATE_FIELDS if field in fields), None)
            # 日期解析读取未转换的原单元格，以保留 XLSX 数值及日期类型的区别。
            date_column = next(
                (i for i, column in enumerate(columns) if column["field_key"] == date_field), None
            )
            raw_date = (
                values[date_column]
                if date_column is not None and date_column < len(values)
                else None
            )
            parsed_date, date_warnings = _date_value(
                raw_date, epoch=format_info["excel_date_system"]
            )
            warnings.extend(date_warnings)
            text_years = sorted(set(TEXT_DATE_YEAR.findall(narrative + "\n" + causal_text)))
            if parsed_date and any(year != parsed_date[:4] for year in text_years):
                warnings.append("narrative_date_year_conflict")
            formula_fields = [
                key
                for key, value in fields.items()
                if isinstance(value, str) and value.startswith("=")
            ]
            if formula_fields:
                warnings.append("formula_cells_preserved_uncomputed")
            markers = _synthetic_markers(fields, title)
            if markers:
                warnings.append("input_contains_explicit_synthetic_markers")
            records.append(
                {
                    "import_id": f"IMPORT-{digest}-{sheet_index:03}-{row_index:06}",
                    "source_sheet": table["name"],
                    "source_row": row_index,
                    "business_fields": fields,
                    "title": title,
                    "title_field": title_field,
                    "narrative": narrative,
                    "narrative_field": narrative_field,
                    "provided_causal_text": causal_text,
                    "provided_causal_text_field": causal_field,
                    "parsed_date": parsed_date,
                    "date_field": date_field,
                    "date_years_in_text": text_years,
                    "warnings": warnings,
                    "provenance": "synthetic" if markers else "user_provided_unverified",
                    "synthetic_markers": markers,
                    "not_independently_verified": True,
                }
            )
            sheet_info["imported_rows"] += 1
        sheet_metadata.append(sheet_info)
    marked = sum(record["provenance"] == "synthetic" for record in records)
    source = {
        "sha256": digest,
        "source_filename": path.name,
        "provenance": "synthetic" if records and marked == len(records) else "user_provided",
        "user_input": True,
        "not_independently_verified": True,
        "contains_synthetic_markers": bool(marked),
        "synthetic_marked_records": marked,
        "record_count": len(records),
        "sheets": sheet_metadata,
        "verification_scope": "仅核对文件及原单元格，不独立核验事件真实性、责任或原因",
        **format_info,
    }
    if not records:
        import_warnings.append({"code": "no_event_text_records_imported"})
    return {"source": source, "records": records, "warnings": import_warnings}


def analyze_event_records(dataset: dict, outdir: str | Path) -> dict:
    """只审计已有文本中的规则短语，输出保留来源的 JSON。"""
    from .text_mining import heuristic_roles

    records = dataset.get("records")
    source = dataset.get("source")
    if (
        not isinstance(records, list)
        or not isinstance(source, dict)
        or not source.get("not_independently_verified")
    ):
        raise ValueError("Expected unverified input records with source metadata")
    audited = []
    warning_counts: Counter = Counter()
    for record in records:
        text_results = {}
        uncertainty = []
        for field in ("narrative", "provided_causal_text"):
            text = record.get(field, "")
            if not isinstance(text, str):
                raise ValueError("Source narrative and provided causal text must be strings")
            # 未计算的表格公式不是调查叙述，不抽取其中的文字常量。
            roles = [] if text.startswith("=") else heuristic_roles(text)
            text_results[field] = [
                {
                    **role,
                    "status": "candidate_text_phrase_requires_review",
                    "source_field": record.get(f"{field}_field"),
                }
                for role in roles
            ]
            uncertainty.extend(
                {
                    "field": field,
                    "phrase": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                }
                for match in UNCERTAINTY_PATTERN.finditer(text)
            )
        warning_counts.update(record.get("warnings", []))
        audited.append(
            {
                "import_id": record["import_id"],
                "source_sheet": record["source_sheet"],
                "source_row": record["source_row"],
                "provenance": record["provenance"],
                "not_independently_verified": True,
                "warnings": record.get("warnings", []),
                "text_audit": {
                    "rule_candidates": text_results,
                    "uncertainty_cues": uncertainty,
                    "requires_expert_review": True,
                },
            }
        )
    report = {
        "source": source,
        "summary": {
            "imported_records": len(records),
            "synthetic_marked_records": source.get("synthetic_marked_records", 0),
            "records_with_rule_candidates": sum(
                any(row["text_audit"]["rule_candidates"].values()) for row in audited
            ),
            "records_with_uncertainty_cues": sum(
                bool(row["text_audit"]["uncertainty_cues"]) for row in audited
            ),
            "warning_counts": dict(warning_counts),
        },
        "records": audited,
        "import_warnings": dataset.get("warnings", []),
        "scope": "只读来源记录的文本审计；规则短语不是核验后的致因或论文网络",
        "events_generated": False,
        "loads_generated": False,
        "causal_network_inferred": False,
        "not_independently_verified": True,
    }
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, value in (("event_records.json", dataset), ("text_audit.json", report)):
        (out / filename).write_text(
            json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report
