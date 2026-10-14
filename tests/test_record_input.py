"""使用最小 CSV 测试夹具检查来源保留、日期解析和文本审计。"""

import csv
from datetime import date, datetime
import hashlib
import json

import pytest

from airport_risk.record_input import (
    _columns,
    _date_value,
    analyze_event_records,
    read_event_records,
)


def _csv_file(tmp_path, rows, name="input.csv", encoding="utf-8-sig"):
    """写入独立的 CSV 单元测试夹具。"""
    path = tmp_path / name
    with path.open("w", encoding=encoding, newline="") as stream:
        csv.writer(stream).writerows(rows)
    return path


def test_bom_csv_header_scan_preserves_source_rows_and_skips_placeholders(tmp_path):
    path = _csv_file(
        tmp_path,
        [
            ["单元测试夹具，不是调查记录"],
            [],
            ["序号", "时间", "标题", "事件经过", "致因抽取数据库"],
            ["1", "2019-01-02", "测试标题甲", "由于测试条件甲，导致测试结果乙。", "原始提供的文本"],
            ["2", "2019-01-03", "", "", ""],
            ["3", "2019-01-04", "测试标题乙", "原因不明。", ""],
        ],
    )
    before = path.read_bytes()
    imported = read_event_records(path)
    assert path.read_bytes() == before
    assert imported["source"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert imported["source"]["source_filename"] == "input.csv"
    assert imported["source"]["provenance"] == "user_provided"
    assert imported["source"]["not_independently_verified"] is True
    assert imported["source"]["record_count"] == 2
    assert imported["source"]["sheets"][0]["header_row"] == 3
    assert imported["source"]["sheets"][0]["skipped_placeholder_rows"] == 1
    assert [record["source_row"] for record in imported["records"]] == [4, 6]
    assert len({record["import_id"] for record in imported["records"]}) == 2
    assert read_event_records(path) == imported
    first = imported["records"][0]
    assert first["business_fields"]["序号"] == "1"
    assert first["narrative"] == "由于测试条件甲，导致测试结果乙。"
    assert first["provided_causal_text"] == "原始提供的文本"
    assert first["parsed_date"] == "2019-01-02"
    assert first["provenance"] == "user_provided_unverified"


def test_duplicate_time_and_multiline_headers_retain_all_original_columns(tmp_path):
    path = _csv_file(
        tmp_path,
        [
            ["序号", "时间", "标题", "鸟击信息", "时间", "航空器\n是否损伤", "备注", "备注"],
            ["1", "2019/2/3", "测试标题", "测试叙述", "05:30", "否", "原文甲", "原文乙"],
        ],
    )
    imported = read_event_records(path)
    columns = imported["source"]["sheets"][0]["columns"]
    assert [column["original_header"] for column in columns].count("时间") == 2
    assert columns[5]["original_header"] == "航空器\n是否损伤"
    fields = imported["records"][0]["business_fields"]
    assert fields["时间"] == "2019/2/3"
    assert fields["时间（时分）"] == "05:30"
    assert fields["航空器是否损伤"] == "否"
    assert fields["备注"] == "原文甲" and fields["备注__2"] == "原文乙"
    assert imported["records"][0]["parsed_date"] == "2019-02-03"
    assert len(set(column["field_key"] for column in columns)) == len(columns)


def test_date_conflict_and_missing_information_are_warned_without_rewriting(tmp_path):
    narrative = "2018年2月3日的测试文本；来源不明，具体日期尚未确定。"
    path = _csv_file(
        tmp_path,
        [
            ["序号", "时间", "标题", "事件经过"],
            ["1", "2019-02-03", "测试标题", narrative],
            ["2", "", "仅标题夹具", ""],
            ["3", "2019-02-30", "无效日期夹具", "测试叙述"],
        ],
    )
    imported = read_event_records(path)
    first, second, third = imported["records"]
    assert first["narrative"] == first["business_fields"]["事件经过"] == narrative
    assert first["parsed_date"] == "2019-02-03"
    assert "narrative_date_year_conflict" in first["warnings"]
    assert second["parsed_date"] is None and second["narrative"] == ""
    assert "missing_date" in second["warnings"] and "missing_narrative" in second["warnings"]
    assert third["parsed_date"] is None
    assert third["business_fields"]["时间"] == "2019-02-30"
    assert "date_parse_failed" in third["warnings"]


def test_existing_synthetic_markers_are_preserved_in_mixed_and_all_marked_input(tmp_path):
    headers = ["序号", "标题", "事件经过", "数据属性", "合成事件ID"]
    marked = ["1", "【合成】测试标题", "测试叙述", "合成", "SYN-RAW-000001"]
    marked_path = _csv_file(tmp_path, [headers, marked], name="marked.csv")
    imported = read_event_records(marked_path)
    assert imported["source"]["provenance"] == "synthetic"
    assert imported["source"]["contains_synthetic_markers"] is True
    assert imported["records"][0]["provenance"] == "synthetic"
    assert imported["records"][0]["business_fields"]["合成事件ID"] == "SYN-RAW-000001"
    mixed_path = _csv_file(
        tmp_path, [headers, marked, ["2", "测试标题", "未核验输入文本", "", ""]], name="mixed.csv"
    )
    mixed = read_event_records(mixed_path)
    assert mixed["source"]["provenance"] == "user_provided"
    assert mixed["source"]["contains_synthetic_markers"] is True
    assert mixed["source"]["synthetic_marked_records"] == 1
    assert [record["provenance"] for record in mixed["records"]] == [
        "synthetic",
        "user_provided_unverified",
    ]


def test_text_audit_only_reports_source_candidates_and_uncertainty(tmp_path):
    path = _csv_file(
        tmp_path,
        [
            ["序号", "时间", "标题", "事件经过", "致因抽取数据库"],
            ["1", "2019-01-02", "测试标题", "由于测试条件甲，导致测试结果乙。", "原因不明。"],
            ["2", "2019-01-03", "测试标题乙", "未证实由于测试条件丙，导致测试结果丁。", ""],
        ],
    )
    imported = read_event_records(path)
    report = analyze_event_records(imported, tmp_path / "private")
    assert report["summary"]["imported_records"] == 2
    assert report["summary"]["records_with_rule_candidates"] == 1
    assert report["summary"]["records_with_uncertainty_cues"] == 2
    candidates = report["records"][0]["text_audit"]["rule_candidates"]["narrative"]
    assert len(candidates) == 1
    assert candidates[0]["cause_phrase"] == "测试条件甲"
    assert candidates[0]["requires_expert_review"] is True
    assert candidates[0]["source_field"] == "事件经过"
    assert report["records"][1]["text_audit"]["rule_candidates"]["narrative"] == []
    assert report["not_independently_verified"] is True
    assert (
        report["events_generated"]
        is report["loads_generated"]
        is report["causal_network_inferred"]
        is False
    )
    assert json.loads((tmp_path / "private" / "event_records.json").read_text()) == imported
    assert json.loads((tmp_path / "private" / "text_audit.json").read_text()) == report
    assert not any("load" in key for record in imported["records"] for key in record)


def test_extra_columns_and_uncomputed_formulas_are_retained_without_extraction(tmp_path):
    formula = '=IF(1,"由于测试条件甲，导致测试结果乙。","")'
    path = _csv_file(
        tmp_path,
        [
            ["序号", "标题", "事件经过"],
            ["1", "测试标题", formula, "多出列中的原始内容"],
        ],
    )
    imported = read_event_records(path)
    record = imported["records"][0]
    assert record["business_fields"]["额外列_4"] == "多出列中的原始内容"
    assert record["narrative"] == formula
    assert "formula_cells_preserved_uncomputed" in record["warnings"]
    report = analyze_event_records(imported, tmp_path / "audit")
    assert report["summary"]["records_with_rule_candidates"] == 0


def test_excel_numeric_date_systems_and_invalid_leap_day_are_explicit():
    assert _date_value(43467, epoch="1900")[0] == "2019-01-02"
    assert _date_value(42005, epoch="1904")[0] == "2019-01-02"
    assert _date_value(60, epoch="1900") == (None, ["excel_serial_60_invalid_calendar_date"])
    assert _date_value(43468, epoch=None) == (None, ["numeric_date_without_excel_epoch"])
    assert _date_value(2019, epoch="1900") == (None, ["year_without_full_date"])
    assert _date_value("2019", epoch="1900") == (None, ["year_without_full_date"])
    assert _date_value(date(2019, 1, 2), epoch="1900")[0] == "2019-01-02"
    assert _date_value(datetime(2019, 1, 2, 3, 4), epoch="1900")[0] == "2019-01-02"


def test_empty_or_unsupported_files_fail_or_report_no_imported_events(tmp_path):
    path = _csv_file(tmp_path, [["只有说明，没有事件表头"]])
    imported = read_event_records(path)
    assert imported["records"] == []
    assert any(
        warning["code"] == "no_event_text_records_imported" for warning in imported["warnings"]
    )
    with pytest.raises(ValueError, match="Only"):
        read_event_records(tmp_path / "unsupported.xls")


def test_header_keys_remain_unique_with_existing_duplicate_suffixes():
    columns = _columns(["时间", "时间（时分）", "时间", "备注", "备注__2", "备注"])
    assert len({column["field_key"] for column in columns}) == len(columns)


def test_category_explanation_is_not_imported_but_numbered_causal_text_is_kept(tmp_path):
    path = _csv_file(
        tmp_path,
        [
            ["序号", "时间", "标题", "事件经过", "致因抽取数据库", "事件主因"],
            ["", "", "", "", "", "字段类别说明，不是事件"],
            ["1", "2019-01-02", "", "", "用户提供的单独致因文本", ""],
        ],
    )
    imported = read_event_records(path)
    assert imported["source"]["record_count"] == 1
    assert imported["source"]["sheets"][0]["skipped_placeholder_rows"] == 1
    row = imported["records"][0]
    assert row["source_row"] == 3
    assert row["provided_causal_text"] == "用户提供的单独致因文本"
    assert row["title"] == row["narrative"] == ""
    assert "missing_narrative" in row["warnings"] and "missing_title" in row["warnings"]
