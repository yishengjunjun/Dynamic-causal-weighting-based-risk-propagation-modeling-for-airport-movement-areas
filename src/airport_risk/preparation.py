"""观测负载与已审核因果阶段标注的样本构建。"""

from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone as utc_timezone
import hashlib
import io
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import networkx as nx

from .data import RiskData
from .nlp import approved_chains


def _marked_synthetic(value):
    if isinstance(value, dict):
        if (
            value.get("provenance") == "synthetic"
            or value.get("contains_synthetic_markers")
            or value.get("synthetic_markers")
        ):
            return True
        return any(_marked_synthetic(item) for item in value.values())
    if isinstance(value, list):
        return any(_marked_synthetic(item) for item in value)
    return isinstance(value, str) and (
        value.startswith(("SYN_", "SYN-RAW-", "SYN_EVT_", "SYN-EVT-"))
        or "【合成】" in value
        or value.casefold() in {"synthetic", "合成"}
    )


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value.strip()


def _nodes(node_ids):
    nodes = [_identifier(node, "node_id") for node in node_ids]
    if len(nodes) < 2 or len(set(nodes)) != len(nodes):
        raise ValueError("至少需要两个不重复的节点 ID")
    return nodes


def _window_settings(history_steps, future_steps, stride):
    for name, value, minimum in (
        ("history_steps", history_steps, 2),
        ("future_steps", future_steps, 1),
        ("stride", stride, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} 必须为不小于 {minimum} 的整数")


def _timestamp(value, timezone):
    # 仅有日期的记录不具备逐时观测含义。
    if not isinstance(value, str) or ("T" not in value and " " not in value):
        raise ValueError("timestamp 需包含完整日期及时间")
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        if timezone is None:
            raise ValueError("无时区 timestamp 需要显式 timezone")
        moment = moment.replace(tzinfo=ZoneInfo(timezone))
        # 排除夏令时跳变造成的不存在或歧义时间。
        back = moment.astimezone(utc_timezone.utc).astimezone(moment.tzinfo)
        if back.replace(tzinfo=None) != moment.replace(tzinfo=None):
            raise ValueError("timestamp 位于不存在的当地时间")
        if moment.utcoffset() != moment.replace(fold=1).utcoffset():
            raise ValueError("歧义当地时间需提供显式 UTC 偏移")
    return moment


def _build(sequences, node_ids, *, history_steps, future_steps, stride, metadata):
    _window_settings(history_steps, future_steps, stride)
    nodes = _nodes(node_ids)
    histories, futures, years, airports, events, samples = [], [], [], [], [], []
    skipped = []
    for event, item in sorted(sequences.items()):
        trajectory = np.asarray(item["loads"], dtype=float)
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != len(nodes)
            or not np.isfinite(trajectory).all()
            or np.any(trajectory < 0)
        ):
            raise ValueError(f"事件 {event} 的负载轨迹无效")
        if len(trajectory) < history_steps + future_steps:
            skipped.append(
                {"event_id": event, "reason": "insufficient_steps", "steps": len(trajectory)}
            )
            continue
        for stop in range(history_steps, len(trajectory) - future_steps + 1, stride):
            histories.append(trajectory[stop - history_steps : stop])
            futures.append(trajectory[stop : stop + future_steps])
            years.append(item["year"])
            airports.append(item["airport"])
            events.append(event)
            samples.append(f"{event}::window:{stop:08d}")
    if not histories:
        raise ValueError("没有事件满足历史和未来窗口长度；不能用填充构造未来观测")
    history, future = np.stack(histories), np.stack(futures)
    # 因子测量与参考序列只使用当前窗口历史，不读取预测目标。
    factors = history.mean(axis=1)
    reference = factors.mean(axis=1)
    meta = {
        **metadata,
        "network_policy": "train_fold",
        "graph_role": "layout_placeholder",
        "capacity_role": "layout_placeholder",
        "node_factor_indices": list(range(len(nodes))),
        "history_steps": history_steps,
        "future_steps": future_steps,
        "stride": stride,
        "split_unit": "event",
        "factor_measurement": "history_window_mean",
        "factor_reference": "history_factor_mean",
        "skipped_events": skipped,
    }
    return RiskData(
        history,
        future[:, 0].copy(),
        np.zeros((len(nodes), len(nodes))),
        np.ones(len(nodes)),
        np.zeros(len(nodes), dtype=bool),
        np.ones(len(nodes)),
        np.array(years),
        np.array(airports),
        np.array(events),
        np.array(nodes),
        factors,
        reference,
        meta,
        sample_ids=np.array(samples),
        future_loads=future,
    ).validate()


def prepare_observations(
    path,
    node_ids,
    *,
    source_id,
    history_steps=4,
    future_steps=1,
    stride=1,
    timezone=None,
    missing="reject",
    annotated_chains=None,
    load_units="source_declared",
    provenance="authorized_real",
):
    """读取长表观测；缺测只可拒绝或沿用已观测的过去状态。"""
    source_id = _identifier(source_id, "source_id")
    load_units = _identifier(load_units, "load_units")
    nodes = _nodes(node_ids)
    if missing not in {"reject", "forward_fill"}:
        raise ValueError("missing 必须为 reject 或 forward_fill")
    payload = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    required = {"event_id", "airport", "timestamp", "node_id", "load"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"观测长表缺少列：{sorted(required-set(reader.fieldnames or []))}")
    grouped = defaultdict(dict)
    airport_map, last_local_year = {}, {}
    synthetic_marked = False
    for line, row in enumerate(reader, 2):
        synthetic_marked |= _marked_synthetic(row)
        event = _identifier(row["event_id"], "event_id")
        airport = _identifier(row["airport"], "airport")
        node = _identifier(row["node_id"], "node_id")
        if node not in nodes:
            raise ValueError(f"第 {line} 行节点 {node} 不在目录中")
        if event in airport_map and airport_map[event] != airport:
            raise ValueError("同一事件不能跨多个机场")
        airport_map[event] = airport
        moment = _timestamp(row["timestamp"], timezone)
        stamp = moment.astimezone(utc_timezone.utc)
        if node in grouped[event].setdefault(stamp, {}):
            raise ValueError("同一事件、时间和节点有重复观测")
        try:
            load = float(row["load"])
        except (ValueError, TypeError) as error:
            raise ValueError(f"第 {line} 行 load 不是数值") from error
        if not np.isfinite(load) or load < 0:
            raise ValueError("load 必须为有限非负数")
        grouped[event][stamp][node] = load
        last_local_year[event] = max(last_local_year.get(event, moment.year), moment.year)
    sequences, timestamps, interval_seconds = {}, {}, {}
    for event, observations in grouped.items():
        loads, last = [], {}
        stamps = sorted(observations)
        for stamp in stamps:
            current = observations[stamp]
            if missing == "reject" and set(current) != set(nodes):
                raise ValueError(f"事件 {event} 时间 {stamp.isoformat()} 缺少节点观测")
            last.update(current)
            if set(last) != set(nodes):
                raise ValueError("前向填充要求第一个时间点具备所有节点观测")
            loads.append([last[node] for node in nodes])
        sequences[event] = {
            "loads": loads,
            "year": last_local_year[event],
            "airport": airport_map[event],
        }
        timestamps[event] = [stamp.isoformat() for stamp in stamps]
        intervals = [(b - a).total_seconds() for a, b in zip(stamps, stamps[1:])]
        # TCN 的一步对应相同观测间隔；不规则序列需先按明确规则重采样。
        if intervals and not np.allclose(intervals, intervals[0], rtol=0, atol=1e-6):
            raise ValueError(f"事件 {event} 的观测间隔不规则，请先显式重采样")
        interval_seconds[event] = intervals[0] if intervals else None
    defined_intervals = {interval for interval in interval_seconds.values() if interval is not None}
    if len(defined_intervals) > 1:
        raise ValueError("不同事件的观测步长必须相同")
    metadata = {
        "provenance": "synthetic" if synthetic_marked else provenance,
        "source_id": source_id,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "measurement_type": "observed_load",
        "time_axis": "timestamp",
        "load_units": load_units,
        "missing_policy": missing,
        "timestamps_utc": timestamps,
        "interval_seconds": interval_seconds,
        "event_year_rule": "latest_observation_local_year",
        "annotated_chains": annotated_chains or {},
    }
    return _build(
        sequences,
        nodes,
        history_steps=history_steps,
        future_steps=future_steps,
        stride=stride,
        metadata=metadata,
    )


def prepare_annotations(
    path,
    node_ids,
    *,
    source_id,
    history_steps=2,
    future_steps=1,
    stride=1,
    provenance="authorized_real",
):
    """将审核后的阶段载荷或有序因果链编码为阶段预测样本。"""
    source_id = _identifier(source_id, "source_id")
    nodes = _nodes(node_ids)
    payload = Path(path).read_bytes()
    document = json.loads(payload.decode("utf-8"))
    if (
        not isinstance(document, dict)
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
        or not isinstance(document.get("events"), list)
    ):
        raise ValueError("标注需使用 schema_version=1 和 events 列表")
    sequences, chains_by_event, reviewers = {}, {}, {}
    for item in document["events"]:
        if not isinstance(item, dict):
            raise ValueError("events 中的每项必须为标注对象")
        event = _identifier(item.get("event_id"), "event_id")
        if event in sequences:
            raise ValueError("标注事件 ID 重复")
        if item.get("reviewed") is not True:
            raise ValueError(f"事件 {event} 尚未审核")
        reviewer = _identifier(item.get("reviewer"), "reviewer")
        year = item.get("year")
        if type(year) is not int or not 1900 <= year <= 2100:
            raise ValueError("审核标注需明确有效事件年份")
        airport = _identifier(item.get("airport"), "airport")
        chains = item.get("causal_chains") or approved_chains(item.get("relations", []))
        if not isinstance(chains, list) or any(
            not isinstance(chain, list)
            or len(chain) < 2
            or len(set(chain)) != len(chain)
            or any(node not in nodes for node in chain)
            for chain in chains
        ):
            raise ValueError("causal_chains 需为无重复节点的目录内有序因果链")
        stages = item.get("stages")
        if stages is not None:
            if not isinstance(stages, list) or not stages:
                raise ValueError("stages 必须为非空列表")
            trajectory = []
            for stage in stages:
                if not isinstance(stage, dict) or set(stage) - set(nodes):
                    raise ValueError("阶段需以节点 ID 为键，未列出的节点按未激活编码")
                values = []
                for node in nodes:
                    value = stage.get(node, 0)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not np.isfinite(value)
                        or value < 0
                    ):
                        raise ValueError("阶段载荷必须为有限非负数")
                    values.append(value)
                trajectory.append(values)
        else:
            if not chains:
                raise ValueError("标注需提供 stages 或 causal_chains")
            # 合并共享节点的链，按最长前驱深度确定阶段，保持所有因果先后。
            graph = nx.DiGraph()
            for chain in chains:
                graph.add_edges_from(zip(chain, chain[1:]))
            if not nx.is_directed_acyclic_graph(graph):
                raise ValueError("循环因果链需要显式 stages，不能自动分配阶段")
            levels = {}
            for node in nx.topological_sort(graph):
                levels[node] = max(
                    (levels[parent] + 1 for parent in graph.predecessors(node)), default=0
                )
            active = set()
            trajectory = [np.zeros(len(nodes)).tolist()]
            for step in range(max(levels.values()) + 1):
                active.update(node for node, level in levels.items() if level == step)
                trajectory.append([float(node in active) for node in nodes])
        sequences[event] = {"loads": trajectory, "year": year, "airport": airport}
        chains_by_event[event], reviewers[event] = chains, reviewer
    metadata = {
        "provenance": "synthetic" if _marked_synthetic(document) else provenance,
        "source_id": source_id,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "measurement_type": "coded_annotations",
        "time_axis": "causal_stage",
        "load_units": "dimensionless_annotation_activation",
        "encoding": "explicit_stage_loads_or_cumulative_dag_level_activation",
        "annotated_chains": chains_by_event,
        "reviewers": reviewers,
        "event_year_rule": "reviewed_event_year",
    }
    return _build(
        sequences,
        nodes,
        history_steps=history_steps,
        future_steps=future_steps,
        stride=stride,
        metadata=metadata,
    )
