"""实验配置校验、输入检查与运行版本记录。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from .data import RiskData, split_indices
from .experiments import experiment_defaults, validate_experiment_config

PREPARATION_DEFAULTS = {
    "history_steps": 2,
    "future_steps": 1,
    "stride": 1,
    "timezone": "Asia/Shanghai",
    "missing": "reject",
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    return load_config_snapshot(path)[0]


def load_config_snapshot(path):
    content = Path(path).read_bytes()
    config = json.loads(content.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("配置文件顶层必须是 JSON 对象")
    return config, hashlib.sha256(content).hexdigest()


def resolve_config(config):
    """解析固定默认值，得到写入运行清单的完整配置。"""
    if not isinstance(config, dict) or set(config) - {
        "schema_version",
        "experiment",
        "preparation",
        "factor_catalog",
    }:
        raise ValueError("配置存在未知顶层字段")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 2:
        raise ValueError("schema_version 必须为 2")
    experiment = validate_experiment_config(config.get("experiment", {}))
    from .network import NetworkBuilder

    NetworkBuilder(experiment)
    preparation = config.get("preparation", {})
    if not isinstance(preparation, dict) or set(preparation) - set(PREPARATION_DEFAULTS):
        raise ValueError("preparation 存在未知字段")
    preparation = {**PREPARATION_DEFAULTS, **preparation}
    for key, minimum in (("history_steps", 2), ("future_steps", 1), ("stride", 1)):
        if type(preparation[key]) is not int or preparation[key] < minimum:
            raise ValueError(f"preparation.{key} 必须为不小于 {minimum} 的整数")
    if preparation["missing"] not in {"reject", "forward_fill"}:
        raise ValueError("preparation.missing 必须为 reject 或 forward_fill")
    if preparation["timezone"] is not None:
        try:
            ZoneInfo(preparation["timezone"])
        except (ZoneInfoNotFoundError, TypeError, ValueError) as error:
            raise ValueError("preparation.timezone 不是有效时区") from error
    if any(seed >= 2**64 for seed in experiment["seeds"]):
        raise ValueError("experiment.seeds 必须小于 2**64")
    if not 0 <= experiment["gamma"] < 1:
        raise ValueError("experiment.gamma 必须位于 [0,1)")
    if not 0 < experiment["eta"] < 1 or any(
        not 0 < experiment["eta"] * value < 1 for value in experiment["eta_scales"]
    ):
        raise ValueError("experiment.eta 及其扰动值必须位于 (0,1)")
    factor_catalog = config.get("factor_catalog", "factors.json")
    if not isinstance(factor_catalog, str) or not factor_catalog.strip():
        raise ValueError("factor_catalog 必须为目录文件路径")
    result = {
        "schema_version": 2,
        "experiment": experiment,
        "preparation": preparation,
        "factor_catalog": factor_catalog,
    }
    json.dumps(result, allow_nan=False)
    return result


def validate_config(config):
    try:
        resolve_config(config)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        return [str(error)]
    return []


def inspect_inputs(config, data):
    config = resolve_config(config)
    parameters = config["experiment"]
    issues = []
    if data.metadata["provenance"] != "authorized_real":
        issues.append("实验入口需有来源声明的数据；测试夹具不可用于实证评价")
    if data.metadata.get("synthetic_airports") or any(
        str(value).startswith(("SYN_", "SYN-RAW-"))
        for values in (data.airports, data.event_ids)
        for value in values
    ):
        issues.append("输入仍含生成数据标记")
    measurement = data.metadata.get("measurement_type")
    if measurement not in {"observed_load", "coded_annotations"}:
        issues.append("metadata.measurement_type 需明确 observed_load 或 coded_annotations")
    if not data.metadata.get("source_id") or not data.metadata.get("load_units"):
        issues.append("数据需明确 source_id 和 load_units")
    if parameters["training_mode"] == "transition" and data.future_loads is None:
        issues.append("transition 训练需要 future_loads，不能填充未观测的未来值")
    if parameters["var_lag"] > data.history.shape[1]:
        issues.append("var_lag 不能超过 history 的历史步数")
    if parameters["network_mode"] == "external_fixed" and not data.metadata.get("network_source"):
        issues.append("外部先验网络需要 metadata.network_source")
    try:
        from .relations import compile_relations

        compile_relations(
            data.node_ids, parameters.get("relations") or data.metadata.get("relations", {})
        )
    except (ValueError, TypeError, KeyError) as error:
        issues.append(f"节点关系配置无效：{error}")
    if (
        parameters["capacity_mode"] == "grey_topology"
        and measurement == "observed_load"
        and data.metadata.get("load_units") != "dimensionless_normalized"
    ):
        issues.append("grey_topology 容量需无量纲归一负载；有量纲观测请使用 train_quantile")
    plans = [
        {
            "protocol": "stratified",
            "train_fraction": parameters["stratified_train_fraction"],
            "seed": seed,
        }
        for seed in parameters["seeds"]
    ]
    enabled = parameters["enabled_protocols"]
    if "additional_fraction" in enabled:
        plans += [
            {"protocol": "stratified", "train_fraction": value}
            for value in parameters["additional_train_fractions"]
        ]
    if "rolling" in enabled:
        plans += [
            {"protocol": "rolling", "test_year": year} for year in parameters["rolling_test_years"]
        ]
    if "airport" in enabled:
        plans += [{"protocol": "airport", "holdout": parameters["airport_holdout"]}]
    if "count" in enabled and parameters["printed_train_count"] >= len(np.unique(data.event_ids)):
        issues.append("printed_train_count 必须小于独立事件总数")
    for plan in plans:
        try:
            train, validation, _ = split_indices(data, **plan)
            if plan["protocol"] == "rolling":
                year = plan["test_year"]
                if set(data.years[train]) != set(range(year - 12, year - 3)) or set(
                    data.years[validation]
                ) != set(range(year - 3, year)):
                    raise ValueError("滚动协议需覆盖完整的九年训练和三年验证窗口")
        except ValueError as error:
            issues.append(f"划分 {plan} 不可执行：{error}")
    return issues


def preflight(config_path, data_path=None):
    config, digest = load_config_snapshot(config_path)
    issues = validate_config(config)
    result = {
        "ready": False,
        "config_sha256": digest,
        "issues": issues,
        "data_fingerprint": None,
        "scope": "configuration_and_data_contract",
    }
    if data_path is None:
        issues.append("缺少 RiskData 数据文件")
    else:
        try:
            data = RiskData.load(data_path)
            result["data_fingerprint"] = data.fingerprint()
            result["measurement_type"] = data.metadata.get("measurement_type")
            if not issues:
                issues.extend(inspect_inputs(config, data))
        except (OSError, ValueError, KeyError, TypeError) as error:
            issues.append(f"数据校验失败：{error}")
    result["ready"] = not issues
    return result


def source_manifest(root):
    root = Path(root)
    if not (root / "src" / "airport_risk").is_dir():
        package = Path(__file__).resolve().parent
        return {
            f"installed/airport_risk/{path.relative_to(package)}": file_sha256(path)
            for path in sorted(package.rglob("*.py"))
        }
    paths = [
        path for directory in ("src", "tests", "configs") for path in (root / directory).rglob("*")
    ]
    paths += [root / "pyproject.toml", root / "requirements-lock-macos-arm64.txt"]
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(paths)
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and not any(part.endswith(".egg-info") for part in path.parts)
    }
