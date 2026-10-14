"""事件标注、样本准备、模型训练及闭环预测的命令入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import RiskData, split_indices
from .experiments import run_experiments, _fit_score
from .models import CCRLModel
from .network import prepare_fold_data
from .nlp import FactorCatalog, ImportedSRLBackend, HanLPSRLBackend, rule_relations
from .preparation import prepare_annotations, prepare_observations
from .record_input import read_event_records, analyze_event_records
from .reconstruction import load_config_snapshot, resolve_config, preflight, source_manifest
from .reporting import write_json


def _catalog(args, config=None):
    path = args.catalog if args.catalog else args.config.parent / config["factor_catalog"]
    return FactorCatalog.load(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="机场飞行区动态因果权重与风险传播模型")
    sub = parser.add_subparsers(dest="command", required=True)
    records = sub.add_parser("analyze-records", help="读取事件表并输出字段审计")
    records.add_argument("--data", type=Path, required=True)
    records.add_argument("--out", type=Path, required=True)
    annotate = sub.add_parser("annotate", help="导出因果候选及人工审核表")
    annotate.add_argument("--data", type=Path, required=True)
    annotate.add_argument("--catalog", type=Path, required=True)
    annotate.add_argument("--airport", required=True, help="来源记录的机场 ID")
    annotate.add_argument("--backend", choices=["rules", "imported", "hanlp"], default="rules")
    annotate.add_argument("--srl-annotations", type=Path)
    annotate.add_argument("--model-path", type=Path)
    annotate.add_argument("--out", type=Path, required=True)
    prepare = sub.add_parser("prepare", help="将观测长表或审核标注转换为 RiskData")
    prepare.add_argument("--kind", choices=["observations", "annotations"], required=True)
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--source-id", required=True)
    prepare.add_argument(
        "--load-units", help="观测负载单位；已无量纲归一时使用 dimensionless_normalized"
    )
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--catalog", type=Path)
    prepare.add_argument("--chains", type=Path, help="已审核事件因果链 JSON 映射")
    prepare.add_argument("--out", type=Path, required=True)
    for command, help_text in (
        ("check", "检查配置与数据契约"),
        ("run", "执行所配置的实验矩阵"),
        ("train", "训练主模型并保存可续训状态"),
    ):
        command_parser = sub.add_parser(command, help=help_text)
        command_parser.add_argument("--config", type=Path, required=True)
        command_parser.add_argument("--data", type=Path, required=command != "check")
        command_parser.add_argument("--out", type=Path, required=command != "check")
        if command == "train":
            command_parser.add_argument("--resume", type=Path)
    validate = sub.add_parser("validate", help="检查 RiskData 数组契约")
    validate.add_argument("--data", type=Path, required=True)
    for command, help_text, suffix in (
        ("infer", "根据历史预测下一时刻负载", ".npy"),
        ("rollout", "逐周期更新权重并预测负载轨迹", ".npz"),
    ):
        command_parser = sub.add_parser(command, help=help_text)
        command_parser.add_argument("--checkpoint", type=Path, required=True)
        command_parser.add_argument(
            "--history", type=Path, required=True, help="NPY [批次,历史步数,节点]"
        )
        command_parser.add_argument("--out", type=Path, required=True)
        if command == "rollout":
            command_parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args(argv)

    if args.command == "analyze-records":
        summary = analyze_event_records(read_event_records(args.data), args.out)
        print(f'已读取 {summary["summary"]["imported_records"]} 条记录：{args.out}')
    elif args.command == "annotate":
        if args.backend == "imported" and args.srl_annotations is None:
            parser.error("imported 后端需要 --srl-annotations")
        if args.backend == "hanlp" and args.model_path is None:
            parser.error("hanlp 后端需要 --model-path")
        backend = (
            ImportedSRLBackend.load(args.srl_annotations)
            if args.backend == "imported"
            else HanLPSRLBackend(args.model_path) if args.backend == "hanlp" else None
        )
        catalog = _catalog(args)
        source = read_event_records(args.data)
        events = []
        for record in source["records"]:
            text = record["narrative"] or record["provided_causal_text"]
            event = record["import_id"]
            relations = backend.extract(text, event_id=event) if backend else rule_relations(text)
            events.append(
                {
                    "event_id": event,
                    "airport": args.airport,
                    "year": int(record["parsed_date"][:4]) if record["parsed_date"] else None,
                    "reviewed": False,
                    "reviewer": None,
                    "causal_chains": [],
                    "date_review_required": bool(record["warnings"]),
                    "source_record": record,
                    "relations": [
                        catalog.annotation_template(relation, event_id=event)
                        for relation in relations
                    ],
                }
            )
        write_json(args.out, {"schema_version": 1, "source": source["source"], "events": events})
        print(f"已导出 {len(events)} 条待审核标注：{args.out}")
    elif args.command == "prepare":
        if args.out.suffix != ".npz":
            parser.error("prepare 输出需使用 .npz 扩展名")
        config = resolve_config(load_config_snapshot(args.config)[0])
        catalog = _catalog(args, config)
        settings = config["preparation"]
        if args.kind == "observations":
            if not args.load_units:
                parser.error("observations 需要 --load-units 声明负载单位")
            import json

            chains = json.loads(args.chains.read_text(encoding="utf-8")) if args.chains else None
            data = prepare_observations(
                args.source,
                catalog.ids,
                source_id=args.source_id,
                annotated_chains=chains,
                load_units=args.load_units,
                **settings,
            )
        else:
            data = prepare_annotations(
                args.source,
                catalog.ids,
                source_id=args.source_id,
                **{key: settings[key] for key in ("history_steps", "future_steps", "stride")},
            )
        data.save(args.out)
        print(
            f"已保存 {len(data.target)} 个窗口、{len(np.unique(data.event_ids))} 个事件：{args.out}"
        )
    elif args.command in {"check", "run", "train"}:
        result = preflight(args.config, args.data)
        if args.command == "check" and args.out:
            write_json(args.out, result)
        if not result["ready"]:
            print("检查未通过：")
            for issue in result["issues"]:
                print(f"- {issue}")
            return 2
        if args.command == "check":
            print(f'检查通过；数据指纹={result["data_fingerprint"]}')
            return 0
        if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
            raise ValueError("实验输出目录必须为空，防止不同运行混用产物")
        raw_config, digest = load_config_snapshot(args.config)
        data = RiskData.load(args.data)
        if digest != result["config_sha256"] or data.fingerprint() != result["data_fingerprint"]:
            raise ValueError("预检查后输入文件发生变化，请重新检查后运行")
        config = resolve_config(raw_config)
        write_json(
            args.out / "run_manifest.json",
            {
                "preflight": result,
                "configuration": config,
                "source_sha256": source_manifest(Path(__file__).resolve().parents[2]),
                "measurement_type": data.metadata["measurement_type"],
                "data_source_id": data.metadata["source_id"],
            },
        )
        parameters = config["experiment"]
        if args.command == "run":
            run_experiments(
                data,
                args.out,
                profile=parameters["profile"],
                config_override=parameters,
                progress=lambda message: print(message, flush=True),
            )
        else:
            seed = parameters["seeds"][0]
            train, validation, test = split_indices(
                data, seed=seed, train_fraction=parameters["stratified_train_fraction"]
            )
            fold, audit = prepare_fold_data(data, train, parameters)
            model, metrics, trace = _fit_score(
                "DWPM",
                fold,
                train,
                test,
                fold.adjacency,
                seed,
                parameters,
                validation=validation,
                checkpoint_path=args.out / "training.pt",
                resume_from=args.resume,
            )
            model.save(args.out / "model.pt")
            write_json(args.out / "network.json", audit)
            write_json(
                args.out / "training.json",
                {
                    "metrics": metrics,
                    "trace": trace,
                    "train_event_ids": np.unique(data.event_ids[train]),
                    "test_event_ids": np.unique(data.event_ids[test]),
                },
            )
        print(args.out)
    elif args.command == "validate":
        data = RiskData.load(args.data)
        print(
            f"校验通过：{len(data.target)} 个窗口、{len(np.unique(data.event_ids))} 个事件、{len(data.capacities)} 个节点；指纹={data.fingerprint()}"
        )
    elif args.command in {"infer", "rollout"}:
        suffix = ".npy" if args.command == "infer" else ".npz"
        if args.out.suffix != suffix:
            parser.error(f"{args.command} 输出需使用 {suffix} 扩展名")
        model = CCRLModel.load(args.checkpoint)
        history = np.load(args.history, allow_pickle=False)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.command == "infer":
            np.save(args.out, model.predict(history))
        else:
            np.savez_compressed(args.out, **model.rollout(history, steps=args.steps))
        write_json(
            args.out.with_suffix(".json"),
            {
                "checkpoint": str(args.checkpoint),
                "node_ids": model.node_ids,
                "scope": "history_only_prediction",
                "weight_update": "recomputed_each_cycle",
            },
        )
        print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
