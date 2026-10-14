"""输出结构化结果并生成中文图表；每张图明确标注数据来源。"""

from pathlib import Path
import csv
import json
import os
import re
import tempfile
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "airport-risk-mpl-cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font


def configure_chinese_font():
    """发现并校验中文字体；缺失时明确报错，避免生成方块字。"""
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "Microsoft YaHei UI",
        "SimHei",
        "WenQuanYi Micro Hei",
        "Heiti SC",
        "Arial Unicode MS",
        "Noto Serif CJK SC",
        "Songti SC",
    ]
    # 检查本模块全部中文字形，覆盖标题、图例、轴标签及页脚。
    required = {
        ord(c) for c in re.findall(r"[\u4e00-\u9fff]", Path(__file__).read_text(encoding="utf-8"))
    }
    paths = []
    for family in candidates:
        entries = [entry for entry in font_manager.fontManager.ttflist if entry.name == family]
        if entries:
            paths.extend(entry.fname for entry in entries)
            continue
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=[family]), fallback_to_default=False
            )
            paths.append(path)
        except ValueError:
            continue
    # 补充常见系统路径，兼容字体缓存尚未登记的系统字体。
    paths.extend(
        str(path)
        for path in [
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        ]
        if path.is_file()
    )
    for path in dict.fromkeys(paths):
        try:
            if not required.issubset(FT2Font(path).get_charmap()):
                continue
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            weights = [
                font_manager.weight_dict.get(entry.weight, entry.weight)
                for entry in font_manager.fontManager.ttflist
                if entry.name == family
            ]
            weights = [weight for weight in weights if isinstance(weight, (int, float))]
            normal_weight = min(weights, key=lambda weight: abs(weight - 400)) if weights else 400
            bold_weight = min(weights, key=lambda weight: abs(weight - 700)) if weights else 700
            plt.rcParams.update(
                {
                    "font.family": [family],
                    "font.sans-serif": [family],
                    "font.weight": normal_weight,
                    "axes.labelweight": normal_weight,
                    "axes.titleweight": normal_weight,
                    "axes.unicode_minus": False,
                }
            )
            return {
                "family": family,
                "path": str(path),
                "weight": normal_weight,
                "bold_weight": bold_weight,
                "verified_chinese_glyphs": len(required),
            }
        except (OSError, RuntimeError, ValueError):
            continue
    raise RuntimeError(
        "未发现覆盖图表中文字形的字体。请安装 Noto Sans CJK SC、思源黑体或微软雅黑后重绘。"
    )


def _model_label(name):
    """保留模型缩写，同时解释图中模型名称。"""
    return {
        "DWPM": "DWPM\n动态权重",
        "NDWPM": "NDWPM\n静态权重",
        "DBN": "DBN\n二分类基线",
        "VAR": "VAR\n岭回归基线",
        "T-GNN": "T-GNN\n时空图基线",
        "NoRL": "移除强化学习",
        "NoRL_NoAttention": "移除强化学习\n及注意力",
    }.get(name, name)


def clean_json(value):
    """将 NumPy 类型转换为 JSON 类型，并以空值表示非有限数值。"""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path, value):
    """使用 UTF-8 写入规范 JSON，保留中文并拒绝非标准数值。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows):
    """汇总全部字段并写入 UTF-8 CSV。"""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean_json(v) for k, v in row.items()})


def plot_results(data, report, outdir):
    """使用本次训练与干预结果绘制中文图表，适配可选实验。"""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    font = configure_chinese_font()
    prefix = "测试构造数据" if data.metadata["provenance"] == "synthetic" else "输入数据"
    plt.rcParams.update(
        {"font.size": 9, "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False}
    )

    def finish(fig, filename):
        fig.text(0.5, 0.01, prefix + "｜本次实验计算结果", ha="center", fontsize=8, color="#52606d")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
        fig.savefig(out / filename)
        plt.close(fig)

    def absent(ax, message="配置未启用此实验"):
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    graph = nx.from_numpy_array(data.adjacency, create_using=nx.DiGraph)
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(graph, seed=10)
    nx.draw_networkx_edges(
        graph, pos, ax=ax, arrowsize=7, width=0.6, alpha=0.35, edge_color="#52606d"
    )
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_size=180, node_color=np.where(data.impedance, "#e9a23b", "#58a6a6")
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        ax=ax,
        labels={i: str(node) for i, node in enumerate(data.node_ids)},
        font_size=7,
        font_family=font["family"],
        font_weight=font["weight"],
    )
    ax.legend(
        handles=[
            Patch(color="#e9a23b", label="阻抗节点"),
            Patch(color="#58a6a6", label="其他节点"),
        ],
        loc="lower right",
    )
    ax.set_title(f"主训练折风险网络：{len(graph)} 个节点，{graph.number_of_edges()} 条边")
    ax.set_axis_off()
    finish(fig, "fig1_graph.png")

    primary = [
        row
        for row in report["benchmarks"]
        if row["protocol"] == report.get("primary_protocol", "primary_stratified")
    ]
    methods = list(dict.fromkeys(row["model"] for row in primary))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    values = [
        [row["mse_normalized"] for row in primary if row["model"] == name] for name in methods
    ]
    labels = [_model_label(name) for name in methods]
    axes[0].boxplot(values, tick_labels=labels)
    axes[0].set(xlabel="模型", ylabel="容量归一化 MSE", title="不同随机种子的下一时刻负载预测误差")
    axes[1].bar(
        labels,
        [
            np.mean([row["event_failure_accuracy"] for row in primary if row["model"] == name])
            for name in methods
        ],
    )
    axes[1].set(xlabel="模型", ylabel="平均准确率", title="事件失效分类准确率", ylim=(0, 1))
    finish(fig, "fig2_prediction.png")

    attacks = report.get("node_attacks", {})
    if attacks and attacks.get("status") == "completed":
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for level, values in attacks["nri"].items():
            axes[0].plot(np.asarray(values, dtype=float), label=f"负载增加 {level}%")
        for level, values in attacks.get(
            "perturbation_excess_over_unperturbed_rollout",
            attacks["perturbation_global_load_change"],
        ).items():
            axes[1].plot(np.asarray(values, dtype=float), label=f"负载增加 {level}%")
        axes[0].set(xlabel="节点索引", ylabel="局部 NRI", title="公式 24：局部节点风险指数")
        axes[1].set(
            xlabel="受扰动节点索引",
            ylabel="相对初始总负载的增量",
            title="逐节点扰动相对于无扰动传播的额外影响",
        )
        for ax in axes:
            ax.legend()
        finish(fig, "fig3_nri.png")
        for metric, filename in (("sri", "fig4_sri.png"), ("ndi", "fig5_ndi.png")):
            fig, ax = plt.subplots(figsize=(9, 4))
            curves = np.asarray(attacks[metric], dtype=float)
            finite_final = np.nan_to_num(curves[:, -1], nan=-np.inf, posinf=np.finfo(float).max)
            top = np.argsort(-finite_final, kind="stable")[: min(8, len(curves))]
            for node in top:
                curve = curves[node].copy()
                curve[~np.isfinite(curve)] = np.nan
                ax.plot(curve, label=str(data.node_ids[node]))
            ax.set(
                xlabel="传播步数",
                ylabel=metric.upper(),
                title=f"{metric.upper()} 扰动传播轨迹：按最终值展示前 {len(top)} 个节点",
            )
            ax.legend(ncol=4, fontsize=7, title="受扰动节点")
            finish(fig, filename)

    comparison = {
        key: value
        for key, value in report.get("control_comparison", {}).items()
        if value.get("status") == "completed"
    }
    if comparison:
        fig, axes = plt.subplots(
            1, len(comparison), figsize=(6 * len(comparison), 4), squeeze=False
        )
        names = {
            "none": "无干预",
            "degree": "按节点度干预",
            "NRI_load_ratio": "NRI 与负载比复合干预",
            "random": "随机干预",
        }
        for ax, (mode, result) in zip(axes.flat, comparison.items()):
            for name, curve in result["curves"].items():
                ax.plot(curve, label=names.get(name, name))
            ax.set(
                xlabel="传播步数",
                ylabel="平均 NDI",
                title={"dynamic": "动态重算权重", "frozen": "训练折平均权重冻结"}[mode]
                + "：等预算干预",
            )
            ax.legend()
        finish(fig, "fig6_control.png")

    if report.get("ablations"):
        fig, ax = plt.subplots(figsize=(9, 4))
        rows = report["ablations"]
        ax.bar(
            [_model_label(row["model"]) for row in rows], [row["mse_normalized"] for row in rows]
        )
        ax.set(xlabel="模型设置", ylabel="容量归一化 MSE", title="独立训练的消融实验")
        finish(fig, "fig7_ablation.png")

    sensitivity = report["sensitivity"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    rows = [row for row in sensitivity["grey_threshold"] if row.get("status") == "completed"]
    if rows:
        axes[0, 0].plot(
            [row["threshold"] for row in rows],
            [row["mse_normalized"] for row in rows],
            marker="o",
            label="各子图全部节点",
        )
        comparable = [row for row in rows if row.get("shared_node_metrics")]
        if comparable:
            axes[0, 0].plot(
                [row["threshold"] for row in comparable],
                [row["shared_node_metrics"]["mse_normalized"] for row in comparable],
                marker="s",
                label="共同节点及统一容量",
            )
        axes[0, 0].set(
            xlabel="灰关联阈值", ylabel="容量归一化 MSE", title="阈值筛选后重建网络并重新训练"
        )
        axes[0, 0].legend()
        axes[0, 1].plot(
            [row["threshold"] for row in rows],
            [row["retained_modeled_nodes"] for row in rows],
            marker="o",
            label="节点数",
        )
        axes[0, 1].plot(
            [row["threshold"] for row in rows],
            [row["retained_edges"] for row in rows],
            marker="s",
            label="边数",
        )
        axes[0, 1].set(xlabel="灰关联阈值", ylabel="数量", title="训练折内实际保留的网络规模")
        axes[0, 1].legend()
    else:
        absent(
            axes[0, 0],
            "无可训练的阈值结果" if sensitivity["grey_threshold"] else "配置未启用灰关联敏感性",
        )
        absent(axes[0, 1])
    rows = sensitivity["capacity_prediction"]
    if rows:
        axes[1, 0].plot(
            [row["scale"] for row in rows], [row["mse_normalized"] for row in rows], marker="o"
        )
        axes[1, 0].set(
            xlabel="模型容量缩放系数",
            ylabel="容量归一化 MSE",
            title="容量扰动后重新训练，评价容量保持一致",
        )
    else:
        absent(axes[1, 0])
    plotted = False
    for key, label in (("capacity_scale", "容量缩放"), ("eta_scale", "初始负载系数缩放")):
        rows = [row for row in sensitivity[key] if row.get("mean_final_ndi") is not None]
        if rows:
            axes[1, 1].plot(
                [row["scale"] for row in rows],
                [row["mean_final_ndi"] for row in rows],
                marker="o",
                label=label,
            )
            plotted = True
    if plotted:
        axes[1, 1].set(
            xlabel="压力场景参数缩放系数",
            ylabel="最终平均 NDI",
            title="传播敏感性：与预测误差独立评价",
        )
        axes[1, 1].legend()
    else:
        absent(axes[1, 1])
    finish(fig, "fig8_sensitivity.png")

    rollout_path = out.parent / "dynamic_rollout.json"
    if rollout_path.exists():
        rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
        loads = np.asarray(rollout["load_trajectory"], dtype=float)
        fig, ax = plt.subplots(figsize=(9, 4))
        image = ax.imshow((loads / data.capacities).T, aspect="auto", cmap="YlOrRd")
        ax.set(
            xlabel="闭环预测步数（零为末次观测）",
            ylabel="节点索引",
            title="观测历史起点的动态负载传播",
        )
        fig.colorbar(image, ax=ax, label="负载与容量之比")
        finish(fig, "fig9_dynamic_rollout.png")


def markdown_report(report):
    """生成训练边界、评价定义、实测结果与完整产物索引。"""
    main = [
        row
        for row in report["benchmarks"]
        if row["protocol"] == report.get("primary_protocol", "primary_stratified")
    ]
    config = report["config"]
    lines = [
        "# 实验报告",
        "",
        f"输入来源声明：**{report['provenance']}**。数值由本次运行计算。",
        "",
        f"数据指纹：`{report['data_fingerprint']}`",
        "",
        f"主实验按事件组和年份分层，训练比例 {config['stratified_train_fraction']:.1%}；随机种子 {config['seeds']}。同一事件的全部窗口分配到同一折。",
        "",
        f"启用协议：`{', '.join(config['enabled_protocols'])}`。机场留出标识：`{config['airport_holdout']}`；滚动测试年份：{config['rolling_test_years']}。",
        "",
        f"网络模式：`{config['network_mode']}`；容量模式：`{config['capacity_mode']}`；训练模式：`{config['training_mode']}`。每折因子筛选、建网和统计容量只读取该折训练域；有来源的外部先验按配置独立接入。",
        "",
        "归一化 MSE 按该折容量计算。节点失效是负载达到容量；事件失效是在该事件任一窗口至少一个所选节点失效。风险分数为截断到 [0,1] 的负载容量比。",
        "",
        "| 模型 | 平均归一化 MSE | 事件失效准确率 | 平均节点数 |",
        "|---|---:|---:|---:|",
    ]
    for model in dict.fromkeys(row["model"] for row in main):
        selected = [row for row in main if row["model"] == model]
        lines.append(
            f"| {model} | {np.mean([row['mse_normalized'] for row in selected]):.6f} | {np.mean([row['event_failure_accuracy'] for row in selected]):.4f} | {np.mean([row['nodes'] for row in selected]):.1f} |"
        )
    lines += [
        "",
        "全部已配置模型及协议的计算结果均保留，设置在评价前确定。",
        "",
        "## 因子筛选与容量敏感性",
        "",
        "每个灰关联阈值独立筛选活动因子、裁剪历史与目标的节点轴、重建边与容量并重新训练。各子图全节点指标与共同节点统一容量指标分别保存；不把不同节点集合的准确率视作相同评价任务。筛选后不足两个节点的阈值记录为不可训练，不强制增加节点。",
        "",
        f"共同评价节点：{report['sensitivity'].get('grey_shared_node_ids',[])}。",
        "",
        "容量预测敏感性在每个缩放条件重新训练，评价使用主训练折名义容量，保持标签与单位一致。容量传播敏感性改变压力场景传播容量；η 只设置拓扑压力场景的初始负载，独立报告 NDI。",
        "",
        "## 动态扰动与干预",
        "",
        f"权重模式：{config['control_weight_modes']}；NRI 排序模式：`{config['control_rank_mode']}`。每种策略使用相同观测历史、攻击集合、控制节点预算和干预强度。随机策略的节点集合、随机种子及摘要指纹在结果中保存。",
        "",
        "动态模式每周期根据当前历史重算权重，执行节点关系门控及传播，再把预测负载追加到历史。冻结模式使用完整训练折的平均权重。干预抑制受控节点的正向新增负载，保留负向恢复量。NRI 未定义时优先使用有限 NRI，不足预算的部分按负载容量比补齐并记录次数。",
        "",
        "压力场景不读取未来目标。单事件闭环预测仅在实际提供未来观测的预测步计算误差；后续没有标签的预测步只保存轨迹。",
        "",
        "## 产物",
        "",
        "`metrics.json` 保存完整配置、各折网络摘要、实测指标和计时；`networks/` 保存训练事件清单、因子选择、边显著性与容量；`splits/` 保存事件组与样本索引；`tables/event_predictions.csv` 保存逐事件评价；`stress_selection.json` 保存压力历史取样；`dynamic_rollout.json` 保存动态权重及负载；`checkpoints/` 保存模型和训练状态；`figures/` 保存本次结果生成的中文图。",
        "",
    ]
    return "\n".join(lines)
