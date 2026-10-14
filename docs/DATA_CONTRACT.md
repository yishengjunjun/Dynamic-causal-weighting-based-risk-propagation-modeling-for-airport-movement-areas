# 数据契约

## RiskData 数组

设 B 为样本窗口数、H 为历史步数、F 为未来步数、N 为活动节点数、K 为候选因子数。

| 字段 | 形状 | 定义 |
|---|---|---|
| history | B×H×N，H≥2 | 当前可见的非负负载历史 |
| target | B×N | 下一步观测或审核阶段载荷 |
| future_loads | B×F×N，可选 | 仅训练／评价可见的未来轨迹；首步等于 target |
| adjacency | N×N | 源节点→目标节点二元矩阵，对角线为零 |
| capacities | N | 严格正容量，与负载单位一致 |
| impedance | N | 0／1 阻抗标记 |
| grades | N | 非负灰关联度，至少一个正值 |
| years / airports | B | 事件年份及来源机场 ID |
| event_ids | B | 独立事件组 ID，同一事件的窗口可重复 |
| sample_ids | B | 唯一窗口 ID；只有一窗口／事件时可省略 |
| node_ids | N | 稳定、唯一节点 ID |
| factor_values | B×K | 历史因子测量 |
| factor_reference | B | 历史参考序列 |
| metadata | JSON 对象 | 来源、单位、时间轴、编码与因子映射 |

所有数值必须有限；负载非负，容量严格正。`metadata.node_factor_indices` 将 N 个节点映射到不同候选列。NPZ 以 Unicode 和数值数组存储，加载禁用 pickle。SHA-256 指纹覆盖全部数组及元数据。

同一事件的全部窗口须共享年份和机场；所有划分以事件为单位。年度分层按独立事件数量分配，再取事件全部窗口。滚动与机场协议也保持事件边界。`future_loads` 不用于预测权重、网络构建、灰关联或容量估计。

## 逐时观测长表

`prepare --kind observations` 读取 UTF-8 CSV：

```text
event_id,airport,timestamp,node_id,load
```

时间需包含日期、时间及 UTC 偏移；无偏移时使用显式配置时区。拒绝只有日期的记录、不存在／歧义的当地时间、同一节点同一时间重复、跨机场事件和不规则观测间隔。事件之间须使用相同采样步长。年份以整事件最后观测的当地年份固定，跨年事件整体分配到较晚年份，防止后年目标进入较早训练期。

默认 `missing=reject` 要求所有节点完整观测。`forward_fill` 仅允许从过去已观测状态延续；首时刻仍须完整，绝不向前读取未来值。短于 H+F 的事件不构造窗口，记录跳过原因；整份输入无可用窗口时返回错误。

窗口历史均值作为因子测量，其跨因子均值作为参考序列。该统计只使用历史。默认 `train_quantile` 容量由训练历史的配置分位数确定，保留原负载单位。采用 `grey_topology` 时，观测负载必须已经按明确规则归一到无量纲，并声明单位 `dimensionless_normalized`。

## 审核因果阶段标注

`prepare --kind annotations` 读取以下结构。示例展示格式，不是业务事件：

```json
{
  "schema_version": 1,
  "events": [{
    "event_id": "event-id",
    "airport": "source-airport-id",
    "year": 2019,
    "reviewed": true,
    "reviewer": "reviewer-id",
    "causal_chains": [["a24", "a17", "a28"]]
  }]
}
```

也可提供 `stages`，每阶段为节点 ID 到非负载荷的字典，未列出的节点按未激活 0 编码。无显式 stages 时，有序链从全零初态开始，将全部链合并为有向无环图，按最长前驱深度分层累积激活，保持共享节点的全部因果先后；循环链必须提供显式 stages。因果阶段不是物理时间，输出定义为 `measurement_type=coded_annotations`、`time_axis=causal_stage`、无量纲阶段激活。

文本规则与 SRL 只导出候选；人工审核确定节点、方向、阶段、日期冲突及审核人后才可准备训练数据。未来目标来自标注的后续阶段，不由模型预测回填。

## 网络与来源

准备阶段写入零图、单位容量作为候选张量的布局；元数据明确 `network_policy=train_fold` 和对应布局角色。训练前 `prepare_fold_data` 使用训练事件重新选择节点、检验审核因果边、计算容量与阻抗，再切片历史和未来节点轴。布局值不会作为最终估计值直接训练。

`external_fixed` 仅用于独立先验，须提供 `metadata.network_source`；所有折仍独立选择活动节点。先验来源须独立于当前测试数据。

`provenance=authorized_real` 表示数据提供者声明可用于分析，程序记录来源并检查契约；`synthetic` 用于明确的软件测试夹具。原始文件、审核记录、NPZ、模型权重和运行结果不随源码分发。
