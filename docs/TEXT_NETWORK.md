# 事件文本、因子目录与训练域因果网络

本流水线依次执行语义候选提取、因子映射、专家审核、训练域因子筛选和有向网络构建。计算模块不访问留出事件的词频、标签、因果链、历史负载或容量信息。

## 因子目录

`configs/factors.json` 包含 98 个有稳定 ID、中文名称、分类、别名与操作定义的候选因素。`a1–a36`、`b1–b8`、`c1–c13`、`d1–d6` 对应论文表 1 的 63 项；`x1–x35` 覆盖疲劳、信息交接、冰雪、鸟类活动、异物、车辆机械故障和管理闭环等候选维度。目录中的 `catalog_group` 区分表 1 节点与扩展候选项。灰关联筛选后的节点数由输入训练数据决定。

四个类别聚合节点与具体节点均保留独立 ID。表 1 中 `c4` 和 `c10` 同名，目录维持两个编号；词面命中不能消除这项歧义，审核人须明确使用哪个 ID。`FactorCatalog.match()` 输出候选集合，只有审核后填写的 `cause_factor_ids` 和 `effect_factor_ids` 可以形成正式因果关系。

## 语义接口与审核协议

`rule_relations()` 使用中文和英文显式因果连接词确定方向，输出方法名称 `explicit_causal_rules`。否定关系保留 `polarity=negated`，疑似、可能等措辞保留 `certainty=uncertain`，初始审核状态为 `pending`。它不将关键词共现解释为因果关系。

神经语义角色标注由 `HanLPSRLBackend` 调用本地 HanLP 多任务模型，运行 `model(text, tasks="srl")`，读取 `tok*` 与 `srl` 字段。语义帧采用 `[短语,角色,起始词元,终止词元]`，终点不包含在区间内；适配器验证区间、短语和 `PRED`。只有因果谓词的 `ARG0→ARG1`，或者明确的 `ARGM-CAU`，可以产生因果候选。一般动作谓词的施事与受事不会自动产生因果边。输出方法记录为 `hanlp_neural_srl`。

HanLP 为可选依赖；将模型及其依赖资源预先保存在本机，再向适配器传入模型路径。项目不上传事件叙述到外部推理服务，不分发 HanLP 权重。可从官方模型表选择 `CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH` 等带分词与 SRL 任务的模型，模型与语料的授权按提供方条款管理。接口依据：[官方多任务 SRL 示例](https://github.com/hankcs/HanLP/blob/doc-zh/plugins/hanlp_demo/hanlp_demo/zh/srl_mtl.ipynb)、[官方数据格式](https://hanlp.hankcs.com/docs/data_format.html)、[官方安装说明](https://hanlp.hankcs.com/docs/install.html)。

离线标注可由 `ImportedSRLBackend` 读取事件 ID 对应的 `text`、`tokens` 和 `frames`；原文变化或词元区间不一致会被拒绝。导入格式遵循相同协议。

`review_relation()` 记录批准、驳回或待审决定，以及审核人、说明和最终因子 ID。否定关系不能批准为肯定边。`approved_chains()` 默认只接受经批准、非否定且非疑似的关系；是否允许经审阅的疑似关系由 `include_uncertain` 显式配置。专家链的结构为：

```json
{"chain":["b2","a7"],"review_status":"approved","reviewer":"审核人标识","polarity":"affirmative","certainty":"asserted"}
```

上述 JSON 仅演示结构，不是事件记录。批量训练元数据使用 `metadata.annotated_chains[event_id] = [[因子ID,...],...]`。其中纯链序列是调用方已完成审核后提交的有序链；也可使用包含审核字段的链对象或含 `relations` 字段的对象。训练网络仅读取本折事件 ID 对应的条目。

## 文本特征的训练域约束

`TextFeatureModel.fit(train_narratives)` 在训练文本上确定语言、TF-IDF 词表、IDF 和 KMeans 中心。中文为字符 2–4 元组，英文为词 1–2 元组并使用英语停用词。`transform()` 与 `predict()` 固定使用已训练状态，留出文本的新词不进入词表。`thematic_clusters()` 是一次性的描述分析接口；用于预测时应采用独立的 fit/transform 接口。

## 灰关联与活动节点

`NetworkBuilder.fit(values_train, reference_train, chains_train, node_ids, history_train=...)` 输入形状分别为 `[训练样本,候选因素]`、`[训练样本]` 和审核因果链。灰关联采用核心公式 1–4；默认严格保留 `grade > 0.3`。另要求训练域出现率至少 `0.01`，排除从未出现的列，并默认排除训练域常量列。常量筛查可在明确需要聚合常量节点时配置关闭。筛选完全来自训练折。

`active_indices` 选择实际节点维度，历史、目标、未来轨迹、阻抗向量和矩阵均按同一选择裁剪。因子输入表仍保留完整候选列，`node_factor_indices` 指向原因子列，便于不同灰阈值重新筛选。不以相邻链节点被剔除为理由创建跨越该节点的新边。少于两个活动节点时，正式预测实验返回可解释的不可训练状态，不强行补选节点。

## 边频率与显著性

每条因果链只贡献相邻有向边。相同事件内同一边出现多次仍计为一个事件支持；不同事件的支持计数相加。默认最小支持为 2。

置换检验在每个事件的每条链内部随机打乱节点顺序，保留链长度、节点出现次数与事件分组。零假设是给定同一链节点集合后的次序随机性，检验对象是有向相邻支持频率；它不是基于随机对照的因果效应识别。使用 999 次置换及固定种子 2026，单侧 `p=(1+次数[置换支持≥观测支持])/(R+1)`。对全部非对角候选有向边执行 Benjamini–Hochberg 校正，默认 `q≤0.05`，再与最小支持条件取交集。

`expert_edges` 可指定批准或驳回的边，须提供 `source`、`target`、`status`、`reviewer` 与 `basis`；覆盖统计决定后的记录完整保留在审计文件中。统计检验不会在边不足时自动降低显著性门槛。空图允许进行公式和接口计算；没有定义的网络指标由实验层明确报告。

图统计包含有向三元组完整 census、入出度形态、弱连通分量，以及无向投影上的贪心模块度社区。三元组是描述性计数，不额外声称具有随机图显著性。

## 容量与阻抗

`capacity_mode=grey_topology` 使用公式 5–7，默认 `omega_be=omega_pr=0.5`。基础容量设正值下限 `1e-6`，再默认归一化为均值 1，最后乘 `capacity_scale`。这一尺度与项目无量纲编码负载配套；若观测负载具有独立物理量纲，应关闭均值归一化或采用训练分位数容量，并据实际单位设置倍率。

`capacity_mode=train_quantile` 使用训练历史所有样本和历史步上的节点 0.95 分位数作为容量，设正值下限并乘容量倍率。该模式保留观测负载量纲，不做均值归一化。两种模式均不使用预测目标或留出历史估计容量。

阻抗标记来自训练历史的恢复证据：某节点在至少 10% 的相邻历史步出现负载下降，下降量严格大于 `impedance_delta_tolerance=1e-9`，则标为阻抗节点。未观察到下降的节点标为累积型节点。此操作定义针对离散负载轨迹，不推断事件叙述中未记录的恢复行为。

## 折接口与外部网络

`prepare_fold_data(data, train_indices, config, grey_threshold=None)` 返回裁剪后的 `RiskData` 与 JSON 可序列化审计。它保留样本及事件 ID，切片 `future_loads`，记录选中节点、训练事件 ID、容量、边的 p/q 值、专家决定、网络指纹与 `train_only`。

默认 `network_mode=train_fitted` 重新估计每个训练折网络。若使用在本研究数据之外独立确定的网络，须显式设置 `network_mode=external_fixed`，并提供非空 `metadata.network_source`；灰关联仍由训练折筛选，但边、容量和阻抗采用外部先验相应子图，审计记为外部来源。`metadata.network_policy` 仅记录输入来源，不覆盖运行配置；即使输入标记为外部先验，`network_mode=train_fitted` 仍按提交的候选观测重新构建本折网络。不得将全数据估计的网络标记为独立外部先验。
