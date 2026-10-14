# 使用指南

本文命令均在仓库根目录执行。事件记录、审核标注、负载数组与模型产物放在私有路径；示例中的文件路径需替换为实际输入。

<a id="installation"></a>

## 1. 安装与环境

需要 Python 3.11 或以上版本。可在虚拟环境中安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,xlsx]'
python -m airport_risk --help
python -m pytest -q
```

以上虚拟环境激活命令适用于 macOS／Linux。Windows PowerShell 使用 `.venv\Scripts\Activate.ps1`。基础计算支持 CPU；所需库及版本范围见 [pyproject.toml](../pyproject.toml)。

中文图表需要 Noto Sans CJK SC、思源黑体、苹方或微软雅黑等覆盖所需字形的字体。绘图入口会检查字体，缺少字形时需要先安装对应中文字体。

默认规则后端与离线 SRL 后端无需下载模型。使用本地 HanLP SRL 时，另行安装 `hanlp`，准备带分词与 SRL 任务的中文多任务模型，并通过 `--model-path` 指定本地资源。模型协议与来源见[文本与网络](TEXT_NETWORK.md)。

<a id="data-preparation"></a>

## 2. 从事件记录到训练样本

### 2.1 读取事件表与导出审核候选

```bash
python -m airport_risk analyze-records \
  --data data/private/events.xlsx \
  --out data/private/record-analysis
python -m airport_risk annotate \
  --data data/private/events.xlsx \
  --catalog configs/factors.json --airport source-airport-id \
  --out data/private/annotations.json
```

`--airport` 使用输入记录对应的机场 ID。导入保留来源文件摘要、工作表、原行号与业务字段；日期冲突和不确定表达随候选保存。候选初始处于待审状态，审核格式见[事件表读取](INPUT_RECORDS.md)。

可通过 `--backend imported --srl-annotations data/private/srl.json` 使用离线 SRL，或通过 `--backend hanlp --model-path private/models/hanlp` 使用本地神经 SRL。上述参数追加到 `annotate` 命令，其他必需参数保持一致。

### 2.2 使用审核因果阶段标注

审核节点映射、关系方向、事件年份与阶段，填写 `reviewed`、`reviewer`，以及有序 `causal_chains` 或 `stages` 后，生成阶段预测样本：

```bash
python -m airport_risk prepare --kind annotations \
  --source data/private/annotations.json \
  --source-id reviewed-events-v1 \
  --config configs/default.json \
  --out data/private/risk.npz
```

该路径输出 `measurement_type=coded_annotations`，时间轴为因果阶段。预测目标是下一标注阶段的载荷；阶段编码按其单位与时间轴评价。详细编码与 JSON 字段见[数据契约](DATA_CONTRACT.md#审核因果阶段标注)。

### 2.3 使用逐节点负载观测

已有带时间戳的负载观测时，准备 UTF-8 CSV 长表，字段为：

```text
event_id,airport,timestamp,node_id,load
```

每个时间点包含全部目录节点，事件间采样间隔保持一致。时间戳、缺测与窗口要求见[数据契约](DATA_CONTRACT.md#逐时观测长表)。

```bash
python -m airport_risk prepare --kind observations \
  --source data/private/loads.csv \
  --source-id monitored-loads-v1 --load-units source-unit \
  --chains data/private/reviewed-chains.json \
  --config configs/default.json \
  --out data/private/risk.npz
```

`--load-units` 填写实际负载单位。`--chains` 接收已审核的事件因果链 JSON 映射，供训练折建网使用；其结构见[文本与网络](TEXT_NETWORK.md)。该参数为可选项，没有链时不会凭空建立因果边。

该路径输出 `measurement_type=observed_load`。默认配置使用训练历史分位数容量以保留负载单位；使用 `grey_topology` 容量时，需提供按明确规则归一化的负载，并声明 `dimensionless_normalized` 单位。

### 2.4 校验输入

```bash
python -m airport_risk validate --data data/private/risk.npz
python -m airport_risk check \
  --config configs/default.json --data data/private/risk.npz
```

`validate` 检查数组契约；`check` 同时检查来源、配置、单位、未来轨迹及启用划分的可执行条件。`transition` 训练需要连续未来观测或后续审核阶段，且 `target == future_loads[:,0]`。同一事件的全部窗口保持在同一数据折。

<a id="train-evaluate"></a>

## 3. 训练与实验

### 3.1 执行实验矩阵

```bash
python -m airport_risk run \
  --config configs/default.json \
  --data data/private/risk.npz \
  --out runs/experiment-001
```

输出目录须为空。默认配置启用年度分层主划分，支持单年份、单机场输入；各随机种子分别执行训练。`enabled_protocols`、`enabled_experiments` 和 `model_names` 控制划分、实验和模型范围。

完整实验包括 DWPM、NDWPM、DBN、VAR、T-GNN 对照，顺序消融、实际裁减节点的灰阈值扫描、容量和初始负载敏感性，以及动态／冻结权重的攻击与等预算干预。训练折内估计因子、图与容量，再应用于验证和测试样本。指标及动态控制定义见[实验协议](EXPERIMENTS.md)。

`configs/paper_protocols.json` 预设额外比例划分、九年训练／三年验证／次年测试及北京首都机场留出。输入须覆盖启用年份与机场，机场 ID 与配置保持一致；该配置的拓扑容量使用无量纲负载。使用时将上述命令的 `--config` 替换为对应路径。

### 3.2 只训练主模型

```bash
python -m airport_risk train \
  --config configs/default.json \
  --data data/private/risk.npz \
  --out runs/train-001
```

该入口使用首个配置随机种子和主划分训练 DWPM，输出 `model.pt`、`training.pt`、网络审计及训练曲线。推理模型与续训状态分别保存。

### 3.3 断点续训

```bash
python -m airport_risk train \
  --config configs/default.json \
  --data data/private/risk.npz \
  --resume runs/train-001/training.pt \
  --out runs/train-002
```

`epochs` 表示本次追加的训练轮数，`warmup_epochs` 表示预热总轮数。续训恢复优化器、局部随机状态和阶段进度，校验训练数组及配置指纹；已经完成的预热轮次不会重复执行。检查点保存边界与配置要求见[模型说明](MODEL.md#6-检查点与精确续训)。

## 4. 推理与闭环预测

历史文件为 NPY 数组，形状 `[批次,历史步数,节点]`；节点顺序与模型保存的 `node_ids` 一致。

```bash
python -m airport_risk infer \
  --checkpoint runs/train-001/model.pt \
  --history data/private/history.npy \
  --out data/private/prediction.npy
python -m airport_risk rollout \
  --checkpoint runs/train-001/model.pt \
  --history data/private/history.npy --steps 12 \
  --out data/private/rollout.npz
```

`infer` 预测下一周期负载。`rollout` 每周期重新编码历史、计算权重、应用关系算子和传播，再将预测负载追加到滚动窗口；输出包含初始周期的负载轨迹及逐周期边权。推理接口仅接收已观测历史。

## 5. 结果与文档

实验输出包含完整配置、输入与源码摘要、数据划分、训练折网络、逐事件评价、负载与权重轨迹、模型检查点及中文图表。无定义指标、不可训练节点集合及不适用实验会保存状态与原因。

实际计算定义见[公式映射](FORMULA_MAP.md)、[模型说明](MODEL.md)和[实验协议](EXPERIMENTS.md)。软件测试记录见 [software_validation.json](../review/software_validation.json)。返回[仓库首页](../README.md)。
