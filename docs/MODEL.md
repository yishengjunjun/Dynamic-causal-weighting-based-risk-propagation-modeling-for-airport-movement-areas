# 模型与训练定义

本文件定义运行实现的数学约定、节点关系、训练回合和检查点契约。网络和容量来自当前训练折，验证集、测试集以及未来目标不参与初始化边权、特征统计或网络选择。

## 1. 输入与方向

- `history[B,H,N]` 为长度至少 2 的已观测非负节点负载。
- `target[B,N]` 为下一周期观测负载。
- `future_loads[B,F,N]` 为与历史连续衔接的观测轨迹；`target == future_loads[:,0]`。
- 邻接和权重轴顺序均为 `[源节点,目标节点]`，邻接为无自环二元掩码。
- 容量 `C[N]` 必须严格为正；阻抗标识 `Z[N]` 为二元量。
- `node_ids` 与数组最后一轴一一对应，节点 ID 不随子图重编号而改变。

本实现的传播量是非负、累积的风险负载。单步传播不引入恢复项；闭环轨迹保持相应的累积语义。观测中的物理计量与容量必须使用相同单位。

## 2. 同步传播

未配置关系时，源状态为 `a_i = 1[L_i >= C_i]`，目标增量为：

`Δ_j = C_j Σ_i a_i W_ij`

公式 12 用稳定等价式计算：

`Θ_j = sigmoid(-L_j) - exp(-C_j) sigmoid(L_j)`

`L'_j = L_j + (1 - Z_j Θ_j) Δ_j`

保留公式 12 的负值。目标节点阻抗影响到达该节点的增量。全部节点使用同一周期的负载计算，禁止边遍历过程中的异步更新。不存在的边权始终为零。

### 2.1 节点关系

`relations` 为可选配置；空配置严格保持上述基础传播。三类关系组相互不重叠，重复节点、未知 ID、组内重复和不足两个成员均拒绝接受。

| 配置键 | 源强度定义 |
|---|---|
| `mutex_groups` | 只保留已激活成员中 `L_i/C_i` 最大者；并列时选择当前节点顺序中索引最小者 |
| `cooccurrence_groups` | 全部成员均达到容量时保留各成员源强度，否则整组源强度为零 |
| `cumulative_groups` | 每个成员使用同一连续强度 `mean_k max(L_k/C_k - 1,0)`，表示组平均正超载量 |

关系强度代替基础公式中的 `a_i`，其余传播项保持一致。累积关系为连续超载强度，可大于 1。二元失效和互斥选择为硬门控；权重梯度始终保留，累积关系和公式 12 对负载保留分段可导的梯度。

### 2.2 运行上下文与表 2

`context[node_id]` 必须同时给出 `years_service`、`workload` 和 `working_hours`。表 2 查询采用左闭右开、末段包含右端点的规则。工时区间 `[1,2]`、`[3,4]`、`[5,6]` 之间的空隙没有有效系数；处于空隙或超出表范围的上下文输入会被拒绝。

三个分段系数按 `context_weights` 加权平均，权重必须非负且总和为 1，默认各为 `1/3`。配置上下文后的吸收项为：

`Θ_context,j = clip(Θ_j + Σ_m α_m θ_m,j, -1, 1)`

这一定义将上下文作为公式 12 基础值的有界加性项，而不是替换公式 12。只有 `Z_j=1` 的节点应用该吸收项。无上下文节点继续使用未经截断的公式 12 基础值。

配置示例：

```json
{
  "mutex_groups": [["A", "B"]],
  "cooccurrence_groups": [["C", "D"]],
  "cumulative_groups": [["E", "F"]],
  "context_weights": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
  "context": {"G": {"years_service": 10, "workload": 0.8, "working_hours": 5}}
}
```

该示例只展示配置格式，不表示任何机场的节点关系。实际声明由私有网络材料或训练折中确定的关系清单提供。

`remap_relations(config, selected_ids)` 将关系截取到训练折子图：至少保留两个成员的关系组继续生效，其余关系组删除，上下文只保留当前节点。随后由 `compile_relations` 按子图节点顺序映射索引并检查冲突。

## 3. CCRL 因果权重

### 3.1 时序编码和注意力

输入通道默认是二元失效状态。`include_load_channel=true` 时附加 `L/C` 连续通道。因果卷积满足：

`h[t] = Σ_k W[k] x[t-dk] + b`

参数 `W[0]` 对应当前周期，左侧缺失历史补零，各层使用膨胀率 `2^layer` 和 `tanh`。节点共享卷积参数，卷积不接触未来位置。

最后一个观测位置的编码经 query/key 线性投影后计算点积注意力，尺度为 `sqrt(hidden)`。只在当前源节点的已有出边上执行 softmax；无出边源的整行注意力为零。关闭注意力的消融使用相同图掩码上的均匀出边注意力。

### 3.2 初始边权

公式 17 使用最后两个观测位置：

`W^0_ij = alpha_ij max(L_last,j - L_prev,j, 0)/(L_prev,i + epsilon)`

边权投影到 `[0,max_weight]`。预测目标不参与该计算。`epsilon`、`max_weight` 和投影方式均为公开运行参数。

### 3.3 连续策略和价值函数

策略状态为已有边的权重向量，顺序由邻接的行优先非零索引决定。Actor 和 critic 为两层 MLP，中间激活为 `tanh`。

策略为独立 Normal 分布：

`μ(s) = action_scale tanh(actor(s))`

`σ = action_scale exp(clip(log_std,-5,1))`

动作是边权增量，更新后投影到 `[0,max_weight]`。Actor 末层初始为零，因此初始均值动作不改变边权。训练的得分函数梯度在与梯度图分离的采样动作处计算。

奖励为原始负载平方误差的负和：`r = -Σ_j (L_pred,j-L_observed,j)^2`。Critic 使用半梯度 TD：`δ = r + gamma V(s') - V(s)`；终止转移的 bootstrap 项为零。Critic 损失为 `mean(δ²)/2`，actor 损失为 `-mean(log π(action|s) stop_gradient(δ))`。

### 3.4 两种训练回合

- `training_mode="refinement"`：同一已观测历史下执行 `refinement_steps` 次权重细化，每次动作均对照同一个下一周期目标；环境历史保持固定。推理使用相同次数的均值动作细化。
- `training_mode="transition"`：必须提供观测 `future_loads`。每周期根据当前历史计算初始边权，采样一次策略动作，用统一关系传播得到下一负载，将预测负载追加到固定长度历史窗口，再计算下一周期状态。每步奖励只与相应周期的观测标签比较；标签不进入状态、边权或历史窗口。推理每周期使用一次均值动作，`rollout` 按同一流程闭环更新历史。

监督预热仅更新 encoder/query/key，损失为容量归一化的单步负载 MSE；actor–critic 阶段更新策略和价值函数。`warmup_epochs=0` 可关闭预热。关闭 RL 时使用预热后的初始边权。训练中的采样使用独立的本地随机生成器，不依赖调用方的随机调用顺序。

## 4. 对照模型

### 4.1 静态传播

对训练历史的每个相邻观测位置计算公式 17 的正增量比，注意力使用均匀出边分配，再对样本及时间取均值得到固定边权。目标负载不参与静态边权估计。预测和闭环传播使用相同节点关系、上下文与同步传播公式。

### 4.2 VAR

取最后 `lag` 个观测周期，按时间升序展平。输入和目标的均值、标准差仅从训练样本估计；常数列尺度固定为 1。带截距的岭回归不惩罚截距。`ridge=0` 使用最小二乘；否则求解带岭项的正规方程。输出是原单位的连续负载。VAR 是统计预测对照，不应用物理传播关系。

### 4.3 二元 DBN

双时间片网络中，每个节点使用自身上一周期状态以及最多 `max_parents` 个空间入邻居状态。空间父节点按节点索引排序截取；空间图可有环，跨时间片转移仍是有向无环的条件概率结构。使用对称 Laplace 平滑估计二元条件概率表。预测负载是两种下一状态的训练条件均值按失效概率混合；缺失状态的条件均值使用该节点训练目标均值。DBN 的概率表不应用物理传播关系。

### 4.4 时序图传播

T-GNN 共享因果 TCN 先编码节点历史，再计算入邻居编码的均值。节点自身编码和入邻居均值拼接后经过 MLP 与 sigmoid，得到目标节点的边强度。按源节点出度归一得到有效边权，再应用与 CCRL 相同的节点关系和同步物理传播。该结构不使用 CCRL 的公式 17 初始化或 actor–critic。单步模式优化容量归一化 MSE；转移模式对闭环预测轨迹全部周期的该损失取均值并反向传播。

## 5. 统一动态接口

`weights(history)` 返回未施加源门控的 `[B,N,N]` 边权。`propagate_numpy(loads,weights,capacities=None)` 使用模型自身关系和上下文；容量 override 用于压力实验，不改变模型编码器的训练容量。`predict`、`rollout` 以及动态控制实验共用这一传播定义。

`rollout` 返回 `loads[B,steps+1,N]`（包含初始周期）及 `weights[B,steps,N,N]`。每周期权重由当前历史重新计算；静态模型的边权固定。模型预测值与新观测值在接口中有明确区分。

## 6. 检查点与精确续训

### 6.1 推理检查点

`save(path)` 保存版本 3 的模型配置和参数；`load(path)` 仅使用 `torch.load(...,weights_only=True)` 读取，严格校验参数名称和形状。版本 1、2 不自动接受，避免卷积索引、关系和训练定义混用。

### 6.2 训练检查点

`save_training(path)` / `load_training(path)` 额外保存：

- 三个 CCRL 优化器或 T-GNN 监督优化器的完整状态。
- 本地随机生成器状态及可审计的 Python、NumPy、CPU/CUDA 全局随机快照。
- 已完成预热轮次、训练轮次和完整学习曲线。
- 实际输入数组的 SHA-256 摘要、模型配置摘要、训练协议以及调用方的数据和配置指纹。

数组摘要覆盖历史、目标和全部未来轨迹的形状及 float32 字节内容；仅传入同名文件并不能绕过检查。续训要求数据、模型配置、批大小、预热总轮次、训练模式和外部指纹一致。预热已完成的轮次不会重复运行；`epochs` 表示本次追加的训练轮次，允许在不改变其他协议的情况下延长训练。`warmup_epochs` 应显式设置为固定的预热总轮次。

```python
model.fit(history, target, epochs=2, warmup_epochs=2, batch_size=64,
          training_mode="transition", future_loads=future_loads,
          checkpoint_path="private/checkpoints/train.pt",
          data_fingerprint=data_id, config_fingerprint=config_id)
resumed = CCRLModel.load_training("private/checkpoints/train.pt")
resumed.fit(history, target, epochs=2, warmup_epochs=2, batch_size=64,
            training_mode="transition", future_loads=future_loads,
            data_fingerprint=data_id, config_fingerprint=config_id)
```

检查点按完整轮次保存，先写同目录临时路径再原子替换。续训使用保存的本地生成器和优化器状态，调用方的全局随机状态不被覆盖。

`reset_training()` 清除续训状态并保留模型参数，可显式开始新的独立训练任务。该操作不用于绕过实验分折或数据指纹检查。

测试入口为 `python -m pytest -q`；测试记录见 [software_validation.json](../review/software_validation.json)。
