# 参考项目与许可来源

本表记录相关结构、接口及许可来源。

| 模块 | 参考项目及许可来源 | 相关结构或接口 |
|---|---|---|
| 因果膨胀卷积 | [locuslab/TCN](https://github.com/locuslab/TCN)，[MIT](https://github.com/locuslab/TCN/blob/master/LICENSE) | 左因果卷积与膨胀滞后 |
| Actor–Critic | [pytorch/examples](https://github.com/pytorch/examples)，[BSD-3-Clause](https://github.com/pytorch/examples/blob/main/LICENSE) | 策略与价值函数分工、得分函数梯度 |
| 连续策略框架 | [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)，[MIT](https://github.com/DLR-RM/stable-baselines3/blob/master/LICENSE) | 连续动作策略接口 |
| 级联失败场景 | [CNRC-NAJU 项目](https://github.com/CNRC-NAJU/Prediction-and-mitigation-of-nonlocal-cascading-failures-using-graph-neural-networks) | 节点失效、攻击与缓解实验 |
| DBN | [pgmpy](https://github.com/pgmpy/pgmpy)，[MIT](https://github.com/pgmpy/pgmpy/blob/dev/LICENSE) | 双时间片条件概率结构 |
| VAR | [statsmodels](https://github.com/statsmodels/statsmodels)，[BSD-3-Clause](https://github.com/statsmodels/statsmodels/blob/main/LICENSE.txt) | 多变量自回归 |
| 时序图模型 | [PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)，[MIT](https://github.com/benedekrozemberczki/pytorch_geometric_temporal/blob/master/LICENSE) | DCRNN／TGCN 时序图输入接口 |

运行依赖为 NumPy、PyTorch、NetworkX、Matplotlib、scikit-learn 等，版本见 `pyproject.toml`。可选 HanLP SRL 接口来源见 [TEXT_NETWORK.md](TEXT_NETWORK.md)。第三方源码与许可范围见 [THIRD_PARTY.md](../THIRD_PARTY.md)。
