# 源码审阅入口

软件版本 0.3.0，文档整理日期 2026-10-14。

| 内容 | 文件 |
|---|---|
| 使用流程与命令 | README.md |
| 来源、单位、窗口与阶段编码 | docs/DATA_CONTRACT.md、src/airport_risk/preparation.py |
| 文本、网络、容量及阻抗 | docs/TEXT_NETWORK.md、src/airport_risk/network.py |
| 数学和关系定义 | docs/FORMULA_MAP.md、docs/MODEL.md |
| 训练、续训和基线 | src/airport_risk/models.py、baselines.py、relations.py |
| 协议、指标和动态干预 | docs/EXPERIMENTS.md、experiments.py、dynamic.py |
| 默认参数与研究协议 | configs/default.json、configs/paper_protocols.json |

运行源码包不包含事件数据、审核标注、训练权重、生成图表或私有实测结果。机器可读验证记录为 [software_validation.json](review/software_validation.json)，其中的测试、发布及 CI 状态为记录日期当天的验证快照；配置与输入检查记录为 [preflight.json](review/preflight.json)。许可证状态见 [LICENSE-NOTICE.md](LICENSE-NOTICE.md)。
