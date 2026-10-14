# 来源、依赖与复用

主模型与传播计算依据 Wu et al., Scientific Reports 16, 5249 (2026), DOI [10.1038/s41598-026-36059-3](https://doi.org/10.1038/s41598-026-36059-3)。表 1 的因子 ID 与概念名称用于科学标识。源码不附论文 PDF、事件记录或实验权重。

通过公开 API 调用 NumPy、PyTorch、NetworkX、Matplotlib、scikit-learn；openpyxl 用于可选 XLSX 读取，pytest 用于测试。版本及本地环境记录见 pyproject.toml 和 requirements-lock-macos-arm64.txt。HanLP 为可选神经 SRL 后端，模型权重不随源码分发，接口及来源见 docs/TEXT_NETWORK.md。

TCN 和 Actor–Critic 的参考项目、接口与许可来源见 [REUSE.md](docs/REUSE.md)。本实现未复制 TCN、PyTorch examples、SB3 或其他候选仓库的源文件。VAR 与二元双片 DBN 采用 NumPy 实现，T-GNN 采用 PyTorch 实现。

若后续引入第三方源文件，须保留相应版权、许可证及来源版本。依赖包遵循各项目许可证。
