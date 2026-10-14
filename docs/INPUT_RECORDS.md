# 事件表读取与审核

XLSX／CSV 导入保留文件 SHA-256、工作表、原行号、业务字段、叙述和来源日期。扫描前十行定位表头，重复表头与无事件身份的说明行单独处理；同名列通过映射保留。公式单元格保持公式文本，不执行。

```bash
python -m airport_risk analyze-records --data data/private/events.xlsx \
  --out data/private/record-analysis
python -m airport_risk annotate --data data/private/events.xlsx \
  --catalog configs/factors.json --airport source-airport-id \
  --out data/private/annotations.json
```

Excel 数字日期按工作簿日期系统解析，原值保留。日期与叙述年份冲突时记录提示，由审核者确定事件年份；不按文件名覆盖。原有加工致因文本单独保存，词面共现不直接成为因果边。

候选关系保留方向、否定、不确定性及语义帧，状态初始为 pending。审核者填写节点、因果阶段、年份和审核人，设置 reviewed 后进入[样本构建](DATA_CONTRACT.md)。单纯修改候选 review_status 不会自动生成物理逐时观测。

输入中的来源类别与测试用途标记随导入结果保留。全部导入产物写入私有路径。
