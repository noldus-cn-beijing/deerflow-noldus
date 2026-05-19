# 输出契约

最终返回的消息必须包含：

- status (completed/failed)
- handoff 文件路径
- 输出文件列表（metrics CSV、statistics JSON、charts PNG）
- data_quality_warnings（如有）

## 脚本 stdout 最终输出契约

`handoff_code_executor.json` 写盘后，**脚本 stdout 必须按以下格式输出 `[gate_signals]` 块**：

```
OK: handoff written to <path>

[gate_signals]
data_quality:
  critical_count: <int>     # 等于 data_quality_warnings 中 severity=="critical" 的条目数
  warning_count: <int>      # severity=="warning" 的条目数
  critical_items:
    - <每条 <80 字的 critical 警告 message，最多 5 条>
    - ...
    （如无 critical 条目，输出"    (none)"占位行）
statistical_validity: ok | warning | failed   # failed = handoff.status=="failed"; warning = 有 critical; 否则 ok
errors_count: <int>         # handoff.errors 数组长度
```

**为什么必须输出**：lead 在 Step 1.5 读这个块的字段做拦截决策，无须 read_file handoff（节省上下文 5-30 KB）。块缺失时 lead 会回退到 read handoff 的兜底路径——能跑通但效率退化。

**确定性约束**：由 Python 代码直接生成，不依赖模型推理。每个范式 reference 的脚本模板已经包含完整的输出代码块，复制使用即可。
