# 输出契约

最终返回的消息必须包含：

- status (completed/failed)
- handoff 文件路径
- 输出文件列表（metrics CSV、statistics JSON、charts PNG）
- data_quality_warnings（如有）

## handoff `inputs.raw_files` 路径约定（必读）

`handoff_code_executor.json` 的 `inputs.raw_files` 字段**必须**是虚拟路径（`/mnt/user-data/uploads/xxx.txt`），**从 `plan_metrics.json.inputs.raw_files` 字段原样抄过来**。

❌ 禁止：`Path(p).resolve()`、`os.path.realpath(p)`、把虚拟路径换成宿主机绝对路径（`/home/.../user-data/uploads/...`）。

✅ 做法：直接 read plan_metrics.json → 取 `inputs.raw_files` → 原样写进 handoff。

**为什么强制**：下游 chart-maker 读 handoff 拿 raw_files 后透传给 `catalog.resolve --raw-files-json`；resolve 会把这些路径原样写进 plan_charts.json 的 `input` 字段；plotting 脚本被 sandbox guardrail 校验调用参数中的路径——宿主路径会触发 `local.host_path_blocked` 拒绝，整轮跑不动。

## 脚本 stdout 最终输出契约

**完成信号**：只有调 `seal_code_executor_handoff` 工具并返回 OK 才算任务完成。`[gate_signals]` 块是中间产物（供 lead 快速决策），不是完成信号。

handoff 落库后，在最终消息中输出 `[gate_signals]` 块：

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
