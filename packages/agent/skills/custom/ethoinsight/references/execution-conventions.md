# EthoInsight 执行约束 (执行类 subagent 必读)

服务对象:code-executor / chart-maker 等"执行类"subagent。

## bash 调用形式

只允许两种形式:

1. **脚本调用**:`python -m ethoinsight.scripts.<paradigm | _common>.<name> --input ... --output ...`
2. **文件操作**:`mkdir / cp / mv / ls / cat / grep / head / tail` (常规 POSIX,不含 `python -c` / `pip install` / `bash -c '...'`)

CLI 例外(可调,但只限本进程):
- `python -m ethoinsight.catalog.resolve --mode metrics ...` — 由 `prep_metric_plan` 工具内部调,subagent 不直接用此 mode
- `python -m ethoinsight.catalog.resolve --mode charts ...` — chart-maker 自跑

其他形式会被 `ScriptInvocationOnlyProvider` 拦截。

## handoff JSON 写入规则

- 文件名严格:`handoff_<subagent_name>.json`(下划线,与 SubagentConfig.name 中的连字符替换后一致)
- 路径:`/mnt/user-data/workspace/handoff_<name>.json`
- 编码:UTF-8,`ensure_ascii=False`,2-space indent
- schema:见各 subagent 自己 skill 的 `templates/output-contract.md` (本文件不重复 schema)

## error recovery

- 脚本 stderr 非空 → 读 traceback → 决定是否重试
- 同一脚本(同一 metric_id / chart_id)最多重试 **2 次**;再失败则把 error 写入 handoff.errors[] 并继续后续步骤
- 不要"探索式地" `ls` skill 目录或 `--help`;该跑哪些脚本由 plan_metrics.json / plan_charts.json 决定

## gate_signals 块通用格式

执行类 subagent 完工后必须在最终 AIMessage 末尾输出:

```
[gate_signals]
constitution_acknowledged: true
<其他字段...>
errors_count: <int>
```

具体字段由各 subagent 自己 contract 决定。lead 用块的存在性判定走 gate_signals 路径。
