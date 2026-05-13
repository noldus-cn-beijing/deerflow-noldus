---
name: ethoinsight-code
description: >
  EthoInsight 数据分析执行手册（给 code-executor subagent 用）。
  按 catalog 指标范式执行脚本。code-executor 读 metric_plan.json 编排执行而不写胶水代码。
  Use when code-executor receives a paradigm-specific analysis task.
type: workflow
---

# EthoInsight 代码执行指南

## 工作流（catalog → plan → execute 架构）

code-executor 的工作流程：

1. **read** `${workspace_path}/metric_plan.json` —— lead 已生成的施工单
2. **for each entry in plan.metrics**：
   ```
   python -m <entry.script> --input <entry.input> --output <entry.output>
   ```
3. **if plan.statistics 非空且 skip_reason is null**：
   ```
   python -m <plan.statistics.script> --inputs ... --groups ... --output ...
   ```
4. **for each chart in plan.charts**：
   ```
   python -m <chart.script> --input ... --output ...
   ```
5. **聚合**：把所有 metrics[].output 的 JSON + charts 路径 + statistics 输出（如有）合并构造 handoff JSON
6. **write handoff**：`write_file ${workspace_path}/handoff_code_executor.json`
7. **输出 `[gate_signals]` 块**给 lead

### 重要约束

- 不要写胶水脚本拼接代码 —— 所有指标 + 统计 + 绘图都在脚本里
- 不要读 catalog YAML —— plan.json 已经把你需要的执行字段（script、input、output）展开
- bash 命令必须是脚本调用（`python -m ethoinsight.scripts.*`）或文件操作（mkdir / cp / mv / ls / cat / grep / head / tail）。其他形式的 bash（包括 `python -c`、`pip install`）会被运行时拦截
- 遇到脚本报错：读 stderr → 查 `references/error-recovery.md` → 决定重试 / 跳过 / 反问 lead
- 如果 plan.metrics[i].required is true 且脚本失败：必须停下来报 lead；required is false 时记 warning 继续

## 通用资源

- `references/error-recovery.md` — 常见错误诊断 + 重试策略
- `references/quality-checks.md` — 数据质量自检清单（NaN / 单位 / 列名缺失）
- `templates/output-contract.md` — handoff JSON 字段规范

## 最终消息约定（必读）

每个脚本 stdout 末尾打印 `[result] {...}` 行。你收集所有脚本的 [result] 行 + 聚合构造 handoff JSON 后，在最终消息中输出 `[gate_signals]` 块。

输出格式见 `templates/output-contract.md` 的 `[gate_signals]` 段。

`[gate_signals]` 块的 data_quality 等字段由你根据 handoff JSON 实际内容计算（不再靠胶水脚本自动生成），确保 lead 不读 handoff 也能做数据质量决策。
