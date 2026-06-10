---
name: ethoinsight-code
description: >
  EthoInsight 数据分析执行手册（给 code-executor subagent 用）。
  按 catalog 指标范式执行脚本。code-executor 读 plan_metrics.json 编排执行而不写胶水代码。
  Use when code-executor receives a paradigm-specific analysis task.
type: workflow
---

# EthoInsight 代码执行指南

> **本 skill 服务于 code-executor，只跑 metrics + stats。图表归 chart-maker（W21）。**
> 执行约束见根 skill references/execution-conventions.md（W7）。

## 工作流（catalog → plan → execute 架构）

code-executor 的工作流程：

1. **读一次** `${workspace_path}/plan_metrics.json`，然后立即开始跑第一个脚本（不要反复 read 同一个文件）
2. **for each entry in plan.metrics**：
   ```
   python -m <entry.script> --input <entry.input> --output <entry.output>
   ```
   **同指标不同 subject 用一条 bash 并行**：`bash -c "python -m ... & python -m ... & wait; echo ALL_DONE_<id>"`
3. **if plan.statistics 非空且 skip_reason is null**：
   ```
   python -m <plan.statistics.script> --inputs ... --groups ... --output ...
   ```
4. **聚合**：把所有 metrics[].output 的 JSON + statistics 输出（如有）合并构造 handoff 数据结构
5. **封存 handoff**：调 system_prompt 指定的 `seal_code_executor_handoff` 工具，将聚合好的数据落库为 `${workspace_path}/handoff_code_executor.json`。**本 skill 不描述 handoff 文件写法**——封存字段结构以 system_prompt + 工具 schema 为权威（SSOT 原则），工具返回 OK 即任务完成。

### 重要约束

- **每个文件最多读一次**：plan_metrics.json 只在开头读一次，不要在执行中途反复 read 确认；每个 metric 的 output JSON 只在聚合时读一次
- 不要写胶水脚本拼接代码 —— 所有指标 + 统计都在脚本里
- 不要读 catalog YAML —— plan.json 已经把你需要的执行字段（script、input、output）展开
- bash 命令必须是脚本调用（`python -m ethoinsight.scripts.*`）或文件操作（mkdir / cp / mv / ls / cat / grep / head / tail）。其他形式的 bash（包括 `python -c`、`pip install`）会被运行时拦截
- 遇到脚本报错：读 stderr → 查 `references/error-recovery.md` → 决定重试 / 跳过 / 反问 lead
- 如果 plan.metrics[i].required is true 且脚本失败：必须停下来报 lead；required is false 时记 warning 继续
- **handoff 中 `inputs.raw_files` 必须是虚拟路径**（`/mnt/user-data/uploads/xxx.txt`），从 `plan_metrics.json.inputs.raw_files` **原样抄过来**，**不要** `Path(...).resolve()` / `realpath` 把它转成宿主机绝对路径。chart-maker 下游会把这些路径透传给 `catalog.resolve --raw-files-json`，宿主路径会被 sandbox guardrail 拦掉整轮跑不动。

## 通用资源

- `references/error-recovery.md` — 常见错误诊断 + 重试策略
- `references/quality-checks.md` — 数据质量自检清单（NaN / 单位 / 列名缺失）
- `templates/output-contract.md` — handoff JSON 字段规范

## 最终消息约定（必读）

每个脚本 stdout 末尾打印 `[result] {...}` 行。你收集所有脚本的 [result] 行 + 聚合构造 handoff 数据结构后，调 `seal_code_executor_handoff` 工具落库。**只有 seal 工具返回 OK 才算任务完成。**

封存完成后，在最终消息中输出 `[gate_signals]` 块（中间产物，用于 lead 快速决策，非完成信号）。

## 并行执行规则（E2E 加速）

当多个脚本满足以下全部条件时，用一个 bash 调用并行执行：

1. 脚本之间互不依赖（不同 subject / 不同指标）
2. 输出文件不同
3. 输入文件不同

示例（OFT 旷场，2 subjects × 5 指标 = 10 个独立脚本）：

```bash
bash -c "
python -m ethoinsight.scripts.oft.compute_center_distance --input s1.xlsx --output cdist_s1.json --parameters-json '{\"center_zone\":\"in_zone\"}' &
python -m ethoinsight.scripts.oft.compute_center_distance --input s2.xlsx --output cdist_s2.json --parameters-json '{\"center_zone\":\"in_zone\"}' &
python -m ethoinsight.scripts.oft.compute_distance_ratio --input s1.xlsx --output dratio_s1.json --parameters-json '{\"center_zone\":\"in_zone\"}' &
python -m ethoinsight.scripts.oft.compute_distance_ratio --input s2.xlsx --output dratio_s2.json --parameters-json '{\"center_zone\":\"in_zone\"}' &
wait
"
```

**效果**: 10 个独立 turns → 1 个 turn。

**注意**：
- 仅在脚本间无依赖时并行；如果某脚本依赖另一脚本的输出，必须串行
- 使用 `&` 后台运行 + `wait` 等待全部完成后才继续
- 如果有脚本失败，`wait` 后检查每个输出 JSON 是否存在来排查，将失败条目记 errors
- 并行脚本的输出不互相干扰（每个输出到不同文件）
- 如果 script 名或 args 来自 plan_metrics.json 的不同 entry，逐条照抄 plan 中的 script + args 即可

## Batch read 规则（E2E 加速）

如果你需要读多个文件（这些文件之间互不依赖），一次性 cat 它们：

```bash
bash cat /mnt/user-data/workspace/plan_metrics.json \
         /mnt/user-data/workspace/experiment-context.json \
         > /mnt/user-data/workspace/code_context_bundle.txt
```

然后 read_file /mnt/user-data/workspace/code_context_bundle.txt 一次拿到全部上下文。

**注意**：
- 不要 batch 读超大文件（如原始轨迹 txt，5MB+）—— 这些保持单独 read_file
- 只 batch 读 JSON 配置文件、handoff JSON 等小文件
- 如果文件数 ≤ 2，直接读即可，batch 优势不大
