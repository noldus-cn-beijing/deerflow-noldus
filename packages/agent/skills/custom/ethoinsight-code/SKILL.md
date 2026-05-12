---
name: ethoinsight-code
description: >
  EthoInsight 数据分析执行手册（给 code-executor subagent 用）。
  按 paradigm 渐进披露脚本清单与决策手册。code-executor 选脚本编排而不写胶水代码。
  Use when code-executor receives a paradigm-specific analysis task.
type: workflow
---

# EthoInsight 代码执行指南

## 工作流（脚本即指标架构）

code-executor 的工作流程：

1. **read** `references/by-paradigm/<paradigm>.md` —— 看可用脚本清单 + 实验设计决策树
2. **裁剪**：根据 lead 给的实验信息（范式、n、分组、用户特殊需求），决定要跑哪些脚本
3. **准备输入**（如需多文件聚合）：`write_file` 生成 `inputs.json` 和 `groups.json`
4. **bash 循环调脚本**：每个脚本一次 bash 调用，形如：
   ```
   python -m ethoinsight.scripts.<paradigm>.<script_name> --input ... --output ...
   ```
5. **收集**：脚本输出 JSON / PNG，stdout 含 `[result] {...}` 行
6. **聚合**：构造 handoff JSON
7. **写 handoff**：`write_file` 到 `${workspace_path}/handoff_code_executor.json`
8. **输出 [gate_signals]** 块给 lead

### 重要约束

- 不要写胶水脚本拼接代码 —— 所有指标计算已经在脚本里，subagent 只是编排者
- bash 命令必须是脚本调用（`python -m ethoinsight.scripts.*`）或文件操作（mkdir / cp / mv / ls / cat / grep / head / tail）。其他形式的 bash（包括 `python -c`、`pip install`、运行自定义脚本）会被运行时拦截
- 遇到脚本报错：读 stderr → 查对应范式 md 的「错误处理」段 → 决定重试 / 跳过 / 反问 lead

## Reference Materials

- `references/by-paradigm/<paradigm>.md` — 每个范式的脚本清单 + 决策手册
- `templates/output-contract.md` — handoff JSON schema 详细约定
- `references/error-recovery.md` — 通用错误恢复指引
- `references/quality-checks.md` — handoff 写入前的自检清单

## 范式渐进披露入口

- **EPM** (高架十字迷宫): `references/by-paradigm/epm.md`
- **OFT** (Open Field): `references/by-paradigm/oft.md`
- **Shoaling** (群体游动): `references/by-paradigm/shoaling.md` *(Phase 2 时撰写)*
- **Zero Maze**: `references/by-paradigm/zero-maze.md`
- **LDB** (Light-Dark Box): `references/by-paradigm/ldb.md`
- **TST** (Tail Suspension): `references/by-paradigm/tst.md`
- **FST** (Forced Swim): `references/by-paradigm/fst.md`

## 通用资源

- `references/error-recovery.md` — 常见错误诊断 + 重试策略
- `references/quality-checks.md` — 数据质量自检清单（NaN / 单位 / 列名缺失）
- `templates/output-contract.md` — handoff JSON 字段规范

## 最终消息约定（必读）

每个脚本 stdout 末尾打印 `[result] {...}` 行。你收集所有脚本的 [result] 行 + 聚合构造 handoff JSON 后，在最终消息中输出 `[gate_signals]` 块。

输出格式见 `templates/output-contract.md` 的 `[gate_signals]` 段。

`[gate_signals]` 块的 data_quality 等字段由你根据 handoff JSON 实际内容计算（不再靠胶水脚本自动生成），确保 lead 不读 handoff 也能做数据质量决策。

