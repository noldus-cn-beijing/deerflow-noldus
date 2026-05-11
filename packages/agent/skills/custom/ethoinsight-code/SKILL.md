---
name: ethoinsight-code
description: >
  EthoInsight 数据分析执行手册（给 code-executor subagent 用）。
  按 paradigm 渐进披露指标函数清单、胶水脚本范例、handoff JSON schema。
  Use when code-executor receives a paradigm-specific analysis task.
type: workflow
---

# EthoInsight 代码执行指南

## 工作模式

你（code-executor）拿到 lead 派遣的任务后:

1. **read** `references/by-paradigm/<paradigm>.md` — 看本范式可用的指标函数清单 + 调用范例 + handoff schema
2. **read** ethoinsight-charts skill（已在你的白名单） — 按数据特性选图
3. **write_file** 写胶水脚本 `analysis.py`（在 `${workspace_path}/` 下）— `import ethoinsight.metrics.<范式>` + 算指标 + 跑统计 + 出图 + 写 `handoff_code_executor.json`
4. **bash** `python ${workspace_path}/analysis.py`
5. 出错时按 `references/error-recovery.md` 处理；脚本崩溃 traceback 会自动回到你的 context，改代码重跑（最多 2 次）

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

胶水脚本 stdout 最后必须包含 `[gate_signals]` 块（已在每个范式 reference 的胶水脚本模板末尾给出代码）。这是 lead 做数据质量决策的依据，不输出会导致 lead 退化到回读 handoff 的兜底路径（仍能跑通但浪费 token）。

输出格式见 `templates/output-contract.md` 的 `[gate_signals]` 段。

由 Python 代码生成，不要靠模型自己加这个块——它一定会忘或写错。

## 反模式（永远禁止）

1. ❌ 跑 `parse_trajectories` / `compute_metrics` / `run_statistics` 等老 langchain 工具（已废弃）
2. ❌ 把整段范式分析包装成 1 个 `analyze_<paradigm>()` 函数（颗粒度错）
3. ❌ 现场实现指标算法（如 run-length encoding）— 工程师已预制在 `ethoinsight.metrics.<范式>`
4. ❌ 读 EthoVision raw txt 前几行确认列名 — 函数内部已固化列名识别（regex 匹配）
