---
name: ethoinsight-code
description: >
  Behavioral data analysis execution guide for code-executor subagent.
  Orchestrates 5 granular tools (parse_trajectories → compute_metrics →
  run_statistics → generate_charts → assess_and_handoff) via intermediate
  files in /mnt/user-data/workspace/. Includes quality checks, method
  validation rules, and error recovery procedures.
version: 2.0.0
author: noldus-insight
---

# EthoInsight 数据分析执行指南

分析流水线由 5 个工具依次完成，中间产物保存在 `/mnt/user-data/workspace/`，
每步结束后读取对应的 `*_summary.json` 验证质量再进入下一步。

## 6 步工作流

### 步骤 1：解析轨迹数据

调用 `parse_trajectories`：
- `file_pattern`：从 lead agent 的任务描述中提取，例如 `/mnt/user-data/uploads/*.txt`

调用后读 `/mnt/user-data/workspace/parsed_summary.json` 检查：
- `n_files` ≥ 3 表示样本量充足
- `columns` 含 x_center_mm、y_center_mm、velocity_mm_s 等核心列
- `quality_warnings` 为空或可接受

出现 `status: failed` 时参考 `references/error-recovery.md` 的"解析失败"一节。

### 步骤 2：计算行为指标

调用 `compute_metrics`：
- `paradigm`：从任务描述推断（shoaling, open_field, epm 等）
- `groups`：JSON 字符串，例如 `'{"control":["Subject 1","Subject 2"],"treatment":["Subject 3","Subject 4","Subject 5"]}'`
- `metrics`：可选，逗号分隔；留空使用范式默认指标

调用后读 `/mnt/user-data/workspace/metrics_summary.json` 检查：
- `computed_metrics` 含核心指标
- 每组 `n ≥ 3`（`quality_warnings` 会自动标注 n<3 的问题）
- 无 `zero variance` 警告

### 步骤 3：组间统计检验

调用 `run_statistics`（无需额外参数，自动从 metrics.pkl 读取）：
- `alpha=0.05`, `correction="bonferroni"`（默认值即可）

调用后读 `/mnt/user-data/workspace/statistics.json` 检查：
- `comparisons` 覆盖每个指标
- `method_warnings` 为空；如有 n<5 + 参数检验的警告需记录，后续写入 handoff

### 步骤 4：生成图表

调用 `generate_charts`：
- `chart_types="box_plot"`（默认）；若用户指定 violin_plot/raincloud_plot 按需传入
- `include_trajectory=True` 生成轨迹图
- `include_timeseries=True` 生成 shoaling 特有的 IID 和 polarity 时序图

调用后读 `/mnt/user-data/workspace/charts.json` 确认 `chart_paths` 非空。

### 步骤 5：领域评估与 handoff

调用 `assess_and_handoff`：
- `paradigm`：与步骤 2 相同
- `groups`：与步骤 2 相同

调用后确认 `handoff_path` 存在，检查 `n_errors` 和 `errors` 预览。

### 步骤 6：返回结果

按 `templates/output-contract.md` 的格式返回给 lead agent，消息含：
- handoff JSON 路径
- 关键输出文件（metrics.csv、statistics.json、charts PNG）
- 质量警告摘要

## 范式不支持时的 Fallback

若推断的范式未被 `compute_metrics` 支持（返回 `status: failed` 且提示 "paradigm not supported"），
切换到 fallback 流程，详见 `references/fallback-workflow.md`：
先用 `get_analysis_template` 获取脚本，write_file 后 bash 执行。

## 质量检查与错误恢复

- 数据质量判断细节：`references/quality-checks.md`
- 每步失败时的排查步骤：`references/error-recovery.md`
- 工具参数快速参考：`references/tool-reference.md`
