# 范式分析工具规格

## 架构概述

code-executor subagent 通过 5 个细粒度工具完成一个范式的分析：

1. `parse_trajectories` — 解析 EthoVision 轨迹文件（自动 UTF-16）
2. `compute_metrics` — 计算范式特定指标
3. `run_statistics` — 组间统计检验（自动选参数/非参数）
4. `generate_charts` — 生成图表（box/violin/raincloud/trajectory/timeseries）
5. `assess_and_handoff` — 领域评估 + 写 handoff JSON

工具间通过 `/mnt/user-data/workspace/` 下的文件传递中间状态：

| 文件 | 由谁写 | 由谁读 |
|------|--------|--------|
| `parsed.pkl` | parse_trajectories | compute_metrics, generate_charts |
| `parsed_summary.json` | parse_trajectories | LLM (read_file), assess_and_handoff |
| `metrics.pkl` | compute_metrics | run_statistics, generate_charts, assess_and_handoff |
| `metrics_summary.json` | compute_metrics | LLM, assess_and_handoff |
| `statistics.json` | run_statistics | generate_charts, assess_and_handoff |
| `charts.json` | generate_charts | assess_and_handoff |
| `handoff_code_executor.json` | assess_and_handoff | lead agent → data-analyst |

详细工具 API 见 `packages/agent/skills/custom/ethoinsight-code/references/tool-reference.md`。

## 兼容性约束

- handoff JSON schema 必须与旧 `run_paradigm_analysis_tool` 输出兼容
- `run_paradigm_analysis_tool` 保留在 `ethoinsight/templates/tool.py` 中（未注册到 config.yaml），仅作为内部 API

## Fallback

范式未实现时走 `get_analysis_template` + write_file + bash 的旧路径，
详见 skill 的 `references/fallback-workflow.md`。
