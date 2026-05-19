# 细粒度 Tool 重构交接文档

> 日期：2026-04-17
> 上一份交接：`docs/handoffs/2026-04-17-session-end-handoff.md`

---

## 1. 本次改动摘要

把 `run_paradigm_analysis`（monolithic tool）拆成 5 个细粒度 tool：
- parse_trajectories
- compute_metrics
- run_statistics
- generate_charts
- assess_and_handoff

中间状态经 `/mnt/user-data/workspace/` 下的 pkl + JSON 文件传递，skill 指导 LLM 编排。

**根因**：`run_paradigm_analysis` 实现在 `ethoinsight/templates/tool.py:410` 但从未在 `config.yaml` 中注册，导致 code-executor 无限循环。

## 2. 修改的文件

| 文件 | 改动 |
|------|------|
| `packages/ethoinsight/ethoinsight/templates/tool.py` | 追加 5 个 @tool wrapper（~400 行）+ 辅助函数 |
| `packages/agent/config.yaml` | tools 区块注册 5 个新 tool |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` | tools 列表改为新 5 个；max_turns 8→12；prompt 精简 |
| `packages/agent/skills/custom/ethoinsight-analysis/SKILL.md` | 全量重写为 6 步编排指南 |
| `packages/agent/skills/custom/ethoinsight-analysis/references/` | 新增 tool-reference.md、error-recovery.md、quality-checks.md；删除旧 data-quality-checks.md |
| `packages/ethoinsight/tests/test_granular_tools.py` | 新增（含 E2E shoaling 测试） |
| `packages/agent/backend/tests/test_ethoinsight_analysis_skill.py` | 新增 |
| `CLAUDE.md` | 更新流水线章节 |
| `docs/specs/paradigm-analysis-tools-spec.md` | 重写 |

## 3. 保留不动的

- `run_paradigm_analysis_tool`（未注册到 config，作为内部 fallback）
- `get_analysis_template_tool`（继续用于未实现的范式）
- `data_analyst.py`、`report_writer.py`、`knowledge_assistant.py`、`lead_agent/prompt.py`
- ethoinsight 库核心函数（parse/metrics/statistics/charts/assess 全不动）

## 4. 验证

- 单元测试：1546 → 1551 通过（新增 5 个 skill 布线测试）
- ethoinsight 测试：114 通过（7 个新 granular_tools 测试中 7 passed）
- E2E：待 `make dev` + demo-data shoaling 测试验证

## 5. 下一步

- E2E 验证通过后提交（分 3 个 commit：feat(tools)、feat(skill)、docs）
- 继续 Phase 0 剩余（EPM/OFT 范式补全、429 重试 5s/15s/30s 策略）
