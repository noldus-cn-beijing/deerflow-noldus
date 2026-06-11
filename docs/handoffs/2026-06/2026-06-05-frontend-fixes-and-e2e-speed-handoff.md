# Handoff — 2026-06-05 前端修复 + E2E 加速

> 给下一个 AI Agent 的上下文总结

## 当前任务目标

修复前端问题 + 加速 E2E 流水线（目标：从 21 min 降到 10-12 min）。

## 已完成的改动（已合入 dev）

### 前端修复（PR #91 已合 dev）

| 改动 | 文件 |
|------|------|
| 消息重复输出修复 | `packages/agent/frontend/src/core/messages/utils.ts` |
| 图表图片 404 修复 | `packages/agent/frontend/src/components/workspace/messages/markdown-content.tsx` |
| MarkdownContent 传 threadId | `packages/agent/frontend/src/components/workspace/messages/message-list.tsx` |
| Subagent 卡片状态滞后修复 | `packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx` |

### Subagent 模型切换（已合 dev，`b4388f69`）

code-executor、chart-maker、report-writer 从 `inherit` (deepseek-v4-pro) → `deepseek-v4-pro-summary` (deepseek-v4-flash, 无 thinking)。data-analyst 保持 deepseek-v4-pro+thinking。

### 前端信息架构（PR #92 已合 dev，`25a6e620`）

- Phase 1: `assistant:subagent` handler 加 `extractQualityWarnings` + `QualityWarningBanner`
- Phase 2a: `assistant:processing` handler 修复 tool-call-only 消息被丢弃
- Phase 2b: `ToolCall` 组件新增 inspect_uploaded_file / prep_metric_plan / set_experiment_paradigm 摘要渲染

## 进行中的改动（在分支上，未合入）

### `perf/e2e-pipeline-speed-optimization` — E2E 加速 4 项优化

**分支**: `perf/e2e-pipeline-speed-optimization`（`5013242f`），**需要提 PR 到 dev**

改动文件：
| # | 优化 | 文件 |
|---|------|------|
| 1 | n=1 快速路径 | `lead_agent/prompt.py` |
| 2 | 反问合并 | `lead_agent/prompt.py` |
| 3 | 并行 bash | `ethoinsight-code/SKILL.md` |
| 4 | batch 读文件 | `chart_maker.py`, `data_analyst.py`, `report_writer.py`, `ethoinsight-code/SKILL.md` |

**已修复**: batch-read 的 `/tmp/` 路径 → `/mnt/user-data/workspace/`（sandbox allowlist 不含 /tmp）；n=1 判定从 phantom `plan_metrics.json.groups` → `groups.json` + `subject_count` 交叉推算。

**待做**: 提 PR 到 dev → dogfood 验证耗时改进。

## 关键设计文档

| 文档 | 路径 | 状态 |
|------|------|------|
| 前端信息架构设计 | `docs/plans/2026-06-04-frontend-info-architecture-fixes.md` | ✅ 已合 dev，已由 opus review |
| E2E 耗时瓶颈分析 | `docs/plans/2026-06-04-e2e-latency-bottleneck-analysis.md` | ✅ 已合 dev |
| E2E 加速 spec | `docs/superpowers/specs/2026-06-04-e2e-pipeline-speed-optimization-spec.md` | ✅ 已合 dev |

## 关键发现

1. **QualityWarningBanner 全链路已实现**（`QualityWarningBroadcastMiddleware` → `extractQualityWarnings` → `QualityWarningBanner`），不是"待实施"。真正的 bug 是 `assistant:subagent` group 不渲染 banner（已在 PR #92 修复）。

2. **`/tmp/` 不在 sandbox path allowlist 中**（`sandbox/tools.py`），只有 `/mnt/user-data/`、`/mnt/skills/`、`/mnt/acp-workspace/` 在白名单。临时文件必须写到 `/mnt/user-data/workspace/`。

3. **`plan_metrics.json` 没有 `groups` 字段**——分组信息在单独的 `groups.json` 中。

4. **E2E ~21 min = ~80 LLM turns × ~13s/turn**。最大瓶颈是 LLM 推理（60%+），其次是串行 subagent 链。

5. **code-executor/chart-maker/report-writer 的 thinking 默认已关闭**（`SubagentConfig.thinking_enabled=False`）。

## 未完成事项

1. **提 PR**: `perf/e2e-pipeline-speed-optimization` → `dev`
2. **Dogfood 验证**: 跑一次 E2E 测实际加速效果
3. **data-analyst 文件数 ≤ 2 的 batch read 值得商榷**（spec 自己说 ≤2 没必要 batch）
4. **Phase 3 (subagent 实时进度 events)**: 需要后端 SSE 改动，未启动

## 建议接手路径

1. 读 `docs/milestone/README.md` 了解项目全局
2. 从 `perf/e2e-pipeline-speed-optimization` 提 PR 到 dev
3. 合入后用 OFT 旷场数据跑一次 E2E，记录耗时
4. 如果还需要继续加速，优先做：lead 反问合并效果验证 → code-executor 并行 bash 效果验证

## 风险与注意事项

- **测试**: 全量测试有 4 个预存在 test isolation 污染失败（与改动无关），不要归因于自己的改动
- **Sandbox 路径**: 临时文件只能用 `/mnt/user-data/workspace/`，不能用 `/tmp/`
- **字段来源**: 改 prompt 前先 grep 后端工具的真实返回字段，不要假设存在
- **DeerFlow sync 规则**: `lead_agent/prompt.py` 和 subagent `__init__.py` 是受保护文件，sync 上游时需 surgical merge
