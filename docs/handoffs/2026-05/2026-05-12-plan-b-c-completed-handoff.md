# 2026-05-12 Plan B 完成 + Plan C 完成 交接

## 已完成（Plan B：6 范式脚本即指标）

按 EPM 模板补齐 OFT / Zero Maze / LDB / FST / TST / Shoaling 6 个范式的脚本即指标：

| 范式 | 脚本目录 | compute | plot | stats | 测试 |
|------|----------|---------|------|-------|------|
| OFT | `scripts/oft/` | center_time_ratio, thigmotaxis_index, center_distance_ratio, center_entry_count | plot_box_center | run_groupwise_stats | 8 tests |
| Zero Maze | `scripts/zero_maze/` | open_zone_time_ratio, open_zone_time, open_zone_distance, hesitation_count | plot_box_open_zone | run_groupwise_stats | 8 tests |
| LDB | `scripts/ldb/` | light_time_ratio, transition_count, light_latency | plot_box_light | run_groupwise_stats | 10 tests |
| FST | `scripts/fst/` | immobility_time, immobility_latency, immobility_bout_count | plot_box_immobility | run_groupwise_stats | 7 tests |
| TST | `scripts/tst/` | immobility_time, immobility_latency, immobility_bout_count | plot_box_immobility | run_groupwise_stats | 7 tests |
| Shoaling | `scripts/shoaling/` | inter_individual_distance, nearest_neighbor_distance, group_polarity | plot_box_iid | run_groupwise_stats | 10 tests |

**总计**：32 个新脚本 + 6 个测试文件。73 script tests 全绿（含已有 EPM 9 tests）。ethoinsight metrics 层 153 passed / 7 skipped，无回归。

### Bug 修复

- `ethoinsight/metrics/fst.py`：`DEFAULT_MOBILITY_COL` 从 `"Mobility_State"` 改为 `"mobility_state"`（与 `normalize_columns` 小写输出一致）
- `ethoinsight/metrics/tst.py`：`DEFAULT_MOBILITY_COL` 从 `"Activity_State"` 改为 `"activity_state"`
- 同步更新 `tests/test_metrics_fst.py` 和 `tests/test_metrics_tst.py` 的列名引用

## 已完成（Plan C：前端 reasoning 重复修复）

- `packages/agent/frontend/src/core/messages/utils.ts`：`groupMessages` 改为互斥 if-else 链，确保 reasoning+content+no-tool_calls 只进入 `assistant` group
- `packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx`：删除 `!rawContent` 特殊分支，统一渲染路径（Reasoning 在 MarkdownContent 上方）
- 新增 `vitest` 依赖 + `utils.test.ts`（6 个分组规则用例）+ `vitest.config.ts`

## 已完成（by-paradigm 文档改写）

6 个范式文档从旧"胶水脚本范例"格式改写为"脚本清单+决策手册"格式（按 EPM `epm.md` 模板）：

- `oft.md` — 4 compute + 通用 + plot_box_center + stats，paradigm key: `open_field`
- `zero-maze.md` — 4 compute + 通用 + plot_box_open_zone + stats
- `ldb.md` — 3 compute + 通用 + plot_box_light + stats，paradigm key: `light_dark_box`
- `fst.md` — 3 compute + 通用 + plot_box_immobility + stats，paradigm key: `forced_swim`
- `tst.md` — 3 compute + 通用 + plot_box_immobility + stats，paradigm key: `tail_suspension`
- `shoaling.md` — 新建，3 compute（多 subject `--inputs`）+ 通用 + plot_box_iid + stats

每个 doc 含：脚本清单表、输入格式约定、实验设计决策树（n=1 / n 3-4 / n≥5）、handoff JSON schema、data_quality_warnings 触发条件、错误处理表、编排流程。

## 未完成事项

### 🔴 高优先级 — 等行为学同事提供真数据后

1. **列名 regex 调校** — 每个范式的 `metrics/<范式>.py` 中 auto-detect regex 需要根据真数据列名调整（接续 `docs/handoffs/2026-05-11-sota-migration-completed-real-data-pending-handoff.md`）
2. **脚本 e2e 验证** — 用真数据跑每个范式的脚本，确认输出合理

### 🟡 中优先级 — files-are-facts e2e dogfooding

3. `docs/handoffs/2026-05/2026-05-11-handoff.md` §5 的三个用例（A 正态流程 / B critical 拦截 / C Guardrail 拦截）待 `make dev` 实测

### 🟢 低优先级

4. `code-executor` SKILL.md 和 system_prompt 确认已切换到脚本编排工作流（Plan A T11-T12 已做，本次未改）

## 关键文件速查

| 用途 | 路径 |
|------|------|
| 脚本目录 | `packages/ethoinsight/ethoinsight/scripts/{epm,oft,zero_maze,ldb,fst,tst,shoaling}/` |
| 脚本 CLI helper | `packages/ethoinsight/ethoinsight/scripts/_cli.py` |
| 脚本测试 | `packages/ethoinsight/tests/scripts/test_{epm,oft,zero_maze,ldb,fst,tst,shoaling}_scripts.py` |
| 测试 fixtures | `packages/ethoinsight/tests/scripts/conftest.py` |
| by-paradigm 文档 | `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/*.md` |
| 前端消息分组 | `packages/agent/frontend/src/core/messages/utils.ts` |
| 前端消息渲染 | `packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx` |
| 前端测试 | `packages/agent/frontend/src/core/messages/utils.test.ts` |
| Spec | `docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md` |
| Plan A (EPM) | `docs/superpowers/plans/2026-05-12-plan-a-script-per-metric-epm.md` |

## 下一位 Agent 的第一步建议

1. 等行为学同事提供真 EthoVision 数据后，按 `docs/handoffs/2026-05-11-sota-migration-completed-real-data-pending-handoff.md` 的 Step 1-6 做列名诊断和 regex 调校
2. 列名调完后，用真数据跑 `cd packages/ethoinsight && uv run pytest tests/scripts/ -v` 验证脚本
3. 启动 `make dev`，做 files-are-facts e2e dogfooding（用例 A/B/C）
4. Plan B/C 全部验证通过后，考虑是否 squash merge 到 main
