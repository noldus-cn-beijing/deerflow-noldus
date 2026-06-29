# Spec: 移除顶层 7 阶段进度轨 + 右上「N 步进行中」运行轨迹徽章

> 状态：待实施（删除清单已完整核盘，无孤儿）
> 归属：前端 dogfood 修复批（2026-06-29）

## Context（为什么做）

用户 dogfood 反馈两条：
1. 右上「N 步进行中」徽章（`RunTraceWidget`）在**流程结束后仍显运行中并持续脉动**，点开抽屉信息模糊，价值低。
2. 顶层 7 阶段横向进度轨（`AnalysisRail`，「上传/范式识别/列对齐 已完成」那条）也被判定**没必要**（与对话流信息重复）。

用户决定：**两个都移除**。移除即一并消灭「状态卡死」bug（无需再修运行态判定），最干脆。

> 注：移除背后的真根因（trace 派生无「run 结束」信号、非终态状态卡死）见 plan `cosmic-growing-wigderson.md`——若将来想保留某一个并修，那条 `runActive` 归一方案在那里；但本 spec 按用户决定走「删除」。

## 删除清单（已核盘，删完 `pnpm check` 应 0 error）

### A. 整目录/文件删除
- `src/components/workspace/analysis-rail/`（analysis-rail.tsx、stage-node.tsx、index.ts）
- `src/components/workspace/trace/`（run-trace-widget/-drawer/-panel、trace-event-item、index.ts）
- `src/core/workflow/`（capability-plan、derive-workflow-stages、stages、use-workflow-stages、index + 各 .test.ts）—— 仅被 AnalysisRail 消费，全孤儿
- `src/core/trace/`（build-run-trace、use-run-trace、types、index + build-run-trace.test.ts）—— 仅被上述两特性消费，全孤儿

### B. 编辑（两路由 page.tsx 去挂载点 + import）
- `src/app/workspace/chats/[thread_id]/page.tsx`：删 `import AnalysisRail`、`import RunTraceWidget`、header 里 `<RunTraceWidget .../>`、**整个** sticky rail wrapper div（含 `<AnalysisRail/>`）及其上方解释注释。删后 `<main>` 从 header 直接接 MessageList wrapper。
- `src/app/workspace/agents/[agent_name]/chats/[thread_id]/page.tsx`：同样三处 + sticky wrapper。

### C. i18n 三文件同删（缺一 typecheck 红）
- `src/core/i18n/locales/types.ts`、`zh-CN.ts`、`en-US.ts` 的 `runTrace.*` 与 `workflowStages.*` 两整段。

### D. 测试删除
- `core/trace/build-run-trace.test.ts`、`core/workflow/{capability-plan,derive-workflow-stages,stages}.test.ts`。

## 必须保留（共享，仍被消息流消费——勿删）
- `src/core/tools/stage-broadcast.ts` —— `getStageBroadcastForBash`（message-group.tsx）、`getStageBroadcastForSubagent`（subtask-card.tsx）仍用。
- `src/core/messages/utils.ts` 的 `extractQualityWarnings`（message-list/-item）、`findToolCallResult`（message-group、artifact-file-detail）、`findToolCallArgs`（message-list）—— 全是消息流渲染依赖。

## 风险 / 注意
- 删 sticky wrapper 时**连带删干净**，别留空 sticky div（否则布局留白/层级残留）。
- 删后两个 page 的 MessageList 高度链（`flex-1 min-h-0`）不受影响（rail 本就是 sticky 兄弟，不在 flex 高度链里），但**实施后须本地 localhost:2026 目测**对话流顶部不塌、滚动正常。
- 全 grep 确认无其它消费者（`useRunTrace`/`useWorkflowStages`/`useCapabilityPlan`/`buildRunTrace`/`TraceEvent`）——调研已确认仅这两特性 + 自身测试用到。

## 验收
- `pnpm check` 0 error（typecheck 抓孤儿 import）；`npx vitest run` 绿（删测后总数下降、无新红）。
- 本地 localhost:2026：header 无 N 步徽章、顶部无进度轨、对话流布局正常、无 console error。
