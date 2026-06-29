# Step 0 实测报告：切回卡顿（in-flight 大消息）根因坐实

> spec：`docs/superpowers/specs/2026-06-29-fix-tab-switchback-jank.md` §Step 0
> 日期：2026-06-29
> 前置核实：#232（extraction-cache，`45cc5b84`）与 #223（挂载成本卡顿修复，`6d921a36`）均在 dev/HEAD（`10b674f8`）。
> 红线：dev build perf 不作数 → 本报告所有数字均 **prod build**（`make start`，next-server v16.1.7，已确认无 dev markers）。

## 结论（先读）

**spec 假设的根因（后台 tab 节流 + SSE 持续灌 state → 切回一次性 flush + in-flight 大消息整条重解析）在本可用 harness 下未能复现。** 两次 prod 复现（一次带、一次不带 `--disable-background-timer-throttling`）切回瞬间**零 longtask**（≤0ms），且冻结期间 in-flight 消息**不增长**（`frozen_grew_by` = -1800 / 0），与"后台期间 SSE 持续灌入"的前提矛盾。

**关键有效性天花板**：headless Chrome 的 `Page.setWebLifecycleState('frozen')` 会**冻结整页 JS 执行**（比真实后台 tab 的 `hidden` 更严），因此 SSE 的回调（`useStream` 在 hooks.ts 处理网络事件）也一并冻结 —— 这正是 `frozen_grew_by ≤ 0` 的原因。真实 OS 级切 tab（用户切到 VS Code）下，hidden tab **只暂停 rAF + 节流 timer，不暂停网络事件回调**，`messages` state 会持续增长，这才是 spec 机制成立的前提。**本 harness 无法忠实复现该不对称性**（headless 无真实窗口失焦）。

**因此按 spec 红线「实施前必须先 CDP 实测坐实」+「最终选择以 Step 0 trace 数据为准」：A/B/C 均不应在未坐实前实施。** 本会话不写实现代码。

## 复现方法

脚本：`scripts/repro/switchback-trace.cjs`（worktree 内）。流程：
1. 上传 EPM Xuhui-28（28 个 xlsx）→ 发分析请求 → 自动答 HITL（answers file，`on_unmatched: generic`，perf 复现非正确性）。
2. 等 in-flight 消息越过阈值（data-analyst 大段流式）。
3. **`cdp.send('Page.setWebLifecycleState', {state:'frozen'})`** 模拟切走（N 秒），测量冻结期 in-flight 增量。
4. **`{state:'active'}`** 模拟切回，重置 longtask buffer 后采 6s 窗口的 `PerformanceObserver('longtask')`。
5. 对照：navigate-away-and-back（#223 场景，挂载成本）。

## 数据（prod build）

| run | 冻结 flag | freeze 时 inflight | 冻结期增长 `frozen_grew_by` | 切回 longtask max / count | nav-switchback longtask max |
|---|---|---|---|---|---|
| 1 | `--disable-background-timer-throttling` | 15271 chars | **-1800** | **0ms / 0** | 447ms |
| 2 | （移除该 flag） | 4880 chars | **0** | **0ms / 0** | 50ms |

in-flight 峰值：run1 在冻结前已涨到 **20315 chars**（data-analyst 多段流式），属"大消息"场景。

### 读数

- **切回零 longtask（两次）**：`frozen → active` 切回 6s 窗口内主线程无 >50ms 阻塞。spec 假设的"切回瞬间长卡顿"未出现。
- **`frozen_grew_by ≤ 0`（两次）**：冻结期消息不增长 → SSE 回调被一并冻结 → **未复现 spec 根因①（后台积压）**。没有积压就没有切回 flush，根因①/③不成立。
- **nav-switchback 仍见 longtask（447ms / 50ms）**：这是 #223 已治的"一次性挂载成本"路径，与 in-flight 重解析无关，且量级与 2026-06-26 handoff 一致。

## 为什么 harness 复现不了（有效性天花板）

spec 机制依赖的**不对称性**：hidden tab 下 **rAF/重渲染暂停，但网络事件 JS 回调继续** → `messages` 持续变大 → 切回积压 flush。

- `Page.setWebLifecycleState('frozen')` ≠ 真实 `hidden`：它**整页冻结**（含 JS 执行），所以 SSE 回调也停 → `frozen_grew_by ≤ 0`。
- CDP 无 `hidden` 状态（只有 `frozen`/`active`）。Chromium issue #342919175：Chrome 125+ 后台 tab `visibilityState` 仍报 `visible`，进一步说明 headless 无真实"切走"语义。
- 真实复现需要 **OS 级窗口失焦**（headed 浏览器 + 真切到别的 app），本环境 headless 无窗口可失焦。
- hooks.ts 确无 `visibilitychange`/节流处理 → SSE 层在真实 hidden 下确会持续灌入（spec 前提**结构上成立**），只是本 harness 复现不出。

## spec 候选 A/B/C 在本结果下的处置

| 候选 | 针对 | 本结果 |
|---|---|---|
| A 冻结前缀+活跃尾部 | 根因②（in-flight 重解析） | 切回零 longtask → 现无可治的卡顿，**不应盲上** |
| B visibilitychange 感知 | 根因①+③ | 后台积压未复现 → 缺触发面，**不应盲上** |
| C 大消息降级渲染 | 兜底 | 同上，**不应盲上** |

守 spec §不做 + `feedback_perf_is_efficient_impl_not_visual_downgrade`（perf=高效实现，先量再改；未证根因前不盲改避免 over-engineering）+ `feedback-handoff-bug-claims-expire-check-head-before-execution`（不执行未坐实的 handoff 论断）。

## 给下一步的建议（三选一，需用户拍板）

1. **真实复现**（推荐）：在**有 GUI 的机器**上 headed Chrome，跑 dogfood 到 data-analyst 大流式，**用 OS 真切 tab**（切到别的窗口 N 秒再切回）录 CDP performance trace / DevTools Performance。这才可能坐实 spec 机制。本 headless 环境做不到。
2. **手工 dogfood**：你自己在浏览器跑一次大流式，切到 VS Code 再切回，主观确认"是否还卡"+ 若卡用 DevTools 录一段 trace 给我。这是最低成本的坐实路径。
3. **搁置**：若实测确实不卡（#223+#232 可能已顺手覆盖了该场景），则本 spec 标记 wontfix / 已被既有修复覆盖，归档。

## 产物

- `scripts/repro/switchback-trace.cjs`（worktree）— 复现脚本，含 HITL 驱动 + frozen/active + longtask 采集。
- `/tmp/switchback-trace-out/`（run1）、`/tmp/switchback-trace-out2/`（run2）— `switchback-trace.json` / `inflight-growth.json` / `run.log` / `clarifications.json`。

> 注：复现期间本地 dev 已切 prod（`make start`）。复现结束未切回 —— 如需恢复 dev，请 `make stop && make dev`。
