# 修复 spec：浏览器崩溃重连后，前端 SSE 卡在已 success 的 run 上空转 / 点暂停才解套 + 误发 cancel 致 409（2026-06-30）

> dogfood 现场新发现（2026-06-30，thread `993e8b83-3d25-405a-a923-3086eac58fe3`，EPM 113 图重负载）。**本 spec 只写不实施，交别的 agent。** 前端 SSE / run 状态机是红线敏感区（见原切回卡顿 spec 的「不动 useStream/mergeMessages/dedupe」），故本 spec **Step 1 要求先在 prod build 复现坐实，再动手**——日志已给出强证据链，但具体修法须以复现为准。

---

## 一、现象（用户实测）

1. 浏览器在重负载页面（113 图 + 长对话）**崩溃**。
2. 重进该 thread，选「是，画图」。
3. **thread 一直没有输出**（卡住、像 hang）。
4. 用户**点「暂停输出」** → 弹 Runtime Error：
   ```
   HTTP 409: {"detail":"Run 57d06e5b-0492-4ca7-b32b-8f2cc40fe987 is not cancellable (status: success)"}
   ```
5. **神奇之处**：点完暂停后，后续 lead agent **反而开始正常输出画图**了。

---

## 二、根因链（服务器日志坐实的部分 + 待复现确认的部分）

### 已坐实（来自 `logs/gateway.log`，2026-06-30 时间线）

| 时间 | 事件 |
|---|---|
| 10:00:23 | run `57d06e5b` 创建（这是**上一轮分析 run**：data-analyst + ask_clarification） |
| 10:03:37 | run `57d06e5b` **`-> success`**（已终态完成） |
| （崩溃，用户重进） | 前端 `reconnectOnMount` 反复 `GET /runs/57d06e5b/stream`（已 success，无新事件可推） |
| 10:??:?? | 用户点暂停 → `POST /runs/57d06e5b/cancel?action=interrupt` → **409 Conflict（status: success 不可取消）** |
| 10:10:44 | 新 run `5a301993` 创建 |
| 10:11:38 | `prep_chart_plan success: charts=113` → 真正开始画图 |

### 根因判断

1. **崩溃重连后，前端 `reconnectOnMount: true`（`hooks.ts:421`）重新订阅了最近的 run `57d06e5b`，但该 run 已 `success`**。前端 join 上去后没有新 SSE 事件，**且未识别「该 run 已终态」→ 未复位 `isLoading` → UI 卡在「运行中」空转**，新交互（画图）发不出去。
2. **点「暂停」误发 cancel 到一个 success run** → 后端正确返回 409 `is not cancellable (status: success)`。这个 409 **现有的 `isRunNotOnThisWorkerError`（`stream-error.ts:46`）匹配不到**——它只匹配 `"is not active on this worker"`（多 worker bridge 路由 409），不覆盖 `"is not cancellable (status: ...)"` 这种**终态 run 的 cancel 409**，于是冒泡成 Runtime Error toast。
3. **点暂停意外解套**：cancel 失败/中止订阅这个动作，让前端放弃了对死 run 的订阅、复位了状态，新画图请求才得以发出（`5a301993`）。这反证了「卡住的根源是前端没主动识别 run 终态」。

> ⚠️ **待 Step 1 在 prod 复现确认**：上面 1、3 是从日志 + `reconnectOnMount` 行为推断的，前端 run-state 状态机的精确卡点（是 `useStream` join 后不发终态、还是 `thread.isLoading` 不复位、还是订阅未清理）需要 prod 复现 + 前端断点/console 坐实，**不可凭日志直接改 useStream 核心**。

---

## 三、Step 1（不可跳）：prod build 复现 + 坐实前端卡点

1. prod build 起（`make start`，dev build 不作数——dev 重负载页面本身就崩，污染复现）。
2. dogfood 重负载页面（thread `993e8b83` 或等价 113 图场景）→ 制造崩溃/强制刷新 → 重进 thread。
3. 观察：是否复现「无输出空转」。开 DevTools Network 看是否反复 `GET /runs/<已success的run>/stream`；前端加 console 打 `thread.isLoading` + 当前订阅的 `runId` + 该 run 的 `status`。
4. **判别**：
   - 若前端 join 一个已 `success` 的 run 后 `isLoading` 仍 `true` 不复位 → 根因 = **未识别 run 终态**（修法 A）。
   - 若是订阅未清理导致 SSE 干等 → 根因 = **订阅生命周期**（修法 B）。
   - 二者可能并存。

**产出**：一份复现小报告（落 `docs/superpowers/reports/`），明确「重连后卡点在哪一层 + isLoading/订阅的实际状态」。**不先坐实不写 useStream 周边的实现代码。**

---

## 四、Step 2：按复现选修法（候选，复现定生死）

### 修法 A（若根因=未识别 run 终态）
重连后若 join 的目标 run 已是终态（`success`/`error`/`cancelled`），**不进入「运行中」UI 态**：复位 `isLoading`、不显示流式指示器、允许立即发起新交互。
- 优先在**渲染消费层 / run-state 派生层**解决，**不动 `useStream`/`mergeMessages`/dedupe 核心**（红线）。
- 可能落点：`hooks.ts` 里基于 `runs` 列表 + 当前 run `status` 派生 `isLoading` 的地方（`findLatestUnloadedRunIndex` 附近、`handleStreamStart`/事件回调），加「目标 run 已终态 → 不置 loading」短路。

### 修法 B（若根因=订阅生命周期）
重连订阅一个已终态 run 时，主动结束订阅、不挂在 `/stream` 上干等。

### 修法 C（必做，正交小修）：扩 409 分类，不弹 toast
`stream-error.ts` 增加一个判定（或扩 `isRunNotOnThisWorkerError` 为更通用的「可静默的 run-state 409」），识别 `"is not cancellable"` / `status: success|error|cancelled` 这类**对一个已终态 run 操作的 409**，**抑制 Runtime Error toast**（终态 run 不可 cancel 不是用户可操作的失败）。
- 有现成单测基建 `stream-error.test.ts`，加断言。
- **注意**：这只是消除误报 toast，**不替代修法 A/B**——根因是空转，409 只是症状。

---

## 五、验收（prod build）

1. **Step 1 复现报告**先行：明确卡点 + 选定修法。
2. 真机复核（prod）：重负载页面崩溃/刷新 → 重进 thread → **不再空转，立即可继续交互**（无需点暂停解套）；若上一轮 run 已 success，重进直接呈现完成态。
3. 点暂停一个已 success 的 run（或重连触发）**不再弹 Runtime Error toast**（409 被静默分类）。
4. `npx vitest run` 绿（含 `stream-error.test.ts` 新断言 + 任何 run-state 派生的新单测）；`pnpm check` 0。
5. **守红线**：`useStream`/`mergeMessages`/dedupe 未碰（git diff 证）。

---

## 六、不做什么

- ❌ 不在 Step 1 prod 复现坐实前，写 `useStream` 周边任何实现代码。
- ❌ 不动流式核心（useStream/mergeMessages/dedupe）。
- ❌ 不改后端 cancel 语义——后端对 success run 返回 409 是**正确**行为，修的是前端别误发 cancel + 别空转。
- ❌ 不把「点暂停才能继续」当成可接受的 workaround 留着。

---

## 七、关联

- 现有 409 处理：`stream-error.ts:isRunNotOnThisWorkerError`（只覆盖 worker-bridge 409，本 spec 扩它或新增同类判定）。
- 现有重连基建：`hooks.ts:reconnectOnMount`、spec `2026-06-26-rejoin-thread-history-merge-disorder-fix`（重连消息合并顺序，与本 spec 的 run-state 卡点正交，别混改）。
- 同期 dogfood 新 bug：`2026-06-30-clarification-awaiting-streaming-dots-fix-spec.md`（dots 指示器），与本 spec 都属「`isLoading` 状态派生不准」家族，但落点不同——可一并实施但分别验收。
