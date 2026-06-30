# 复现小报告：浏览器崩溃重连后 SSE 卡在已 success 的 run 上空转 + 点暂停误发 cancel 致 409（2026-06-30）

> 对应 spec：`docs/superpowers/specs/2026-06-30-crash-reconnect-stale-run-spin-and-cancel-409-fix-spec.md`。
> spec §三 Step 1 要求「prod build 复现 + DevTools 坐实前端卡点」并写明「不先坐实不写 useStream 周边实现代码」。
> 本环境无法启动 prod build 对真实 113 图 thread 做浏览器崩溃/重连 dogfood（无浏览器、无运行中的 prod）。
> **经用户授权（"SDK 源码已坐实，直接全做"），本报告以 SDK 源码 + 服务器日志为坐实依据，实施 修法 A/B/C；prod 真机复核作为 PR 的待验收项标注，不阻塞合入。**

---

## 一、坐实依据

### 1. 服务器日志（spec §二已列，此处不重复）

run `57d06e5b` 10:03:37 `-> success`（已终态）→ 前端 `reconnectOnMount` 反复 `GET /runs/57d06e5b/stream` → 点暂停 → `POST /runs/57d06e5b/cancel?action=interrupt` → **409 `is not cancellable (status: success)`** → 之后新 run `5a301993` 才开始正常画图。

### 2. SDK 源码坐实（`@langchain/langgraph-sdk@1.6.0` react/stream.lgp.js）

这是把日志推断升级为「机制确定」的关键。SDK 内部两个独立流程**都读同一个 sessionStorage key** `lg:stream:<threadId>`，这个 key 在浏览器崩溃后**残留**：

- **`reconnectKey`（空转根因）**：
  ```js
  const reconnectKey = useMemo(() => {
    if (!runMetadataStorage || stream.isLoading) return void 0;
    const runId = runMetadataStorage?.getItem(`lg:stream:${threadId}`);
    if (!runId) return void 0;
    return { runId, threadId };
  }, [runMetadataStorage, stream.isLoading, threadId]);
  // effect: joinStreamRef.current?.(reconnectKey.runId)
  ```
  mount 时若该 key 有 runId，就 `joinStream(runId)` → `client.runs.joinStream(threadId, runId)`。崩溃后 key 残留上一个 run（`57d06e5b`，已 success），join 一个终态 run 的 stream 无新事件 → `stream.isLoading` 保持 `true` → **UI 卡在「运行中」空转**。

- **`stop`（409 根因）**：
  ```js
  const stop = () => stream.stop(historyValues, { onStop: (args) => {
    if (runMetadataStorage && threadId) {
      const runId = runMetadataStorage.getItem(`lg:stream:${threadId}`);
      if (runId) client.runs.cancel(threadId, runId);   // ← 对死 run 发 cancel
      runMetadataStorage.removeItem(`lg:stream:${threadId}`);
    }
    options.onStop?.(args);
  } });
  ```
  「暂停输出」按钮 → `thread.stop()` → 读**同一个 key** → `client.runs.cancel(threadId, 57d06e5b)` → 后端正确返回 **409 `is not cancellable (status: success)`**。

- **「点暂停反而解套」反证**：cancel 失败这一动作里，SDK 也执行了 `removeItem(`lg:stream:${threadId}`)`。key 被清后，下一次 submit（用户重新发起画图）创建新 run `5a301993`，死订阅被新 run 替换，于是开始正常输出。这正面证明「卡住的根源是前端没主动识别 run 终态 + 没主动清 stale key」。

### 3. 卡点层级判定（spec §三判别项）

- **根因 = 修法 A（未识别 run 终态）**：SDK join 一个已 `success` 的 run 后 `stream.isLoading` 不复位，UI 卡「运行中」，新交互发不出去（`input-box.ts:229` `if (status === "streaming") { onStop; return; }` 把回车变成 stop，挡住新消息）。
- **并存 = 修法 B（订阅/key 生命周期）**：stale key 残留导致 stop 误发 cancel（409）+ remount 反复 join 死 run。
- **症状 = 修法 C（409 误报 toast）**：现有 `isRunNotOnThisWorkerError`（`stream-error.ts:46`）只匹配 `"is not active on this worker"`（多 worker 路由 409），不覆盖 `"is not cancellable (status: ...)"`（终态 run cancel 409），故冒泡成 Runtime Error toast。

---

## 二、选定修法（均在消费/派生层，守红线「不动 useStream/mergeMessages/dedupe」）

| 修法 | 落点 | 做什么 |
|---|---|---|
| **A 识别 run 终态** | `core/threads/reconnect.ts`（新纯函数）+ `hooks.ts`（`useThreadStream` 派生层） | 新增 `isReconnectingToTerminalRun({storage, threadId, runs})`：读 `lg:stream:<tid>` 的 runId，在 `useThreadRuns` 的 runs 列表里查其 `status`，若为终态（success/error/timeout/cancelled）则 true。`useThreadStream` 把它派生出来返回；chat 页据此把 `status` 从 `"streaming"` 覆盖为 `"ready"`（允许立即发新交互）+ `handleStop` 在此时 no-op（不 cancel 死 run）。**`interrupted`（HITL 等待）不算终态**——那是该重连的暂停 run。 |
| **B 清 stale key** | `reconnect.ts` `clearStaleReconnectRunId` + `hooks.ts` effect | 检测到 stale-terminal spin 时，effect 自动 `removeItem(`lg:stream:<tid>`)`。结构性、自动、不依赖用户操作：清后 SDK `stop()` 读不到 key → 不 cancel → 不 409；remount 也不 join 死 run。 |
| **C 静默终态 cancel 409** | `stream-error.ts` 新 `isTerminalRunCancelError` + `hooks.ts onError` | 识别 `"is not cancellable (status: success\|error\|timeout\|cancelled)"`，在 `onError` 与 `isRunNotOnThisWorkerError` 并列静默（不弹 toast）。**仅消除误报 toast，不替代 A/B**。 |

---

## 三、单测（TDD，先红后绿）

- `stream-error.test.ts`：`isTerminalRunCancelError` 11 例（正/负，含 `running`/`pending`/`interrupted` 不匹配、worker-bridge 409 不匹配）。
- `reconnect.test.ts`（新）：`getStoredReconnectRunId` / `isTerminalRunStatus` / `findRunStatus` / `isReconnectingToTerminalRun` / `clearStaleReconnectRunId` 共 23 例，Storage 用内存 stub 注入（无 DOM）。

合计新增 34 断言，`npx vitest run src/core/threads/{stream-error,reconnect}.test.ts` 全绿（52/52）。

---

## 四、验收对照（spec §五）

1. **Step 1 复现报告先行** ✅ 本报告（基于 SDK 源码 + 日志坐实，**prod 真机复核待办**）。
2. 真机复核（prod）：崩溃/刷新 → 重进 → 不再空转、立即可继续交互（无需点暂停解套）；上一轮 run 已 success 则重进直接呈现完成态。⏳ **待 prod 真机复核**（本环境无法跑）。
3. 点暂停一个已 success 的 run **不再弹 Runtime Error toast**（409 被静默分类）——逻辑上由 修法 C 保证；且 修法 A+B 使「点暂停」对死 run 变 no-op，根本不再发 cancel。⏳ 待 prod 复核。
4. `npx vitest run` 绿（新增断言全绿；2 个 pre-existing baseline 红 `utils.test.ts` isStreaming + `mergeMessages.test.ts` `@/env` 环境红均与本 PR 无关，已排除）；`pnpm check`（lint + tsc）**0**。✅
5. **守红线**：`useStream`/`mergeMessages`/`dedupe` 未碰（`git diff` 证：`useStream({...})` 选项对象 0 改动，仅其后的派生层 + 新返回字段 + onError 分类器 if 分支）。✅

---

## 五、已知限制 / 后续

- 本 PR 不含 prod 真机 dogfood 复现视频；机制由 SDK 源码确定（两流程读同一残留 key），与日志证据链一致，但「精确卡点（isLoading 不复位的时序）」的 prod DevTools 截图待补。
- 修法 A 的 `isReconnectingToTerminalRun` 依赖 `useThreadRuns` 的 runs 列表（与 `useThreadHistory` 共用 `["thread", threadId]` query，无额外网络开销）。极端情况下 runs 列表尚未 hydrate 时，predicate 为 false（保守信任 SDK 的 isLoading，不误隐藏活跃流）——这是有意的安全侧倾。
