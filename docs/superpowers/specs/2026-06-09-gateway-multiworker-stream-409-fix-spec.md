# Spec ② — 修 Gateway 多 worker 切 thread 再切回报 HTTP 409

> 日期：2026-06-09 ｜ 作者：实施前 spec（待 review，**不直接实施**）
> 主仓库 dev HEAD：`1a819e7e` ｜ 配套：spec ①（`2026-06-09-deerflow-sync-21-commits-spec.md`）正交、独立 worktree/PR
> 来源 handoff：`docs/handoffs/2026-06/2026-06-09-two-specs-sync-and-409-fix-handoff.md`
> 架构前提：**Gateway-embedded 多 worker**（无独立 langgraph 容器，跟随上游；CLAUDE.md 运行时模式段）

---

## 0. 现象（用户实测，线上部署后）

从**正在运行**的 thread 页面切走（切到别的 thread）→ 再切回 → 前端报：
```
Failed to load resource: the server responded with a status of 409
HTTP 409: {"detail":"Run 7fb55505-... is not active on this worker and cannot be streamed"}
```
报错栈指向打包文件 `9cf7c8666eaa7a30.js`（已回源码定位，见 §2.2）。

---

## 1. ✅ 已核实根因（现场读码确证 + 本 spec 作者复核）

### 1.1 409 抛出点（确证）

`packages/agent/backend/app/gateway/routers/thread_runs.py`：
- **line 268**（`join_run`，`GET /{thread_id}/runs/{run_id}/join`）：`if record.store_only: raise HTTPException(409, "...is not active on this worker and cannot be streamed")`。
- **line 308**（`stream_existing_run`，`GET`/`POST /{thread_id}/runs/{run_id}/stream`）：`if record.store_only and action is None: raise HTTPException(409, "...is not active on this worker...")`。
  - 该 endpoint docstring 明确：**"The LangGraph SDK's `joinStream` and `useStream` stop button both use POST to this endpoint."** → 确证前端走 SDK。
  - `action`（interrupt/rollback）非 None 时**豁免** 409（先 cancel 再 stream 残留事件）——即停止按钮路径不报 409，只有**纯重连 join**（action=None）报。

### 1.2 根因链：Gateway-embedded 多 worker + 内存 StreamBridge（确证）

1. **多 worker**：`packages/agent/docker/docker-compose.yaml:69`：
   ```
   uvicorn app.gateway.app:app --host 0.0.0.0 --port 8001 --workers ${GATEWAY_WORKERS:-4}
   ```
   → 默认 **4 个 worker 进程**。
2. **实时 SSE 流只有内存实现**：`runtime/stream_bridge/` 下仅 `memory.py`（`MemoryStreamBridge`）+ `base.py` + `async_provider.py`。`async_provider.py:55-56` 明确：
   ```python
   if config.type == "redis":
       raise NotImplementedError("Redis stream bridge planned for Phase 2")
   ```
   → **跨 worker 共享流上游自己标 Phase 2 未实现**。流是 **worker 本地内存**。
3. **run 元数据有持久化**：`runtime/runs/manager.py`：
   - `RunRecord.store_only: bool = False`（manager.py:93）。
   - `_record_from_store(row)`（manager.py:~227）从持久 store 行重建 record 时置 **`store_only=True`**（manager.py:~246）。
4. **结果**：切回来时 nginx/uvicorn 把重连请求轮询到**另一个 worker**（不是当初跑 run 的那个）→ 该 worker 内存无此 run 的 bridge → `run_mgr.get(run_id)` 内存找不到、但从持久 store 查到行 → 重建 `store_only=True` record → **409**（不是 404——元数据查得到）。

**`store_only` 语义**：「元数据可查、但实时流不在本 worker、无法 join」。这是上游有意设计，**不是 bug**——只是上游没做跨 worker 共享流。

### 1.3 ⭐ 关键补充洞察（本 spec 作者核实，handoff 未点透）：run **内容**已跨 worker 可读，仅**实时流**是 worker-local

`thread_runs.py` 的这些读端点**不 gate on `store_only`**，对 `store_only` record 照常返回：
- `GET /{thread_id}/runs/{run_id}`（line 212，`get_run`）→ 返回 run 最终状态（从 `run_mgr.get`，store_only record 可用）。
- `GET /{thread_id}/runs/{run_id}/messages`（line 384）→ 从 **event store**（`get_run_event_store`）读 run 的消息（持久层，跨 worker）。
- `GET /{thread_id}/runs/{run_id}/events`（line 410）→ 同上，run 的事件。
- `GET /{thread_id}/messages`（line 339）→ thread 历史消息。

**含义**：run 的**已产出内容**（messages/events）通过 event store **已经是跨 worker 可读**的；**只有正在进行的实时 SSE 增量**是 worker-local。→ 这让「前端优雅降级」(方案 A) 不仅可行，而且**改动很小**（数据源现成）。

### 1.4 ⭐ 前端实际行为核实（修正 handoff 对「加载失败」的描述）

核实 `frontend/src/core/threads/hooks.ts` 后，切回 thread 的真实链路：
- `useStream`（hooks.ts:389）配置 **`reconnectOnMount: true`** + **`fetchStateHistory: { limit: 1 }`**。
  - `reconnectOnMount: true` = 切回时 SDK **自动重连**正在跑的 run 流（POST `/stream`，action=None）→ 命中别的 worker → **409**。
  - `fetchStateHistory: { limit: 1 }` = 切回时 SDK **同时**拉取持久化的 thread state → `thread.messages`（被 `message-list.tsx:62` 等消费）**仍从 state history 渲染出来**。
- 409 经 SDK 抛到 `onError(error)`（hooks.ts:528）→ 当前行为 `toast.error(getStreamErrorMessage(error))`（hooks.ts:530）→ **弹一个误导性错误 toast**。
- 浏览器 console 的 `Failed to load resource ... 409` 是**失败 fetch 的浏览器级日志**（非 app 层 `console.error`；核实 app 层 stream 路径无主动 console.error，唯一在 hooks.ts:984 是无关路径）。

**结论（修正）**：切回时 thread **内容并不会真的「加载失败」**——历史内容照常渲染（`fetchStateHistory`）。真正的缺陷是 **(a) 误导性错误 toast** + **(b) 浏览器 console 的 409 噪声** + **(c) 正在跑的 run 的实时增量看不到**（要等结束或手动刷新）。方案 A 主要消除 (a)，缓解 (c)，(b) 是 SDK 重连行为的副产物（见 §3.1 诚实边界）。

### 1.5 生产 worker 数核实（handoff §3.3 gap #3，已查）

`~/ethoinsight-prod/agent.env` 存在但**未设 `GATEWAY_WORKERS`** → 生产走 compose 默认 **4 workers**。grep 全仓 + prod env 确认：除 compose:69 与 handoff 外无其他 `GATEWAY_WORKERS` 赋值。
→ **生产确实多 worker（4），409 根因成立、可复现**。（未 cat prod env 文件——含密钥；针对性 grep 单变量即足够、尊重边界。）

---

## 2. 关键文件清单（确证行号）

| 文件 | 行 | 说明 |
|---|---|---|
| `app/gateway/routers/thread_runs.py` | 267-268（join_run）/ 307-308（stream_existing_run） | 409 抛出点（`store_only` 判据） |
| `app/gateway/routers/thread_runs.py` | 212 / 339 / 384 / 410 | **降级数据源**：run 状态 / thread 消息 / run 消息 / run 事件（均不 gate store_only） |
| `runtime/runs/manager.py` | 93 / ~227 / ~246 | `store_only` 字段 + `_record_from_store` 置位 |
| `runtime/stream_bridge/memory.py` + `async_provider.py` | async_provider:55-56 | 内存 StreamBridge（唯一实现；Redis = `NotImplementedError("Phase 2")`，方案 C 要新增） |
| `packages/agent/docker/docker-compose.yaml` | 69 | `--workers ${GATEWAY_WORKERS:-4}`（方案 B） |
| `frontend/src/core/threads/hooks.ts` | 389（useStream）/ 528-530（onError → toast）/ 301（getStreamErrorMessage）/ 188（buildRunMessagesUrl）| 方案 A 改动点 |
| `frontend/src/core/api/api-client.ts` | 50-56 | `joinStream` SDK wrapper（CSRF + sanitize；方案 A 可选挂钩点）|

### 2.2 报错栈回源

打包文件 `9cf7c8666eaa7a30.js` 对应 `@langchain/langgraph-sdk` 的 `joinStream`/`useStream` 实现（SDK 内部对 `/stream` 的 fetch）。app 层入口是 `frontend/src/core/threads/hooks.ts:389` 的 `useStream(...)` 配置（`reconnectOnMount: true`）。catch 点是同文件 `onError`（528）。

---

## 3. 修复方案三选项（取舍表，建议 A+B，C 列 backlog）

用户已了解三选项（当时转向先看上游有无修复，未拍板）。最终选型留 review。

| 方案 | 做法 | 真实代价 | 评价 |
|---|---|---|---|
| **A 前端优雅降级**（建议默认） | `onError` 识别 409「is not active on this worker」→ **不弹错误 toast**（内容已由 `fetchStateHistory` 渲染）；可选主动刷新 run 状态/消息（端点现成 §1.3）。消除误导性报错 | 最小、最稳、不动后端基础设施；约 1 个文件 + 1 个分类函数 + vitest 单测 | 缺点：切回时「还在跑的 run」看不到实时增量（要等结束或手动刷新）。对当前体验已是**大幅改善**（不再吓人的红 toast）。**浏览器 console 的 409 仍会有一行**（SDK 重连副产物，§3.1） |
| **B GATEWAY_WORKERS=1** | compose 把默认改 1，单 worker 所有 run 同进程内存，重连必命中，**409 彻底消失** | 改 compose 一行（+ 文档）；丢多 worker 并发吞吐（4→1） | 适合**当前用户量小**的过渡，彻底无 409。**不是终态**（高并发 Gateway 成瓶颈）。memory `feedback_dev_prod_behavior_alignment`：必须落 compose，不能只丢 prod .env 让用户配 |
| **C 共享 StreamBridge** | 实现 Redis pub/sub（或 DB 轮询）版 `StreamBridge`，任意 worker 可 join 别 worker 的流 | 引 Redis 依赖 + 写新 bridge 实现（上游已留 `config.type=="redis"` 钩子 + Phase 2 占位）+ 改部署 + 充分测试 | 彻底解决跨 worker 实时续流，**真正终态**，工作量大。**可能届时跟随上游**（若上游补了 Phase 2 Redis bridge，spec① 类 sync 即可吃下） |

### 3.1 ⚠️ 方案 A 的诚实边界

- A 消除**误导性错误 toast**（主要痛点）+ 保证历史内容渲染（本就如此）。
- A **不能完全消除浏览器 console 的那一行 409**——它是 SDK `reconnectOnMount` 发起的 fetch 失败的浏览器级日志，除非改 SDK 重连行为或上游修。可接受（console 噪声 ≠ 用户可见错误）。若 review 要求连 console 也干净 → 需配合 B（无 409）或 C（流可 join）。
- A **不提供实时增量续流**——切回正在跑的 run 看不到新增量，得等 `onFinish` 或手动刷新。若这是硬需求 → C。

### 3.2 建议组合

- **短期**：**A（前端降级兜底）** 必做（消除可见错误）；**B（过渡期可选 workers=1）** 视当前并发量决定——用户量小则 B 直接彻底无 409，A 作为「将来 workers>1 时仍优雅」的长期兜底。
- **Backlog**：**C** 留待真需要多 worker 实时续流时做（或跟随上游 Phase 2）。
- 让 review 拍板 A 单独 / A+B。

---

## 4. 方案 A 具体改动（前端）

### 4.1 改动点：`frontend/src/core/threads/hooks.ts`

**(1) 新增 409「worker-local」分类函数**（放在 `getStreamErrorMessage` 附近，hooks.ts:~301）：
```ts
/**
 * True when the error is the Gateway's "run is active on another worker"
 * 409 — emitted by thread_runs.py join_run / stream_existing_run when a
 * store_only RunRecord is reached (multi-worker + in-memory StreamBridge).
 *
 * Switching back to a still-running thread makes the SDK (reconnectOnMount)
 * POST /stream; nginx may route it to a worker that doesn't hold the run's
 * in-memory bridge, yielding this 409. The thread's persisted content is
 * already rendered via fetchStateHistory, so this is NOT a user-facing
 * failure — we suppress the error toast for it.
 */
function isRunNotOnThisWorkerError(error: unknown): boolean {
  const msg = getStreamErrorMessage(error);
  // Match the server detail string (thread_runs.py:268,308). Also tolerate a
  // status code on the error object if the SDK surfaces one.
  const status =
    typeof error === "object" && error !== null
      ? Reflect.get(error, "status") ?? Reflect.get(error, "statusCode")
      : undefined;
  return (
    msg.includes("is not active on this worker") ||
    (status === 409 && msg.toLowerCase().includes("worker"))
  );
}
```
> 注：分类**主认服务端 detail 串**（最稳，契约来自 thread_runs.py:268/308 的字面量），状态码作辅助。两处 detail 串一致，单一匹配即可。

**(2) `onError` 短路**（hooks.ts:528-530）：
```ts
onError(error) {
  setOptimisticMessages([]);
  if (isRunNotOnThisWorkerError(error)) {
    // Cross-worker re-join: content already shown via fetchStateHistory.
    // Don't alarm the user with a red toast. (Optional: trigger a state
    // refresh so the just-finished run's tail is pulled in — see (3).)
    // Intentionally no toast.error here.
  } else {
    toast.error(getStreamErrorMessage(error));
  }
  pendingUsageBaselineMessageIdsRef.current = new Set(
    messagesRef.current.map(messageIdentity).filter((id): id is string => Boolean(id)),
  );
  if (threadIdRef.current && !isMock) {
    void queryClient.invalidateQueries({ queryKey: threadTokenUsageQueryKey(threadIdRef.current) });
  }
},
```

**(3) 可选增强**：识别到该 409 后，主动 `invalidateQueries` / 重新 `fetchStateHistory`（或调 `buildRunMessagesUrl` 拉 run 消息）把「切走期间该 run 新产出的内容」补渲染（端点 §1.3 现成）。若 `useStream` 的 `fetchStateHistory` 已在每次切回拉最新 state（limit:1 = 最新一条 state，含累积 messages），通常已够；增强属锦上添花，可留 review 决定是否纳入首版。

### 4.2 不改 `api-client.ts`

`joinStream` wrapper（api-client.ts:50-56）只做 CSRF + sanitize，**不在此层吞 409**（那样会影响所有 joinStream 调用方、且丢失 onError 的统一处理）。集中在 `onError` 分类是正确切面。

### 4.3 i18n

新增**不**引入用户可见文案（A 是「不显示」错误），无需加翻译键。

---

## 5. 方案 B 具体改动（若纳入）

### 5.1 `packages/agent/docker/docker-compose.yaml:69`

把默认 worker 数从 4 改 1（过渡期）：
```yaml
command: sh -c "cd backend && PYTHONPATH=. /app/backend/.venv/bin/uvicorn app.gateway.app:app --host 0.0.0.0 --port 8001 --workers ${GATEWAY_WORKERS:-1}"
```
> memory `feedback_dev_prod_behavior_alignment`：worker 数是**产品属性不是调试开关**，必须落 compose 默认，不能只让用户在 prod `.env` 配。改默认值后，想多 worker 的部署仍可用 `GATEWAY_WORKERS=N` 覆盖。

### 5.2 文档

在 `docs/sop/deploy-via-tar-sop.md`（或部署文档）记一句：`GATEWAY_WORKERS` 默认 1（单 worker 内存共享、无跨 worker 409）；调大需先有共享 StreamBridge（方案 C）否则切 thread 会触发 409（已由方案 A 在前端优雅降级，但实时续流仍不可用）。

### 5.3 B 的权衡提示（写给 review）

- workers=1 = 所有 thread 的 run 串行在一个进程的 event loop 上。当前用户量小可接受；**若已有「多个研究员同时跑分析」的并发**，单 worker 会让 Gateway 成瓶颈（一个 run 的重 CPU/IO 拖慢其他）。决策依据 = 当前真实并发量。
- A 已兜底「workers>1 时不报可见错误」，所以 B 不是必须；B 的增量价值 = **连 console 409 噪声也消除** + **重连必命中（实时续流也恢复，因为本就同进程）**。

---

## 6. ✅ 已核实：不能靠 sync 上游解决（与 spec① 正交）

grep spec① 的 21 个上游 commit 全量 diff 的 `stream/worker/join/409/store_only/sse/StreamBridge` 关键词，**无一条修「跨 worker SSE join」**。`async_provider.py:55-56` 仍是 Redis `NotImplementedError("Phase 2")`。最相关的 `268fdd69`（shutdown drain）/`f725a963`（singleton 保护）只让 run 元数据更可恢复，**不消除流的 409**。
→ **spec② 是独立修复，不被 spec① 阻塞**。两 spec 各自 worktree/PR。
（spec① §5.1 已提示：合 `268fdd69` 后复核 manager.py 的 `store_only`/`_record_from_store` 段未被上游破坏，spec② 依赖其不变。）

---

## 7. 测试

### 7.1 前端单测（vitest，**修正 handoff**：前端有 vitest）

核实：`frontend/package.json` 有 `"test": "vitest run"`，且存在 `src/core/api/stream-mode.test.ts`。→ 方案 A 的 409 分类器**可加 vitest 单测**（比 handoff §3.6「playwright only」更轻、更确定）：
- 新建 `src/core/threads/<hooks-or-stream-error>.test.ts`（或就近测 `isRunNotOnThisWorkerError`，需把它 export 或抽到可测模块）：
  - 输入服务端 detail 串 `"Run X is not active on this worker and cannot be streamed"` → `true`。
  - 输入 `{ status: 409, message: "...worker..." }` → `true`。
  - 输入普通错误（超时/网络/500）→ `false`（确保不误吞真错误，仍弹 toast）。
- 运行：`cd packages/agent/frontend && pnpm test`（或 `npm run test`）。

> 抽取建议：把 `isRunNotOnThisWorkerError` + `getStreamErrorMessage` 放到一个**无 React 依赖**的小模块（如 `src/core/threads/stream-error.ts`）便于 vitest 直接 import（hooks.ts 含 React hook，整文件难单测）。

### 7.2 行为验证（playwright / 手动，复现链路）

`frontend/CLAUDE.md` 注明「No test framework configured」指无 E2E 框架——行为验证走 playwright（本仓库 chrome-devtools-mcp / playwright MCP 可用）或手动：
- **多 worker 起**（`GATEWAY_WORKERS=4`）：开一个 thread 发分析任务（run 进行中）→ 切到别的 thread → 切回 → **改前**：红错误 toast + console 409；**改后（A）**：无错误 toast、历史内容在、（console 仍可能一行 409）。
- **方案 B 验证**：`GATEWAY_WORKERS=1` 同样操作 → **无任何 409**（toast + console 皆净）。

### 7.3 后端

方案 A/B **不动后端逻辑**（B 只改 compose 启动参数）。若 review 选 C 才涉及后端新 bridge → 那时 `make test` + 新 bridge 单测/集成测另写（不在本 spec 范围）。

---

## 8. dogfood 验证（必做，验收门槛）

线上/本地**多 worker**（`GATEWAY_WORKERS=4`，与生产一致）起：
1. 开 thread A，发一个会跑一阵的分析任务（保持 run 进行中）。
2. 切到 thread B（或新建），再切回 thread A。
3. **验收（方案 A）**：① 无红色错误 toast；② thread A 历史消息/产出正常渲染；③ run 结束后内容完整（`onFinish` 或刷新后实时增量补齐）。
4. **若纳入 B（workers=1）**：额外确认浏览器 console **无** 409 一行（彻底干净）。
5. 记录到 handoff/milestone：复现步骤 + 改前后对比截图。

---

## 9. 验收 checklist

- [ ] 方案 A：`onError` 对「is not active on this worker」409 不再 `toast.error`，普通错误仍弹（vitest 单测覆盖正/负例）。
- [ ] 方案 A：切走再切回正在跑的 thread，无用户可见错误 toast，历史内容渲染正常（dogfood §8）。
- [ ] （若 B）compose `GATEWAY_WORKERS` 默认改 1，部署文档同步；多 worker 仍可 `GATEWAY_WORKERS=N` 覆盖。
- [ ] （若 B）workers=1 下切 thread 全程无 409（toast + console 皆净）。
- [ ] 前端 lint/typecheck 通过；`pnpm test`（vitest）绿。
- [ ] 不误吞真实 stream 错误（500/超时/网络断仍弹 toast）——负例测试守住。
- [ ] C 作为 backlog 记录（不在本次实施）。

---

## 10. 关联 memory（实施 agent 必读）

- `feedback_dev_prod_behavior_alignment` — worker 数是产品属性，落 compose 不放 prod .env（§5.1）
- `feedback_deploy_compose_per_service_image_tag` — 部署链镜像/配置对齐（B 改 compose 后重新 deploy 验证）
- `feedback_async_io_blocks_event_loop` — workers=1 时单 event loop 更敏感于同步 IO 阻塞（B 的并发权衡背景）
- `feedback_grill_handoff_must_be_verified` — 本 spec 已现场核实，实施 agent 仍抽查 §1 关键点
- `project_2026-06-08_three_specs_review_acdf` — **worktree 必须 `git worktree add <path> -b <name> dev`**（默认 origin/main 落后、PR 卷入误删）

---

## 11. 实施 agent 第一步

1. 读 §10 memory + 本 spec。
2. **建 worktree（红线）**：`git worktree add <path> -b fix-gateway-multiworker-409 dev`（**显式基于 dev**）。前端改动需 `cd packages/agent/frontend && pnpm install`（或 npm）。
3. 确认 review 选型（A / A+B）。先实施 A（`hooks.ts` + 抽 `stream-error.ts` + vitest 单测）。
4. （若 A+B）改 compose 默认 + 文档。
5. §7 测试 + §8 dogfood（多 worker 复现）。
6. 交 review，**不直接 merge 到 main**（先进 dev，再 PR）。

---

## 附：根因一句话

> 多 worker（4）+ 实时 SSE 流只存在于「跑 run 的那个 worker 的内存」（StreamBridge 无共享实现，Redis 版上游标 Phase 2）。切回 thread 时 SDK（`reconnectOnMount`）重连被轮询到别的 worker，该 worker 查得到 run 元数据（持久 store → `store_only=True`）但没有流 → 409。**run 已产出内容本就跨 worker 可读（event store），只有实时增量是 worker-local**——所以前端优雅降级（A）小而稳，单 worker（B）彻底无 409，共享 bridge（C）是终态。
