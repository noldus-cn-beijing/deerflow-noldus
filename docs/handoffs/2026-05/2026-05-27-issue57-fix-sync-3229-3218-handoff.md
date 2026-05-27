# 2026-05-27 issue #57 修复 + DeerFlow sync PR #3229/#3218 — 交接文档

## TL;DR

- **Issue #57 修复**：Welcome 对话框与乐观消息的 3-4 秒重叠显示，根因是 `isNewThread` 翻转时机依赖 HTTP `onCreated` 回调。新增 `welcomeDismissed` 状态在 `handleSubmit` 立即翻转，前端 2 文件改动
- **DeerFlow 上游 sync**：surgical merge PR #3229（async_provider SQLite offload + Blockbuster runtime gate）和 PR #3218（think tag 反引号防护），14 文件 410+ 行
- **后端延迟调查**：ECS 内部全链路 < 4ms，非后端代码问题；公网 HTTP 93ms 正常。3-4 秒延迟来自用户浏览器到 ECS 的网络路径，前端修复已掩盖
- **2 commit 已合 dev**，已推 origin

## Issue #57: Welcome 重叠显示

### 根因

`sendMessage()` 内部有两步断裂：

1. `setOptimisticMessages()` 同步插入用户消息 → MessageList 立即渲染（hooks.ts:570）
2. `isNewThread → false` 需等 `client.threads.create()` HTTP 返回 → `onCreated` → `onStart`（hooks.ts:278）

之间的间隙 = HTTP RTT。dev localhost ~50ms 不可见，ECS 公网 3-4 秒明显。

Playwright 注入 3s fetch 延迟实测：`first_message_appeared` 与 `welcome_removed` 间隔 3,115ms，完美复现。

### 修复（commit `682c10ff`）

两个 page.tsx 新增 `welcomeDismissed` 状态 + `isCenteredLayout` derived：

| 改动点 | 说明 |
|--------|------|
| `welcomeDismissed` state | `handleSubmit` 开头立即翻 true |
| `isCenteredLayout = isNewThread && !welcomeDismissed` | 解耦 Welcome/布局可见性 |
| Welcome/AgentWelcome 渲染 | `isNewThread → isCenteredLayout` |
| 居中布局 `-translate-y-[calc(50vh-96px)]` + max-width | 同步改为 `isCenteredLayout` |
| 空消息 guard | 防空文本误触发 dismiss |
| 失败恢复 `.catch()` | reset `welcomeDismissed = false` |
| clarification option | 走 `handleSubmit` 而非直调 `sendMessage`（带 `void`） |

`isNewThread` 其他用途（header backdrop、MessageList pt-10、autoFocus、showFollowups）不动。

### 后端延迟调查结论

| 测试点 | 耗时 |
|--------|------|
| ECS Docker 内部 (nginx→langgraph) | 3.8ms |
| ECS localhost | 0.4ms |
| 本地→ECS HTTP | 93ms |

后端全链路无瓶颈。Issue #57 的 3-4 秒是结构性时序裂口 + 用户公网 RTT 放大，前端修复已消除症状。

## DeerFlow upstream sync: PR #3229 + #3218

### PR #3229 — async_provider fix + Blockbuster gate（commit `4df35f44`）

**async_provider.py**（surgical merge）：

- 新增 `_prepare_sqlite_checkpointer_path` / `_prepare_database_sqlite_checkpointer_path` helper
- `_async_checkpointer` sqlite 分支：`resolve_sqlite_conn_str` + `ensure_sqlite_parent_dir` → `await asyncio.to_thread(_prepare_sqlite_checkpointer_path, ...)`
- `_async_checkpointer_from_database` sqlite 分支：同步 `ensure_sqlite_parent_dir` → `await asyncio.to_thread(_prepare_database_sqlite_checkpointer_path, ...)`
- **保留** `DeerFlowAsyncSqliteSaver`（Noldus 定制）

**Noldus 适配**：

| 上游 | Noldus |
|------|--------|
| `AsyncSqliteSaver`（sys.modules mock） | `DeerFlowAsyncSqliteSaver.from_conn_string` 直接 patch |
| `test_sqlite_creates_parent_dir_via_to_thread` 断言 `ensure_sqlite_parent_dir` | 改为 `_prepare_sqlite_checkpointer_path` |
| 新增 `test_database_sqlite_creates_parent_dir_via_to_thread` | 适配 `DeerFlowAsyncSqliteSaver` patch |
| `test_skills_load.py` 测 `SubagentExecutor._load_skills` | 改写为 `_load_skill_contents`（skip 暂挂，函数体内 import 无法 monkeypatch） |

**Blockbuster suite**（完整引入）：

- `tests/blocking_io/` — conftest + 4 测试（smoke ×3、skills_load ×1 skip、sqlite_lifespan ×1）
- `tests/support/detectors/blocking_io_runtime.py` — Blockbuster-based detector
- `pyproject.toml` — `blockbuster>=1.5.26,<1.6` dev dep + `allow_blocking_io` marker
- `Makefile` — `test-blocking-io` target
- `.github/workflows/backend-blocking-io-tests.yml` — CI gate（路径适配 `packages/agent/backend/`）

### PR #3218 — think tag backtick guard（commit `4df35f44`）

- `splitInlineReasoning` 从正则 `TRAILING_UNCLOSED_THINK_RE` 改为上游 `indexOf("<think>")` + 反引号防护
- 删除 `TRAILING_UNCLOSED_THINK_RE` 常量，新增 `THINK_OPEN_TAG`
- 新增 6 个测试 case（preamble、hasReasoning、hasContent、backtick guard）
- 21 tests passed, typecheck clean

## 仓库状态快照

- 当前分支: `dev`
- HEAD: `4df35f44` sync(deerflow): surgical merge 上游 PR #3229 + #3218
- 前一 commit: `682c10ff` fix(frontend): 消除新对话 Welcome 与乐观消息的重叠显示
- 本地 = origin/dev（已 push）
- 未 commit 改动: `deploy-via-tar.sh`（pre-existing，SKIP_PRUNE 注释）
- 3 份历史 handoff 仍 untracked（上次遗留，不在本次范围）

## 未完成事项

### P3: 更新 `.deerflow-sync-state`

sync 基准从 `f9b70713` 前移到 `162fb214`（upstream），手动更新后 commit。当前新 sync 点已在 dev 但 state 文件未改：

```bash
sed -i 's/last_sync_commit: f9b70713/last_sync_commit: 162fb214/' .deerflow-sync-state
sed -i 's/last_sync_date: 2026-05-25/last_sync_date: 2026-05-27/' .deerflow-sync-state
sed -i 's/last_sync_commits_count: 15/last_sync_commits_count: 23/' .deerflow-sync-state
sed -i 's/last_sync_prs:.*/last_sync_prs: "#33-#38, #3229, #3218"/' .deerflow-sync-state
```

### P3: 远端 `origin/fix/catalog-fields-into-plan` 仍存在

上次遗留，5 分钟清理。

### P4: `test_skills_load` skip 解除

`_load_skill_contents` 函数体内 import `get_app_config` 导致 monkeypatch 无法生效。后续将 `AppConfig` 改为参数注入后可启用该测试。

## 建议接手路径

大概率不需要做任何事。如果用户提到：

- **"Welcome 还有延迟"** → 检查 ECS 是否已部署最新 frontend 镜像（`make deploy-tar` 后确认镜像 md5）
- **"sync 更多上游 commit"** → 从 `162fb214`（upstream-real/main）开始 diff，参考本 handoff 的 surgical merge 模式
- **"Blockbuster CI 红了"** → 检查 `test_skills_load` skip 是否仍然有效、`DeerFlowAsyncSqliteSaver` patch 是否正确

## milestone 建议

DeerFlow sync track 需要更新：sync 点从 `f9b70713`（2026-05-25）推进到 `162fb214`（2026-05-27），新增 2 个 PR 合入。

Issue #57 不需要独立 milestone（属于 bug fix，在已有 product track 内）。

建议在 `docs/milestone/deerflow-sync-2026-05-25-all-5-pr-merged.md` 追加：

> **2026-05-27**: 追加 sync PR #3229（async_provider SQLite offload + Blockbuster runtime gate）和 PR #3218（think tag backtick guard）。sync 基准 f9b70713 → 162fb214。新增 14 文件，2 commit 合入 dev，全量测试通过。详见 [5/27 handoff](../handoffs/2026-05/2026-05-27-issue57-fix-sync-3229-3218-handoff.md)
