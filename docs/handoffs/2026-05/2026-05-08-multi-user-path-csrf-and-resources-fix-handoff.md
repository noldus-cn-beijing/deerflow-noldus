# 2026-05-08 多用户路径解析 user_id 漏传 + CSRF + 缺失资源 — 端到端验证完成

## TL;DR

继上午 `subagent ContextVar 跨线程丢失`修复（commit `7ff27483`）之后，端到端验证 shoaling 流水线时又暴露 **4 个独立 bug**，全部修复并端到端验证通过：

1. **`/api/threads/{tid}/archived-messages` 路由 user_id 漏传** → 旧 thread 看不到归档前的最初对话
2. **`/api/threads/{tid}/artifacts/...` 路由 user_id 漏传** → 图表 / report.md 全部 404
3. **`present_files` tool user_id 漏传** → lead agent 卡死，**根本不派遣 report-writer**，report.md 永远不生成
4. **`POST /api/threads/{tid}/suggestions` 403 (CSRF token 缺失)** → 前端 `input-box.tsx` 用了原生 `fetch()` 没经过 `fetchWithAuth` helper

外加 1 个 UX 装饰修复：
- 补 `noldus-emblem.svg`（品牌定制时 workspace-header.tsx 改了引用但没传文件）

**改动范围**：5 个文件 + 1 个新资源，~10 行净增。

**验证**：新 thread `8d71e03f-2749-46ce-9228-7e0dc936f6cd` 端到端 shoaling 流水线全程跑通：上传 → planning → code-executor → data-analyst → **report-writer 成功派遣并生成 report.md** → 前端图表全部加载、报告显示正确。

## 改动清单

| 文件 | 改动 | 修复 bug |
|---|---|---|
| `packages/agent/backend/app/gateway/path_utils.py` | helper 加 `user_id=get_effective_user_id()` | bug 2（覆盖 artifacts router + skills router 全部调用方） |
| `packages/agent/backend/app/gateway/routers/threads.py` | archived-messages 路由 `paths.thread_dir(tid, user_id=...)` + 加 `@require_permission("threads", "read", owner_check=True)` + `request: Request` 形参 | bug 1 + 安全规范一致性 |
| `packages/agent/backend/packages/harness/deerflow/tools/builtins/present_file_tool.py` | `resolve_virtual_path(tid, fp, user_id=get_effective_user_id())` | bug 3 |
| `packages/agent/backend/tests/test_present_file_tool_core_logic.py` | mock 适配新增的 `user_id` kwarg | 测试同步 |
| `packages/agent/frontend/src/components/workspace/input-box.tsx` | suggestions POST 改用 `fetchWithAuth`（自动注入 X-CSRF-Token） | bug 4 |
| `packages/agent/frontend/public/images/noldus-emblem.svg` | 新增（来源：`/home/wangqiuyang/resources/images/Noldus Emblem.svg`） | UX 装饰 |

## 根因分类

**Bug 1/2/3 是同一类**：上游 better-auth 同步引入 `users/<uid>/threads/<tid>/...` 多用户隔离结构，要求每个调用 `paths.*` 的点显式传 `user_id`。**写**端（uploads / archiving middleware / sandbox / artifacts 服务等）大多在 sync 时逐个修过，但**读**端的几个调用点漏掉了：

- `app/gateway/path_utils.py:resolve_thread_virtual_path`（artifacts + skills 共用 helper）
- `app/gateway/routers/threads.py:get_archived_messages`（noldus 自加路由）
- `tools/builtins/present_file_tool.py:_normalize_presented_filepath`（lead agent 工具）

`get_effective_user_id()` 的 silent fallback 到 `"default"` 是**bug 放大器**——所有遗漏点不会立即崩，而是写到/读到 `users/default/threads/<tid>/...` 这条幽灵路径，长得和真路径一模一样，前端只看到 404 / "不存在"，真正的根因要直到文件系统出现两份目录才会暴露。

**Bug 3 的二级影响**最严重：`present_files` 失败让 lead agent 在"展示图表"那一步 hang 住，没派遣 report-writer，**整个流水线中途中断**——前端看到 7 张图缩略图但永远点不开 report，因为 report 根本没被生成。

**Bug 4 是另一类**：better-auth 同时引入了 CSRF Double-Submit Cookie 保护。前端有专门的 `fetchWithAuth` helper（`core/api/fetcher.ts`）自动注入 `X-CSRF-Token` 头，但 `input-box.tsx:356` 用了原生 `fetch()` 绕过 helper，state-changing POST 直接 403。

## 验证

### 后端测试
```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_present_file_tool_core_logic.py tests/test_artifacts_router.py tests/test_threads_router.py tests/test_archiving_summarization.py -q
# 37 passed in 1.09s
```

### 前端类型检查 + lint
```bash
cd packages/agent/frontend
pnpm typecheck   # 0 error
pnpm lint        # 0 error
```

### 端到端浏览器验证（实测）

新 thread `8d71e03f-2749-46ce-9228-7e0dc936f6cd`，跑斑马鱼 shoaling 全流程：

```
.deer-flow/users/cd95effa-.../threads/8d71e03f-.../user-data/
├── uploads/                        ✅ 5 个轨迹文件
├── workspace/
│   ├── handoff_planning.json       ✅
│   ├── handoff_code_executor.json  ✅
│   ├── handoff_data_analyst.json   ✅
│   ├── handoff_report_writer.json  ✅  ← 关键：报告生成阶段成功执行
│   ├── conversation_summary.md
│   ├── parsed.pkl / metrics.pkl / charts.json / statistics.json / metrics_summary.json
│   └── experiment-context.json
└── outputs/
    ├── distance_moved_box_plot.png
    ├── mean_nnd_box_plot.png
    ├── velocity_stats_box_plot.png
    ├── trajectory.png
    ├── inter_individual_distance_timeseries.png
    ├── group_polarity_timeseries.png
    ├── metrics.csv
    ├── statistics.json
    └── report.md                   ✅  ← 报告文件存在并可前端加载
```

服务器日志确认所有 artifact GET 200 OK：
```
GET /api/threads/8d71e03f-.../artifacts/mnt/user-data/outputs/trajectory.png             200 OK
GET /api/threads/8d71e03f-.../artifacts/mnt/user-data/outputs/distance_moved_box_plot.png 200 OK
GET /api/threads/8d71e03f-.../artifacts/mnt/user-data/outputs/mean_nnd_box_plot.png       200 OK
GET /api/threads/8d71e03f-.../artifacts/mnt/user-data/outputs/velocity_stats_box_plot.png 200 OK
GET /api/threads/8d71e03f-.../artifacts/mnt/user-data/outputs/group_polarity_timeseries.png 200 OK
GET /api/threads/8d71e03f-.../artifacts/mnt/user-data/outputs/inter_individual_distance_timeseries.png 200 OK
POST /api/threads/8d71e03f-.../suggestions   200 OK   ← CSRF 修复验证
GET /api/threads/8d71e03f-.../archived-messages 200 OK
```

## 上游归属判定

**这 4 个全部是本地 fork 的 regression，不是上游 deerflow 的问题。** 不需要给上游提 issue：

- `archived-messages` 路由本身是 noldus 自加 extension（threads.py 注释明写 `# Noldus extension`）
- `artifacts` / `present_files` 的 user_id 漏传是 better-auth 同步轮 G/H 时本地调用点遗漏
- `input-box.tsx` 的原生 fetch 用法是 noldus 写的（上游可能没有 suggestions route）
- `noldus-emblem.svg` 是品牌定制资源

## 经验沉淀

1. **silent fallback 是 bug 放大器**：`get_effective_user_id()` 在 ContextVar 未设时静默回退到 `"default"`，让"已认证但 ContextVar 丢失"和"匿名"两种语义合并到一条路径。所有漏传 user_id 的点不会立刻崩——它们写/读到 `users/default/threads/<tid>/`，文件系统沉默地形成幽灵副本，根因要靠文件系统出现两份目录才暴露。建议下一轮基础设施加固：
   - `get_effective_user_id()` 改成 `require_*` 抛错版本，所有 sandbox/路径操作必须用 require 版
   - 文件系统层加守门：写入 `users/default/...` 时 emit 一个明显的 warning 日志（生产应该极少触发）

2. **better-auth 同步的检查清单还没完全跑过**：bug 1/2/3 都是 better-auth 多用户改造时的 sync 遗漏。轮 G/H 修过的"明显路径解析点"覆盖了大部分调用，但还有 2-3 处隐蔽的 helper（如 `path_utils.resolve_thread_virtual_path` 这种 1 行小函数）和 noldus-extension 路由没扫到。下次同步前用 `grep -rn "resolve_virtual_path\|paths\.thread_dir\|paths\.sandbox_" backend/ | grep -v "user_id="` 跑一遍能机械性发现遗漏。
   - 已知**仍未修**的 1 处：`app/channels/manager.py:334`（IM channel 入口，目前未启用所以不阻塞）

3. **state-changing fetch 必须走 fetchWithAuth**：原生 `fetch()` 一律会被 CSRFMiddleware 403。前端 18+ 处用原生 `fetch()` 的 POST/PUT/DELETE 都是潜在 bug——只是大多数走 LangGraph SDK 那条路径（SDK 本身有 CSRF 处理），不经过这条 fetcher 就漏。建议加 ESLint rule 禁用裸 `fetch(` 调用 state-changing 方法，或者在下次完整 sweep 时 grep 检查全部调用点。

4. **`present_files` 这种走 ToolRuntime 的工具也会受 ContextVar 影响**：lead agent 主线程的 ContextVar 是 OK 的（`make_lead_agent` set），但 subagent 通过任务工具调过来时会跨线程——上午刚修的 ContextVar fix 已经覆盖了这条路径，加 user_id 显式传参是 belt-and-braces 双保险。

## 后续 / 不在本次范围

- `app/channels/manager.py:334` 漏传 user_id（IM channel 未启用，不阻塞）
- 前端其他 ~18 处用原生 `fetch()` 的 POST/PUT/DELETE 调用，下一轮统一改为 `fetchWithAuth`
- 评估 `get_effective_user_id()` 改成 require 抛错版本（涉及 CLI/migration 路径，需小心）
- 旧 thread `05377611-...` 的 report.md 永久缺失（当时 present_files bug 卡住，report-writer 没派遣），不可恢复——此 thread 的图表本次修复后能加载，但报告无法补救

## 待办

- [ ] `git add` + commit（commit message 中文，按 noldus 规范），等用户决定 push 时机
- [ ] 是否把今天的 4 个 fix 和上午的 ContextVar fix 一起 push 到 origin/dev 由用户决定

## 相关文档

- 上午的 ContextVar fix 完成交接：[2026-05-08-subagent-contextvar-fix-completed-handoff.md](2026-05-08-subagent-contextvar-fix-completed-handoff.md)
- 上午的 ContextVar fix 实施计划：[../superpowers/plans/2026-05-08-subagent-contextvar-fix-plan.md](../superpowers/plans/2026-05-08-subagent-contextvar-fix-plan.md)
- 上传文件 Agent 看不到 fix（lead agent ContextVar 桥接，commit `02547092`）：[2026-05-08-upload-file-not-recognized-FIXED-handoff.md](2026-05-08-upload-file-not-recognized-FIXED-handoff.md)
- Tier 4 better-auth 同步轮 3 完成交接：[2026-05-08-deerflow-tier234-round3-completed-handoff.md](2026-05-08-deerflow-tier234-round3-completed-handoff.md)
- 前端 tab 崩溃 fix（`useUpdateSubtask` render 期 setState）：[2026-05-08-frontend-tab-crash-on-clarification-and-subtask-FIXED-handoff.md](2026-05-08-frontend-tab-crash-on-clarification-and-subtask-FIXED-handoff.md)
