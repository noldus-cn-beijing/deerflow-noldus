# 2026-05-18 DeerFlow Sync 进度

> 同步轮次：第 3 轮（2026-05-18）
> Worktree: `.claude/worktrees/deerflow-sync-2026-05-18`
> 分支：`sync/deerflow-2026-05-18`
> 基线：dev `7e61478f`（P0 合 + G4 frontend 合 + 5 个新提交全部已 push）
> 上游：`deerflow/main` `39f901d3`
> 战略 plan：`/home/wangqiuyang/.claude/plans/home-wangqiuyang-noldus-insight-docs-ha-drifting-moth.md`

## Plan A（11 个 commit，复用 `2026-05-15-deerflow-upstream-sync-plan-A-safe-batch.md`）

| Task | Commit | 状态 | 落地 commit hash | 备注 |
|---|---|---|---|---|
| A-1 | `2a1ac06b` | ✅ | `d11720c9` | persistence token usage 分组表达式 |
| A-2 | `2eb11f97` | ⏸ BLOCKED | | 本地 journal.py（382 行）缺 caller bucketing + dedup（上游 500 行），A-2 commit 依赖那些 base 才能工作。推迟到 Plan B 阶段统一处理 journal.py。 |
| A-3 | `f1a0ab69` | ✅ | `78eab548` | tool_search promotions 重入保护 |
| A-4 | `20d2d2b3` | ✅ | `4330586d` | dangling tool_call middleware（含 invalid_tool_calls 处理）|
| A-5 | `2b5bece7` | ✅ 代码 / ⏸ 测试 | `bae6c690` | sandbox singleton lifecycle reset（代码合，测试推迟）|
| A-6 | `e9deb6c2` | ⏸ BLOCKED | | thread metadata SQL 优化大 PR：新增 json_compat.py（195 行）+ 多轮 Copilot review，改动密集；本地 threads.py 有 87 行 Tier 4 G better-auth + 路径修复 drift，高冲突风险。推迟到 Plan B。|
| A-7 | `7caf03e9` | ✅ 文案 / ⏸ 测试 | `c865705f` | postgres extra 安装指引（仅 4 文件文案，测试推迟）|
| A-8 | `9892a7d4` | 🔶 PENDING | | subagent token 归并：journal.py +68 / task_tool.py +141（受保护）/ executor.py +16（受保护）。task_tool.py 本地有 handoff 占位符定制，需 surgical merge。**集中在 A-8~A-11 批次处理**。|
| A-9 | `eab7ae3d` | 🔶 PENDING | | subagent token 流到 header（依赖 A-8）|
| A-10 | `813d3c94` | 🔶 PENDING | | system_prompt + skills 合并（受保护 executor.py）|
| A-11 | `7de9b582` | 🔶 PENDING | | Runtime type alias（受保护 sandbox/tools.py + task_tool.py）|
| A-4 | `20d2d2b3` | ⏳ | | dangling tool_call middleware |
| A-5 | `2b5bece7` | ⏳ | | sandbox singleton lifecycle reset |
| A-6 | `e9deb6c2` | ⏳ | | thread metadata 过滤 SQL 优化 |
| A-7 | `7caf03e9` | ⏳ | | postgres extra 配置 |
| A-8 | `9892a7d4` | ⏳ | | subagent token 归并 |
| A-9 | `eab7ae3d` | ⏳ | | subagent token 流到 header |
| A-10 | `813d3c94` | ⏳ | | system_prompt + skills 合并 |
| A-11 | `7de9b582` | ⏳ | | Runtime type alias |

## Plan A-ext（14 个 commit，本轮新增）

### Backend SAFE（6 个）

| Task | Commit | 状态 | 落地 commit hash | 备注 |
|---|---|---|---|---|
| AE-1 | `a814ab50` | ✅ | `a948205a` | security_scanner.py JSON robustness |
| AE-2 | `e74e126e` | ✅ backend / ⏸ docker | `7b1ee998` | sandbox PVC user scoping（docker/ 跳过，本地有 deployment 定制）|
| AE-3 | `7a2670ea` | ✅ | `5318e42a` | gateway artifact preview cap |
| AE-4 | `722c690f` | ✅ 代码+新测试 / ⏸ 老测试 | `8ece7676` | memory queue agent isolation（含补 resolve_runtime_user_id）|
| AE-5 | `45060a9f` | ✅ | `f6b9f3fd` | postgres row lock 规避 |
| AE-6 | `181d8365` | ✅ 已并入 A-4 | A-4 (`4330586d`) | dangling_tool_call_middleware adjacency — A-4 cp 整文件已包含 |

### Frontend SAFE（4 个，G4 已合 origin/dev）

| Task | Commit | 状态 | 落地 commit hash | 备注 |
|---|---|---|---|---|
| AE-7 | `4538c322` | ✅ | `fe563601` | thinking content type fix（1 行）|
| AE-8 | `c0233cae` | ✅ | `4f739130` | login 闪屏 + ResizeObserver（2 文件 6 行）|
| AE-9 | `7c42ab3e` | ⏸ BLOCKED | | wait async chat submit：上游改 5 文件，本地 input-box.tsx drift 569 行 / 2 个 page.tsx drift 198~211 行（全是 Noldus 业务定制）。clearSubmittedState 抽取复杂，surgical merge 高风险。推迟到 frontend 团队评估。 |
| AE-10 | `6d3cffb4` | ⏸ BLOCKED | | hooks.ts 上游改 173 行，本地 drift 803 行（Noldus 重定制 thread streaming 逻辑）。surgical merge 高风险。推迟到 frontend 团队评估。 |

### PROTECTED surgical merge（2 个）

| Task | Commit | 状态 | 落地 commit hash | 备注 |
|---|---|---|---|---|
| AE-11 | `380255f7` | ⏸ BLOCKED | | /mnt/user-data contract 大架构改动：LocalSandboxProvider 从 singleton → PerThreadCache（+235 行）、sandbox/tools.py 10 行（受保护 + Noldus `{{shared://}}` 占位符）、新增 366 行测试。与本地架构冲突大，推迟到 Plan B 专门 worktree。|
| AE-12 | `0c37509b` | ⏳ | | todo_middleware.py 222 行融合 |

### EVAL（2 个，默认跳过）

| Task | Commit | 状态 | 落地 commit hash | 备注 |
|---|---|---|---|---|
| AE-13 | `6d611c2b` | ✅ | `0ce7529a` | JWT secret persist（本地无 Noldus 定制，整文件 cp）|
| AE-14 | `39f901d3` | ⏸ BLOCKED | | runs manager 重启恢复：上游改 5 文件（manager 加 store backing + user_id 透传，base.py / memory.py / thread_runs.py 签名联动）。本地 manager.py 是远古版本（drift 153 行 — 缺 store backing），不是单纯 fix 而是架构升级。推迟到 Plan B。|

## 永久 SKIP（写入 SOP 黑名单）

- `48e038f7` channels Discord mention-only — Noldus 不用
- `0d1053ca` Windows uploads symlink — Linux 部署不用
- `2b1fcb3e` 删 max_turns — 与本地硬限制冲突

## 推迟到第 4 轮（Plan B）

- `c1b7f1d1` + `08ee7ade` + `881ff712` + `f76e4e35` — DynamicContextMiddleware 链（改 prompt.py，等 spec-phase-1 合 dev）
- `de253e4a` model_name 链路 — Alembic migration
- `68d8caec` update_agent user_id — EVAL
- `94da8f67` serve.sh + uv extras — 与本地 30→90s 冲突
- `bedbf229` mcp async sync wrapper — 受保护 mcp/tools.py
- `30a58462` write_file --append — 受保护 sandbox/tools.py
- `5127f08e` token usage 默认开 — EVAL
- `1c96a6af` bootstrap user scope — EVAL

## 验证 Gate（每 task）

- `cd packages/agent/backend && make test`（pre-existing 7 失败除外，不引入新失败）
- `make lint`
- 改 frontend：`cd packages/agent/frontend && pnpm typecheck && pnpm test`
- 改 sandbox/path：手动 `make dev` 看 ImportError

## 收尾验证（合 dev 前）

- Dogfood EPM 数据端到端：lead 调 prep_metric_plan → 4 subagent 跑到 → 报告生成
- G4 阶段播报 4 stage 都出现
- 训练数据飞轮录到 `.deer-flow/training-data/auto-collected/`
