# DeerFlow 上游 Tier 2/3/4 同步 - 轮 3 设计文档(better-auth)

**日期**: 2026-05-07
**状态**: 已批准,待实施
**前置依赖**: [2026-05-07-deerflow-tier234-round2-completed-handoff.md](../../handoffs/2026-05-07-deerflow-tier234-round2-completed-handoff.md)

---

## 0. 目标

按"上游能用就用、不重复造轮子"原则,本轮把上游 better-auth 多用户体系合入 noldus,接通轮 2 已铺好的 persistence/user_context 桩,得到一个**可登录、可隔离、可生产部署**的 EthoInsight v0.1 多用户研究助手版本。

合入的上游 commit(8 个,合并到 4 个 phase):

| Phase | 上游 commit | 内容 |
|---|---|---|
| **G** 后端 auth | `94eee95f` (后端核心 + 部分 wiring) + `da174dfd` (Gateway internal auth + CSRF) + `4e4e4f92` (安全加固) + `78633c69` (agent_name 传 setup_agent) + `ed9ebfac` (require_auth 强 request) + `88d47f67` (nginx auth route) | 后端 auth 模块 + 中间件 + LangGraph hook + lifespan ensure_admin |
| **H** 前端 auth + setup wizard | `848ace98` (first-boot wizard) + `94eee95f` (前端 core/auth) + `98a5b34f` (better-auth npm 库拆除) | 拆 better-auth + 接 core/auth + 登录页 + 设置页 + 安装向导 |
| **I** 品牌中文化 | (无上游对应) | EthoInsight 品牌字符串 + auth namespace 中文翻译 |
| **J** 部署文档收尾 | (无上游对应) | Multi-user 部署 SOP + 阿里云 PG 配置示例 + 弃用文档更新 |

总计 **20-25h**, 产出 **4 个 commit**, 不 push origin。

**显式不做**:
- 阿里云 Postgres 实际部署(只产文档,不实际部署)
- LangGraph PostgresSaver 切换(thread state 持久化, 留轮 4)
- Sandbox 切 AioSandbox 容器化(留轮 4)
- 前端 auth 视觉细节优化(等用户反馈再调)
- Multi-user 性能压测(10 人并发, 留轮 4)

---

## 1. 关键决策

### 1.1 不重复造轮子,直接用上游

上游 `94eee95f` 已经实现完整 auth 体系,包括 JWT 签发/验证、bcrypt 密码哈希(SHA-256 预哈希防 72 字节截断)、SQLite/Postgres UserRepository、Provider Factory 模式、reset_admin CLI、12 个单元测试。`848ace98` 加了 first-boot setup wizard。`da174dfd` 加了 process-local internal auth + CSRF。`98a5b34f` 拆除前端 better-auth npm 依赖。`4e4e4f92` 做安全加固(SHA-256 预哈希、错误信息脱敏、setup-status rate limit、auto-migrate 旧密码哈希)。

**结论**: 全套照拉,无需 surgical merge(noldus 后端 auth 一片空白)。前端有 4 个 `server/better-auth/` 文件但**没有 component 真用过**(已 grep 确认),实质单用户模式,直接拆除 + 接上游 `core/auth/`。

### 1.2 commit 拆分:按"功能完整性"而非"上游边界"

轮 1/轮 2 是按上游 commit 一一对应拆,因为 noldus 端有大量定制需要 surgical 保留。轮 3 不同——上游 auth 代码我们不改,所以**按"完整可工作的 unit"拆**:

| Commit | 工时 | 内容 |
|---|---|---|
| 轮 3 后端 (G) | 12-15h | 一次合 5+ 个上游 commit 的所有后端代码,deps 装好,启动通,后端测试 ≥ 1980 passed |
| 轮 3 前端 (H) | 6-8h | 一次合所有前端代码,better-auth 拆除,登录页通,前端 typecheck 0 error |
| 轮 3 品牌 (I) | 3-4h | EthoInsight 品牌 + 中文翻译追加 |
| 轮 3 部署文档 (J) | 1-2h | Multi-user 部署 SOP + 阿里云 PG 示例 + 4-23 文档收尾 |

每 commit 内部仍按 task 步骤推进,但**所有 task 完成才 commit**——避免半成品状态污染 git 历史。

### 1.3 数据库后端:本地 SQLite,生产阿里云 Postgres

- **本地开发**: `database.backend: sqlite`, 数据落 `.deer-flow/data/deerflow.db`
- **生产**: `database.backend: postgres`, `database.postgres_url: $DATABASE_URL`,部署时 env 注入阿里云 PG connection string `postgresql+asyncpg://user:pass@rm-xxx.pg.rds.aliyuncs.com:5432/ethoinsight`
- **轮 3 实施**: 默认 sqlite, Postgres 配置只在 config.example.yaml 给示例,实际部署在 J phase 的 SOP 中说明

### 1.4 受保护的不是文件,是语义内容

轮 3 的 surgical merge 判定基准是 **grep 关键字残量**, 不是 diff 行数。新 agent 可以重构、拆分、合并文件,只要受保护的语义内容仍在(grep 找得到、行为仍生效),就是合格的。

---

## 2. 架构变化

### 2.1 后端架构(轮 3 后)

```
backend/
├── app/
│   ├── gateway/
│   │   ├── auth/                          ✨ 新增 (上游来)
│   │   │   ├── __init__.py               # exports
│   │   │   ├── config.py                 # AuthConfig: jwt_secret, token_expire, ...
│   │   │   ├── credential_file.py        # /etc 下凭证文件读写
│   │   │   ├── errors.py                 # AuthErrorCode + AuthErrorResponse
│   │   │   ├── jwt.py                    # create_access_token / decode_token
│   │   │   ├── local_provider.py         # LocalAuthProvider 主入口
│   │   │   ├── models.py                 # User pydantic + DB row 类
│   │   │   ├── password.py               # bcrypt + SHA-256 预哈希(72 字节防截断)
│   │   │   ├── providers.py              # AuthProvider ABC + factory
│   │   │   ├── repositories/
│   │   │   │   ├── base.py              # UserRepository ABC
│   │   │   │   └── sqlite.py            # SqliteUserRepository
│   │   │   └── reset_admin.py           # CLI 工具
│   │   ├── auth_middleware.py        ✨ 新增 (JWT 验证 + 401 + ed9ebfac/4e4e4f92)
│   │   ├── csrf_middleware.py        ✨ 新增 (Double-submit cookie, da174dfd)
│   │   ├── internal_auth.py          ✨ 新增 (process-local internal token, da174dfd)
│   │   ├── langgraph_auth.py         ✨ 新增 (langgraph.json hook)
│   │   ├── deps.py                   ✏ 改: get_current_user_from_request 等
│   │   ├── app.py                    ✏ 改: register middlewares + lifespan ensure_admin
│   │   ├── authz.py                  ✨ 新增 (require_auth 装饰器 + owner_filter)
│   │   └── routers/
│   │       ├── auth.py              ✨ 新增 (register/login/me/setup-status, 848ace98)
│   │       ├── threads.py           ✏ 改: thread.owner_id 过滤
│   │       ├── thread_runs.py       ✏ 改: owner check
│   │       ├── uploads.py           ✏ 改: owner check
│   │       ├── artifacts.py         ✏ 改: owner check
│   │       ├── feedback.py          ✏ 改: owner check
│   │       └── suggestions.py       ✏ 改: owner check
│   └── channels/manager.py          ✏ 改: 用 internal_auth 调 LangGraph (da174dfd)
└── packages/harness/deerflow/
    └── persistence/user/             ✨ 已存在 (D 阶段铺骨架, 轮 3 接通 UserRepository)
```

### 2.2 前端架构(轮 3 后)

```
frontend/
├── src/
│   ├── app/
│   │   ├── (auth)/                   ✨ 新增路由组
│   │   │   ├── layout.tsx
│   │   │   ├── login/page.tsx
│   │   │   └── setup/page.tsx
│   │   ├── workspace/                ✏ 改: 接 AuthProvider
│   │   └── api/auth/[...all]/       ❌ 删 (better-auth catch-all)
│   ├── core/
│   │   ├── auth/                     ✨ 新增 (上游来)
│   │   │   ├── AuthProvider.tsx
│   │   │   ├── gateway-config.ts
│   │   │   ├── proxy-policy.ts
│   │   │   ├── server.ts
│   │   │   └── types.ts
│   │   ├── api/
│   │   │   ├── api-client.ts        ✨ 新增 (带 JWT header)
│   │   │   └── fetcher.ts           ✨ 新增 (401 → redirect /login)
│   │   └── i18n/locales/zh-CN.ts    ✏ 追加 auth namespace
│   ├── components/
│   │   ├── workspace/settings/
│   │   │   └── account-settings-page.tsx  ✨ 新增 (改密码)
│   │   └── landing/hero.tsx        ✏ 不动 (EthoInsight 已就绪)
│   └── server/better-auth/          ❌ 整目录删
└── package.json                     ✏ 改: 删 better-auth + 加 fetcher 依赖
```

### 2.3 数据流(登录 → 调 agent)

```
1. 前端 → POST /api/auth/login {email, password}
2. Gateway auth router → LocalAuthProvider.authenticate
   → SHA-256 预哈希 → bcrypt.checkpw + JWT signing
   → returns {access_token, user}
3. 前端 store token (httpOnly cookie + AuthProvider context)
4. 前端 → POST /api/threads/{id}/runs (Authorization: Bearer xxx)
5. AuthMiddleware → decode JWT → set request.state.user
6. langgraph_auth.py hook (langgraph.json 注入) → set thread.metadata.owner_id
7. ThreadDataMiddleware → 用 metadata.owner_id 设 runtime.user_context
8. Agent 跑 → 所有 IO 走 /users/{user_id}/threads/{tid}/...
```

### 2.4 与轮 2 已就绪部分的关系

轮 2 D 阶段已经铺好的桩:
- ✅ `persistence/user/model.py` UserRow ORM model 文件存在
- ✅ `runtime/user_context.py` `set_current_user / get_effective_user_id` 已能用
- ✅ `paths.py` 全套 `user_id` 参数已支持
- ✅ memory storage / repositories 已 user-aware
- ✅ `client.py` IO 调用已传 user_id(轮 2 收尾修复)

轮 3 做"接通":
- 引入 `app/auth/LocalAuthProvider` 写入 user 表
- AuthMiddleware 把 JWT user 写入 `runtime.user_context` contextvar
- LangGraph hook 把 `request.state.user.id` 写入 `thread.metadata.owner_id`
- ThreadDataMiddleware 改用 `metadata.owner_id` 而非默认值

---

## 3. 受保护语义清单

**核心规则**: 受保护的是**语义内容**,不是文件本身。新 agent 可以改文件结构,只要 grep 能找到关键字、行为仍生效,就是合格的。

### 3.1 必须保留的 noldus 价值内容

| 受保护语义 | 物理位置(目前) | grep 验证命令 | 期望残量 |
|---|---|---|---|
| 中文调度规则 + Gate 反问 + EV19 模板路径 + 4 个 ethoinsight 子代理描述 | `agents/lead_agent/prompt.py` | `grep -c "中文\|按以下\|EV19\|Gate" prompt.py` | ≥ 5 |
| 4 个 ethoinsight 子代理注册 | `subagents/builtins/__init__.py` | `grep -c "code-executor\|data-analyst\|report-writer\|knowledge-assistant" __init__.py` | ≥ 4 |
| 中间件链(Archiving/ThinkTag/TrainingData/GateEnforcement/LoopDetection) | `agents/lead_agent/agent.py` | `grep -c "ArchivingSummarizationMiddleware\|ThinkTagMiddleware\|TrainingDataMiddleware\|GateEnforcementMiddleware\|LoopDetectionMiddleware" agent.py` | ≥ 5 |
| `recursion_limit` + `max_turns` + `{{shared://}}` | `subagents/executor.py` | `grep -c "recursion_limit\|max_turns\|shared://" executor.py` | ≥ 5 |
| `{{shared://}}` + `SHARED_PATH_PREFIX` + `mask_local_paths_in_output` | `sandbox/tools.py` | `grep -c "{{shared://}}\|SHARED_PATH_PREFIX\|mask_local_paths_in_output" tools.py` | ≥ 3 |
| 总超时 + circuit breaker | `llm_error_handling_middleware.py` | `grep -c "circuit_breaker\|total_timeout" llm_error_handling_middleware.py` | ≥ 1 |
| 4096 字符截断 | `mcp/tools.py` | `grep -c "4096\|truncate" mcp/tools.py` | ≥ 1 |
| `BeforeSummarizationHook` + memory_flush hook + archive hook | `summarization_middleware.py` | `grep -c "BeforeSummarizationHook" summarization_middleware.py` | ≥ 1 |
| `/mnt/shared` + `shared_dir()` + `shared_path` 字段 | `config/paths.py` + `thread_state.py` + `thread_data_middleware.py` | `grep -c "/mnt/shared\|shared_dir\|shared_path"` (合并三文件) | ≥ 5 |
| 5 个 ethoinsight skill 的 markdown 内容 | `packages/agent/skills/custom/` | `ls packages/agent/skills/custom/ \| wc -l` | = 5 |
| EthoInsight 品牌 | `frontend/src/components/landing/hero.tsx` | `grep -c "EthoInsight\|EthoVision" hero.tsx` | ≥ 2 |
| 中文 i18n locale | `frontend/src/core/i18n/locales/zh-CN.ts` | `grep -c "zhCN\|常用\|首页\|设置"` | ≥ 4 |

### 3.2 不动的内容(完全不修改)

- 5 个 ethoinsight custom skill 的 markdown(`compaction-recovery / ethoinsight / ethoinsight-code / ethoinsight-charts / ethoinsight-planning`)
- `agents/lead_agent/prompt.py` 中文调度规则一字不动
- `subagents/builtins/__init__.py` 4 个 ethoinsight 子代理注册不动
- 中间件链顺序不动

### 3.3 可改的语义(轮 3 实施时,但要保证 grep 残量达标)

| 文件 | 轮 3 改动 | 改后 grep 验证 |
|---|---|---|
| `app/gateway/app.py` | 追加 AuthMiddleware + CSRFMiddleware 注册 + lifespan ensure_admin hook | `grep -c "AuthMiddleware\|CSRFMiddleware\|ensure_admin" app.py` ≥ 3 |
| `app/gateway/deps.py` | 追加 `get_current_user_from_request` 等 dep | `grep -c "get_config\|get_current_user_from_request" deps.py` ≥ 2 |
| `app/gateway/routers/threads.py` | 加 `owner_filter` 过滤 + `metadata['owner_id']` | 改后 noldus 定制残量: `grep -c "shared_path\|noldus\|EthoInsight"` 与改前一致 |
| `agents/middlewares/memory_middleware.py` | 用 `metadata['owner_id']` 注入 user_context | `grep -c "agent_name\|user:" memory_middleware.py` 改前=改后 |
| `langgraph.json` | 加 `auth: {path: "...langgraph_auth.py:auth"}` 字段 | `grep -c "auth" langgraph.json` ≥ 1 |
| `frontend/package.json` | 删 `better-auth: "^1.3"` 一行 | `grep -c "better-auth" package.json` = 0 |
| `frontend/src/env.js` | 删 4 行 BETTER_AUTH env | `grep -c "BETTER_AUTH" env.js` = 0 |
| `frontend/src/app/workspace/layout.tsx` | 包 `<AuthProvider>` | `grep -c "AuthProvider" layout.tsx` ≥ 1,noldus 现有 UI 改动 grep 残量不变 |
| `frontend/src/core/i18n/locales/zh-CN.ts` | **追加** auth namespace,**不重写** | `grep -c "auth:" zh-CN.ts` ≥ 1,noldus 现有 keys 不丢 |
| `config.yaml` | 加 `auth: {enabled: true, ...}` + `database: {backend: sqlite, ...}` | `grep -c "^auth:\|^database:" config.yaml` ≥ 2 |
| `backend/pyproject.toml` | 追加 deps: `bcrypt / pyjwt / sqlalchemy / asyncpg / aiosqlite / alembic / email-validator / itsdangerous` | `grep -c "bcrypt\|pyjwt\|sqlalchemy" pyproject.toml` ≥ 3 |

### 3.4 整文件覆盖 OK 的(无 noldus 定制)

| 文件 | 来源 |
|---|---|
| `app/gateway/auth/*` 全部(11 个文件) | 上游 94eee95f / 4e4e4f92 |
| `app/gateway/auth_middleware.py` | 上游 94eee95f / 4e4e4f92 |
| `app/gateway/csrf_middleware.py` | 上游 da174dfd |
| `app/gateway/internal_auth.py` | 上游 da174dfd |
| `app/gateway/langgraph_auth.py` | 上游 94eee95f |
| `app/gateway/authz.py` | 上游 94eee95f / ed9ebfac |
| `app/gateway/routers/auth.py` | 上游 94eee95f |
| `frontend/src/core/auth/*` | 上游 94eee95f |
| `frontend/src/core/api/api-client.ts` + `fetcher.ts` | 上游 94eee95f |
| `frontend/src/app/(auth)/*` | 上游 94eee95f / 848ace98 |
| `frontend/src/components/workspace/settings/account-settings-page.tsx` | 上游 94eee95f |
| `frontend/pnpm-lock.yaml` | 上游 98a5b34f(整覆盖,不本地 lock) |
| `nginx.local.conf` 加 catch-all 路由 | 上游 88d47f67(可整覆盖,无 noldus 定制) |
| `backend/uv.toml` | 上游 94eee95f(pinned PyPI index) |
| 上游所有新增 test_*.py(`test_auth.py / test_auth_config.py / test_ensure_admin.py / test_owner_isolation.py / ...`) | 上游 |

---

## 4. 测试策略

### 4.1 起点 / 终点

- **起点**: `2 failed, 1877 passed, 14 skipped`(轮 2 末尾)
- **终点**: `2 failed, ≥1980 passed, 14 skipped`(上游 5 个 auth commit 共带来 ~100 个新测试)

### 4.2 每 commit 验证标准

| Commit | 必过测试 | 必过 grep | 启动 smoke |
|---|---|---|---|
| **G 后端 auth** | `test_auth_config / test_auth_errors / test_auth / test_auth_middleware / test_auth_type_system / test_ensure_admin / test_langgraph_auth / test_owner_isolation / test_user_context` 全过 + 全量 ≤ 2 failed,passed ≥ 1980 | 受保护语义清单(§3.1)全部达标 | `make dev` 起来,`curl /api/auth/setup-status` 返 200 |
| **H 前端 auth** | 后端基线不变 + `pnpm typecheck` 0 error + `pnpm lint` 0 error | hero.tsx EthoInsight ≥ 2、zh-CN.ts 中文 keys ≥ 100 | `make dev` 浏览器看到登录页 |
| **I 品牌中文** | 后端基线不变 + 前端 typecheck 0 error | hero.tsx EthoInsight ≥ 2,zh-CN.ts auth namespace 完整,**前端代码中 "DeerFlow" 字符串残量 = 0** | 浏览器登录页全中文 + EthoInsight |
| **J 部署文档** | 不跑代码测试 | docs SOP 文件存在,含阿里云 PG 配置示例 | N/A |

### 4.3 关键回归测试(每 commit 必跑)

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -3
```

**< 1877 passed** 立即停。

### 4.4 端到端验证(用户手动做,不在测试套件)

| 场景 | 验证目标 | 触发 commit |
|---|---|---|
| 首次启动看到 setup wizard | 创建 admin | H |
| 登录后跳转 workspace | session 持久化 | H |
| 上传文件、跑分析、生成 artifact | user_id 隔离 | G+H 后 |
| 改密码 / 退出登录 | account-settings 工作 | H |
| 浏览器 console 无 better-auth 报错 | 旧库彻底拆 | H |
| 登录页 EthoInsight + 中文 | 品牌适配 | I |
| Channels (slack/feishu/telegram) 可发消息 | internal_auth 工作 | G |
| 阿里云 PG 连接(有部署机器时) | postgres backend | J |

### 4.5 已知 pre-existing failures(绝不修)

- `test_ethoinsight_planning_skill.py::test_planning_skill_is_enabled_in_config`
- `test_lead_prompt_interactive_pipeline.py::TestDefaultPipelineIsTwoStep::test_usage_example_shows_ask_clarification_between_analyst_and_writer`

### 4.6 Lint

```bash
cd packages/agent/backend
PYTHONPATH=. uv run ruff check app/gateway/auth/ app/gateway/ packages/harness/deerflow/persistence/user/ 2>&1 | tail -3
```

每 commit 后 0 error。

---

## 5. 风险与回滚

### 5.1 高风险点

| # | 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|---|
| 1 | `_migrate_orphaned_threads` 把现有 thread 全错配给 admin user | 中 | 用户已有分析 thread"消失"或归属错乱 | 先 dry-run 写迁移脚本,跑前用户 `cp -r .deer-flow .deer-flow.bak.$(date)` |
| 2 | `pyproject.toml` 加 deps 后 `uv sync` 与 `langgraph>=1.0.6,<1.0.10` 版本冲突 | 中 | 启动失败 | 加完 deps 立即 `uv lock --check`,有 conflict 立即停下问用户 |
| 3 | 前端 `pnpm install` 删 better-auth 后 lock 大 diff | 高 | PR review 困难 | 用上游 `pnpm-lock.yaml` 整覆盖,不本地重 lock |
| 4 | AuthMiddleware 注册顺序错(在 CORS 之后)→ OPTIONS preflight 失败 | 中 | 前端调任何 API 都报 CORS error | 严格按上游 `app.py` 顺序: CORS → Auth → CSRF → 业务 |
| 5 | langgraph.json 加 auth hook 后老 thread 没有 metadata.owner_id → 列表为空 | 中 | 用户登录后看不到历史 thread | `_migrate_orphaned_threads` lifespan helper 把没有 owner_id 的 thread 归给 admin |
| 6 | nginx `/api/auth/*` 路由没加 catch-all → login 404 | 低 | 登录失效 | `88d47f67` 补丁的 `/api/` catch-all 必须合 |
| 7 | Channels 用旧 LangGraph SDK 路径,被 AuthMiddleware 401 | 中 | IM 集成失效 | `internal_auth.py` 必拉,channels/manager.py 改用 internal token |
| 8 | gateway/routers/threads.py surgical merge 丢 noldus 定制(`shared_path` / 训练数据 hook 等) | 高 | thread 创建后 sandbox 路径不对 | 改前 `cp threads.py /tmp/threads-backup.py`,改后 grep 验证 noldus 定制残量 |
| 9 | 前端 `core/auth/AuthProvider` 包 `workspace/layout.tsx` 时 break noldus 现有的 layout 改动 | 中 | UI 错乱或 SSR hydration error | surgical: 只在 layout 顶层包 Provider,不动 noldus 内部 children 结构 |

### 5.2 回滚机制

```bash
# 单 commit 回滚(每 phase 独立)
git revert <commit-sha>

# 完整回滚到轮 2 末尾(本 spec 写时)
git reset --hard ed33afee
```

#### G commit 回滚特殊性

G 是最大 commit(预计 +5000/-2000 行)。出问题时 **revert 重做** 比中途修改安全——逻辑分散在 11+ 文件,半成品调试时间会比 revert 重做更久。

#### 数据回滚

```bash
# G commit 前必须备份
cp -r packages/agent/backend/.deer-flow packages/agent/backend/.deer-flow.bak.round3

# 回滚后恢复
rm -rf packages/agent/backend/.deer-flow
mv packages/agent/backend/.deer-flow.bak.round3 packages/agent/backend/.deer-flow
```

### 5.3 中途中断协议

某 phase 卡 4h+ 测试不通过:
1. `git stash` 当前未提交
2. 写中断 handoff: `docs/handoffs/2026-XX-XX-tier234-round3-INTERRUPTED.md`,记录:phase / 已 commit / 卡点 stack trace / 已尝试 debug
3. **不 commit 半成品**
4. 通知用户

### 5.4 不要做的事

- ❌ 不 push origin/dev (留 dev 等用户决定)
- ❌ 不修 2 个 pre-existing failures
- ❌ 不动 5 个 ethoinsight custom skill markdown
- ❌ 不动中文 prompt / Gate 反问 / EV19 模板路径
- ❌ 不重写 zh-CN.ts(只追加 auth namespace)
- ❌ 不重新生成 pnpm-lock.yaml(用上游)
- ❌ 不 `uv lock --upgrade-package langgraph` 自作主张

### 5.5 完成定义

- [ ] 4 个 commit 在本地 dev 分支
- [ ] 测试: `2 failed, ≥1980 passed, 14 skipped`
- [ ] `make lint` 0 error(后端) + `pnpm check` 0 error(前端)
- [ ] `make dev` 启动成功
- [ ] 浏览器看到登录页(EthoInsight + 中文)
- [ ] Setup wizard 能创建 admin
- [ ] 登录后跳 workspace,能跑分析
- [ ] 受保护语义清单(§3.1)grep 全部达标
- [ ] 写新 handoff `docs/handoffs/2026-XX-XX-deerflow-tier234-round3-completed-handoff.md`
- [ ] **不 push** — 留 dev 等用户决定

---

## 6. 与轮 4 衔接

轮 3 完成后,留给轮 4 的事:
1. **阿里云 Postgres 实际部署**(本轮只产文档,不实施)
2. **LangGraph PostgresSaver 切换**(thread state 也持久化到 PG)
3. **Sandbox 切 AioSandboxProvider**(每用户容器隔离,4-23 计划提过)
4. **Multi-user 性能压测**(10 人并发)
5. **前端 auth 视觉细节优化**(等用户反馈再调)

---

## 7. 参考

- 轮 2 完成 handoff: `docs/handoffs/2026-05-07-deerflow-tier234-round2-completed-handoff.md`
- 轮 2 设计 spec: `docs/superpowers/specs/2026-05-07-deerflow-tier234-round2-design.md`
- 弃用的 user-backend 计划: `docs/plans/2026-04-23-multi-user-deployment.md`
- 105 commit 分类: `docs/handoffs/2026-05-07-tier234-walkthrough-data/all-105-commits-classified.txt`
- noldus 上游同步规则: `CLAUDE.md` L123-L174
- 上游核心 commit:
  - `94eee95f` feat(auth): release-validation pass for 2.0-rc — 12 blockers + simplify follow-ups
  - `848ace98` feat: replace auto-admin creation with secure interactive first-boot setup
  - `da174dfd` feat: implement process-local internal authentication for Gateway and enhance CSRF handling
  - `98a5b34f` fix: resolve merge conflict in pnpm-lock.yaml and clean up better-auth dependencies
  - `4e4e4f92` fix(security): harden auth system and fix run journal logic bug
  - `78633c69` fix(agents): propagate agent_name into ToolRuntime.context for setup_agent
  - `88d47f67` fix(nginx): add catch-all /api/ location for auth routes
  - `ed9ebfac` fix: enforce 'request' parameter requirement in require_auth decorator
