# DeerFlow 上游 Tier 2/3/4 同步 - 轮 3 实施 Plan (better-auth)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把上游 better-auth 多用户体系合入 noldus,接通轮 2 已铺好的 persistence/user_context 桩,得到一个可登录、可隔离、可生产部署的 EthoInsight v0.1 多用户研究助手版本。

**Architecture:** 4 个 phase 4 个 commit:G 后端 auth(一次合 5+ 个上游 commit 的所有后端代码,deps 装好,启动通,后端测试 ≥ 1980 passed)→ H 前端 auth(拆 better-auth + 接 core/auth + 登录页)→ I 品牌中文化(EthoInsight + 中文翻译追加)→ J 部署文档收尾。每 commit 内仍按 task 步骤推进,但**所有 task 完成才 commit**——避免半成品状态污染 git 历史。

**Tech Stack:** Python 3.12 / pytest / SQLAlchemy 2.0 + asyncpg + aiosqlite (persistence) / bcrypt + pyjwt + email-validator + itsdangerous (auth) / Alembic (migrations) / Next.js + React (frontend) / pnpm

**前置依赖:**
- 当前 HEAD 应为 `da8df421 docs: 修正轮 3 spec 中 auth 路径` 或更新(轮 3 spec 已 commit)
- 轮 2 末尾测试基线: `2 failed, 1877 passed, 14 skipped`
- 设计文档: `docs/superpowers/specs/2026-05-07-deerflow-tier234-round3-design.md`

**关键路径:**
```
NOLDUS=/home/wangqiuyang/noldus-insight
BACKEND=$NOLDUS/packages/agent/backend
HARNESS=$BACKEND/packages/harness/deerflow
GATEWAY=$BACKEND/app/gateway
TESTS=$BACKEND/tests
FRONTEND=$NOLDUS/packages/agent/frontend
SKILLS_CUSTOM=$NOLDUS/packages/agent/skills/custom
```

**通用测试命令:**
```bash
cd $BACKEND
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

**已知 pre-existing failures(绝对不要修):**
- `test_ethoinsight_planning_skill.py::test_planning_skill_is_enabled_in_config`
- `test_lead_prompt_interactive_pipeline.py::TestDefaultPipelineIsTwoStep::test_usage_example_shows_ask_clarification_between_analyst_and_writer`

**Noldus 受保护语义清单(任何时候都必须 grep 验证残量):**

| 文件 | grep 命令 | 期望残量 |
|---|---|---|
| `agents/lead_agent/prompt.py` | `grep -c "中文\|按以下\|EV19\|Gate" $HARNESS/agents/lead_agent/prompt.py` | ≥ 5 |
| `agents/lead_agent/agent.py` | `grep -c "ArchivingSummarizationMiddleware\|ThinkTagMiddleware\|TrainingDataMiddleware\|GateEnforcementMiddleware\|LoopDetectionMiddleware" $HARNESS/agents/lead_agent/agent.py` | ≥ 5 |
| `subagents/executor.py` | `grep -c "recursion_limit\|max_turns\|shared://" $HARNESS/subagents/executor.py` | ≥ 5 |
| `subagents/builtins/__init__.py` | `grep -c "code-executor\|data-analyst\|report-writer\|knowledge-assistant" $HARNESS/subagents/builtins/__init__.py` | ≥ 4 |
| `sandbox/tools.py` | `grep -c "{{shared://}}\|SHARED_PATH_PREFIX\|mask_local_paths_in_output" $HARNESS/sandbox/tools.py` | ≥ 3 |
| `config/paths.py` | `grep -c "/mnt/shared\|shared_dir" $HARNESS/config/paths.py` | ≥ 4 |
| `agents/middlewares/summarization_middleware.py` | `grep -c "BeforeSummarizationHook" $HARNESS/agents/middlewares/summarization_middleware.py` | ≥ 1 |
| `frontend/src/components/landing/hero.tsx` | `grep -c "EthoInsight\|EthoVision" $FRONTEND/src/components/landing/hero.tsx` | ≥ 2 |
| `skills/custom/` | `ls $SKILLS_CUSTOM \| wc -l` | = 5 |

**General rules:**
- ❌ 永远不要 `git show <sha> > <file>` 整文件覆盖含 noldus 受保护语义的文件
- ❌ 永远不要 push origin 或 deerflow remote
- ❌ 永远不要修 pre-existing failures
- ❌ 永远不要重写 zh-CN.ts(只 append auth namespace)
- ❌ 永远不要 `uv lock --upgrade-package langgraph` 自作主张
- ✅ 每完成一个 task 跑相关单测;每完成一个 phase 跑全量测试 + lint
- ✅ 测试失败数从 2 涨到 3+,**立即停下**,99% 是 mock path 漏改

---

## Phase 0: 基线确认 (Task 0)

### Task 0: 验证起点

**Files:** 无改动,纯验证

- [ ] **Step 1: 验证 git HEAD**

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline -1
```

期望 HEAD 是 `da8df421` (轮 3 spec 修正提交) 或更新的 commit。

如果 HEAD 比 `ed33afee` (轮 2 末尾) 还旧,**立即停下问用户**。

- [ ] **Step 2: 验证工作区干净**

```bash
git status
```

期望: `无文件要提交,干净的工作区`。

如果有 `archiving_summarization.py / thread_data_middleware.py / uploads_middleware.py / test_uploads_middleware_core_logic.py` 4 个文件未提交,**问用户**:
- "这 4 个文件改动是 D.3 漏掉的 user_id surgical merge,需要先 commit/stash 吗?"
- 不要自动 commit,等用户确认。

如果有其他未跟踪改动,**立即停下问用户**。

- [ ] **Step 3: 验证 deerflow remote**

```bash
git remote -v | grep deerflow
git fetch deerflow main 2>&1 | tail -2
```

期望: deerflow remote 存在,fetch 成功。

- [ ] **Step 4: 跑测试基线**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -3
```

期望:
```
2 failed, 1877 passed, 14 skipped
```

如果失败数 ≠ 2 或 passed < 1870,**立即停下问用户**。

- [ ] **Step 5: 验证前端 build 基线**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm install --frozen-lockfile 2>&1 | tail -3
pnpm typecheck 2>&1 | tail -5
```

期望: typecheck 0 error。

如果有 error,记录下来(后续 H phase 改完前端后重跑作对比)。

- [ ] **Step 6: 备份 .deer-flow 数据目录**

```bash
cd /home/wangqiuyang/noldus-insight
DATE=$(date +%Y%m%d-%H%M%S)
cp -r packages/agent/backend/.deer-flow packages/agent/backend/.deer-flow.bak.round3-$DATE
ls -la packages/agent/backend/ | grep .deer-flow
```

期望: `.deer-flow` 和 `.deer-flow.bak.round3-XXX` 都存在。

⚠️ **此备份用于回滚现有 thread/memory 数据**。Phase G 引入 `_migrate_orphaned_threads` 后老 thread 会被全部归给 admin user,如果出问题用此备份恢复。

---

## Phase G: 后端 auth 合入 (Task G.1 - G.13)

> 上游 commit (按合入顺序):
> - `94eee95f` feat(auth): release-validation pass for 2.0-rc — 12 blockers (主体)
> - `da174dfd` feat: process-local internal authentication + CSRF
> - `4e4e4f92` fix(security): harden auth system + run journal logic bug
> - `78633c69` fix(agents): propagate agent_name into ToolRuntime.context
> - `ed9ebfac` fix: enforce 'request' parameter requirement in require_auth decorator
>
> **总冲击文件 (按 commit 94eee95f stat):**
> ```
> 后端新增 (整文件覆盖):
>   app/gateway/auth/{__init__, config, credential_file, errors, jwt,
>     local_provider, models, password, providers, reset_admin}.py
>   app/gateway/auth/repositories/{base, sqlite}.py
>   app/gateway/auth_middleware.py / authz.py / csrf_middleware.py /
>     internal_auth.py / langgraph_auth.py
>   app/gateway/routers/auth.py (418 行)
>
> 后端 surgical (含 noldus 定制或调用方):
>   app/gateway/app.py (158 行 diff)
>   app/gateway/deps.py (97 行 diff)
>   app/gateway/routers/{threads, thread_runs, uploads, artifacts, feedback,
>     suggestions}.py
>   app/channels/manager.py (12 行)
>
> 配置:
>   backend/pyproject.toml (加 deps)
>   backend/uv.toml (PyPI index pin)
>   backend/langgraph.json (加 auth hook)
>   config.yaml / config.example.yaml (加 auth + database)
>
> 测试新增 (整文件覆盖):
>   test_auth.py (654 行) / test_auth_config.py (54 行) /
>   test_auth_errors.py (75 行) / test_auth_middleware.py (222 行) /
>   test_auth_type_system.py (701 行) / test_ensure_admin.py (319 行) /
>   test_langgraph_auth.py (312 行) / test_owner_isolation.py (465 行) /
>   test_user_context.py (69 行)
>
> 测试 surgical:
>   test_artifacts_router.py / test_suggestions_router.py /
>   test_thread_meta_repo.py / test_threads_router.py /
>   test_uploads_router.py / test_channels.py /
>   test_lead_agent_prompt.py / test_title_middleware_core_logic.py
> ```
>
> **noldus 受保护文件:** `gateway/routers/threads.py` (614 行 noldus diff,**必须 surgical merge**)。

### Task G.1: 添加后端依赖

**Files:**
- Modify: `packages/agent/backend/packages/harness/pyproject.toml`
- Modify: `packages/agent/backend/pyproject.toml` (如果有,可能没有)

- [ ] **Step 1: 看上游 pyproject.toml 加了什么**

```bash
cd /home/wangqiuyang/noldus-insight
git show 94eee95f -- 'backend/packages/harness/pyproject.toml' | head -40
```

预期:`bcrypt>=4.0.0`、`pyjwt>=2.9.0`、`email-validator>=2.0.0`、`sqlalchemy>=2.0.36`、`alembic>=1.14.0`、`asyncpg>=0.30.0`、`aiosqlite>=0.20.0`、`itsdangerous>=2.2.0`。

- [ ] **Step 2: 看 noldus 现状**

```bash
grep -E "bcrypt|pyjwt|sqlalchemy|alembic|asyncpg|aiosqlite|email-validator|itsdangerous" \
  packages/agent/backend/packages/harness/pyproject.toml
```

期望: 所有 grep 0 行(D 阶段没合入这些依赖)。

- [ ] **Step 3: 用 Edit tool 把上游加的依赖追加到 noldus pyproject.toml**

按 step 1 看到的版本号,在 noldus `packages/harness/pyproject.toml` 的 `[project] dependencies` 列表中追加(逐行 Edit):

具体加这 8 个依赖到 `dependencies` 列表(已经有的不重复):
```toml
"bcrypt>=4.0.0",
"pyjwt>=2.9.0",
"email-validator>=2.0.0",
"sqlalchemy>=2.0.36",
"alembic>=1.14.0",
"asyncpg>=0.30.0",
"aiosqlite>=0.20.0",
"itsdangerous>=2.2.0",
```

- [ ] **Step 4: 查 backend/pyproject.toml 是否需要也加**

```bash
cat packages/agent/backend/pyproject.toml | grep -E "^name|^dependencies" | head -5
```

如果它是 workspace member 引用 harness package,通常不需要在这里再加。如果它有独立 deps 列表,按 step 3 同样处理。

- [ ] **Step 5: 拉上游 backend/uv.toml(如果存在)**

```bash
git show 94eee95f -- 'backend/uv.toml' 2>&1 | head -10
ls packages/agent/backend/uv.toml 2>&1
```

如果 noldus 端没有 `uv.toml` 但上游有,整文件 拉取:
```bash
git show deerflow/main:backend/uv.toml > packages/agent/backend/uv.toml
```

- [ ] **Step 6: 运行 uv lock --check 看冲突**

```bash
cd packages/agent/backend
uv lock --check 2>&1 | tail -10
```

期望: 列出新依赖会被加入 lock,无 conflict 报错。

如果有 `langgraph` 版本冲突或类似:**立即停下问用户**,不要 `--upgrade-package langgraph`。

- [ ] **Step 7: 跑 uv sync 装包**

```bash
cd packages/agent/backend
uv sync 2>&1 | tail -10
```

期望: 安装成功。

- [ ] **Step 8: 验证 import**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
import bcrypt
import jwt
import sqlalchemy
import alembic
import asyncpg
import aiosqlite
import itsdangerous
from email_validator import validate_email
print('OK all imports')
"
```

期望: `OK all imports`。

任何 ImportError,**立即停下** — 看是哪个包没装上,可能 step 3 写错版本号。

### Task G.2: 拉 app/gateway/auth/ 11 个新文件

**Files:**
- Create: `packages/agent/backend/app/gateway/auth/__init__.py`
- Create: `packages/agent/backend/app/gateway/auth/config.py`
- Create: `packages/agent/backend/app/gateway/auth/credential_file.py`
- Create: `packages/agent/backend/app/gateway/auth/errors.py`
- Create: `packages/agent/backend/app/gateway/auth/jwt.py`
- Create: `packages/agent/backend/app/gateway/auth/local_provider.py`
- Create: `packages/agent/backend/app/gateway/auth/models.py`
- Create: `packages/agent/backend/app/gateway/auth/password.py`
- Create: `packages/agent/backend/app/gateway/auth/providers.py`
- Create: `packages/agent/backend/app/gateway/auth/reset_admin.py`
- Create: `packages/agent/backend/app/gateway/auth/repositories/__init__.py`
- Create: `packages/agent/backend/app/gateway/auth/repositories/base.py`
- Create: `packages/agent/backend/app/gateway/auth/repositories/sqlite.py`

- [ ] **Step 1: 创建目录骨架**

```bash
cd /home/wangqiuyang/noldus-insight
mkdir -p packages/agent/backend/app/gateway/auth/repositories
ls packages/agent/backend/app/gateway/auth/
```

期望: `repositories/` 子目录创建成功,目录中暂无文件。

- [ ] **Step 2: 拉 11 个 auth 文件(整文件,无 noldus 定制)**

```bash
cd /home/wangqiuyang/noldus-insight
GATEWAY=packages/agent/backend/app/gateway

for f in __init__.py config.py credential_file.py errors.py jwt.py \
         local_provider.py models.py password.py providers.py reset_admin.py \
         repositories/__init__.py repositories/base.py repositories/sqlite.py; do
    git show "deerflow/main:backend/app/gateway/auth/$f" \
        > "$GATEWAY/auth/$f" 2>&1 || echo "FAIL: $f"
done

ls -R $GATEWAY/auth/
```

期望: 13 个文件全部就位,无 FAIL 输出。

- [ ] **Step 3: import smoke test**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from app.gateway.auth import config, errors, jwt, local_provider, models, password, providers
from app.gateway.auth.repositories import base, sqlite as sqlite_repo
from app.gateway.auth.credential_file import read_credential_file
print('OK')
"
```

期望: `OK`。

任何 ImportError, **立即停下** — 通常是某文件 import 了 noldus 没有的 sibling, 看错误信息排查。

### Task G.3: 拉 5 个 gateway-level auth 文件

**Files:**
- Create: `packages/agent/backend/app/gateway/auth_middleware.py`
- Create: `packages/agent/backend/app/gateway/authz.py`
- Create: `packages/agent/backend/app/gateway/csrf_middleware.py`
- Create: `packages/agent/backend/app/gateway/internal_auth.py`
- Create: `packages/agent/backend/app/gateway/langgraph_auth.py`

- [ ] **Step 1: 拉 5 个文件**

```bash
cd /home/wangqiuyang/noldus-insight
GATEWAY=packages/agent/backend/app/gateway

for f in auth_middleware.py authz.py csrf_middleware.py internal_auth.py langgraph_auth.py; do
    git show "deerflow/main:backend/app/gateway/$f" \
        > "$GATEWAY/$f"
done

wc -l $GATEWAY/auth_middleware.py $GATEWAY/authz.py $GATEWAY/csrf_middleware.py \
       $GATEWAY/internal_auth.py $GATEWAY/langgraph_auth.py
```

期望:
- auth_middleware.py ~135 行
- authz.py ~262 行
- csrf_middleware.py ~112 行
- internal_auth.py ~26 行
- langgraph_auth.py ~106 行

- [ ] **Step 2: import smoke test**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from app.gateway.auth_middleware import AuthMiddleware
from app.gateway.authz import require_auth
from app.gateway.csrf_middleware import CSRFMiddleware
from app.gateway.internal_auth import InternalAuthBearer, get_internal_token
from app.gateway.langgraph_auth import auth
print('OK')
"
```

期望: `OK`。

### Task G.4: 拉 routers/auth.py

**Files:**
- Create: `packages/agent/backend/app/gateway/routers/auth.py`

- [ ] **Step 1: 拉 routers/auth.py(整文件,418 行,无 noldus 端历史)**

```bash
cd /home/wangqiuyang/noldus-insight
git show deerflow/main:backend/app/gateway/routers/auth.py \
    > packages/agent/backend/app/gateway/routers/auth.py
wc -l packages/agent/backend/app/gateway/routers/auth.py
```

期望: ~480 行(94eee95f 加 418 + 848ace98 加 ~70)。

- [ ] **Step 2: import smoke test**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from app.gateway.routers.auth import router
print('routes:', [r.path for r in router.routes])
"
```

期望: 列表包含 `/api/auth/login`、`/api/auth/register`、`/api/auth/me`、`/api/auth/logout`、`/api/auth/setup-status`、`/api/auth/setup`。

### Task G.5: surgical merge gateway/deps.py

**Files:**
- Modify: `packages/agent/backend/app/gateway/deps.py`

⚠️ noldus deps.py 在轮 2 收尾时已加 `get_config()` 函数,**不要整覆盖**,只加上游新增的函数。

- [ ] **Step 1: 看上游加了什么**

```bash
cd /home/wangqiuyang/noldus-insight
git show 94eee95f -- 'backend/app/gateway/deps.py' | head -100
```

记录上游加的新函数:`get_local_provider`、`get_current_user_from_request`、`get_optional_user_from_request`(以及可能的 helper)。

- [ ] **Step 2: 看 noldus 现状**

```bash
cat packages/agent/backend/app/gateway/deps.py
```

记录 noldus 现有的:`get_config`、`get_current_user`(轮 2 收尾加的)、其他 dep 函数。

- [ ] **Step 3: 用 Edit 把上游新函数追加到 noldus deps.py**

策略:
- 保留 noldus `get_config()` 函数原样
- 在文件末尾(最后一个函数后)用 Edit 追加上游 step 1 看到的新函数定义,完整复制 import + 函数体
- 顶部 import 区域追加上游需要的 import(注意去重)

具体:用 Edit tool 在 `def get_config` 函数后插入这些新函数,从 step 1 输出复制。

- [ ] **Step 4: 验证 noldus 定制保留**

```bash
grep -c "def get_config" packages/agent/backend/app/gateway/deps.py
grep -c "def get_current_user_from_request\|def get_optional_user_from_request\|def get_local_provider" \
     packages/agent/backend/app/gateway/deps.py
```

期望: 第 1 行 `1`,第 2 行 `≥ 3`(三个新函数都有)。

- [ ] **Step 5: import smoke test**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from app.gateway.deps import get_config, get_current_user_from_request, get_optional_user_from_request, get_local_provider
print('OK')
"
```

期望: `OK`。

### Task G.6: surgical merge gateway/app.py

**Files:**
- Modify: `packages/agent/backend/app/gateway/app.py`

⚠️ noldus app.py 在轮 2 收尾时已加 `app.state.config = cfg`,**不要整覆盖**。需 surgical 加:
1. AuthMiddleware + CSRFMiddleware 注册(注意顺序:CORS → Auth → CSRF → 业务)
2. lifespan hook `_ensure_admin_user`
3. lifespan hook `_migrate_orphaned_threads`
4. auth router 注册

- [ ] **Step 1: 看上游 app.py 完整**

```bash
cd /home/wangqiuyang/noldus-insight
git show 94eee95f -- 'backend/app/gateway/app.py' > /tmp/app_py_94eee95f.diff
git show da174dfd -- 'backend/app/gateway/app.py' > /tmp/app_py_da174dfd.diff
git show 848ace98 -- 'backend/app/gateway/app.py' > /tmp/app_py_848ace98.diff
git show deerflow/main:backend/app/gateway/app.py > /tmp/app_py_upstream.py
wc -l /tmp/app_py_upstream.py packages/agent/backend/app/gateway/app.py
```

记录:上游 app.py 总行数 vs noldus app.py 总行数。

- [ ] **Step 2: 看 noldus 现有定制**

```bash
diff /tmp/app_py_upstream.py packages/agent/backend/app/gateway/app.py | head -100
grep -c "noldus\|EthoInsight\|ethoinsight\|app.state.config" packages/agent/backend/app/gateway/app.py
```

如果 noldus 定制 grep ≥ 1,**必须 surgical merge**。

- [ ] **Step 3: 备份 noldus 现有 app.py**

```bash
cp packages/agent/backend/app/gateway/app.py /tmp/app_py_noldus_backup.py
```

- [ ] **Step 4: 应用上游改动到 noldus app.py(逐处 Edit)**

按 patch /tmp/app_py_94eee95f.diff、/tmp/app_py_da174dfd.diff、/tmp/app_py_848ace98.diff 中所有 `+` 行,用 Edit tool 逐处加入 noldus app.py:

具体改动:
1. 顶部 imports 加: `from app.gateway.auth_middleware import AuthMiddleware`、`from app.gateway.csrf_middleware import CSRFMiddleware`、`from app.gateway.routers import auth as auth_router`、`from app.gateway.auth import _ensure_admin_user_from_credential_file`(具体函数名以上游为准)
2. 在 lifespan 函数中加 `await _ensure_admin_user(app, config)` 调用 + `await _migrate_orphaned_threads(...)` 调用
3. 在 `app = FastAPI(...)` 后,**严格按顺序**: 已有 CORS → 加 `app.add_middleware(AuthMiddleware, ...)` → 加 `app.add_middleware(CSRFMiddleware, ...)` → 已有业务 middleware
4. 在 router include 区加 `app.include_router(auth_router.router, prefix="/api/auth")`(具体 prefix 看上游)
5. 保留 noldus `app.state.config = cfg`(轮 2 收尾加的)

⚠️ **AuthMiddleware 必须在 CORS 之后,CSRFMiddleware 之后,但在业务中间件之前**。顺序错会导致 OPTIONS preflight 失败。

- [ ] **Step 5: 验证 noldus 定制 + 上游新内容都在**

```bash
grep -c "app.state.config\|AuthMiddleware\|CSRFMiddleware\|_ensure_admin_user\|auth_router\|include_router.*auth" \
     packages/agent/backend/app/gateway/app.py
```

期望: ≥ 5。

如果 < 5,看 grep 结果哪一项缺,补上;如 noldus 定制丢失(`app.state.config` grep = 0),**立即 git restore 并重做**。

- [ ] **Step 6: import smoke test**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from app.gateway.app import app
print('OK', app.title)
"
```

期望: `OK <some title>`。

任何 ImportError 或 startup error,**立即停下**,看错误信息排查。

### Task G.7: surgical merge channels/manager.py

**Files:**
- Modify: `packages/agent/backend/app/channels/manager.py`

- [ ] **Step 1: 看上游改动 (da174dfd)**

```bash
cd /home/wangqiuyang/noldus-insight
git show da174dfd -- 'backend/app/channels/manager.py'
```

预期: 12 行改动,加 `from app.gateway.internal_auth import get_internal_token`,在创建 LangGraph SDK client 时传 `Authorization: Bearer <internal_token>`。

- [ ] **Step 2: 用 Edit 应用同样改动**

按 step 1 patch,在 noldus channels/manager.py 中:
1. 顶部加 `from app.gateway.internal_auth import get_internal_token`
2. 在 `client = get_client(...)` 调用前加 token 获取
3. 把 token 通过 `headers={"Authorization": f"Bearer {token}"}` 传入 client

具体行号看 patch + noldus 现状。

- [ ] **Step 3: 验证 noldus 频道定制保留**

```bash
grep -c "feishu\|slack\|telegram" packages/agent/backend/app/channels/manager.py
```

期望: ≥ 3(noldus 现有的频道处理保留)。

- [ ] **Step 4: import smoke test**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from app.channels.manager import ChannelManager
print('OK')
"
```

### Task G.8: surgical merge gateway/routers/threads.py(高风险)

**Files:**
- Modify: `packages/agent/backend/app/gateway/routers/threads.py`

⚠️ **noldus threads.py 614 行 diff(轮 2 已识别),含大量 noldus 定制(shared_path 处理 / 训练数据 hook / 自定义 response 等)。绝对不能整覆盖。**

- [ ] **Step 1: 备份 noldus threads.py**

```bash
cp packages/agent/backend/app/gateway/routers/threads.py /tmp/threads_noldus_backup.py
wc -l /tmp/threads_noldus_backup.py
```

- [ ] **Step 2: 看上游改动**

```bash
cd /home/wangqiuyang/noldus-insight
git show 94eee95f -- 'backend/app/gateway/routers/threads.py' > /tmp/threads_94eee95f.diff
wc -l /tmp/threads_94eee95f.diff
head -100 /tmp/threads_94eee95f.diff
```

预期: 32 行小改动,主要是:
- 加 `from app.gateway.deps import get_current_user_from_request`
- 在 list/get/delete 路由签名加 `current_user: User = Depends(get_current_user_from_request)`
- 把 `metadata['user_id']` 重命名为 `metadata['owner_id']`
- 在 list 函数中加 `owner_filter = current_user.id`

- [ ] **Step 3: 看 noldus threads.py 现状**

```bash
grep -n "def list_threads\|def get_thread\|def delete_thread\|metadata\[.user_id.\]\|metadata\[.owner_id.\]" \
     packages/agent/backend/app/gateway/routers/threads.py
```

记录每个路由函数的行号。

- [ ] **Step 4: 应用上游改动(逐处 Edit)**

对每个上游 patch 中的 `+` 行,用 Edit tool 在 noldus threads.py 对应位置加。具体:

(a) 顶部 imports 加(如还没有):
```python
from app.gateway.deps import get_current_user_from_request
```

(b) 把所有 `metadata["user_id"]` 重命名为 `metadata["owner_id"]`(用 Edit replace_all=true,但仅限 metadata 字典访问,不动 noldus 自定义其他 user_id 用法):

```bash
grep -n 'metadata\["user_id"\]\|metadata\[.user_id.\]' packages/agent/backend/app/gateway/routers/threads.py
```

逐个用 Edit:`metadata["user_id"]` → `metadata["owner_id"]`。

(c) 在 list_threads / get_thread / delete_thread 函数签名加 `current_user` dep,在函数体内加 `owner_filter`:

按上游 patch 提供的具体代码,Edit 加入。

⚠️ **保留 noldus 所有 shared_path / 训练数据 / EthoInsight 相关代码**。改完每处后立即 grep 验证。

- [ ] **Step 5: 验证 noldus 定制 + 新增 owner check 都在**

```bash
NOLDUS_CUSTOMS=$(grep -c "shared_path\|training\|ethoinsight\|EthoInsight" /tmp/threads_noldus_backup.py)
NOLDUS_CUSTOMS_AFTER=$(grep -c "shared_path\|training\|ethoinsight\|EthoInsight" packages/agent/backend/app/gateway/routers/threads.py)
echo "noldus customs before: $NOLDUS_CUSTOMS"
echo "noldus customs after: $NOLDUS_CUSTOMS_AFTER"

OWNER_REFS=$(grep -c "owner_id\|current_user\|get_current_user_from_request" packages/agent/backend/app/gateway/routers/threads.py)
echo "owner refs: $OWNER_REFS"
```

期望:
- `noldus customs after >= noldus customs before`(no 丢失)
- `owner_refs >= 5`(新增 owner check 都在)

如果 customs 减少了,**立即 `cp /tmp/threads_noldus_backup.py packages/agent/backend/app/gateway/routers/threads.py` 重做**。

- [ ] **Step 6: import smoke test**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from app.gateway.routers.threads import router
print('OK', len(router.routes), 'routes')
"
```

期望: 无 ImportError,routes 数量与之前一致。

### Task G.9: surgical merge 其他 routers (thread_runs, uploads, artifacts, feedback, suggestions)

**Files:**
- Modify: `packages/agent/backend/app/gateway/routers/thread_runs.py`
- Modify: `packages/agent/backend/app/gateway/routers/uploads.py`
- Modify: `packages/agent/backend/app/gateway/routers/artifacts.py`
- Modify: `packages/agent/backend/app/gateway/routers/feedback.py`
- Modify: `packages/agent/backend/app/gateway/routers/suggestions.py`

每个 router 改动小(2-15 行),都是加 owner check。

- [ ] **Step 1: 处理 thread_runs.py**

```bash
cd /home/wangqiuyang/noldus-insight
git show 94eee95f -- 'backend/app/gateway/routers/thread_runs.py' | head -40
grep -c "中文\|EthoInsight\|noldus" packages/agent/backend/app/gateway/routers/thread_runs.py
```

如果 noldus 定制 grep = 0 且上游 diff < 50 行,可直接整文件覆盖:
```bash
git show deerflow/main:backend/app/gateway/routers/thread_runs.py \
    > packages/agent/backend/app/gateway/routers/thread_runs.py
```

否则 surgical:按 patch 逐处 Edit 加 owner check。

- [ ] **Step 2-5: uploads.py / artifacts.py / feedback.py / suggestions.py 同样流程**

每个文件:
1. `git show 94eee95f -- 'backend/app/gateway/routers/<file>'` 看 patch
2. `grep -c "中文\|EthoInsight\|noldus" $GATEWAY/routers/<file>` 看 noldus 定制
3. 无定制且小 → 整覆盖;有定制 → surgical

- [ ] **Step 6: import smoke test 全部 routers**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from app.gateway.routers import auth, threads, thread_runs, uploads, artifacts, feedback, suggestions
print('OK')
"
```

### Task G.10: 接通 persistence/user

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/user/__init__.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/user/model.py`
- Possibly: 新增 repository 文件

D 阶段已铺骨架,验证它能用,补缺失的部分。

- [ ] **Step 1: 看 D 阶段 user/ 现状**

```bash
cd /home/wangqiuyang/noldus-insight
ls packages/agent/backend/packages/harness/deerflow/persistence/user/
cat packages/agent/backend/packages/harness/deerflow/persistence/user/__init__.py
cat packages/agent/backend/packages/harness/deerflow/persistence/user/model.py
```

期望: model.py 已含 UserRow ORM 类,__init__.py 有 export。

- [ ] **Step 2: 看上游 user/ 完整结构**

```bash
git show deerflow/main:backend/packages/harness/deerflow/persistence/user/__init__.py
git show deerflow/main:backend/packages/harness/deerflow/persistence/user/model.py
git show deerflow/main:backend/packages/harness/deerflow/persistence/user/ 2>&1 | head -10
```

如果上游 user/ 还有其他文件(repository.py 等 — 由前一步 `git show ... 2>&1 | head -10` 输出列出),按列出的文件名逐个拉:

```bash
# 示例: 假设 step 1 输出列出 noldus 端缺 'repository.py' 和 'sql.py' 两个文件:
for f in repository.py sql.py; do
    git show "deerflow/main:backend/packages/harness/deerflow/persistence/user/$f" \
        > "packages/agent/backend/packages/harness/deerflow/persistence/user/$f"
done
```

⚠️ 实际要拉的文件名以 step 1 的 `git show ... 2>&1 | head -10` 输出为准。如该命令显示 noldus 端已含上游全部文件(diff = 0),跳过此处。

- [ ] **Step 3: 验证 noldus user/ 与上游一致**

```bash
diff <(git show deerflow/main:backend/packages/harness/deerflow/persistence/user/__init__.py) \
     packages/agent/backend/packages/harness/deerflow/persistence/user/__init__.py
diff <(git show deerflow/main:backend/packages/harness/deerflow/persistence/user/model.py) \
     packages/agent/backend/packages/harness/deerflow/persistence/user/model.py
```

期望: 无差异 或 只有微小差异(D 阶段已合上游 head)。

如果差异大,**整覆盖到上游**(无 noldus 定制此模块):
```bash
git show deerflow/main:backend/packages/harness/deerflow/persistence/user/__init__.py \
    > packages/agent/backend/packages/harness/deerflow/persistence/user/__init__.py
git show deerflow/main:backend/packages/harness/deerflow/persistence/user/model.py \
    > packages/agent/backend/packages/harness/deerflow/persistence/user/model.py
```

- [ ] **Step 4: import smoke**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.persistence.user.model import UserRow
print('UserRow:', UserRow)
"
```

期望: `UserRow: <class ...>`。

### Task G.11: 配置文件 (config.yaml, langgraph.json, nginx)

**Files:**
- Modify: `packages/agent/config.yaml`
- Modify: `packages/agent/config.example.yaml`
- Modify: `packages/agent/backend/langgraph.json`
- Modify: `packages/agent/docker/nginx/nginx.local.conf`

- [ ] **Step 1: 看上游 config.example.yaml 加了什么 auth 段**

```bash
cd /home/wangqiuyang/noldus-insight
git show deerflow/main:config.example.yaml | grep -A 15 "^auth:" | head -20
git show deerflow/main:config.example.yaml | grep -A 8 "^database:" | head -10
```

记录上游 `auth:` 段和 `database:` 段的具体字段。

- [ ] **Step 2: 看 noldus config.yaml 现状**

```bash
grep -n "^auth:\|^database:\|^channels:" packages/agent/config.yaml
```

期望: `database:` 段轮 2 D 阶段已加,`auth:` 段不存在。

- [ ] **Step 3: 用 Edit 把 auth: 段追加到 config.yaml**

按 step 1 看到的字段,在 noldus config.yaml 的 `database:` 段后追加:
```yaml
# ============================================================================
# Authentication (轮 3 引入)
# ============================================================================
auth:
  enabled: true
  jwt_secret: $JWT_SECRET                # 部署时通过 env 注入
  token_expire_minutes: 1440             # 24h
  cookie_name: deerflow_auth_token
  csrf_cookie_name: deerflow_csrf_token
  cookie_secure: false                   # 本地 http 开发用 false, 上 https 切 true
  cookie_samesite: lax
  bcrypt_rounds: 12
```

⚠️ **具体字段以上游 config.example.yaml 为准**, step 1 看到的具体内容覆盖此模板。

- [ ] **Step 4: 同步改 config.example.yaml**

把 step 3 的段也加到 config.example.yaml,但 `jwt_secret` 写明文示例(不用 env var):
```yaml
auth:
  enabled: true
  jwt_secret: "your-jwt-secret-here-min-32-chars-please-change"
  ...
```

- [ ] **Step 5: 改 langgraph.json**

```bash
cat packages/agent/backend/langgraph.json
git show deerflow/main:backend/langgraph.json
```

如果上游 langgraph.json 有 `"auth"` 字段,在 noldus langgraph.json 中追加:
```json
"auth": {
  "path": "./app/gateway/langgraph_auth.py:auth"
}
```

⚠️ **不动 noldus 现有的 graphs / dependencies / dockerfile_lines** 等字段。

- [ ] **Step 6: 改 nginx.local.conf (88d47f67 补丁)**

```bash
git show 88d47f67 -- 'docker/nginx/nginx.local.conf' | head -30
grep -c "noldus\|EthoInsight" packages/agent/docker/nginx/nginx.local.conf
```

如果 noldus nginx.local.conf 无定制(grep = 0),整覆盖:
```bash
git show deerflow/main:docker/nginx/nginx.local.conf \
    > packages/agent/docker/nginx/nginx.local.conf
```

否则按 patch surgical merge,加 catch-all `/api/` location 块。

- [ ] **Step 7: 验证 config.yaml 可解析**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run python -c "
from deerflow.config import get_app_config
import os
os.environ.setdefault('JWT_SECRET', 'test-secret-for-validation-only-32chars')
cfg = get_app_config()
print('auth enabled:', cfg.auth.enabled)
print('database backend:', cfg.database.backend)
"
```

期望:
```
auth enabled: True
database backend: memory  (或 sqlite)
```

如果 `AppConfig has no attribute 'auth'` 或类似:**立即停下**,可能 G.10 的 user/ ORM 没接通 AuthConfig 注册到 AppConfig。看 `packages/harness/deerflow/config/app_config.py` 是否需要追加 `auth: AuthConfig = Field(default_factory=AuthConfig)`。

### Task G.12: 拉 9 个新测试 + surgical 8 个测试

**Files:**
- Create: `packages/agent/backend/tests/test_auth.py`
- Create: `packages/agent/backend/tests/test_auth_config.py`
- Create: `packages/agent/backend/tests/test_auth_errors.py`
- Create: `packages/agent/backend/tests/test_auth_middleware.py`
- Create: `packages/agent/backend/tests/test_auth_type_system.py`
- Create: `packages/agent/backend/tests/test_ensure_admin.py`
- Create: `packages/agent/backend/tests/test_langgraph_auth.py`
- Create: `packages/agent/backend/tests/test_owner_isolation.py`
- Create: `packages/agent/backend/tests/test_user_context.py`
- Modify: `packages/agent/backend/tests/test_artifacts_router.py`
- Modify: `packages/agent/backend/tests/test_suggestions_router.py`
- Modify: `packages/agent/backend/tests/test_thread_meta_repo.py`
- Modify: `packages/agent/backend/tests/test_threads_router.py`
- Modify: `packages/agent/backend/tests/test_uploads_router.py`
- Modify: `packages/agent/backend/tests/test_channels.py`
- Modify: `packages/agent/backend/tests/test_lead_agent_prompt.py`
- Modify: `packages/agent/backend/tests/test_title_middleware_core_logic.py`

- [ ] **Step 1: 拉 9 个新测试(整文件,无 noldus 历史)**

```bash
cd /home/wangqiuyang/noldus-insight
TESTS=packages/agent/backend/tests

for t in test_auth.py test_auth_config.py test_auth_errors.py test_auth_middleware.py \
         test_auth_type_system.py test_ensure_admin.py test_langgraph_auth.py \
         test_owner_isolation.py test_user_context.py; do
    git show "deerflow/main:backend/tests/$t" > "$TESTS/$t"
done
wc -l $TESTS/test_auth*.py $TESTS/test_ensure_admin.py $TESTS/test_langgraph_auth.py \
       $TESTS/test_owner_isolation.py $TESTS/test_user_context.py
```

期望: 9 个文件都创建成功,行数与 spec §3.4 列出的接近。

- [ ] **Step 2: 处理 8 个 surgical 测试文件 (循环)**

对每个 surgical 测试文件:

```bash
TEST=test_artifacts_router.py
cd /home/wangqiuyang/noldus-insight
diff <(git show deerflow/main:backend/tests/$TEST) packages/agent/backend/tests/$TEST | head -30
grep -c "中文\|EthoInsight\|ethoinsight" packages/agent/backend/tests/$TEST
```

判定:
- noldus 定制 grep = 0 且 diff < 100 行 → 整覆盖:
  ```bash
  git show deerflow/main:backend/tests/$TEST > packages/agent/backend/tests/$TEST
  ```
- 其他 → surgical edit (加 mock current_user 等)

按此流程处理 8 个: `test_artifacts_router / test_suggestions_router / test_thread_meta_repo / test_threads_router / test_uploads_router / test_channels / test_lead_agent_prompt / test_title_middleware_core_logic`。

⚠️ `test_lead_agent_prompt.py` 是 noldus 价值核心,**先 grep 中文 ≥ 1**。如有 noldus 定制,严格 surgical。

- [ ] **Step 3: 跑新增 auth 测试**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_auth.py tests/test_auth_config.py tests/test_auth_errors.py \
    tests/test_auth_middleware.py tests/test_auth_type_system.py tests/test_ensure_admin.py \
    tests/test_langgraph_auth.py tests/test_owner_isolation.py tests/test_user_context.py \
    -v 2>&1 | tail -20
```

期望: 全过(allowed skipped),~186 passed。

如有 fail,看错误信息:
- `ImportError` → 漏改的 conftest 或 fixture 路径
- `AssertionError` 与 db 相关 → SQLAlchemy 没起来,看 G.11 config 是否对
- `Unauthorized` → AuthMiddleware 没注册或顺序错,看 G.6

- [ ] **Step 4: 跑 surgical 测试**

```bash
PYTHONPATH=. uv run pytest tests/test_artifacts_router.py tests/test_suggestions_router.py \
    tests/test_thread_meta_repo.py tests/test_threads_router.py tests/test_uploads_router.py \
    tests/test_channels.py tests/test_lead_agent_prompt.py tests/test_title_middleware_core_logic.py \
    -v 2>&1 | tail -20
```

期望: 全过。

如有 fail,看 mock 路径是否更新到了 `current_user` dep。

### Task G.13: 跑全量回归 + lint + 启动 smoke

- [ ] **Step 1: 跑全量测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -5
```

期望: `2 failed, X passed, 14 skipped`, **X ≥ 1980**。

如果 X < 1970,**立即停下**,看具体失败原因。常见:
- `test_threads_router.py` 失败 → G.8 surgical 没改 owner_id 重命名
- `test_channels.py` 失败 → G.7 internal_auth 没接入
- 大量 401 错误 → G.6 AuthMiddleware 注册顺序错

- [ ] **Step 2: 跑 lint**

```bash
PYTHONPATH=. uv run ruff check app/gateway/auth/ app/gateway/auth_middleware.py \
    app/gateway/csrf_middleware.py app/gateway/internal_auth.py app/gateway/langgraph_auth.py \
    app/gateway/authz.py app/gateway/routers/auth.py 2>&1 | tail -5
```

期望: `All checks passed!` 或 0 error。

- [ ] **Step 3: 验证 noldus 受保护语义残量**

```bash
HARNESS=packages/harness/deerflow

echo "=== prompt.py ==="
grep -c "中文\|按以下\|EV19\|Gate" $HARNESS/agents/lead_agent/prompt.py

echo "=== executor.py ==="
grep -c "recursion_limit\|max_turns\|shared://" $HARNESS/subagents/executor.py

echo "=== agent.py ==="
grep -c "ArchivingSummarizationMiddleware\|ThinkTagMiddleware\|TrainingDataMiddleware\|GateEnforcementMiddleware\|LoopDetectionMiddleware" $HARNESS/agents/lead_agent/agent.py

echo "=== sandbox/tools.py ==="
grep -c "{{shared://}}\|SHARED_PATH_PREFIX\|mask_local_paths_in_output" $HARNESS/sandbox/tools.py

echo "=== config/paths.py ==="
grep -c "/mnt/shared\|shared_dir" $HARNESS/config/paths.py

echo "=== threads.py noldus customs ==="
grep -c "shared_path\|EthoInsight\|noldus" app/gateway/routers/threads.py

echo "=== app.py noldus state ==="
grep -c "app.state.config" app/gateway/app.py

echo "=== skills/custom/ ==="
ls /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ | wc -l
```

期望残量(全部 ≥ 阈值):
- prompt.py ≥ 5
- executor.py ≥ 5
- agent.py ≥ 5
- sandbox/tools.py ≥ 3
- config/paths.py ≥ 4
- threads.py noldus customs ≥ 1
- app.py app.state.config = 1
- skills/custom/ = 5

任何一项 < 阈值,**立即停下** — surgical merge 中丢失了 noldus 价值,需要重做。

- [ ] **Step 4: 启动 smoke**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop 2>&1 || true
make dev 2>&1 | tee /tmp/g-smoke.log &
sleep 30

curl -s http://localhost:8001/api/auth/setup-status 2>&1 | head -3
curl -s http://localhost:8001/health 2>&1 | head -3

make stop
```

期望:
- `/api/auth/setup-status` 返 JSON,内容含 `{"setup_required": true}`(首次启动) 或 `{"setup_required": false}`
- `/health` 返 `{"status":"ok"}`

如果启动失败,看 `/tmp/g-smoke.log` 排查。最常见:
- `ImportError: cannot import name X` → 漏改的 import,具体看 stack
- `JWT_SECRET environment variable not set` → 启动时设环境变量: `export JWT_SECRET=test-secret-32-chars-minimum-please`
- `database error` → SQLite 数据库初始化失败,检查 `.deer-flow/data/` 目录

- [ ] **Step 5: Commit Phase G**

```bash
cd /home/wangqiuyang/noldus-insight

git add packages/agent/backend/app/gateway/auth/ \
        packages/agent/backend/app/gateway/auth_middleware.py \
        packages/agent/backend/app/gateway/authz.py \
        packages/agent/backend/app/gateway/csrf_middleware.py \
        packages/agent/backend/app/gateway/internal_auth.py \
        packages/agent/backend/app/gateway/langgraph_auth.py \
        packages/agent/backend/app/gateway/app.py \
        packages/agent/backend/app/gateway/deps.py \
        packages/agent/backend/app/gateway/routers/auth.py \
        packages/agent/backend/app/gateway/routers/threads.py \
        packages/agent/backend/app/gateway/routers/thread_runs.py \
        packages/agent/backend/app/gateway/routers/uploads.py \
        packages/agent/backend/app/gateway/routers/artifacts.py \
        packages/agent/backend/app/gateway/routers/feedback.py \
        packages/agent/backend/app/gateway/routers/suggestions.py \
        packages/agent/backend/app/channels/manager.py \
        packages/agent/backend/packages/harness/deerflow/persistence/user/ \
        packages/agent/backend/packages/harness/pyproject.toml \
        packages/agent/backend/uv.toml \
        packages/agent/backend/uv.lock \
        packages/agent/backend/langgraph.json \
        packages/agent/backend/tests/ \
        packages/agent/config.yaml \
        packages/agent/config.example.yaml \
        packages/agent/docker/nginx/nginx.local.conf

git commit -m "$(cat <<'EOF'
sync deerflow upstream Tier 4 G: better-auth 后端 (94eee95f + da174dfd + 4e4e4f92 + 78633c69 + ed9ebfac + 88d47f67)

合入 6 个上游 commit 的所有后端代码:

新增 (整文件覆盖, 无 noldus 历史):
- app/gateway/auth/ 13 个文件 (config / jwt / password / local_provider /
  models / providers / repositories / reset_admin CLI 等)
- app/gateway/auth_middleware.py / authz.py / csrf_middleware.py /
  internal_auth.py / langgraph_auth.py
- app/gateway/routers/auth.py (register/login/me/logout/setup-status/setup, 480 行)
- 9 个 auth 测试 (test_auth* / test_ensure_admin / test_langgraph_auth /
  test_owner_isolation / test_user_context, 共 ~3000 行)

Surgical merge (含 noldus 定制):
- app/gateway/app.py: 注册 AuthMiddleware + CSRFMiddleware (CORS 后顺序),
  lifespan ensure_admin + migrate_orphaned_threads, auth router 注册;
  保留 noldus app.state.config (轮 2 收尾)
- app/gateway/deps.py: 加 get_local_provider / get_current_user_from_request /
  get_optional_user_from_request; 保留 noldus get_config (轮 2 收尾)
- app/gateway/routers/threads.py: surgical 加 owner_filter (614 行 noldus
  diff 全保留, shared_path / training hook 全在)
- app/gateway/routers/{thread_runs,uploads,artifacts,feedback,suggestions}.py:
  加 owner check
- app/channels/manager.py: 用 internal_auth 调 LangGraph

新增依赖:
- bcrypt>=4.0.0 / pyjwt>=2.9.0 / email-validator>=2.0.0 /
  sqlalchemy>=2.0.36 / alembic>=1.14.0 / asyncpg>=0.30.0 /
  aiosqlite>=0.20.0 / itsdangerous>=2.2.0

接通 D 阶段桩:
- persistence/user/ ORM model 已就位 (D 阶段铺), LocalAuthProvider 实装
  SqliteUserRepository, AuthMiddleware 通过 contextvar 写 user_id

config.yaml 加 auth + database 段, langgraph.json 加 auth hook,
nginx.local.conf 加 catch-all /api/ 路由 (88d47f67)

保留 Noldus 全部价值:
- agents/lead_agent/prompt.py 中文调度 + Gate + EV19
- agents/lead_agent/agent.py 中间件链 (Archiving/ThinkTag/TrainingData/
  GateEnforcement/LoopDetection)
- subagents/executor.py recursion_limit + max_turns + {{shared://}}
- sandbox/tools.py {{shared://}} + SHARED_PATH_PREFIX
- config/paths.py /mnt/shared + shared_dir
- 5 个 ethoinsight custom skill 原样

测试: 2 failed, ≥1980 passed, 14 skipped (轮 2 末尾 1877 → 增 ~100)

详见 docs/superpowers/specs/2026-05-07-deerflow-tier234-round3-design.md §2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: 验证 commit**

```bash
git log --oneline -1
git diff --stat HEAD~1 HEAD | tail -3
```

期望: HEAD 是 "sync deerflow upstream Tier 4 G: better-auth 后端...",30+ 文件改动。

---

## Phase H: 前端 auth 合入 (Task H.1 - H.7)

> 上游 commit:
> - `94eee95f` 主体: 拉 frontend/src/core/auth/* + frontend/src/app/(auth)/* + 删 server/better-auth/
> - `848ace98` setup wizard: 改 (auth)/setup/page.tsx + (auth)/login/page.tsx + workspace/layout.tsx
>
> **冲击文件:**
> ```
> 新增:
>   frontend/src/core/auth/{AuthProvider, gateway-config, proxy-policy, server, types}.tsx
>   frontend/src/core/api/{api-client, fetcher}.ts
>   frontend/src/app/(auth)/{layout, login/page, setup/page}.tsx
>   frontend/src/components/workspace/settings/account-settings-page.tsx
>
> 删除:
>   frontend/src/server/better-auth/{client,config,index,server}.ts
>   frontend/src/app/api/auth/[...all]/route.ts
>
> Surgical (含 noldus 改动):
>   frontend/src/app/workspace/layout.tsx (77 行 diff)
>   frontend/src/app/workspace/workspace-content.tsx (35 行)
>   frontend/src/components/workspace/input-box.tsx (24 行)
>   frontend/src/components/workspace/settings/settings-dialog.tsx (14 行)
>   frontend/src/core/agents/api.ts (7 行)
>   frontend/src/core/mcp/api.ts (16 行)
>   frontend/src/core/memory/api.ts (37 行)
>   frontend/src/core/skills/api.ts (18 行)
>   frontend/src/core/threads/hooks.ts (3 行)
>   frontend/src/core/uploads/api.ts (5 行)
>   frontend/src/core/i18n/locales/{en-US, zh-CN, types}.ts (2 行 each)
>   frontend/src/env.js (10 行)
>   frontend/package.json (1 行 - 删 better-auth)
>   frontend/pnpm-lock.yaml (大量 - 整覆盖上游)
> ```

### Task H.1: 删除 better-auth 旧代码

**Files:**
- Delete: `packages/agent/frontend/src/server/better-auth/{client,config,index,server}.ts`
- Delete: `packages/agent/frontend/src/app/api/auth/[...all]/route.ts`

- [ ] **Step 1: 备份 server/better-auth 以防需要**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
cp -r src/server/better-auth /tmp/noldus-better-auth-backup
ls /tmp/noldus-better-auth-backup/
```

- [ ] **Step 2: 删除 better-auth 目录和 catch-all route**

```bash
cd /home/wangqiuyang/noldus-insight
rm -rf packages/agent/frontend/src/server/better-auth/
rm -f packages/agent/frontend/src/app/api/auth/\[...all\]/route.ts
rmdir packages/agent/frontend/src/app/api/auth/\[...all\]/ 2>/dev/null || true
rmdir packages/agent/frontend/src/app/api/auth/ 2>/dev/null || true

ls packages/agent/frontend/src/server/ 2>&1
ls packages/agent/frontend/src/app/api/ 2>&1
```

期望: `server/` 下 better-auth 不存在,`api/` 下 auth 子目录也清掉。

### Task H.2: 拉 frontend core/auth + core/api 新文件

**Files:**
- Create: `packages/agent/frontend/src/core/auth/AuthProvider.tsx`
- Create: `packages/agent/frontend/src/core/auth/gateway-config.ts`
- Create: `packages/agent/frontend/src/core/auth/proxy-policy.ts`
- Create: `packages/agent/frontend/src/core/auth/server.ts`
- Create: `packages/agent/frontend/src/core/auth/types.ts`
- Create: `packages/agent/frontend/src/core/api/api-client.ts`
- Create: `packages/agent/frontend/src/core/api/fetcher.ts`

- [ ] **Step 1: 创建目录**

```bash
cd /home/wangqiuyang/noldus-insight
mkdir -p packages/agent/frontend/src/core/auth
mkdir -p packages/agent/frontend/src/core/api
```

- [ ] **Step 2: 拉 5 个 core/auth 文件**

```bash
cd /home/wangqiuyang/noldus-insight
FE=packages/agent/frontend

for f in AuthProvider.tsx gateway-config.ts proxy-policy.ts server.ts types.ts; do
    git show "deerflow/main:frontend/src/core/auth/$f" \
        > "$FE/src/core/auth/$f"
done
wc -l $FE/src/core/auth/*.tsx $FE/src/core/auth/*.ts
```

期望:
- AuthProvider.tsx ~165 行
- gateway-config.ts ~34 行
- proxy-policy.ts ~55 行
- server.ts ~57 行(94eee95f) 或 ~86 行(848ace98 修补后)
- types.ts ~72 行(94eee95f) 或 ~101 行(848ace98 修补后)

- [ ] **Step 3: 拉 2 个 core/api 文件**

```bash
for f in api-client.ts fetcher.ts; do
    git show "deerflow/main:frontend/src/core/api/$f" \
        > "$FE/src/core/api/$f"
done
wc -l $FE/src/core/api/*.ts
```

期望: api-client.ts ~26 行,fetcher.ts ~104 行。

### Task H.3: 拉 (auth) 路由组 3 个文件

**Files:**
- Create: `packages/agent/frontend/src/app/(auth)/layout.tsx`
- Create: `packages/agent/frontend/src/app/(auth)/login/page.tsx`
- Create: `packages/agent/frontend/src/app/(auth)/setup/page.tsx`

- [ ] **Step 1: 创建目录**

```bash
cd /home/wangqiuyang/noldus-insight
mkdir -p "packages/agent/frontend/src/app/(auth)/login"
mkdir -p "packages/agent/frontend/src/app/(auth)/setup"
```

⚠️ 路径中的 `(auth)` 是 Next.js 路由组语法,**括号是文件名一部分**,要在 mkdir 里加引号。

- [ ] **Step 2: 拉 3 个文件**

```bash
cd /home/wangqiuyang/noldus-insight
FE=packages/agent/frontend

git show 'deerflow/main:frontend/src/app/(auth)/layout.tsx' \
    > "$FE/src/app/(auth)/layout.tsx"
git show 'deerflow/main:frontend/src/app/(auth)/login/page.tsx' \
    > "$FE/src/app/(auth)/login/page.tsx"
git show 'deerflow/main:frontend/src/app/(auth)/setup/page.tsx' \
    > "$FE/src/app/(auth)/setup/page.tsx"

wc -l "$FE/src/app/(auth)/layout.tsx" \
       "$FE/src/app/(auth)/login/page.tsx" \
       "$FE/src/app/(auth)/setup/page.tsx"
```

期望:
- layout.tsx ~46 行
- login/page.tsx ~233 行(94eee95f 183 + 848ace98 50)
- setup/page.tsx ~324 行(94eee95f 115 + 848ace98 209)

### Task H.4: 拉 account-settings-page

**Files:**
- Create: `packages/agent/frontend/src/components/workspace/settings/account-settings-page.tsx`

- [ ] **Step 1: 拉文件**

```bash
cd /home/wangqiuyang/noldus-insight
git show 'deerflow/main:frontend/src/components/workspace/settings/account-settings-page.tsx' \
    > 'packages/agent/frontend/src/components/workspace/settings/account-settings-page.tsx'
wc -l 'packages/agent/frontend/src/components/workspace/settings/account-settings-page.tsx'
```

期望: ~132 行。

### Task H.5: surgical merge frontend 调整文件

**Files:**
- Modify: `packages/agent/frontend/src/app/workspace/layout.tsx`
- Modify: `packages/agent/frontend/src/app/workspace/workspace-content.tsx`
- Modify: `packages/agent/frontend/src/components/workspace/input-box.tsx`
- Modify: `packages/agent/frontend/src/components/workspace/settings/settings-dialog.tsx`
- Modify: `packages/agent/frontend/src/core/agents/api.ts`
- Modify: `packages/agent/frontend/src/core/mcp/api.ts`
- Modify: `packages/agent/frontend/src/core/memory/api.ts`
- Modify: `packages/agent/frontend/src/core/skills/api.ts`
- Modify: `packages/agent/frontend/src/core/threads/hooks.ts`
- Modify: `packages/agent/frontend/src/core/uploads/api.ts`
- Modify: `packages/agent/frontend/src/core/i18n/locales/types.ts`
- Modify: `packages/agent/frontend/src/env.js`
- Modify: `packages/agent/frontend/package.json`
- Modify: `packages/agent/frontend/pnpm-lock.yaml`

- [ ] **Step 1: 处理 11 个 core/<module>/api.ts 类(改 fetch 用 fetcher.ts)**

每个文件改动 < 40 行,主要是把 `fetch(url)` 改为 `apiClient.get(url)`。

对每个文件:

```bash
cd /home/wangqiuyang/noldus-insight
FILE=packages/agent/frontend/src/core/agents/api.ts
diff <(git show deerflow/main:frontend/src/core/agents/api.ts) $FILE | head -20
grep -c "EthoInsight\|noldus\|中文" $FILE
```

无 noldus 定制 → 整覆盖:
```bash
git show deerflow/main:frontend/src/core/agents/api.ts > $FILE
```

按此处理:`agents/api.ts / mcp/api.ts / memory/api.ts / skills/api.ts / threads/hooks.ts / uploads/api.ts / i18n/locales/types.ts`。

- [ ] **Step 2: surgical merge workspace/layout.tsx**

```bash
cd /home/wangqiuyang/noldus-insight
FE=packages/agent/frontend
diff <(git show deerflow/main:frontend/src/app/workspace/layout.tsx) "$FE/src/app/workspace/layout.tsx" | head -50
grep -c "EthoInsight\|noldus\|中文" "$FE/src/app/workspace/layout.tsx"
```

如果 noldus 定制 grep > 0,**surgical merge**:
- 只在最外层加 `<AuthProvider>` 包裹
- 保留 noldus 加的所有内部 children / className / theme 等

如果 grep = 0 且 diff < 100 行,整覆盖:
```bash
git show deerflow/main:frontend/src/app/workspace/layout.tsx > "$FE/src/app/workspace/layout.tsx"
```

- [ ] **Step 3: 处理 workspace-content.tsx / input-box.tsx / settings-dialog.tsx**

对每个: diff + grep noldus 定制 + 决定整覆盖或 surgical。

⚠️ `settings-dialog.tsx` 加了 account settings tab,确认 noldus 现有 settings tabs 不丢。

- [ ] **Step 4: 改 env.js (删 BETTER_AUTH 4 行)**

```bash
grep -n "BETTER_AUTH" packages/agent/frontend/src/env.js
```

记录行号,用 Edit tool 删除这 4 行(以及顶部 import 中相关的)。改完:
```bash
grep -c "BETTER_AUTH" packages/agent/frontend/src/env.js
```

期望: `0`。

- [ ] **Step 5: 改 package.json (删 better-auth 一行)**

```bash
grep -n "better-auth" packages/agent/frontend/package.json
```

用 Edit tool 删除这一行(包括前面的逗号要处理,JSON 格式化)。改完:
```bash
grep -c "better-auth" packages/agent/frontend/package.json
```

期望: `0`。

- [ ] **Step 6: pnpm-lock.yaml 整覆盖到上游**

```bash
git show deerflow/main:frontend/pnpm-lock.yaml \
    > packages/agent/frontend/pnpm-lock.yaml
wc -l packages/agent/frontend/pnpm-lock.yaml
```

⚠️ **不本地重 lock**(spec §5.1 风险点 3)。直接用上游版本。

- [ ] **Step 7: pnpm install 装新依赖**

```bash
cd packages/agent/frontend
pnpm install --frozen-lockfile 2>&1 | tail -10
```

期望: install 成功,无 conflict。

如有 lock 不匹配错误,**立即停下问用户** — 可能 noldus 还有自己加的 npm dep 在 package.json 但不在上游 lock。

### Task H.6: 处理 i18n locales

**Files:**
- Modify: `packages/agent/frontend/src/core/i18n/locales/zh-CN.ts` (**只追加 auth namespace,不重写**)
- Modify: `packages/agent/frontend/src/core/i18n/locales/en-US.ts` (**只追加 auth namespace,不重写**)

- [ ] **Step 1: 看上游加的 auth 翻译**

```bash
cd /home/wangqiuyang/noldus-insight
git show 94eee95f -- 'frontend/src/core/i18n/locales/zh-CN.ts'
git show 94eee95f -- 'frontend/src/core/i18n/locales/en-US.ts'
```

预期: 加 `auth: { login: "...", register: "...", password: "...", ... }` 一段。

- [ ] **Step 2: 看 noldus zh-CN.ts 现状**

```bash
wc -l packages/agent/frontend/src/core/i18n/locales/zh-CN.ts
grep -c "zhCN\|常用\|首页\|设置" packages/agent/frontend/src/core/i18n/locales/zh-CN.ts
```

记录行数(轮 3 后行数应该接近 此数 + 上游 step 1 看到的行数)。

- [ ] **Step 3: 用 Edit 在 zh-CN.ts 末尾追加 auth namespace**

按上游 step 1 输出,在 noldus zh-CN.ts 的最后一个 namespace 后,在结尾的 `}` 之前追加 auth 段。

⚠️ **不要 replace_all,不要整覆盖,只追加**。

具体步骤:
1. 用 Read tool 读 zh-CN.ts 末尾 30 行
2. 确认结尾结构(如 `}; export default zhCN;` 或类似)
3. 用 Edit tool 在最后一个 namespace 后插入 auth

- [ ] **Step 4: 同步处理 en-US.ts**

同样流程,加上游英文 auth 翻译。

- [ ] **Step 5: types.ts 已在 H.5 step 1 处理(整覆盖,因为是类型定义)**

- [ ] **Step 6: 验证 noldus 现有翻译保留**

```bash
grep -c "zhCN\|常用\|首页\|设置\|分析\|工作区" packages/agent/frontend/src/core/i18n/locales/zh-CN.ts
grep -c "auth:" packages/agent/frontend/src/core/i18n/locales/zh-CN.ts
```

期望:
- 第 1 行: ≥ 4 (noldus 中文翻译没丢)
- 第 2 行: ≥ 1 (auth namespace 已加)

### Task H.7: 跑前端 typecheck + 启动 smoke + Commit

- [ ] **Step 1: pnpm typecheck**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm typecheck 2>&1 | tail -20
```

期望: 0 error。

如有 error:
- `Cannot find module '@/server/better-auth'` → H.5 漏改的 import,grep 后 Edit
- `Cannot find module '@/core/auth'` → H.2 文件没拉,重做

- [ ] **Step 2: pnpm lint**

```bash
pnpm lint 2>&1 | tail -10
```

期望: 0 error。

- [ ] **Step 3: 启动全栈 smoke**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop 2>&1 || true
export JWT_SECRET=test-secret-32-chars-minimum-please
make dev 2>&1 | tee /tmp/h-smoke.log &
sleep 60

curl -s http://localhost:2026/login 2>&1 | head -3
curl -s http://localhost:2026/api/auth/setup-status 2>&1 | head -3
```

期望:
- `/login` 返 HTML 含登录表单
- `/api/auth/setup-status` 返 JSON

⚠️ 用户应该浏览器手动验证: 打开 `http://localhost:2026`,看到登录页或 setup wizard。

- [ ] **Step 4: 跑后端测试基线确认 H 没破坏**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -3
```

期望: `2 failed, ≥1980 passed` (与 G 末尾一致)。

- [ ] **Step 5: 验证受保护语义残量(再跑一次 G.13 step 3)**

(全部命令复制 G.13 step 3,确认仍达标)

- [ ] **Step 6: 停服务**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop 2>&1 | tail -3
```

- [ ] **Step 7: Commit Phase H**

```bash
cd /home/wangqiuyang/noldus-insight

git add packages/agent/frontend/src/core/auth/ \
        packages/agent/frontend/src/core/api/ \
        'packages/agent/frontend/src/app/(auth)/' \
        packages/agent/frontend/src/components/workspace/settings/account-settings-page.tsx \
        packages/agent/frontend/src/app/workspace/ \
        packages/agent/frontend/src/components/ \
        packages/agent/frontend/src/core/agents/api.ts \
        packages/agent/frontend/src/core/mcp/api.ts \
        packages/agent/frontend/src/core/memory/api.ts \
        packages/agent/frontend/src/core/skills/api.ts \
        packages/agent/frontend/src/core/threads/hooks.ts \
        packages/agent/frontend/src/core/uploads/api.ts \
        packages/agent/frontend/src/core/i18n/ \
        packages/agent/frontend/src/env.js \
        packages/agent/frontend/package.json \
        packages/agent/frontend/pnpm-lock.yaml

# 显式删除已删除的文件
git add -u packages/agent/frontend/src/server/better-auth/ 2>/dev/null
git add -u packages/agent/frontend/src/app/api/auth/ 2>/dev/null

git commit -m "$(cat <<'EOF'
sync deerflow upstream Tier 4 H: better-auth 前端 (94eee95f + 848ace98 + 98a5b34f)

合入上游 better-auth 前端体系:

新增:
- core/auth/ {AuthProvider, gateway-config, proxy-policy, server, types}
- core/api/ {api-client, fetcher} (带 JWT header + 401 redirect)
- app/(auth)/ {layout, login/page, setup/page} (Next.js 路由组)
- components/workspace/settings/account-settings-page.tsx (改密码)

删除:
- server/better-auth/ 4 个文件 (旧 npm 库, 实质未启用)
- app/api/auth/[...all]/route.ts (旧 catch-all)
- package.json 的 better-auth dep
- env.js 的 BETTER_AUTH_* env vars

Surgical merge:
- workspace/layout.tsx: 包 <AuthProvider>, 保留 noldus UI 改动
- 6 个 core/<module>/api.ts: 改用 apiClient.get/post 带 JWT
- i18n locales/{zh-CN, en-US}.ts: 追加 auth namespace,
  保留 noldus 全部现有翻译

pnpm-lock.yaml 整覆盖到上游 (不本地重 lock).

保留 Noldus 全部价值:
- hero.tsx EthoInsight 品牌不动
- zh-CN.ts 全部现有中文翻译保留
- workspace/layout.tsx noldus UI 改动保留

测试: 后端基线不变 2 failed, ≥1980 passed; 前端 typecheck + lint 0 error.

详见 docs/superpowers/specs/2026-05-07-deerflow-tier234-round3-design.md §2.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 8: 验证 commit**

```bash
git log --oneline -2
git diff --stat HEAD~1 HEAD | tail -3
```

期望: HEAD 是 "sync deerflow upstream Tier 4 H: better-auth 前端...",30+ 文件改动。

---

## Phase I: 品牌中文化 (Task I.1 - I.3)

> 上游 i18n 是英文为主,zh-CN 仅有部分翻译。noldus 已大量中文化(轮 1)+ 已加 EthoInsight 品牌(landing/hero.tsx)。
> 轮 3 这步: 替换前端代码中残存的 "DeerFlow" 字符串为 "EthoInsight"(品牌一致性), 补全 i18n 中 auth 相关的中文翻译细节。

### Task I.1: 替换 "DeerFlow" 为 "EthoInsight"

**Files:**
- Modify: 多个前端文件,凡含 "DeerFlow" 字符串(除技术 ID 外)

- [ ] **Step 1: 找出所有"DeerFlow"出现位置**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
grep -rn "DeerFlow\|deer-flow\|deerflow" src/ --include="*.tsx" --include="*.ts" | grep -v node_modules | head -40
```

记录所有出现位置。**注意区分**:
- ✅ 替换为 "EthoInsight" 的: 用户可见文案、页面标题、品牌字符串
- ❌ 保留 "deerflow" 不动的: 技术 ID(`deerflow_auth_token` cookie name、`@deerflow-harness` package 名、API path 段)

- [ ] **Step 2: 替换用户可见的 "DeerFlow"**

对每个用户可见的位置,用 Edit tool 替换:
- `"DeerFlow"` → `"EthoInsight"`
- `"Welcome to DeerFlow"` → `"欢迎使用 EthoInsight"`(或对应中文翻译)

具体文件可能包括:
- `(auth)/login/page.tsx` 的标题
- `(auth)/setup/page.tsx` 的欢迎语
- `metadata` (page.tsx 中 export const metadata)
- 任何 brand banner / footer

- [ ] **Step 3: 验证残量**

```bash
cd packages/agent/frontend
grep -rn "DeerFlow" src/ --include="*.tsx" --include="*.ts" | grep -v node_modules
grep -rn "EthoInsight" src/ --include="*.tsx" --include="*.ts" | grep -v node_modules | wc -l
```

期望:
- DeerFlow 残量: 0(全替换)
- EthoInsight 残量: ≥ 5(已替换 + landing/hero 原有)

⚠️ **如果 DeerFlow 仍有残量,逐处确认**:技术 ID 保留,用户可见替换。

### Task I.2: 补全 auth 翻译中文化

**Files:**
- Modify: `packages/agent/frontend/src/core/i18n/locales/zh-CN.ts`

H.6 已追加上游 auth namespace, 但上游 zh-CN 翻译可能不完整或不符合 noldus 风格。

- [ ] **Step 1: 看 H.6 后的 zh-CN.ts auth 段**

```bash
cd /home/wangqiuyang/noldus-insight
grep -A 50 "^  auth:" packages/agent/frontend/src/core/i18n/locales/zh-CN.ts
```

- [ ] **Step 2: 检查 (auth) 页面中所有 t(...) 调用**

```bash
grep -rn "t([\"']" 'packages/agent/frontend/src/app/(auth)/' --include="*.tsx" | head -30
```

记录所有 i18n key,如 `t("auth.login.title")`、`t("auth.errors.invalidCredentials")` 等。

- [ ] **Step 3: 验证 zh-CN.ts auth 段覆盖所有 key**

对每个 step 2 的 key,在 zh-CN.ts 中 grep 确认存在:

```bash
grep "title:\|invalidCredentials:" packages/agent/frontend/src/core/i18n/locales/zh-CN.ts | head -10
```

如有缺漏,用 Edit tool 在 zh-CN.ts auth 段中补 key。具体翻译用户可定;模板:
- `登录` / `注册` / `邮箱` / `密码` / `创建管理员` / `首次设置`
- 错误信息: `用户名或密码错误` / `无效令牌` / `请先登录`

- [ ] **Step 4: 验证 noldus 全量翻译保留**

```bash
grep -c "首页\|设置\|工作区\|分析\|分享\|删除\|编辑" packages/agent/frontend/src/core/i18n/locales/zh-CN.ts
```

期望: ≥ 5 (noldus 现有翻译完整)。

### Task I.3: typecheck + smoke + Commit

- [ ] **Step 1: pnpm typecheck**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend
pnpm typecheck 2>&1 | tail -10
pnpm lint 2>&1 | tail -10
```

期望: 0 error。

- [ ] **Step 2: 启动 smoke,看登录页**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop 2>&1 || true
export JWT_SECRET=test-secret-32-chars-minimum-please
make dev 2>&1 | tee /tmp/i-smoke.log &
sleep 60

curl -s http://localhost:2026/login | grep -E "EthoInsight|登录|email|password" | head -5
```

期望: HTML 中含 EthoInsight + 中文翻译。

⚠️ 用户应该浏览器手动验证: 登录页全中文 + EthoInsight 品牌, 没有 DeerFlow 残留。

- [ ] **Step 3: 后端测试不动**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -3
```

期望: `2 failed, ≥1980 passed` (不变)。

- [ ] **Step 4: 停服务**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop 2>&1 | tail -3
```

- [ ] **Step 5: Commit Phase I**

```bash
cd /home/wangqiuyang/noldus-insight

git add packages/agent/frontend/src/

git commit -m "$(cat <<'EOF'
sync deerflow upstream Tier 4 I: 品牌 EthoInsight + 中文化补全

- 替换前端代码中所有用户可见的 "DeerFlow" 为 "EthoInsight"
  (技术 ID 如 cookie name / package name / API path 不动)
- 补全 zh-CN.ts auth namespace 中文翻译, 覆盖 (auth) 页面所有 t(...) key
- 验证 noldus 全部现有翻译 + landing/hero EthoInsight 品牌仍在

测试: typecheck + lint 0 error, 后端基线不变.

详见 docs/superpowers/specs/2026-05-07-deerflow-tier234-round3-design.md §2.2

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase J: 部署文档收尾 (Task J.1 - J.3)

### Task J.1: 写 multi-user 部署 SOP

**Files:**
- Create: `docs/sop/multi-user-deployment-sop.md`

- [ ] **Step 1: 写文档**

用 Write tool 创建 `docs/sop/multi-user-deployment-sop.md`,内容包含:

1. **概述**: EthoInsight v0.1 多用户部署架构(单机 Docker Compose,10 人内并发)
2. **依赖**: 阿里云 PostgreSQL RDS, JWT secret, nginx, Let's Encrypt
3. **首次部署步骤**:
   - 准备阿里云 PG: `CREATE DATABASE ethoinsight; CREATE USER ...`
   - 设置 env: `JWT_SECRET / DATABASE_URL / DEER_FLOW_HOST`
   - 改 `config.yaml`: `database.backend: postgres`, `database.postgres_url: $DATABASE_URL`
   - `docker compose up -d`
   - 首次访问触发 setup wizard 创建 admin
4. **运维**:
   - 加用户: `uv run python -m app.gateway.auth.reset_admin --create-user user@x.com`(或前端注册)
   - 重置密码: `reset_admin` CLI
   - 备份: `pg_dump ethoinsight > backup.sql` + `cp -r .deer-flow .deer-flow.bak`
5. **排障**: 401 错误、CSRF 失败、PG 连接失败、orphaned threads
6. **数据隔离保证**: 每用户 `.deer-flow/users/{user_id}/threads/` 独立目录, PG 行级 owner_id 过滤

每个段写具体命令,**不留 placeholder**。

### Task J.2: 更新弃用的 4-23 计划文档

**Files:**
- Modify: `docs/plans/2026-04-23-multi-user-deployment.md`

- [ ] **Step 1: 在文档顶部加更明确的 deprecated 信息**

用 Edit tool, 在 `> ⚠️ **DEPRECATED 2026-05-07**` 段后追加:

```markdown
**轮 3 (2026-05-07) 已合入上游 better-auth 实现**, 替代本计划 Task 1-15。
- 实施 spec: `docs/superpowers/specs/2026-05-07-deerflow-tier234-round3-design.md`
- 实施 plan: `docs/superpowers/plans/2026-05-07-deerflow-tier234-round3-plan.md`
- 完成 handoff: `docs/handoffs/2026-XX-XX-deerflow-tier234-round3-completed-handoff.md`
- 部署 SOP: `docs/sop/multi-user-deployment-sop.md`

本文档保留作历史参考, 体现"造轮子 vs 用上游"的决策路径。
```

### Task J.3: 写完成 handoff + Commit

**Files:**
- Create: `docs/handoffs/2026-XX-XX-deerflow-tier234-round3-completed-handoff.md`

- [ ] **Step 1: 写 handoff**

用 Write tool 创建 handoff,日期用当天 (`date +%Y-%m-%d`)。结构参考轮 2 完成 handoff:

1. **TL;DR**: 4 个 commit, 测试基线变化 (1877 → 1980+), better-auth 已就绪
2. **已完成 commits 表格** (G/H/I/J)
3. **跳过的内容** (实际 PG 部署 / LangGraph PostgresSaver / AioSandbox 切换 → 留轮 4)
4. **关键状态验证**: 测试基线 / 受保护语义残量 / git log
5. **轮 4 backlog**:
   - 阿里云 PG 实际部署
   - LangGraph PostgresSaver 切换 (thread state 持久化)
   - Sandbox 切 AioSandboxProvider (容器隔离)
   - Multi-user 性能压测 (10 人并发)
   - 前端 auth 视觉细节优化
6. **风险/已知问题**:
   - bcrypt 哈希默认 12 rounds, 阿里云低配 PG 可能慢, 监控登录响应时间
   - JWT secret 必须 ≥ 32 字符且不能泄露
   - orphaned threads 迁移只跑一次, lifespan hook 加 idempotent 检查
7. **不要 push**

- [ ] **Step 2: Commit Phase J(包含 handoff + SOP + 弃用更新)**

```bash
cd /home/wangqiuyang/noldus-insight

git add docs/sop/multi-user-deployment-sop.md \
        docs/plans/2026-04-23-multi-user-deployment.md \
        docs/handoffs/2026-*-deerflow-tier234-round3-completed-handoff.md

git commit -m "$(cat <<'EOF'
docs: 轮 3 deerflow 同步 better-auth 完成交接 + multi-user 部署 SOP

- 新增 docs/sop/multi-user-deployment-sop.md: 阿里云 PG 部署完整步骤
- 更新 docs/plans/2026-04-23-multi-user-deployment.md deprecated 信息,
  指向轮 3 实际实施文档
- 新增完成 handoff, 轮 4 backlog (实际 PG 部署 / PostgresSaver / AioSandbox)

不 push origin, 留 dev 等用户决定.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: 验证 4 个 commit 序列**

```bash
git log --oneline -5
```

期望:
```
<sha> docs: 轮 3 deerflow 同步 better-auth 完成交接 + multi-user 部署 SOP
<sha> sync deerflow upstream Tier 4 I: 品牌 EthoInsight + 中文化补全
<sha> sync deerflow upstream Tier 4 H: better-auth 前端 (94eee95f + 848ace98 + 98a5b34f)
<sha> sync deerflow upstream Tier 4 G: better-auth 后端 (94eee95f + da174dfd + 4e4e4f92 + 78633c69 + ed9ebfac + 88d47f67)
<sha> docs: 修正轮 3 spec 中 auth 路径 (app/auth → app/gateway/auth)
```

---

## 总完成定义 (Done Criteria)

- [ ] 4 个本地 commit (G / H / I / J), 全部在 dev 分支
- [ ] 测试基线: `2 failed, ≥1980 passed, 14 skipped`
- [ ] 后端 `make lint` 0 error
- [ ] 前端 `pnpm typecheck` + `pnpm lint` 0 error
- [ ] `make dev` 启动成功 (LangGraph + Gateway + Frontend + Nginx)
- [ ] 浏览器访问 `localhost:2026` 看到登录页 (EthoInsight + 中文)
- [ ] Setup wizard 能创建 admin 账号
- [ ] 登录后跳转 workspace, 能跑分析 (上传 → 分析 → artifact)
- [ ] noldus 受保护语义残量全部达标 (按文头清单 grep 验证)
- [ ] handoff 文档已写
- [ ] **不 push** — 留 dev 等用户决定

---

## 应急处置

### 任何 phase 失败超过 4h 且测试不通过

1. `git status` 看未提交改动
2. 写中断 handoff: `docs/handoffs/2026-XX-XX-tier234-round3-INTERRUPTED.md` 记录:
   - 当前 phase + task
   - 已 commit 的部分(列 SHA)
   - 卡住的具体错误 (stack trace)
   - 已尝试的 debug 步骤
3. **不 commit 半成品**
4. `git stash` 当前改动备份
5. 通知用户

### 测试失败数从 2 涨到 3+

1. **立即停下**, 不要继续往下做
2. 跑失败的测试单独看错误信息:
   ```bash
   PYTHONPATH=. uv run pytest tests/<failing_test>.py -v 2>&1 | tail -30
   ```
3. 99% 是 mock path 没更新或 import 漏改, 不是真正的逻辑 bug
4. 修完了再继续

### 启动失败 (make dev 报 ImportError)

1. 看 `/tmp/g-smoke.log` (或对应 phase 的 log) 找 import error 行
2. 通常原因:
   - SQLAlchemy / asyncpg / bcrypt 没装 → `cd backend && uv sync`
   - JWT_SECRET 没设 → `export JWT_SECRET=test-secret-32-chars-minimum-please`
   - `app.state.config` AttributeError → G.6 surgical merge 把 `app.state.config = cfg` 弄丢了, git restore 重做
3. ImportError of `runtime.user_context` → G.6 langgraph_auth 没接通,看 langgraph.json

### Surgical merge 把 noldus 定制弄丢了

1. `git diff` 看具体哪个文件丢失了什么
2. `git checkout HEAD -- <file>` 恢复整个文件 (前提是已 commit 过该文件的 noldus 状态)
3. 如果未 commit, 用 step 1 备份: `cp /tmp/<file>_noldus_backup.py <real path>`
4. 重新做这个文件的 surgical merge, 这次更小心

### 阿里云 PG 部署相关问题(轮 3 不实施,但用户问到时)

J.1 SOP 中说明的步骤已涵盖。具体:
- env DATABASE_URL 格式: `postgresql+asyncpg://user:pass@rm-xxxxx.pg.rds.aliyuncs.com:5432/ethoinsight`
- 注意 SSL: 阿里云 PG 强制 SSL, asyncpg 需 `?sslmode=require`
- 网络: ECS 同 VPC 直连, 跨 VPC 走 VPN 或公网+白名单
- 备份: 阿里云控制台启用自动备份 + 定期 pg_dump
