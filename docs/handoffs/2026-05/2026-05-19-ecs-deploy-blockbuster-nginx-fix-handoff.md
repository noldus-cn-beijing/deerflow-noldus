# Handoff: ECS 多用户 Docker 部署 — nginx / blockbuster / build context 7 处修复

**日期**: 2026-05-19
**ECS**: i-2ze446h4ats7y8bgjn4w · 39.105.231.16 · 华北 2 · `/root/noldus-insight/` (dev 分支)
**前置交接**: [2026-05-19-ecs-deploy-gateway-pypi-fix-handoff.md](2026-05-19-ecs-deploy-gateway-pypi-fix-handoff.md) ← **该文档诊断错误，作废**

---

## TL;DR

一次会话修了 **7 个独立问题**，其中 **4 个是 deerflow 上游缺陷**（langgraph_api 0.7.65 启用 blockbuster middleware 后暴露的 ASGI 同步 syscall 问题）。当前 ECS Docker 部署可注册、可登录、可完整对话（验证 `Background run succeeded`）。

**所有修复以 hot-patch 形式应用到 ECS 容器**，本地文件已就位但 **未 commit 到 dev 分支** — 下次 `docker compose build` 会丢失所有改动。**必须立即 commit 持久化**（见末尾「必做收尾」）。

---

## 上一任 handoff 错在哪

`2026-05-19-ecs-deploy-gateway-pypi-fix-handoff.md` 说："gateway 容器 running 但 8001 不监听 → uv sync 拉不到包 → 改 aliyun 镜像 + 清 builder cache"。

实际：

- `.venv` 已经 1.1GB（依赖装齐了）
- gateway 日志 `Uvicorn running on http://0.0.0.0:8001` + 4 worker 全启动
- `docker exec nginx wget http://gateway:8001/health` 直接拿到 `{"status":"healthy"}`

**gateway 工作正常**。502 不是 gateway 的问题。

诊断陷阱：上一任用 `docker exec deer-flow-gateway python -c "import fastapi"` 检查依赖，得到 `ModuleNotFoundError`。但**这条命令走的是容器全局 `/usr/local/bin/python`，不是 `.venv/bin/python`** — venv 没激活当然 import 不到。正确诊断必须用 `uv run python -c ...` 或 `.venv/bin/python -c ...`。这条错误诊断让前一任把根因写成 "uv sync 拉不到 pypi"，浪费了一轮修复。

---

## 真实根因链 — 7 个独立问题

### 1️⃣ nginx 静态 upstream 在 docker-compose 容器重建后 502

**症状**: 浏览器访问 `http://39.105.231.16:2026/` 拿 502，nginx 错误日志 `connect() failed (111: Connection refused) while connecting to upstream: "http://172.19.0.2:3000/login"`。但同时 `docker exec nginx wget http://frontend:3000/` 拿到完整 HTML。

**根因**: `nginx.conf` 用 `upstream frontend { server frontend:3000; }` 这种**静态 upstream 块**。nginx:alpine（非 Plus 版）只在 nginx 启动时解析一次 hostname，之后**永远不再查 DNS**。docker-compose `up -d --force-recreate frontend gateway langgraph` 三个应用容器拿了新 IP，但 nginx 没重启所以持旧 IP，所有上游请求 → connection refused。

**修复**: `docker/nginx/nginx.conf` — 删 3 个 `upstream {}` 块，在 server 块开头声明 `set $gateway_upstream "gateway:8001"` 等变量，所有 `location` 改用 `proxy_pass http://$gateway_upstream$request_uri`。变量形式强制 nginx 每请求走 resolver 127.0.0.11 查 docker DNS。langgraph location 因为前面有 `rewrite ... break`，用 `$uri$is_args$args` 而非 `$request_uri`。

**为什么本地 make dev 不复现**: 本地用 loopback IP (`localhost:3000`)，永不漂移。Docker 模式才有 IP 漂移。

---

### 2️⃣ nginx.conf 缺 `/api/` catch-all → `/api/v1/auth/*` 404

**症状**: 登录页加载 OK 但点击注册按钮，`POST /api/v1/auth/register 404`。

**根因**: deerflow 上游 2.0-rc 新增 `auth.router = APIRouter(prefix="/api/v1/auth")`，但维护者**只更新了 `docker/nginx/nginx.local.conf`**（本地 `make dev` 用），**没更新 `docker/nginx/nginx.conf`**（Docker `make up` 用）。两份文件分开维护、`nginx.local.conf` 有 catch-all `location /api/` → gateway，`nginx.conf` 没有。

证据：`nginx.local.conf:L231-248` 注释亲口承认：
> "Covers the auth module (/api/v1/auth/login, /me, /change-password, ...), plus feedback / runs / token-usage routes that **2.0-rc added without updating this nginx config**."

**修复**: `docker/nginx/nginx.conf` 在所有具体 location 之后、catch-all `location /` 之前，加：

```nginx
location /api/ {
    proxy_pass http://$gateway_upstream$request_uri;
    ...
    proxy_pass_header Set-Cookie;   # auth HttpOnly cookies 必需
}
```

跟 `nginx.local.conf` 对齐。longest-prefix 匹配保证具体路由（`/api/models` `/api/threads` 等）仍优先命中。

---

### 3️⃣ `uv run uvicorn` 在 ECS 上 pre-flight sync 卡住

**症状**: gateway 容器 `Up 8 minutes`，但 `docker top` 显示只有 `sh -c ...` 和 `uv run uvicorn ...` 进程，**没有 uvicorn worker 进程**。`docker logs` **完全空白**。

**根因**: `uv run` 首次执行会做一次 lock check / sync。ECS 在国内无法访问某个网络源（推测是 ghcr.io，因为 PyPI 已走 aliyun 镜像），sync hang 在那里。`uv run` 卡住没切到 uvicorn，stdout 也没产生任何输出。

**修复**: `docker/docker-compose.yaml` 的 gateway / langgraph `command` 改成直接走 venv 二进制：

```yaml
# 改前
command: sh -c "cd backend && PYTHONPATH=. uv run uvicorn app.gateway.app:app ..."
# 改后
command: sh -c "cd backend && PYTHONPATH=. /app/backend/.venv/bin/uvicorn app.gateway.app:app ..."
```

绕开 `uv run` 的 sync 阶段，直接用 build 时装好的 uvicorn。

---

### 4️⃣ `backend/packages/ethoinsight` 是 symlink，跨出 Docker build context

**症状**: `make up` 重 build gateway 报 `error: failed to query metadata of file '/app/backend/packages/ethoinsight': No such file or directory`。

**根因**: 本地仓库 `packages/agent/backend/packages/ethoinsight` 是 symlink 指向 `../../../ethoinsight`（仓库内 sibling 目录）。docker build context 是 `packages/agent/`，**symlink 目标在 context 之外**，build 时 symlink 被原样复制但 link 目标在容器里不存在。

`rsync -av` 默认保留 symlink，所以 ECS 上同步过去后仍是 symlink，build 必败。

**临时修复（已应用到 ECS）**: 在 ECS 上 `rm ethoinsight` 删 symlink，`cp -r /root/noldus-insight/packages/ethoinsight ./ethoinsight` 复制真目录进来。

**长期修复（未应用）**: 本地 rsync 加 `-L` 解引用 symlink：

```bash
rsync -aLvz --delete \
  /home/wangqiuyang/noldus-insight/packages/agent/ \
  root@39.105.231.16:/root/noldus-insight/packages/agent/
```

`-L` 让 rsync 把 symlink 当真文件复制，目标端永远是真目录。

---

### 5️⃣ `database_config.py` `Path.resolve()` 在 ASGI 路径触发 blockbuster

**症状**: 浏览器登录后任何对话操作（`POST /threads/search` / `POST /threads`）返回 500。langgraph 日志一长串 `BlockingError: Blocking call to os.getcwd`，stack trace 终点 `database_config.py:74 → Path(self.sqlite_dir).resolve()`。

**根因**: deerflow 上游写了

```python
@property
def _resolved_sqlite_dir(self) -> str:
    return str(Path(self.sqlite_dir).resolve())   # ← Path.resolve 底层调 os.getcwd
```

而 `AppConfig.from_file()` 在 langgraph **每个请求的 authenticate 路径**里都被调用，每次构造新的 `DatabaseConfig`，每次访问 `sqlite_path` → 触发 `Path.resolve()` → `os.getcwd()` → blockbuster 抓 → 500。

langgraph 0.7.65 启用 BlockBuster middleware 检测 ASGI 事件循环上的同步阻塞调用。`make dev` 走 `LANGGRAPH_ALLOW_BLOCKING=1` 绕开（Makefile L113），所以**本地不复现**。**多用户部署不能用 ALLOW_BLOCKING** — event loop 被任何慢 syscall block 都会拖垮所有用户。

**走过的弯路**:

- ❌ 尝试 1: `model_post_init` 缓存到 PrivateAttr。失败：`AppConfig.from_file()` 每请求都重新构造 DatabaseConfig 实例，instance-level 缓存毫无意义。stack trace 上 `model_post_init` 仍然调 `Path.resolve()` 触发 BlockingError。
- ❌ 尝试 2: 模块加载期 prewarm cache。半失败：如果 sqlite_dir 是相对路径且 cache 未命中，第一次仍调 `Path.resolve()`。
- ❌ `os.path.abspath()` 替代 `Path.resolve()`：abspath 内部也调 `os.getcwd()`，同样被拦。

**最终修复**: `backend/packages/harness/deerflow/config/database_config.py`

```python
@lru_cache(maxsize=128)
def _resolve_sqlite_dir(raw: str) -> str:
    if os.path.isabs(raw):           # 字符串判断，不调 syscall
        return raw
    cwd = os.environ.get("PWD") or "/"   # dict lookup,不调 syscall
    return os.path.normpath(os.path.join(cwd, raw))  # 纯字符串操作
```

`os.environ['PWD']` 是 dict lookup（不 syscall），`os.path.isabs` / `os.path.normpath` / `os.path.join` 都是纯字符串操作。整条路径**永不触发 blockbuster**。

PWD 在容器里由 Dockerfile `WORKDIR /app` + compose `working_dir: /app` 设置，可靠。fallback `/` 是 defensive，正常 case 走不到。

**测试**: `backend/tests/test_database_config.py` — 8 个 case，最关键的 `test_no_path_resolve_or_getcwd_at_any_step` 用 mock 拦截 `Path.resolve` / `os.getcwd` / `os.path.abspath`，断言构造+所有 property 访问后调用次数 = 0。

---

### 6️⃣ `init_engine` 没有幂等 guard，每请求重跑 + 6 触发的同步 syscall

**症状**: 修完 #5 后浏览器继续 500，新 stack trace：`engine.py:93 → os.makedirs → BlockingError: Blocking call to os.mkdir`。

**根因**: `get_local_provider()` 在每个 langgraph_auth.authenticate 都调一次：
```python
cfg = AppConfig.from_file()
await init_engine_from_config(cfg.database)   # ← 每请求都来
```

`init_engine()` 没有 `_engine is not None` 短路，每请求重跑：(a) `os.makedirs(sqlite_dir)`（同步 syscall）；(b) `create_async_engine(...)`；(c) WAL pragma 监听器注册；(d) `Base.metadata.create_all`。每次都泄露 engine + WAL listener。

**修复 (一)**: `backend/packages/harness/deerflow/persistence/engine.py:init_engine` 函数体开头加：

```python
global _engine, _session_factory
if _engine is not None:
    return
```

幂等 guard：engine 已存在则直接返回，不重跑任何 init 逻辑。

**测试**: `backend/tests/test_persistence_engine_idempotent.py` — 3 个 case：
- 已 init 状态下再调 init_engine，断言 `os.makedirs` 调用 0 次
- `init_engine_from_config` 也走 guard
- 第一次 init 必须真跑（建目录、建 engine）

---

### 7️⃣ 即便有 guard，第一次 makedirs 仍是同步 syscall

**症状**: 加完 guard 后浏览器还是 500，stack trace 仍然 `engine.py:93 → os.makedirs`。

**根因**: idempotent guard 只在 `_engine is not None` 时短路。**第一次请求** `_engine` 还是 None，所以会真跑 `os.makedirs` —— 同步 syscall，trip blockbuster。多 worker 时每个 worker 进程各自跑一次第一次 init，所以浏览器看到的不是"前几次 500 之后好"，而是"持续 500"（每个 worker 各炸一次后总有新 worker 接力）。

**修复 (二)**: 同一文件 L106 改成：

```python
await asyncio.to_thread(os.makedirs, sqlite_dir or ".", exist_ok=True)
```

把同步 syscall 扔到 worker thread。**blockbuster 只监控主 event loop**，worker thread 里的 mkdir 不抓。这正是 langgraph dev BlockingError 提示的 "Quick fix: Move blocking operations to a separate thread"。

**验证**: 部署后日志 `Starting background run` + `Background run succeeded run_exec_ms=11655`，**没有 BlockingError**，完整 agent 执行链跑通。

---

## 改动文件汇总

### 本地（已修但未 commit）

| 文件 | 改动 |
|---|---|
| `packages/agent/docker/nginx/nginx.conf` | 删 3 upstream，改 set + per-request resolve；加 catch-all `/api/` |
| `packages/agent/docker/docker-compose.yaml` | gateway/langgraph `uv run uvicorn` → `.venv/bin/uvicorn` |
| `packages/agent/backend/packages/harness/deerflow/config/database_config.py` | 改用 PWD + lru_cache 纯字符串路径解析 |
| `packages/agent/backend/packages/harness/deerflow/persistence/engine.py` | 加幂等 guard + `os.makedirs` 走 `asyncio.to_thread` |
| `packages/agent/backend/tests/test_database_config.py` | 新文件，8 cases |
| `packages/agent/backend/tests/test_persistence_engine_idempotent.py` | 新文件，3 cases |

### ECS 端（hot-patch 已应用）

- `database_config.py` `engine.py` 通过 `tar | docker exec -i ... tar -x` 注入到容器 + 清 `__pycache__` + `docker restart` 重启
- `backend/packages/ethoinsight` symlink 改为真目录
- `docker-compose.yaml` 同步本地版本

---

## 必做收尾

### 1. Commit 改动持久化（最重要）

ECS 容器是 hot-patched，下次 `docker compose build` 会用 git 版本重建镜像，**所有修复全丢**。立即 commit：

```bash
cd /home/wangqiuyang/noldus-insight

# 反向 rsync 把 ECS 上 docker-compose.yaml 拿回本地（uv run → .venv/bin/...）
rsync -avz root@39.105.231.16:/root/noldus-insight/packages/agent/docker/docker-compose.yaml \
  packages/agent/docker/docker-compose.yaml

git status
git diff --stat packages/agent/docker/ \
  packages/agent/backend/packages/harness/deerflow/ \
  packages/agent/backend/tests/

git add packages/agent/docker/nginx/nginx.conf \
        packages/agent/docker/docker-compose.yaml \
        packages/agent/backend/packages/harness/deerflow/config/database_config.py \
        packages/agent/backend/packages/harness/deerflow/persistence/engine.py \
        packages/agent/backend/tests/test_database_config.py \
        packages/agent/backend/tests/test_persistence_engine_idempotent.py

git commit -m "$(cat <<'EOF'
fix(deploy): ECS 多用户 Docker 部署链路 7 处修复

nginx (docker-compose 模式专属):
- 静态 upstream 改 per-request resolve,避免容器重建 IP 漂移后 502
- 加 /api/ catch-all + Set-Cookie pass-through,对齐 nginx.local.conf

docker-compose:
- gateway/langgraph command 改走 .venv/bin/uvicorn,绕开 uv run 在 ECS 上 sync 卡住

deerflow harness (Tier 4 persistence + langgraph 0.7.65 blockbuster):
- database_config.py: Path.resolve() 改用 PWD + lru_cache 纯字符串路径解析,
  避免 ASGI 路径触发 BlockingError(os.getcwd)
- persistence/engine.py: init_engine 加幂等 guard + os.makedirs 走 asyncio.to_thread,
  避免每请求重复 init + 第一次 mkdir trip blockbuster

新增测试 11 case 覆盖关键不变量(filesystem-free property 访问、init_engine 幂等性).

详见 docs/handoffs/2026-05/2026-05-19-ecs-deploy-blockbuster-nginx-fix-handoff.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### 2. 后续 rsync 用 `-aL` 防 symlink 复发

```bash
rsync -aLvz --delete \
  /home/wangqiuyang/noldus-insight/packages/agent/ \
  root@39.105.231.16:/root/noldus-insight/packages/agent/
```

`-L` 让 rsync 解引用 symlink 复制真内容。这样 ECS 端永远是真目录，不会再触发问题 #4。

### 3. 下次 deerflow sync 必须 surgical merge 这两个文件

`backend/packages/harness/deerflow/config/database_config.py` 和 `backend/packages/harness/deerflow/persistence/engine.py` 已经偏离上游。下次 `./scripts/sync-deerflow.sh` 必须把它们当作"受保护文件"做 surgical merge：

- 保留 `_resolve_sqlite_dir` 函数 + PWD 实现
- 保留 `init_engine` 顶部的 `if _engine is not None: return`
- 保留 `await asyncio.to_thread(os.makedirs, ...)`
- 上游对这两个文件的非冲突修复手动合入

---

## 长期改进（建议，不阻塞当前可用）

### ⭐ 重要后置发现：Gateway mode 本可完全绕开今天 5/6/7

会话末尾用户问"deerflow 不是应该有部署最佳实践吗"，我重新调研 README + docker-compose + deploy.sh，发现 deerflow 有**两条 prod 路径**：

| 模式 | 命令 | langgraph 容器 | blockbuster | 我们今天踩的坑 |
|---|---|---|---|---|
| Standard | `make up` (`deploy.sh`) | 有，跑 `langgraph dev` | **强制开** | 5/6/7 全踩 |
| Gateway | **`make up-pro`** (`deploy.sh --gateway`) | **无**，runtime 嵌进 gateway 进程 | **不存在** | 全部规避 |
| Licensed | — (TODO in compose L119) | `langchain/langgraph-api:<ver>` (需 LangSmith license) | 关 | 全部规避 |

证据：

- `docker-compose.yaml:119`: `# TODO: switch to langchain/langgraph-api (licensed) once a license key is available. # For now, use 'langgraph dev' (no license required)`
- `README.md:253-254`: "The LangGraph agent server **currently runs via `langgraph dev`** (the open-source CLI server)."
- `backend/CLAUDE.md` "Runtime Modes": Gateway mode 用 deerflow 自己写的 `RunManager` + `StreamBridge`,**no LangGraph Server**
- `scripts/deploy.sh:259-262`: `--gateway` 模式 `services="frontend gateway nginx"`,完全跳过 langgraph 容器

**也就是说今天 5/6/7 三个 BlockingError 问题如果用 `make up-pro` 完全不会出现。** 一开始就该用 Gateway mode。

为什么本任仍跑完了 Standard mode 修复:
- backend/CLAUDE.md 把 Gateway mode 标 **experimental**,本任会话末用户选择"不动架构"(选项 γ)
- 今天修的 PWD + idempotent guard + asyncio.to_thread **三处仍是 Standard mode 必须**,也是好的上游 deerflow PR 候选
- Gateway mode 切换是下一个 sprint 的事

### 下一 sprint 方向(用户最终决策: γ + β 混合)

**γ. 短期**: 今天 7 处修复全部 commit + memory/handoff 写完(本任完成)

**β. 中期**: CI/CD 镜像发布流程
- GitHub Actions / 阿里云 CodePipeline build → push 阿里云 ACR
- ECS 上 `docker compose pull && up -d`,不再本机 build
- 解决今天 #3 (uv run sync 卡)、#4 (symlink) 这种 ECS 本机 build 才有的问题
- 镜像里固化所有 hot-patch,不再 docker cp 热补丁

**不在本 sprint 范围**:
- 试 Gateway mode (用户选 γ 不动架构)
- 买 LangSmith license

### 其他改进(原内容保留)

#### A. AppConfig 缓存（治本）

当前 `app/gateway/deps.py:get_local_provider()` 每个 authenticate 请求都跑 `AppConfig.from_file()` → 重新构造 DatabaseConfig → 进 init_engine（虽然有 guard 但仍多 1 个 lookup）→ 浪费 CPU/IO。

**应该改的**：把 `AppConfig` 缓存到 module level 或 FastAPI dependency cache，进程生命周期共用一份。这样 #5 #6 #7 都从根上变成"不可能发生"。

**为什么没做**：scope 大，要考虑 config hot-reload 行为（CLAUDE.md backend/CLAUDE.md 第 "Config Caching" 部分明确写了 mtime-based reload），动这一层要谨慎。留给下一任。

#### B. asyncio.Lock 防 init race condition

多个并发请求同时进 `init_engine`，都看到 `_engine is None`，都跑 init。`os.makedirs(exist_ok=True)` 是幂等的没事，但 `create_async_engine` 会创建多个 engine 对象，只有最后一个被 `_engine` 持有，其他泄露。

**修法**：

```python
_init_lock = asyncio.Lock()

async def init_engine(...):
    async with _init_lock:
        if _engine is not None:
            return
        ...
```

不是当前 500 的根因，但是健壮性提升。

#### C. blockbuster middleware 在 deerflow Tier 4 体系全审计

`Path.resolve` 和 `os.mkdir` 是被发现的两处。Tier 4 持久化层 / 事件 store / runs manager 还有多少同步 syscall 在 ASGI 路径上是未知数。建议跑一次 `LANGGRAPH_ALLOW_BLOCKING=0` 的压力测试，把全部 BlockingError 抓出来 batch fix。

**或者**: 切到 Gateway mode 让这整类问题消失(见上面 Runtime Modes 表)。

---

## 验证录像

```
2026-05-19T13:21:12 Starting background run    thread_id=247d8b0f run_id=019e4065...
2026-05-19T13:21:23 Background run succeeded   run_exec_ms=11655
2026-05-19T13:21:31 Starting background run    [继续下一轮]
```

浏览器 console 干净，没有 500，没有 BlockingError。

---

## 诊断方法学教训（给下一任）

1. **`docker exec container python -c ...` 不是 venv 的 python**。要用 `uv run python` 或 `.venv/bin/python` 才看到真实运行时状态。前一任 handoff 误诊根因就栽在这条。

2. **路径前缀能立刻区分 log 来源**。`/home/wangqiuyang/...` 是本地 dev，`/root/noldus-insight/...` 是 ECS host，`/app/backend/...` 是 docker 容器内。混淆会导致追错链路。

3. **`docker cp` 在含 mount 的容器上偶尔报 `mkdirat var/run: file exists`**。用 `tar -C srcdir -c file | docker exec -i container tar -C destdir -x` 流式注入更稳。

4. **改 Python 源码后必须 `find __pycache__ -name "module*" -delete` 清字节码**，否则 `docker restart` 后 Python 可能加载旧 `.pyc`。

5. **看 BlockingError stack trace 要看 stack 最后一行"deerflow.xxx"而不是中间的 starlette/uvicorn 帧**，那才是同步 syscall 真正的位置。修完一处会跳到下一处。

6. **`make dev` 用 `LANGGRAPH_ALLOW_BLOCKING=1` 兜底**，所以本地任何 BlockingError 都不会浮现。本地全绿不代表生产能跑。生产是多用户 ASGI，blockbuster 是真朋友。

---

## 相关文档

- 上一份（已作废）: [2026-05-19-ecs-deploy-gateway-pypi-fix-handoff.md](2026-05-19-ecs-deploy-gateway-pypi-fix-handoff.md)
- DeerFlow 同步 SOP: [docs/sop/deerflow-sync-sop.md](../../sop/deerflow-sync-sop.md)
- CLAUDE.md 第 13 条（Tier 4 体系已合入本仓库）: [CLAUDE.md](../../../CLAUDE.md)
- Backend CLAUDE.md（langgraph 0.7.65 + blockbuster 上下文）: [packages/agent/backend/CLAUDE.md](../../../packages/agent/backend/CLAUDE.md)
