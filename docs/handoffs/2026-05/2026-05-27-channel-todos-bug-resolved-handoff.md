# 2026-05-27 ECS Channel 'todos' bug 根因修复 + deploy 链路 3 层缺陷统一修复

## TL;DR

本次会话连续修复了 3 个独立缺陷,全部属于 **部署链路** 沉默失败一类:

1. **bug 表象**: ECS `is_plan_mode=True` 的所有 FST 请求报 `ValueError: Channel 'todos' already exists with a different type`
2. **真根因 1 (镜像漏推)**: `scripts/deploy-via-tar.sh` 只 tag/save 了 `deer-flow-gateway` + `deer-flow-frontend`,**漏了 `deer-flow-langgraph`**;远端 langgraph 容器一直跑旧代码(commit 2ef93308 `_TodoState` 修复 5-25 合 dev 后,langgraph 镜像从未真正部署成功)
3. **真根因 2 (blocking 豁免缺失)**: 修复根因 1 后又出新错 `Blocking call to ScandirIterator.__next__`。`make dev` 命令行带 `BG_JOB_ISOLATED_LOOPS=true + --allow-blocking`,但 ECS `docker-compose.yaml` 把 `--allow-blocking` 做成可选(默认关),且 `BG_JOB_ISOLATED_LOOPS` 根本没 export;dev 与 prod 行为不对齐
4. **真根因 3 (compose 不重建容器)**: 镜像或 env 变了,但 `docker compose up -d` 看 service spec 没变就跳过 recreate。这是镜像漏推 + env 不生效两类故障的共同放大器

**最终修复**:
- `docker-compose.yaml` langgraph 服务 command 写死 `--allow-blocking`,environment 加 `BG_JOB_ISOLATED_LOOPS=true`(与 `backend/Makefile dev` 对齐,**做成产品属性,不再做环境开关**)
- `deploy-via-tar.sh` 加 `LANGGRAPH_IMAGE` tag/save + 远端 `docker compose up -d` 加 `--force-recreate`(永久解决沉默跳过容器重建)
- 用户跑一次 `NO_CACHE=1 make deploy-tar`,**所有 3 个根因一次到位**,无需手动 SSH 操作。前端实测 FST 端到端请求通过

---

## 故障定位过程(也是教训本身)

### 前一轮 handoff 的根因假设是错的

`2026-05-26-channel-todos-bug-diagnosis-handoff.md` 主推"set 迭代顺序 + `PYTHONHASHSEED` 不确定性"。本轮 grill 后发现:

1. 本地 `make_lead_agent(is_plan_mode=True)` 跑 11 个不同 PYTHONHASHSEED(0/1/2/3/4/5/42/100/999/12345)**全部 OK**,如果真是 hash 决定一定有翻车的 seed
2. `_resolve_schema` 本地 dump 显示:进入 `state_schemas` set 的 12 个 schema 里**只有 ThreadState 贡献 todos**,无 OmitFromInput 标记的竞争版本,无 set 顺序敏感性
3. 前一轮 handoff 末尾提到的 "ECS `/app/backend/` 顶层残留 `todo_middleware.py` 和 `test_todo_middleware.py`" 这条线索一查 — md5 = `0228fe3bd0c2160c064a45de71fcceec`(**本地新版**),反而 canonical 路径 `/app/backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py` md5 = `7e1d7fb32a489dd3dec6c855bf817f73`(**旧版,缺 _TodoState**)
4. 那两个顶层 .py 是开发者之前 SSH 进容器手动 `cp` 上去想 hotpatch 的,但 import path 是 `deerflow.agents.middlewares.todo_middleware`(包路径),不是顶层模块,**完全没生效**

### 真正的故障链(根因 1 — 镜像漏推)

```
本地 commit 2ef93308 修复 _TodoState (5-25)
  ↓
make deploy-tar 多次执行
  ↓
docker compose build 同时建出 deer-flow-gateway:latest 和 deer-flow-langgraph:latest 两个独立 image ID
  ↓
deploy-via-tar.sh 只 docker tag/save 了 gateway+frontend,漏 langgraph
  ↓
ECS 远端 docker load 只覆盖 gateway+frontend,langgraph:latest 还是上次的老 image ID
  ↓
docker compose up -d 检查 langgraph 服务的 image ID — 没变 — 跳过容器重建
  ↓
langgraph 容器一直跑老代码(旧 todo_middleware.py 不含 _TodoState)
  ↓
is_plan_mode=True 时 TodoMiddleware 父类 state_schema=PlanningState
     + ThreadState 同时进 state_schemas set
     + PlanningState.todos: Annotated[..., OmitFromInput] (langchain 自带,无 reducer)
     + ThreadState.todos: Annotated[list|None, merge_todos]
  ↓
_resolve_schema(StateSchema, omit_flag=None) 两个 todos 都进 all_annotations,后到的覆盖
     → 取决于 set 迭代顺序,todos 可能解析为 LastValue
_resolve_schema(InputSchema, omit_flag='input') PlanningState 那个被 OmitFromInput 排掉
     → todos 永远是 BinaryOperatorAggregate
  ↓
StateGraph.__init__ 先 add state_schema (todos=LastValue) 再 add input_schema (todos=BinaryOperatorAggregate)
     → channel 类型冲突
     → ValueError: Channel 'todos' already exists with a different type
```

### 根因 2 — blocking 豁免缺失(用户问"为什么 make dev 没有这种问题")

修完根因 1 后,新错出现:`Error: Blocking call to ScandirIterator.__next__`。这是 3 个独立缺陷叠加:

| # | 问题 | 是否 deerflow 上游缺陷 | 是否部署环境缺陷 | 谁的责任 |
|---|---|---|---|---|
| 1 | langgraph dev 拒绝同步 blocking I/O | **不是** | **不是** | langgraph 框架的默认行为(故意 feature) |
| 2 | code-executor 实际跑 `os.scandir` 等同步 I/O | **是** | 不是 | deerflow + ethoinsight(短期不修) |
| 3 | `docker-compose.yaml` 反向默认 + agent.env 漏配 | 不是 | **是** | noldus-insight 部署疏漏 |

**所以 `make dev` 从来没遇到这个错,不是因为代码没问题,而是因为 `backend/Makefile` 命令行写死了两个豁免**:
```makefile
dev:
    BG_JOB_ISOLATED_LOOPS=true uv run langgraph dev ... --allow-blocking ...
```

**ECS 部署链丢了 make dev 已做出的"产品需要 blocking 豁免"决策**。`docker-compose.yaml` 当年写的时候把它做成 `if [ "$${LANGGRAPH_ALLOW_BLOCKING:-0}" = "1" ]; then args="$$args --allow-blocking"; fi`(默认关),`BG_JOB_ISOLATED_LOOPS` 在 environment 段根本不存在。

### 根因 3 — compose 不重建容器是 1+2 的放大器

即便修了 1 和 2,如果不加 `--force-recreate`:
- 镜像变了但 image ID 检测漏了 → 容器跑旧代码
- env_file 内容变了但 inline `environment:` 没变 → 容器跑旧 env

这是 `docker compose up -d` 的设计行为(spec-hash diff 不检测 env_file 内容、不检测 image ID 变化历史),不是 bug,但**沉默失败的代价远大于强制重建几秒**,所以选 force-recreate。

### 关键证据(本地实测)

```
$ md5sum /home/wangqiuyang/noldus-insight/packages/agent/backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py
0228fe3bd0c2160c064a45de71fcceec  # ← 本地 dev HEAD 94867afd, 含 _TodoState 修复

# ECS langgraph 容器(本次部署前)
$ docker exec deer-flow-langgraph md5sum /app/backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py
7e1d7fb32a489dd3dec6c855bf817f73  # ← 旧版

# 本地 docker images(注意 langgraph 没有 :94867afd tag)
deer-flow-gateway:94867afd     fcd30ee88e7d
deer-flow-gateway:latest       fcd30ee88e7d   ← 同步
deer-flow-langgraph:latest     4d6f384afebe   ← 完全不同 ID,没有 :94867afd!
```

---

## 修复实施(三层全部完成)

### Layer 1: 镜像漏推 — `scripts/deploy-via-tar.sh`

```diff
- BACKEND_IMAGE="deer-flow-gateway"  # gateway and langgraph share the backend image
+ BACKEND_IMAGE="deer-flow-gateway"
+ LANGGRAPH_IMAGE="deer-flow-langgraph"
```

```diff
  docker tag "${BACKEND_IMAGE}:latest"  "${BACKEND_IMAGE}:${DEPLOY_TAG}"
+ docker tag "${BACKEND_IMAGE}:latest" "${LANGGRAPH_IMAGE}:latest"
+ docker tag "${BACKEND_IMAGE}:latest" "${LANGGRAPH_IMAGE}:${DEPLOY_TAG}"

  docker save \
      "${FRONTEND_IMAGE}:latest" "${FRONTEND_IMAGE}:${DEPLOY_TAG}" \
      "${BACKEND_IMAGE}:latest"  "${BACKEND_IMAGE}:${DEPLOY_TAG}" \
+     "${LANGGRAPH_IMAGE}:latest" "${LANGGRAPH_IMAGE}:${DEPLOY_TAG}" \
      | gzip -c > "$TAR_PATH"
```

### Layer 2: blocking 豁免 — `docker/docker-compose.yaml` langgraph 服务

```diff
- command: sh -c 'cd /app/backend && args="--no-browser --no-reload --host 0.0.0.0 --port 2024 --n-jobs-per-worker $${LANGGRAPH_JOBS_PER_WORKER:-10}" && if [ "$${LANGGRAPH_ALLOW_BLOCKING:-0}" = "1" ]; then args="$$args --allow-blocking"; fi && /app/backend/.venv/bin/langgraph dev $$args'
+ # --allow-blocking is mandatory: ethoinsight code-executor + deerflow harness contain
+ # sync I/O ... Mirrors backend/Makefile `dev`. Do not gate this behind an env var — the
+ # exemption is a property of this product, not a per-environment toggle.
+ command: sh -c 'cd /app/backend && /app/backend/.venv/bin/langgraph dev --no-browser --no-reload --host 0.0.0.0 --port 2024 --n-jobs-per-worker ${LANGGRAPH_JOBS_PER_WORKER:-10} --allow-blocking'

  environment:
    - DEER_FLOW_SANDBOX_HOST=host.docker.internal
+   - BG_JOB_ISOLATED_LOOPS=true
```

**核心理念**: 把豁免从 "用户在 .env 里配置" 变成 "产品代码内置",dev/prod 行为对齐。

### Layer 3: 容器一定重建 — `scripts/deploy-via-tar.sh`

```diff
- docker compose -p deer-flow -f docker-compose.yaml up -d --remove-orphans \
+ # --force-recreate is mandatory: compose normally skips rebuilding ... (long comment
+ # explaining sleuthing session cost)
+ docker compose -p deer-flow -f docker-compose.yaml up -d --remove-orphans --force-recreate \
      frontend gateway langgraph nginx
```

### 验证

部署后用户实测:
- langgraph 容器 `/app/backend/packages/harness/deerflow/agents/middlewares/todo_middleware.py` md5 = `0228fe3bd0c2160c064a45de71fcceec` ✅
- langgraph 启动日志: `Welcome to LangGraph` + `0.0.0.0:2024`,无 ValueError ✅
- 前端实测 FST 强迫游泳端到端分析请求: ✅ 无 `Channel 'todos' already exists`,无 `Blocking call to ScandirIterator`

---

## 经验沉淀(已写入 memory)

### 教训 1: compose build 同 Dockerfile 也产独立镜像 tag

`gateway` 和 `langgraph` 共用 `backend/Dockerfile`,但 `docker compose build` 会为**每个服务**独立产出 `deer-flow-<service>:latest`。如果 deploy 链路只 save/ship 其中一个,另一个的镜像 tag 在远端永远停留在上次的 image ID,**compose up 会沉默地跳过容器重建**。

→ `feedback_deploy_compose_per_service_image_tag.md`

### 教训 2: dev 和 prod 行为决策必须对齐到代码

`backend/Makefile` 的 `dev` 命令行写死了 `BG_JOB_ISOLATED_LOOPS=true + --allow-blocking`,但 `docker-compose.yaml` 没把这俩复制过去 — 部署链路"丢"了产品属性决策。

→ `feedback_dev_prod_behavior_alignment.md`(新建)

### 教训 3: 手动 `cp` 文件进容器 hotpatch 几乎从不生效

容器内 Python import 走的是 `deerflow.*` 包路径,不是文件名顶层模块。把 `todo_middleware.py` 拷到 `/app/backend/` 顶层**不会**被 import 系统使用,只会迷惑后续诊断。这次开发者的 hotpatch 尝试浪费了上一轮 agent 大量推理时间(误导出 "set hashseed" 假设)。

**正确做法**: 永远走 deploy 流程,不要 SSH 进容器 patch 代码。

### 教训 4: deploy 后必须验镜像 md5

报"X 部署了但行为没变"的 bug 时,**第一步**永远是 `docker exec <容器> md5sum <key 文件>` 与本地 dev HEAD 比对,**不是**调试代码。镜像漏推 / 容器没重建是最便宜的检查,代价是 5 秒。

---

## 仓库状态

- **dev HEAD**: `94867afd`(本次未 commit)
- **未提交改动**:
  - `packages/agent/docker/docker-compose.yaml` — langgraph command 写死 `--allow-blocking`,environment 加 `BG_JOB_ISOLATED_LOOPS=true`(本次新增,根因 2)
  - `packages/agent/scripts/deploy-via-tar.sh` — LANGGRAPH_IMAGE tag/save(本次新增,根因 1) + `--force-recreate`(本次新增,根因 3)
  - `packages/agent/scripts/deploy.sh` — `NO_CACHE=1` 支持(上次会话留下,保留)
- **生产 ECS**: 已部署最新 dev + 三层修复,所有验证通过
- **未消化的 handoff**:
  - `docs/handoffs/2026-05/2026-05-26-channel-todos-bug-diagnosis-handoff.md` — 根因假设有误(set hashseed),建议加 redirect 指向本文档

---

## 下一 agent 的待办

1. **commit + push** 本次三处改动,提交信息示例:
   ```
   fix(deploy): 三层修复 — langgraph 镜像漏推 + blocking 豁免缺失 + compose 不重建

   - docker-compose.yaml: langgraph 写死 --allow-blocking, BG_JOB_ISOLATED_LOOPS=true (与 make dev 对齐)
   - deploy-via-tar.sh: 加 LANGGRAPH_IMAGE tag/save + --force-recreate
   - 解决长期潜伏的"deploy 报成功但容器跑旧代码"沉默失败问题

   修复: ECS Channel 'todos' bug + Blocking call to ScandirIterator
   ```
2. **更新前一份 handoff**: 在 `2026-05-26-channel-todos-bug-diagnosis-handoff.md` 顶部加 redirect,标注 hashseed 假设证伪,真根因见本文档
3. **可选 — 加 deploy 自检脚本**: deploy 完跑一遍 `docker exec <container> md5sum <key file>` 比对本地与远端,不一致 exit non-zero。这次 bug 浪费了多轮 agent 推理时间,加 1 分钟自检值得

---

## milestone 建议

部署链路 track 本轮发生重要 checkpoint(同时修复 3 个独立缺陷,部署链路从此不再有"沉默成功"故障类),建议更新 `docs/milestone/deployment-pipeline.md`(如不存在则新建),关键摘要:

- **2026-05-26**: 部署流水线建立(make deploy-tar / deploy-via-tar.sh / 1Panel + nginx)
- **2026-05-27**: 三层修复 — langgraph 镜像不再漏推 + blocking 豁免硬编码 + 容器强制重建。dev 和 prod 行为完全对齐
- **下一里程碑候选**: ACR pipeline 启用后(`.github/workflows/build-push-acr.yml`),make deploy-tar 退役;此时把 docker-compose.yaml 改回 langgraph-api(带 license),`--allow-blocking` 可移除

