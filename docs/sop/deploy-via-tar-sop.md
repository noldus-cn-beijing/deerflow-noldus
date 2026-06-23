# 通过镜像 tar 部署到远程 (1Panel / 自管 Docker, 不用镜像仓库)

**适用场景**:
- 你本地能 build 镜像 (Linux + Docker), 远程有 Docker
- 不想用 ACR/GHCR 等镜像仓库 (省钱 / 没权限)
- 不想 rsync 整个仓库 (太慢)

**核心流程**:

```
本地 (Linux)                     ECS 服务器
  ├─ docker compose build         │
  ├─ docker save → .tar.gz        │
  ├─ rsync 镜像 tar  ─────────────► /opt/ethoinsight/images/
  ├─ rsync compose/nginx/skills    │
  ├─ rsync config.yaml/extensions  │
  │                                ├─ docker load
  │                                ├─ docker compose up -d
  │                                └─ 1Panel 看状态/日志
```

---

## 1. 一次性准备 (远程)

### 1.1 ECS 上准备目录

```bash
ssh deploy@your-ecs
sudo mkdir -p /opt/ethoinsight/{images,docker,skills,runtime,deer-flow-home}
sudo chown -R $USER:$USER /opt/ethoinsight
```

### 1.2 1Panel 注册这套 stack

1Panel 控制台 → **容器 → 编排** → 添加:
- **名称**: `ethoinsight`
- **Compose 文件路径**: `/opt/ethoinsight/docker/docker-compose.yaml`
- **保存** (此时还跑不起来, 镜像没传)

之后每次部署后, 1Panel 编排页面会自动检测到容器状态更新。

### 1.3 域名 + HTTPS (1Panel 反代)

1Panel → **网站 → 创建网站 → 反向代理**:
- 域名: `ethoinsight.yourdomain.com`
- 代理目标: `http://127.0.0.1:2026` (容器 nginx 暴露的端口)
- 申请 Let's Encrypt 证书 (1Panel 一键)

容器内 nginx 不再需要管 HTTPS, 只对内 listen 2026。

---

## 2. 一次性准备 (本地)

### 2.1 SSH 免密 + Host 别名 (推荐)

`~/.ssh/config`:

```
Host eth-ecs
    HostName 1.2.3.4
    User deploy
    IdentityFile ~/.ssh/id_ed25519
```

测试: `ssh eth-ecs echo ok`

### 2.2 保存生产配置文件 (本地, 在 git 之外)

`config.yaml` 和 `extensions_config.json` 含 API key, **不能进 git**。
建议放在你机器上专门一个目录, 比如:

```bash
mkdir -p ~/ethoinsight-prod
cp packages/agent/config.example.yaml ~/ethoinsight-prod/config.yaml
cp packages/agent/extensions_config.json ~/ethoinsight-prod/extensions_config.json
# 编辑 ~/ethoinsight-prod/config.yaml, 填入真实 API key
chmod 600 ~/ethoinsight-prod/*
```

### 2.3 设环境变量 (推荐: 加到 `.envrc` 或 `~/.bashrc`)

```bash
export DEPLOY_HOST="eth-ecs"
export DEPLOY_PATH="/opt/ethoinsight"
export DEPLOY_CONFIG="$HOME/ethoinsight-prod/config.yaml"
export DEPLOY_EXTENSIONS="$HOME/ethoinsight-prod/extensions_config.json"
```

---

## 3. 日常部署

```bash
cd packages/agent
make deploy-tar
```

脚本会:
1. `linux/amd64` 平台 build 镜像 (frontend + backend)
2. 用 git short hash 打 tag, 导出双镜像 + gzip 到 `/tmp/ethoinsight-<tag>.tar.gz`
3. rsync 镜像 tar + docker compose + nginx + skills + config 到远程
4. 远程 `docker load` + `docker compose up -d`
5. 清理 7 天前未使用的旧镜像

整个过程预期 ~3-5 分钟 (主要是 build 和首次镜像传输; 后续 rsync 增量很快)。

### 仅 build 不部署 (测试构建)

```bash
SKIP_SHIP=1 KEEP_TAR=1 make deploy-tar
```

### 复用已有 build (改了 config 只想推过去)

```bash
SKIP_BUILD=1 make deploy-tar
```

### 指定版本 tag (回滚场景)

```bash
DEPLOY_TAG=v0.1.0 make deploy-tar
```

---

## 3.5 数据库 schema 迁移 (⚠️ 加了新列/新表的版本必看)

> **2026-06-22 起:`make deploy-tar` 已自动跑迁移**(见下「自动迁移」)。本节保留是为了
> ① 理解原理 ② 手动补迁移(老部署、首次接入、自动步骤被跳过时)③ 排障。

应用启动时只跑 `Base.metadata.create_all`,它是 **create-if-not-exists**:
- **全新表** → 建表时带上所有列(含新列),没问题。
- **已存在的表**(如 `feedback`、`threads_meta`、`runs`)→ **不会补新列**。代码一旦查/写新列就会 `sqlite3.OperationalError: no such column: ...`(典型:`runs.token_usage_by_model` 缺失 → `/chats/new` 每次 500;`feedback.paradigm` 缺失 → 反馈/prior_corrections 报错)。

> ECS 上 `runs`/`feedback` 等表早有历史数据,属于"已存在的表"。生产真实库是 checkpointer 与 app 共用的 `${DEER_FLOW_HOME}/data/deerflow.db`(容器内 `/app/backend/.deer-flow/data/deerflow.db`),**不是** `alembic.ini` 里硬编码的 `./data/deerflow.db`。

### 自动迁移(默认,无需手动操作)

`scripts/deploy-via-tar.sh` 在 `docker load` 之后、`docker compose up` **之前**,会用刚加载的 gateway 镜像跑一次 `scripts/run-db-migrations.sh`(镜像里自带 alembic + 迁移文件;脚本随部署 rsync 到 `$DEPLOY_PATH/scripts/` 并 bind-mount 进容器),对宿主持久化的真实库执行迁移:

```
→ Running DB migrations (alembic) inside gateway image before service start
✓ DB migrations applied (or already up to date)
```

- **幂等**:已在 head 则空跑;只应用未执行的 revision。
- **自动 stamp 老库**:生产库历史上靠 `create_all` 建表、**没有 `alembic_version` 表**。脚本会按"已存在哪些迁移加的列"探测真实 schema 停在哪个 revision,先 `alembic stamp <该 revision>` 补上版本表,再 `upgrade head`。**这样补一次后版本链就接上了,以后 sync 加列自动 `upgrade head` 即可。**
- **失败即中止部署**:迁移失败会 `exit 1`,**不启动新容器**——宁可旧代码(可用)继续跑,也不让新代码撞未迁移的库。这时看报错按下面「手动补迁移」排查。

> ⚠️ 千万别绕过自动步骤后直接 `alembic upgrade head`:生产库没有 `alembic_version` 表,直接 upgrade 会从最早的 revision 重放、撞上 `create_all` 已建好的列报 `duplicate column`。必须先 stamp(脚本已自动处理)。

### 手动补迁移(老部署 / 自动步骤被跳过 / 排障)

把脚本拷进运行中的 gateway 容器再跑(它会先备份再迁移;`docker cp` 比 `exec -v` 可靠——`exec` 不支持临时挂卷):

```bash
ssh eth-ecs
# 1) 确保本地脚本已推到 ECS(老部署可能没有)
#    rsync -avz scripts/run-db-migrations.sh eth-ecs:/opt/ethoinsight/scripts/
# 2) 拷进容器并执行
docker cp /opt/ethoinsight/scripts/run-db-migrations.sh deer-flow-gateway:/app/scripts/run-db-migrations.sh
docker compose -p deer-flow exec gateway sh -c \
  "chmod +x /app/scripts/run-db-migrations.sh && /app/scripts/run-db-migrations.sh"
```

若服务已停(或想在不影响旧容器的前提下迁移),用一次性容器 `run --rm`(支持 `-v` 挂卷):

```bash
cd /opt/ethoinsight/docker
docker compose -p deer-flow run --rm --no-deps \
  -v /opt/ethoinsight/scripts/run-db-migrations.sh:/app/scripts/run-db-migrations.sh:ro \
  --entrypoint sh gateway \
  -c "chmod +x /app/scripts/run-db-migrations.sh && /app/scripts/run-db-migrations.sh"
```

脚本输出会显示走了哪条路径(`upgrade head` / `Legacy DB detected ... stamp <rev>`)并在结尾校验 `alembic_version = 20260622_1700`(当前 head)。

**最后兜底**(脚本不可用时,内联 Python——⚠️ 仅当库**已有** `alembic_version` 表时可直接 upgrade;否则先 stamp):

```bash
docker compose -p deer-flow exec gateway sh -c 'cd /app/backend && PYTHONPATH=. uv run python - <<PY
from alembic.config import Config
from alembic import command
from deerflow.config import get_app_config

mig = "/app/backend/packages/harness/deerflow/persistence/migrations"
url = get_app_config().database.app_sqlalchemy_url   # 真实库,与运行时同一文件
cfg = Config(f"{mig}/alembic.ini")
cfg.set_main_option("script_location", mig)
cfg.set_main_option("sqlalchemy.url", url)
command.current(cfg)              # 先看停在哪;若空 → 该库无 version 表,需先 command.stamp(cfg, "<baseline>")
command.upgrade(cfg, "head")
print("done")
PY'
```

> 取 url 的 import 路径若与版本不符,把 `url = ...` 换成写死路径 `url = "sqlite+aiosqlite:////app/backend/.deer-flow/data/deerflow.db"`(sqlite 绝对路径是 **4 个斜杠**)。
> baseline 怎么定:看真实 schema 已有哪些迁移加的列——有 `runs.token_usage_by_model` → 已在 head;只有 `feedback.paradigm` → `20260601_1500`;只有 `feedback.verdict` → `20260512_1200`;都没有 → `base`。`run-db-migrations.sh` 已把这套探测自动化,优先用它。

校验列已加:

```bash
docker compose -p deer-flow exec gateway sh -c \
  "uv run python -c \"import sqlite3,os; db='/app/backend/.deer-flow/data/deerflow.db'; print('runs:', [r[1] for r in sqlite3.connect(db).execute('PRAGMA table_info(runs)')])\""
# 输出应含 'token_usage_by_model'
```

> alembic env.py 已 scope 到 deerflow 自己的表,**不碰 LangGraph checkpointer 表**(两者共用同一 .db 文件,但 checkpointer schema 由 LangGraph 自管)。
> 迁移用 `op.batch_alter_table`(SQLite ALTER 必需),`add_column(nullable=True)`,历史行新列留 NULL,不破坏老数据。

### 回滚时

代码回滚到加列之前的镜像,**但已迁移的 DB 仍带新列** —— `nullable=True` 的新列对旧代码无害(旧代码不读它),所以**通常无需 downgrade**。确需回退 schema 才 `alembic downgrade -1`。

---

## 4. 远程操作 (无需 ssh)

通过 1Panel 面板:

| 操作 | 1Panel 路径 |
|---|---|
| 看容器状态 | 容器 → 容器 |
| 看日志 | 容器 → 容器 → 选 `deer-flow-gateway` → 日志 |
| 重启 | 容器 → 编排 → ethoinsight → 重启 |
| 改环境变量 | **不要在 1Panel 改** — 改本地 `~/ethoinsight-prod/config.yaml` 然后 `make deploy-tar` |
| 备份 .deer-flow 数据 | 计划任务 → 添加 → tar `/opt/ethoinsight/deer-flow-home/` 到 OSS |

### 命令行调试 (ssh)

```bash
ssh eth-ecs
cd /opt/ethoinsight/docker
docker compose -p deer-flow logs -f gateway     # 看 gateway 日志
docker compose -p deer-flow ps                  # 看状态
docker compose -p deer-flow restart gateway     # 重启某个服务
```

---

## 5. 回滚

每次部署的镜像 tag 会保留在 `/opt/ethoinsight/images/`。回滚:

```bash
ssh eth-ecs
cd /opt/ethoinsight
ls images/                                      # 看历史 tag
# 编辑 docker/docker-compose.yaml, 把 image: deer-flow-frontend
# 改成 deer-flow-frontend:<旧 tag>, deer-flow-gateway 同理
docker compose -p deer-flow up -d
```

也可以在本地用 `DEPLOY_TAG=<旧 tag> SKIP_BUILD=1 make deploy-tar` 重新指定 tag 推一次。

---

## 6. 常见问题

### Q: ECS 磁盘满了
镜像每次 build 都会留 layer 缓存。脚本内部跑 `docker image prune -af --filter "until=168h"` 自动清 7 天前未用镜像。手动:
```bash
ssh eth-ecs docker image prune -af
ssh eth-ecs rm -rf /opt/ethoinsight/images/ethoinsight-*.tar.gz  # 清旧 tar
```

### Q: 部署后访问 502 / 不通
1. 看容器是否起来: `docker compose -p deer-flow ps`
2. 看 gateway 日志: `docker compose -p deer-flow logs gateway --tail 100`
3. 常见原因: `config.yaml` 里 API key 没填 / model 配置错; 或 `config.yaml` 没 rsync 过去

### Q: frontend 报 `BETTER_AUTH_SECRET` 错误
脚本会在远程 `$DEER_FLOW_HOME/.better-auth-secret` 自动生成并持久化。如果你想自己指定:
```bash
ssh eth-ecs "echo 'YOUR_SECRET' > /opt/ethoinsight/deer-flow-home/.better-auth-secret && chmod 600 $_"
```

### Q: 本地 build 出来在 ECS 跑不起来 (架构不匹配)
脚本已经强制 `DOCKER_DEFAULT_PLATFORM=linux/amd64`。如果你的本地是 Mac M 系列, 第一次 build 会下载 amd64 base image (慢), 后续走缓存。

### Q: 想换公网入口端口 (默认 2026)
远程 `/opt/ethoinsight/docker/docker-compose.yaml` 的 `nginx.ports` 可改, 或在 ssh 时设 `export PORT=8080`。不过推荐保持 2026, 让 1Panel 反代处理公网端口 (80/443)。

### Q: GATEWAY_WORKERS 应该设多少？
默认 **1**（单 worker、所有 run 同进程内存、切 thread 重连无 409）。调大（如 `GATEWAY_WORKERS=4`）需先有共享 StreamBridge（Redis 版，上游标 Phase 2 未实现），否则切 thread 再切回会触发 HTTP 409（前端已优雅降级不弹错误 toast，但正在跑的 run 的实时增量看不到）。详见他 spec：`docs/superpowers/specs/2026-06-09-gateway-multiworker-stream-409-fix-spec.md`。

---

## 7. 不再使用 ACR / GitHub Actions

部署完整体走本地 + ssh, **CLAUDE.md 第 13 条提到的 ACR pipeline 暂不启用**。如果未来要切到 ACR (比如有多 ECS 节点, 或要给团队多人 push 镜像), 重新启用 `.github/workflows/build-push-acr.yml` 并把这套 `deploy-via-tar.sh` 切到 `docker compose pull` 即可, 配置文件部分 (config.yaml/extensions/skills) 仍然走 rsync。
