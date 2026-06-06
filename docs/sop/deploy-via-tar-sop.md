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

**部署链不自动跑 alembic。** `make deploy-tar` 只做 `docker load` + `docker compose up -d`;生产 `gateway` 容器启动命令里**没有** `alembic upgrade head`。

应用启动时只跑 `Base.metadata.create_all`,它是 **create-if-not-exists**:
- **全新表** → 建表时带上所有列(含新列),没问题。
- **已存在的表**(如 `feedback`、`threads_meta`、`runs`)→ **不会补新列**。代码一旦查/写新列就会 `sqlite3.OperationalError: no such column: ...`。

> ECS 上 `feedback` 表自训练飞轮上线后一直有历史数据(见 CLAUDE.md 飞轮条),属于"已存在的表"。

### 何时需要手动迁移

本次部署的代码**给已存在的表加了列/约束**时(典型信号:`persistence/*/model.py` 改了 `mapped_column`,且 `migrations/versions/` 下有新 revision 文件)。新表、纯逻辑改动无需迁移。

**已知需要迁移的版本**:
- **S8 feedback `paradigm` 列**(`20260601_1500_feedback_paradigm.py`):给已有 `feedback` 表加 `paradigm`。不跑 → 提交反馈 + lead 查 prior_corrections 全部 500/报错。

### 操作(部署后,在 ECS 上跑一次)

⚠️ **不要直接 `alembic upgrade head`,也不要靠 `-x sqlalchemy.url=`**:
- `alembic.ini` 里 `sqlalchemy.url` 硬编码成 `./data/deerflow.db`,**不是**生产真实 DB。真实库是 checkpointer 与 app 共用的 `${DEER_FLOW_HOME}/data/deerflow.db`(容器内 `/app/backend/.deer-flow/data/deerflow.db`)。
- 本仓库的 `env.py` 用 `config.get_main_option("sqlalchemy.url")` 取 url,**既不读 `-x` 参数也不读环境变量**,所以 `alembic -x sqlalchemy.url=...` 会被无视、仍迁错库。

正确做法:用内联 Python 调 `command.upgrade` 并 `set_main_option` 显式覆盖真实 url(下面这段已实测可用):

```bash
ssh eth-ecs
cd /opt/ethoinsight/docker

docker compose -p deer-flow exec gateway sh -c 'cd /app/backend && PYTHONPATH=. uv run python - <<PY
from alembic.config import Config
from alembic import command
from deerflow.config import get_app_config

mig = "/app/backend/packages/harness/deerflow/persistence/migrations"
url = get_app_config().database.app_sqlalchemy_url   # 真实库,与运行时同一文件
cfg = Config(f"{mig}/alembic.ini")
cfg.set_main_option("script_location", mig)
cfg.set_main_option("sqlalchemy.url", url)
print("current:")
command.current(cfg)
print(f"upgrading head on: {url}")
command.upgrade(cfg, "head")
print("done")
PY'
```

> **兜底**(取 url 的 import 路径与版本不符时):把上面 `url = ...` 那行换成写死的真实路径
> `url = "sqlite+aiosqlite:////app/backend/.deer-flow/data/deerflow.db"`
> (sqlite 绝对路径是 **4 个斜杠**:`sqlite+aiosqlite:////app/...`)

校验列已加(以 `feedback.paradigm` 为例):

```bash
docker compose -p deer-flow exec gateway sh -c \
  "uv run python -c \"import sqlite3; print([r[1] for r in sqlite3.connect('/app/backend/.deer-flow/data/deerflow.db').execute('PRAGMA table_info(feedback)')])\""
# 输出应含 'paradigm'
```

> alembic env.py 已 scope 到 deerflow 自己的表,**不碰 LangGraph checkpointer 表**(两者共用同一 .db 文件,但 checkpointer schema 由 LangGraph 自管)。
> 迁移用 `op.batch_alter_table`(SQLite ALTER 必需),`add_column(nullable=True)`,历史行新列留 NULL,不破坏老数据。重复跑只执行未应用的 revision(`command.current` 先看停在哪)。

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

---

## 7. 不再使用 ACR / GitHub Actions

部署完整体走本地 + ssh, **CLAUDE.md 第 13 条提到的 ACR pipeline 暂不启用**。如果未来要切到 ACR (比如有多 ECS 节点, 或要给团队多人 push 镜像), 重新启用 `.github/workflows/build-push-acr.yml` 并把这套 `deploy-via-tar.sh` 切到 `docker compose pull` 即可, 配置文件部分 (config.yaml/extensions/skills) 仍然走 rsync。
