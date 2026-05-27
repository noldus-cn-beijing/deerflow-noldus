# 2026-05-26 会话交接 — 部署流水线建立 + ev19_facts 容器修复

## 当前任务目标

建立 EthoInsight 生产部署流水线（不依赖 ACR），并修复生产容器中 FST 端到端报 `_facts.json` 缺失的问题。

## 当前进展

### ✅ 全部已合入 dev

| PR | 内容 | 合并 commit |
|---|---|---|
| #44 path-pollution-defense | 三层防御封堵 subagent handoff 宿主机路径泄漏 | `0cdbc37c` |
| #45 chart-maker-fst-contract | 统一 plot 脚本 --inputs JSON 契约 + user_intent 过滤 + 失败硬规范 | `bf534017` |
| #47 deploy-via-tar | 本地 build → 镜像 tar → ECS docker load，不用 ACR | `454ad507` |
| #49 ev19-facts-package-relative-path | `_facts.json` 改用包内自包含路径，修复容器内路径解析失败 | `919cad1d` |

**dev HEAD**: `919cad1d`

### ECS 生产环境现状

- ECS IP: `39.105.231.16`，用户: `root`
- 部署路径: `/opt/ethoinsight/`
- 4 个容器 Running: `deer-flow-nginx` / `deer-flow-gateway` / `deer-flow-langgraph` / `deer-flow-frontend`
- 1Panel 已配置: OpenResty 反代 `http://39.105.231.16 → 127.0.0.1:2026`
- **最后一次 deploy**: PR #49 合并后 `make deploy-tar` 全量 build + 部署（含 `_facts.json` 修复）

## 关键上下文

### 日常部署（一句话）

```bash
cd ~/noldus-insight && git pull
cd packages/agent && make deploy-tar
```

### 环境变量（~/.bashrc 已配）

```bash
export DEPLOY_HOST="root@39.105.231.16"
export DEPLOY_PATH="/opt/ethoinsight"
export DEPLOY_CONFIG="$HOME/ethoinsight-prod/config.yaml"
export DEPLOY_EXTENSIONS="$HOME/ethoinsight-prod/extensions_config.json"
export DEPLOY_AGENT_ENV="$HOME/ethoinsight-prod/agent.env"
```

### 生产配置（git 外，~/ethoinsight-prod/）

- `config.yaml` — 模型 API key 等
- `extensions_config.json`
- `agent.env` — AUTH_JWT_SECRET / GITHUB_TOKEN / GITHUB_REPO

**⚠️ GITHUB_TOKEN 在本次会话中明文出现过**，需要去 https://github.com/settings/tokens 撤销并重新生成，然后更新 `~/ethoinsight-prod/agent.env`。

### 手动重启容器（带环境变量）

```bash
ssh root@39.105.231.16 "
export DEER_FLOW_HOME=/opt/ethoinsight/deer-flow-home
export DEER_FLOW_CONFIG_PATH=/opt/ethoinsight/runtime/config.yaml
export DEER_FLOW_EXTENSIONS_CONFIG_PATH=/opt/ethoinsight/runtime/extensions_config.json
export DEER_FLOW_REPO_ROOT=/opt/ethoinsight
export DEER_FLOW_DOCKER_SOCKET=/var/run/docker.sock
export BETTER_AUTH_SECRET=\$(cat /opt/ethoinsight/deer-flow-home/.better-auth-secret)
export HOME=/root
cd /opt/ethoinsight/docker
docker compose -p deer-flow -f docker-compose.yaml up -d frontend gateway langgraph nginx
"
```

## 关键发现

### ev19_facts.py 路径 bug（已修复）

**根因**: 原代码 `Path(__file__).parent ×4 / "docs/..."` 爬仓库根找 `_facts.json`。
- `make dev`：host 上 `docs/` 存在 ✅
- Docker 容器：Dockerfile 只 `COPY backend/`，`docs/` 不存在 ❌

**修复**: 复制到 `packages/ethoinsight/ethoinsight/_facts.json`（包内），路径改为 `Path(__file__).parent / "_facts.json"`。

**教训**: Python 包资源文件只能放包内 + `Path(__file__).parent` 或 `importlib.resources`。

### deploy-via-tar.sh 踩坑

1. `frontend/.env` 必须存在 → 已加 rsync
2. `provisioner` 不能 build（目录不在远程）→ compose up 显式列 4 个服务
3. `nginx.conf` rsync 会变目录 → 已改为显式传单文件
4. `agent.env` 对应 compose 的 `env_file: - ../.env` → 传到 `/opt/ethoinsight/.env`

### 1Panel 集成边界

- ✅ 看状态 / 日志 / 重启
- ✅ OpenResty 反代（已配好）
- ❌ 不要在 1Panel 里改 compose 或环境变量

## 未完成事项

### 🟡 优先验证

1. FST 端到端跑通（上传两个 Arena txt 文件，确认 `set_experiment_paradigm` 不报 `_facts.json` 缺失）
2. 访问 `http://39.105.231.16` 界面正常

### 🟢 后续

1. HTTPS 域名：有域名时 1Panel → 网站 → SSL → Let's Encrypt
2. ACR pipeline：权限到位后启用 `.github/workflows/build-push-acr.yml`
3. `_facts.json` 同步脚本：SSOT 在 `docs/review-packages/2026-04-29-ev19-templates/_facts.json`，同事改后需 `cp` 到包内再 PR，可加 Makefile target

## 下一位 Agent 的第一步建议

```bash
# 确认生产状态
ssh root@39.105.231.16 "docker ps --filter name=deer-flow --format 'table {{.Names}}\t{{.Status}}'"

# 确认 dev HEAD
cd ~/noldus-insight && git log --oneline -3
# 期待: 919cad1d Merge pull request #49 ...

# 访问验证
# 浏览器打开 http://39.105.231.16
# 上传 FST 数据文件，跑端到端
```

## 相关文档

- [docs/sop/deploy-via-tar-sop.md](../sop/deploy-via-tar-sop.md) — 完整部署 SOP
- [docs/sop/aliyun-deployment-guide.md](../sop/aliyun-deployment-guide.md) — ECS 基础设施参考
- [packages/agent/scripts/deploy-via-tar.sh](../../packages/agent/scripts/deploy-via-tar.sh) — 部署脚本
