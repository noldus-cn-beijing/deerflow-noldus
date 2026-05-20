# Sprint: ECS Docker 部署 CI/CD 链路建设

**Sprint 起点**: 2026-05-20
**目标 window**: 2 周（~2026-06-03）
**前置 handoff**: [docs/handoffs/2026-05/2026-05-19-ecs-deploy-blockbuster-nginx-fix-handoff.md](../handoffs/2026-05/2026-05-19-ecs-deploy-blockbuster-nginx-fix-handoff.md)
**前置 SOP**: [docs/sop/aliyun-deployment-guide.md](../sop/aliyun-deployment-guide.md), [docs/sop/multi-user-deployment-sop.md](../sop/multi-user-deployment-sop.md)

---

## 背景

2026-05-19 在 ECS 部署 hot-patch 修了 7 处问题，但**所有修改都是 docker cp 到运行中容器，没固化进镜像**。次日（5-20）docker daemon re-create 容器后，hot-patch 全部丢失，500 复发——5 分钟会话内又踩一次。

证据见 handoff `2026-05-19-ecs-deploy-blockbuster-nginx-fix-handoff.md` "事后调研" 节。

这迫使把"CI/CD 工业化部署"从 v0.1 后置任务**提前到当前 sprint**。

---

## Sprint 决策（已锁定）

| 维度 | 选定方案 | 备选 |
|---|---|---|
| CI 平台 | **阿里云 Flow（云效）** | GitHub Actions / GitLab CI |
| 镜像 Registry | **阿里云 ACR（个人版起步，必要时升企业版）** | ghcr.io / Docker Hub |
| Build 触发 | **PR merge 到 `main` 分支** | git tag / push dev / 手动 |
| ECS 拉镜像 | **watchtower 自动 pull + restart** | ssh 手动 / GHA ssh 跳板 |

为什么这套组合：
- Flow + ACR 都在阿里云内网，跟 ECS 同 VPC build/push/pull 全走内网，规避 GitHub Actions → ACR 的公网 token 管理痛点
- 阿里云 Flow 直接支持 GitHub repo 触发（不需要镜像仓库中转）
- PR-merge 触发让 `main` 始终代表"生产可部署"，`dev` 仍是日常开发流
- watchtower 5 分钟 poll 对 v0.1 内部使用够用，不引入 K8s/Argo 复杂度

---

## Sprint 范围 — SMART 目标

**Specific**: 把 ECS 部署从"本机 build + docker cp hot-patch"切到"PR merge → 阿里云 Flow build → ACR push → watchtower pull → ECS 跑新版本"。

**Measurable** — 验证标准（sprint 结束必须满足）:

1. ✅ main 分支 PR merge 后，阿里云 Flow 在 15 分钟内 build + push 完三个镜像（gateway / langgraph / frontend），ACR 看到新 tag
2. ✅ ECS 上 watchtower 5 分钟 poll 内发现新镜像，自动 `docker compose pull && up -d`，完成滚动重启
3. ✅ 整个 PR merge → ECS 跑新版本 ≤ 20 分钟
4. ✅ 今天 7 处修复（commit `e706435d`）固化进新镜像；docker daemon 重启容器后仍生效
5. ✅ ECS 上 `make up` 路径**废止**，文档替换为 `docker compose pull && up -d`
6. ✅ 团队成员（>=2 人）跑通"提 PR → merge → 看 Flow 跑 → ECS 自动更新"的完整流

**Achievable** — 不在本 sprint 范围（明确排除）:
- 切 Gateway mode (`make up-pro`) — 后置，单独评估 experimental 风险
- LangSmith license / `langgraph up` — v0.1 后看预算决定
- K8s / Helm — v0.1 单机够
- HA / 多机 / 灾备 — v0.1 单机研究助手不需要
- E2E 自动化测试 — 部署链路稳定后再做

**Time-bound**: 2 周 sprint window。

---

## Phase 0 — 前置准备（Day 0，~2 小时）

### 0.1 push 当前 dev 到 origin

```bash
cd /home/wangqiuyang/noldus-insight
git fetch origin dev
git log --oneline origin/dev..dev   # 应该看到 ad40f260 + e706435d
git push origin dev
```

### 0.2 确认 main 分支状态

`main` 分支已存在（5-20 调研发现）。检查它跟 dev 的差距：

```bash
git fetch origin
git log --oneline origin/main..origin/dev | wc -l   # 看 dev 领先多少 commit
```

**决策点**: dev 当前包含今天 7 处修复 + 之前所有特性。要不要：

- (a) **直接把 dev 全部 merge 到 main 当首个 sprint 基线**（推荐，省事）
- (b) cherry-pick 部分稳定 commit 到 main（保守，但 dev 跟 main diff 会很久不收敛）

我推荐 (a)。在 GitHub 上：

```
1. 在 GitHub UI 提 PR：dev → main
2. 标题: "chore: 初始化 main 分支为 sprint 基线（含 e706435d 7 处部署修复）"
3. 用 squash merge 或 merge commit 都可，rebase 容易把今天的 commit message 搞乱
4. merge 后本地：git fetch origin main
```

### 0.3 GitHub branch protection

GitHub repo settings → branches → add rule for `main`:

- ✅ Require a pull request before merging
- ✅ Require approvals: 1（如果团队 >1 人）
- ✅ Require status checks to pass before merging（sprint 后期加 Flow 检查）
- ✅ Restrict deletions

### 0.4 更新 CLAUDE.md 双分支模型

`CLAUDE.md` "Git" 节当前写：

```
- 主分支：`dev`（当前工作分支）
```

改成：

```
- 分支模型：
  - `main` — 生产分支。PR merge 到 main 触发阿里云 Flow build & push ACR
  - `dev` — 日常开发分支。所有 commit 先进 dev
  - Sprint/feature 完成后从 dev 提 PR 到 main
- 提交前跑 `make test` 和 `make lint`
- commit message 用中文，简洁描述改动意图
```

### 0.5 阿里云资源准备

跟现有 [docs/sop/aliyun-deployment-guide.md](../sop/aliyun-deployment-guide.md) 用同一个阿里云账号。已有 ECS i-2ze446h4ats7y8bgjn4w，需新增：

- **阿里云 Flow**（云效）项目，绑定 GitHub repo `noldus-cn-beijing/noldus-insight`
- **ACR 个人版**镜像仓库（免费），创建 namespace `ethoinsight`，三个镜像仓库 `ethoinsight/gateway` / `ethoinsight/langgraph` / `ethoinsight/frontend`
- **RAM 子账号**：给 Flow 一个 ACR push 权限 token；给 ECS 一个 ACR pull 权限 token

⚠ ACR 个人版限制：每个 namespace 最多 3 个仓库（够用），每仓库镜像总大小限额，需要清理策略

---

## Phase 1 — 阿里云 Flow pipeline（Day 1-3）

### 1.1 Flow 项目初始化

阿里云控制台 → 云效 Flow → 创建流水线 → 选择"代码仓库触发"。

- 仓库源：GitHub `noldus-cn-beijing/noldus-insight`
- 触发分支：`main`
- 触发事件：push to main（PR merge 后自动触发 push 事件）

### 1.2 Pipeline 步骤定义

```yaml
# 概念性 YAML，实际界面操作或写 .alpine-ci.yaml
stages:
  - name: checkout
    steps:
      - git clone (自动)
      - 解 symlink：
          run: |
            # backend/packages/ethoinsight 是 symlink，跨 docker build context 必败
            # 见 handoff #4
            rm packages/agent/backend/packages/ethoinsight
            cp -rL packages/ethoinsight packages/agent/backend/packages/ethoinsight

  - name: build-images
    parallel: true
    matrix: [gateway, langgraph, frontend]
    steps:
      - docker build:
          context: packages/agent
          dockerfile: packages/agent/{backend|frontend}/Dockerfile
          target: prod    # 三 stage Dockerfile 取 prod
          args:
            APT_MIRROR: mirrors.aliyun.com
            UV_INDEX_URL: https://mirrors.aliyun.com/pypi/simple/
            NPM_REGISTRY: https://registry.npmmirror.com
          tags:
            - registry.cn-beijing.aliyuncs.com/ethoinsight/{service}:main-${COMMIT_SHA}
            - registry.cn-beijing.aliyuncs.com/ethoinsight/{service}:latest

  - name: push-acr
    steps:
      - docker login registry.cn-beijing.aliyuncs.com -u $ACR_USER -p $ACR_TOKEN
      - docker push 三个 tag

  - name: notify
    steps:
      - 发钉钉/邮件通知 build 完成 + commit sha + ACR tag
```

⚠ Flow 实际界面是"可视化拖拽节点"，不是写 YAML。第一次熟悉一下界面。

### 1.3 凭证管理

Flow 凭证库：
- `ACR_USER` = ACR RAM 子账号用户名
- `ACR_TOKEN` = ACR RAM 子账号 password（不是 AK！是 ACR 专用 password）

⚠ **绝不要 commit 凭证到代码仓库**。所有凭证都进 Flow 凭证管理。

### 1.4 Phase 1 验收

1. 在本地 dev 提个 trivial PR（改 README 一个字）→ merge 到 main
2. Flow 自动触发 → 15 分钟内 build 完三镜像
3. ACR 控制台看到 `ethoinsight/gateway:main-<sha>` 等 6 个 tag
4. 收到通知（如配了）

如果失败，常见 3 个坑：
- symlink 没解 → "No such file or directory" → 检查 0.2 步的解 symlink 命令是否真跑
- 国内网络拉 base image 慢 → 配 aliyun image accelerator（Flow 默认应该有）
- ACR push 401 → 凭证错或 RAM 权限不够

---

## Phase 2 — watchtower 上 ECS（Day 4-5）

### 2.1 ECS docker login ACR

```bash
ssh root@39.105.231.16
docker login registry.cn-beijing.aliyuncs.com -u <acr_user> -p <acr_token>
# 凭证写进 /root/.docker/config.json
cat /root/.docker/config.json   # 验证
```

### 2.2 docker-compose.yaml 改用 ACR 镜像

```yaml
# 当前
services:
  gateway:
    build:
      context: ../
      dockerfile: backend/Dockerfile
    image: deer-flow-gateway   # 本机 build 出来的本地 tag

# 改后
services:
  gateway:
    image: registry.cn-beijing.aliyuncs.com/ethoinsight/gateway:latest
    pull_policy: always   # 每次 up 都 pull
    labels:
      - "com.centurylinklabs.watchtower.enable=true"
    # build: 这一段保留也行，让本机仍能 build；但 ACR 优先
```

三个服务（gateway / langgraph / frontend）同理改。**不要标 nginx 和 provisioner 的 watchtower label** — 避免 watchtower 重启 nginx 中断流量、重启 provisioner 中断 sandbox 任务。

### 2.3 加 watchtower 服务

`docker/docker-compose.yaml` 末尾加：

```yaml
  watchtower:
    image: containrrr/watchtower:latest
    container_name: deer-flow-watchtower
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /root/.docker/config.json:/config.json:ro
    environment:
      - WATCHTOWER_LABEL_ENABLE=true                # 仅监控带 label 的容器
      - WATCHTOWER_POLL_INTERVAL=300                # 5 分钟 poll
      - WATCHTOWER_CLEANUP=true                     # 自动清旧镜像
      - WATCHTOWER_INCLUDE_RESTARTING=false         # 不动 restarting 容器
      - WATCHTOWER_NOTIFICATIONS=shoutrrr           # 可选：钉钉通知
      - WATCHTOWER_NOTIFICATION_URL=<钉钉 webhook>  # 可选
    restart: unless-stopped
```

### 2.4 Phase 2 验收

```bash
# ECS 上启 watchtower
docker compose -p deer-flow -f docker/docker-compose.yaml up -d watchtower
docker logs --tail 20 deer-flow-watchtower   # 看到 "scheduling first run in 300s"

# Phase 1 跑一次 trivial PR → main，5-20 分钟后
docker logs --tail 50 deer-flow-watchtower 2>&1 | grep -iE "pull|update|stop|start"
# 应该看到 watchtower 检测到新镜像 → pull → stop old → start new

# 容器内验证已是新镜像
docker exec deer-flow-gateway grep "_resolve_sqlite_dir" /app/backend/packages/harness/deerflow/config/database_config.py
```

---

## Phase 3 — 镜像固化今天 hot-patch 验证（Day 6）

这一步是 Phase 1+2 的端到端验证。今天 commit `e706435d` 7 处修复**已经在 dev 分支**，所以 Phase 0.2 完成后 main 也有这些修复。Phase 1 build 出的镜像里**应该**包含它们。

验证方式：

```bash
# 在 ECS 上,直接看新镜像里的代码（不进容器）
docker run --rm registry.cn-beijing.aliyuncs.com/ethoinsight/gateway:latest \
  cat /app/backend/packages/harness/deerflow/config/database_config.py | grep -c "_resolve_sqlite_dir"
# 期待 3

docker run --rm registry.cn-beijing.aliyuncs.com/ethoinsight/gateway:latest \
  cat /app/backend/packages/harness/deerflow/persistence/engine.py | grep -c "asyncio.to_thread"
# 期待 1
```

如果 0 / 0 → Phase 1 build 时没拉到最新代码。检查 Flow 触发的 git ref 是不是 main HEAD。

### 3.1 故意把容器干掉再起，验证不再丢

```bash
docker stop deer-flow-gateway deer-flow-langgraph
docker rm deer-flow-gateway deer-flow-langgraph    # 完全销毁容器对象
docker compose -p deer-flow -f docker/docker-compose.yaml up -d gateway langgraph
sleep 30

# 验证还是新代码
docker exec deer-flow-gateway grep "_resolve_sqlite_dir" /app/backend/packages/harness/deerflow/config/database_config.py
# 期待: 3（不再是 0）
```

**Phase 3 验收**: 容器销毁重建后，hot-patch 不再丢。这是今天 5-20 早晨踩坑的根因彻底消除的证据。

---

## Phase 4 — 切流量 + 废止本机 build 工作流（Day 7-8）

### 4.1 改 Makefile

`packages/agent/Makefile` 加新 target：

```makefile
.PHONY: pull-up
pull-up:
	@echo "Pulling latest images from ACR..."
	@docker compose -p deer-flow -f docker/docker-compose.yaml pull
	@docker compose -p deer-flow -f docker/docker-compose.yaml up -d
	@echo "✓ Updated to latest"
```

保留 `make up`（本机 build）作为离线/应急 fallback，但文档主推 `make pull-up`。

### 4.2 写 deploy SOP

新建 `docs/sop/deploy-sop.md`：

```markdown
# 部署 SOP

## 日常部署（PR merge → 生产）
1. dev 分支 commit + push
2. GitHub UI 提 PR：dev → main
3. 至少 1 人 review approve
4. Merge（squash 或 merge commit 都行）
5. 阿里云 Flow 自动跑 build（~15 分钟）
6. ECS watchtower 5 分钟内 pull + restart
7. 浏览器访问 39.105.231.16:2026 验证

## 紧急回滚
1. ACR 找到上一个稳定 tag `main-<old-sha>`
2. ECS 上 docker compose stop 三个服务
3. 手动 docker pull <old-tag> 替换 :latest tag
4. docker compose up -d
5. 查根因后正常发 PR 修复

## 应急本机 build（Flow 挂或网络故障）
仅在 Flow 不可用时使用：make up（保留兼容）
```

### 4.3 通知团队

钉钉/邮件 + Slack（如有）通知：

> EthoInsight 部署链路已上 CI/CD。从今天起：
> - 改代码 → push dev → 提 PR 到 main → merge
> - 不要在 ECS 上 docker cp 或本机 build
> - 5-20 分钟后线上自动更新
> - 详见 docs/sop/deploy-sop.md

---

## Sprint 风险表

| 风险 | 可能性 | 影响 | 缓解 |
|---|---|---|---|
| 阿里云 Flow GitHub trigger 不稳定 | 中 | 高（PR merge 不触发 build） | Phase 1 Day 1 先做最小验证，确认 webhook 收得到 |
| ACR 个人版镜像容量满 | 低 | 中 | 保留 latest + 最近 3 个 main-<sha>，老 tag 自动 GC；不够升企业版（按量付费） |
| watchtower restart 时中断在线用户对话 | 中 | 中 | (a) 先 `--monitor-only` 跑 1 天观察 (b) v0.1 后 langgraph thread checkpoint 入 PG，用户对话不丢 |
| ACR pull token 在 ECS `/root/.docker/config.json` 泄露 | 低 | 高 | RAM 子账号最小权限（只 read，不 push）；定期 rotate token |
| watchtower 拉到坏镜像导致全部 500 | 中 | 高 | rollback SOP 见 Phase 4.2；Flow build 前加 lint+test 关卡（sprint 后期） |
| Flow build 时国内拉 base image 失败（python:3.12-slim, node:22-alpine, ghcr.io/astral-sh/uv）| 高 | 高 | Phase 1 第一次 build 必中。修法：Flow 配阿里云 image accelerator（默认自带），或 pre-push base image 到 ACR |
| Dev 团队不习惯 PR 流程 | 中 | 中 | 给团队 1 天 hands-on 演示；保留 dev 直接 push 作为日常迭代，main 才走 PR |

---

## 验收 checklist（Sprint 结束 self-review）

- [ ] origin/dev 包含今天 commit `e706435d` + `ad40f260`（push 已完成）
- [ ] origin/main 已建立，包含 dev 全部内容
- [ ] CLAUDE.md "Git" 节更新为双分支模型
- [ ] 阿里云 Flow 项目跑通一次 PR-trigger build
- [ ] ACR 三个镜像仓库各至少 2 个 tag（latest + 一个 main-<sha>）
- [ ] watchtower 在 ECS 上跑，poll log 正常
- [ ] 验证：故意销毁容器后，watchtower 自动恢复，新代码生效
- [ ] `docs/sop/deploy-sop.md` 写完
- [ ] 团队至少 2 人各跑通一次 PR → 线上更新流程
- [ ] handoff `2026-05-19-ecs-deploy-blockbuster-nginx-fix-handoff.md` 末尾 "下一 sprint" 节标记完成

---

## 后置（sprint 完成后）

把这份 plan 移到 `docs/milestone/2026-05-cicd-rollout.md`，记录实际遇到的坑和解决方案。

下一个 sprint 候选：
1. 切 Gateway mode 评估
2. langgraph checkpoint 从 InMemorySaver 换 sqlite/PG
3. blockbuster 在 deerflow Tier 4 体系全审计（PR 回上游）

---

## 给执行 agent 的指引

如果你是接手执行这个 sprint 的 agent，按以下顺序操作：

1. 先读完本文档全文
2. 读前置 handoff `2026-05-19-ecs-deploy-blockbuster-nginx-fix-handoff.md`
3. 跟用户确认 Phase 0 准备情况（main 分支是否建好、阿里云资源是否就绪）
4. 按 Phase 顺序推进，每个 Phase 完成后给用户做验收 demo
5. 任何 Flow / ACR 实际操作步骤都通过用户在阿里云控制台手动完成，agent 只给步骤指引和验证脚本
6. ECS 上的命令可通过 ssh 远程跑，但首次操作建议让用户在终端跑、agent 看回显
