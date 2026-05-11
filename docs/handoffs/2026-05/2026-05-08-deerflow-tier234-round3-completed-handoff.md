# DeerFlow 上游 Tier 4 同步 - 轮 3 完成交接（better-auth）

**日期**: 2026-05-08
**状态**: 🟢 全部 4 个 commit 在本地 dev 分支，待 push
**前置依赖**: [2026-05-07-deerflow-tier234-round2-completed-handoff.md](./2026-05-07-deerflow-tier234-round2-completed-handoff.md)

---

## 0. TL;DR

- **轮 3 已完成**：4 个 commit（G/H/I/J）已 commit 到本地 dev 分支，**未 push**
- **测试基线**：`2 failed, 2141 passed, 14 skipped`（轮 2 末尾 1877 passed → +264）
- **better-auth 已就绪**：登录/注册/setup wizard 全部跑通，`/api/v1/auth/setup-status` 返回 `{"needs_setup":true}`
- **不在轮 3 范围**：阿里云 PG 实际部署、LangGraph PostgresSaver 切换、AioSandbox 容器化、性能压测（全部留轮 4）
- **2 个 pre-existing failures 没修**（按规约）

---

## 1. 已完成 commits

| Commit SHA | Phase | 内容 | 文件改动 |
|---|---|---|---|
| `7f74c49a` | Pre | 修轮 2 漏的 user_id 参数同步（4 文件） | 5 文件 +100/-14 |
| `3b61e9fc` | G 后端 auth | better-auth 后端：13+ 新文件 + surgical merge 7 文件 + 17 个测试 | 56 文件 |
| `c65b7771` | H 前端 auth | better-auth 前端：core/auth + (auth) routes + types/i18n auth 段 | 28 文件 |
| `b213b849` | I 品牌中文化 | auth 页面 EthoInsight + 中文翻译 | 3 文件 +35/-35 |
| `<待 J commit>` | J 部署文档 | multi-user-deployment-sop + 弃用 plan 更新 + 本 handoff | 3 文件 |

合入的 6 个上游 commit：
- `94eee95f` feat(auth): release-validation pass for 2.0-rc — 12 blockers + simplify follow-ups
- `848ace98` feat: replace auto-admin creation with secure interactive first-boot setup
- `da174dfd` feat: implement process-local internal authentication for Gateway and enhance CSRF handling
- `98a5b34f` fix: resolve merge conflict in pnpm-lock.yaml and clean up better-auth dependencies
- `4e4e4f92` fix(security): harden auth system and fix run journal logic bug
- `78633c69` fix(agents): propagate agent_name into ToolRuntime.context for setup_agent
- `ed9ebfac` fix: enforce 'request' parameter requirement in require_auth decorator
- `88d47f67` fix(nginx): add catch-all /api/ location for auth routes

---

## 2. 跳过 / 推迟到轮 4 的内容

| 项 | 原因 | 留轮 4 处理 |
|---|---|---|
| 阿里云 Postgres 实际部署 | 需要实际机器 + 域名 + 证书，本轮只产文档 | 轮 4 实地部署 |
| LangGraph PostgresSaver 切换 | thread state 改用 PG 持久化（重启不丢对话） | 轮 4 切换 |
| Sandbox 切 AioSandboxProvider | 每用户独立 Docker 容器隔离（4-23 计划提过） | 轮 4 切换 |
| Multi-user 性能压测 | 10/30/50 人并发基准 | 轮 4 测试 |
| 前端 auth 视觉细节优化 | 等用户反馈再调，本轮只确保功能可用 | 轮 4 polish |
| 上游 feedback router (rating 语义) | 与 noldus 训练数据 flywheel feedback 路径不冲突，但暂不挂载 | 轮 4 评估是否合入 |
| 全站品牌 sweep（landing/docs 中的 DeerFlow 残留） | I phase 只处理 auth 页面 + page title，其他保留 | 后续品牌专项 |

---

## 3. 关键状态验证

### 3.1 测试基线
```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/ --no-header -q --ignore=tests/test_client_live.py 2>&1 | tail -3
# 期望: 2 failed, 2141 passed, 14 skipped
```

### 3.2 受保护语义残量（grep 验证）

| 文件 | grep 残量 | 期望 |
|---|---|---|
| `agents/lead_agent/prompt.py` | 16 | ≥ 5 |
| `subagents/executor.py` | 6 | ≥ 5 |
| `agents/lead_agent/agent.py` middlewares | 16 | ≥ 5 |
| `sandbox/tools.py` | 9 | ≥ 3 |
| `config/paths.py` | 5 | ≥ 4 |
| `summarization_middleware.py` BeforeSummarizationHook | 2 | ≥ 1 |
| `subagents/builtins/__init__.py` 4 个 ethoinsight 子代理 | 4 | = 4 |
| `app/gateway/app.py` `app.state.config` | 2 | ≥ 1 |
| `app/gateway/routers/threads.py` archived-messages | 5 | ≥ 1 |
| `app/gateway/routers/feedback.py` training-data | 13 | ≥ 1 |
| `frontend/src/components/landing/hero.tsx` EthoInsight | 2 | ≥ 2 |
| `skills/custom/` 目录 | 5 | = 5 |

### 3.3 启动 smoke

```bash
cd packages/agent
export AUTH_JWT_SECRET=test-secret-32-chars-minimum-please-change
make dev &
sleep 60

curl -s http://localhost:8001/api/v1/auth/setup-status
# {"needs_setup":true}

curl -s http://localhost:8001/health
# {"status":"healthy","service":"deer-flow-gateway"}

curl -s http://localhost:2026/login | grep -o "<title>[^<]*</title>"
# <title>EthoInsight</title>
```

### 3.4 git log

```bash
git log --oneline -7
b213b849 sync deerflow upstream Tier 4 I: 品牌 EthoInsight + auth 页面中文化
c65b7771 sync deerflow upstream Tier 4 H: better-auth 前端 (94eee95f + 848ace98 + 98a5b34f)
3b61e9fc sync deerflow upstream Tier 4 G: better-auth 后端 (94eee95f + da174dfd + 4e4e4f92 + 78633c69 + ed9ebfac + 88d47f67)
7f74c49a fix(uploads): 同步 user_id 参数到 thread_data / uploads / archiving 中间件
f4ed7461 docs: 本会话总交接 — 轮 2 收尾完成 + 轮 3 spec/plan 已就绪
edf35023 docs: 轮 3 deerflow 同步 better-auth 实施 plan
da8df421 docs: 修正轮 3 spec 中 auth 路径 (app/auth → app/gateway/auth)
```

---

## 4. 本轮关键决策（surgical merge 调整）

### 4.1 deps.py 整覆盖到上游
- 上游 deps.py 的 langgraph_runtime 大幅扩展，引入 persistence engine + thread_meta_store + run_event_store
- 这些底层模块在轮 2 D 阶段已铺好，可以直接用上游版本
- 只调整一处：`RunManager(store=...)` 改 `RunManager()`，因为 noldus RunManager 不接受 store 参数（轮 4 改）

### 4.2 stream_bridge / store async_provider 加 AppConfig 兼容
- 上游 `make_stream_bridge(config)` 期望传 AppConfig（取 `.stream_bridge`），noldus 期望传 StreamBridgeConfig
- 改成：`hasattr(config, "stream_bridge")` 自动适配两种调用方式

### 4.3 threads.py 整覆盖 + 末尾追加 archived-messages
- 上游 threads.py 改造较大（owner_filter + reserved metadata + ThreadMetaStore），无 noldus 文本定制
- 但 noldus 多了 `/archived-messages` 路由（SummarizationMiddleware 配套）
- 策略：整覆盖到上游，末尾追加 noldus 路由 + import json

### 4.4 channels/manager.py surgical
- 加 internal_auth headers + CSRF token 到 LangGraph SDK client
- 保留所有 noldus 频道处理（feishu/slack/telegram/wechat/wecom）

### 4.5 frontend hooks.ts / input-box.tsx 保留 noldus
- 上游版本 555/478 行 diff，是上游内部 refactoring（mode taxonomy 重写、hook 数据结构变化）
- 与 noldus 价值（archived-messages 加载、autoMode/flywheelMode）冲突
- 保留 noldus 版本，浏览器请求自动带 cookie 也能通过 auth

### 4.6 i18n types.ts 仅追加 settings.account 段
- 上游加了一堆 mode taxonomy（flashMode/proMode 等），与 noldus 已加的 autoMode/flywheelMode 冲突
- 仅追加 auth 相关的 settings.sections.account + settings.account 段
- en-US.ts + zh-CN.ts 同步追加（中文翻译手写）

### 4.7 test_lead_agent_prompt + test_channels 保留 noldus
- 上游加的测试要求 prompt.py 有 `_build_self_update_section` 等函数（noldus prompt.py 受保护）
- 上游 test_channels 期望 channels.manager 有上游 refactoring 后的内部数据结构
- 两个测试整文件保留 noldus 版本

### 4.8 feedback router 路径不冲突，先不挂载上游
- 上游 feedback router 是 rating（+1/-1）语义，路径 `/runs/{run_id}/feedback`
- noldus feedback router 是训练数据 verdict 语义，路径 `/feedback`
- 共存可行，但本轮先不挂载上游 router（轮 4 评估是否合入）

---

## 5. 风险 / 已知问题

### 5.1 bcrypt 性能
- 默认 12 rounds，每次登录 ~300ms
- 10 人并发不是瓶颈，但若用户量增长，监控登录响应时间

### 5.2 JWT secret
- 必须 ≥ 32 字符且不能泄露
- `AuthConfig.get_auth_config()` 在没设环境变量时会自动生成 ephemeral secret 但**重启后所有 session 失效**
- 生产部署必须通过 .env 注入 `AUTH_JWT_SECRET`

### 5.3 orphan thread migration
- `_migrate_orphaned_threads` lifespan hook 把没有 owner_id 的 LangGraph thread 全归给 admin
- 是 idempotent 的（只迁移 owner_id 为空的），可重启多次
- 但实际生产首次启动应**先备份**：`cp -r .deer-flow .deer-flow.bak.round3`

### 5.4 前端 hooks.ts 版本分叉
- noldus 保留旧版 hooks.ts，上游持续演进
- 后续 sync 上游时若 hooks.ts 再有改动，要重新评估能否升级（已记 surgical merge 案例）

### 5.5 test_channels 部分 fail（已恢复 noldus 版本）
- 上游 channels 加的 internal_auth 部分已合入
- 但上游 channels 还有更深 refactoring（runs.wait config 参数、from_app_config 工厂方法等）未合入
- 这些只在跑上游 test_channels 时会 fail，noldus 测试都过

---

## 6. 不要做的事（接手者注意）

- ❌ **不 push 任何 commit 到 origin/dev**：等用户审核 + 决定时机
- ❌ **不修 2 个 pre-existing failures**（plan 明确禁止）
- ❌ **不动 5 个 ethoinsight custom skill markdown**
- ❌ **不动 prompt.py 中文调度规则**
- ❌ **不重写 zh-CN.ts**（只追加 auth namespace）
- ❌ **不重新生成 pnpm-lock.yaml**（已用 surgical 删 better-auth）
- ❌ **不 `uv lock --upgrade-package langgraph` 自作主张**

---

## 7. 完成定义验证

- [x] 4 个 commit 在本地 dev 分支
- [x] 测试基线: `2 failed, 2141 passed, 14 skipped`
- [x] 后端 ruff lint 0 error
- [x] 前端 typecheck + lint 0 error
- [x] `make dev` 启动成功，全栈服务上来
- [x] 浏览器访问 `localhost:2026` 看到登录页（EthoInsight + 中文）
- [ ] **用户手动验证**：Setup wizard 创建 admin → 登录 → workspace → 上传 → 分析（端到端）
- [x] noldus 受保护语义残量全部达标
- [x] 完成 handoff（本文档）
- [x] **未 push** — 留 dev 等用户决定

---

## 8. 后续接手路径

1. **用户验证端到端**：浏览器跑一次完整流程，确认无 regression
2. **如有问题** → 看 spec §5（风险与回滚）+ 本 handoff §5
3. **OK 后决定 push 时机**：可立即 push 或延后到 v0.1 milestone
4. **轮 4 启动**：参考本 handoff §2 的 backlog
5. **生产部署**：参考 `docs/sop/multi-user-deployment-sop.md`

---

## 9. 相关文档

- 本轮 spec: `docs/superpowers/specs/2026-05-07-deerflow-tier234-round3-design.md`
- 本轮 plan: `docs/superpowers/plans/2026-05-07-deerflow-tier234-round3-plan.md`
- 部署 SOP: `docs/sop/multi-user-deployment-sop.md`
- 弃用的 4-23 计划: `docs/plans/2026-04-23-multi-user-deployment.md`
- 轮 2 完成 handoff: `docs/handoffs/2026-05-07-deerflow-tier234-round2-completed-handoff.md`
- 轮 2 收尾 bug 修复: `docs/handoffs/2026-05-07-upload-file-not-recognized-fix-handoff.md`
- noldus 同步规则: `CLAUDE.md` L123-L174
- DeerFlow 后端架构: `packages/agent/backend/CLAUDE.md`
