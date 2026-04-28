# 在线部署端到端实施计划（阿里云全栈 + 火山引擎 LLM）

> **给 Claude 的说明**：本计划是**任务级大纲**，不是 TDD 步骤级。每个任务明确了目标、涉及文件、验收标准。实际执行每个任务时，子 agent 可以按需细化到 TDD。
>
> **REQUIRED SUB-SKILL**：执行本计划请用 `superpowers:executing-plans`。

**Goal**：把 EthoInsight 的在线部署从"概念"落到"可访问的 HTTPS 地址"，用户能注册登录、上传数据、agent 分析、看到报告。

**Architecture**：阿里云全托管全家桶承载 user-backend + DeerFlow backend，火山方舟 API 承载 LLM 推理。**最大程度复用阿里云已有托管能力，最小化自建代码**。

**Tech Stack**：
- 主栈：阿里云 IDaaS CIAM / OSS（含 CSI 挂载）/ RDS PostgreSQL / Redis / SAE（含原生 ARMS）/ KMS / SLS / 短信服务
- LLM：阿里云百炼（主）+ 智谱官方 API + 火山方舟（Qwen 候选）
- 代码：FastAPI（user-backend，**瘦身到 ~1500 行**）+ DeerFlow（`bytedance/deer-flow` subtree，**几乎不改**）+ Next.js（前端，改造登录流 + 上传流）

---

## 核心原则：复用阿里云已有基建

本计划严格遵循 [项目决策](/home/qiuyangwang/.claude/projects/-home-qiuyangwang/memory/project_cloud_platform_decision.md) 的"全托管优先"原则。以下是"**能用托管的就不自建**"的具体映射：

| 能力 | ❌ 自建方案（不采用） | ✅ 阿里云托管轮子（采用） |
|---|---|---|
| 登录/注册/改密/MFA/找回 | 自建表单 + 调 API | **IDaaS CIAM 托管登录页**（OIDC 授权码重定向）|
| 密码存储和校验 | PG 表 + bcrypt | **IDaaS 管密码**，我们不存 |
| 短信验证码发送/校验 | 自建 verification_store + Redis TTL | **IDaaS 内置短信验证码流程** |
| JWT 签发/刷新 | 自建 JWT 逻辑 | **IDaaS 发 JWT**，我们只验 JWKS |
| 容器文件系统 ↔ 对象存储 | 自建同步服务 | **OSS CSI 挂载**，SAE 原生支持 |
| 前端文件上传 | 经 user-backend 中转 | **OSS presigned URL 前端直传** |
| APM / Trace | 自建 OTLP 接入 | **SAE 勾选"接入 ARMS"即可** |
| 日志收集 | 自建 logstash | **SAE 容器 stdout → SLS 自动接入** |
| 配置管理 | 启动时拉 KMS | **SAE 配置管理原生集成 KMS** |
| Auth Provider 抽象 | 自建 AuthProvider 基类 | **authlib 标准库**本身就是抽象层 |

**保留的业务中间层**（合理，不算造轮子）：
- user-backend 的 SSE 代理层：为将来做计费/审计/内容过滤预留，不是纯转发，有业务价值
- user-backend 的用户级限流：SAE WAF 做不了 per-user 限流，应用层实现合理
- thread 归属校验：DeerFlow 不改动，就必须由 user-backend 挡在前面做授权

---

## 代码改动规模对比

| 模块 | 原文件大小 | 改动后 | 方式 |
|---|---|---|---|
| `cognito.py` | 625 行 | **0 行** | 删除，换 authlib + 30 行 OIDC 回调 |
| `security.py` | 233 行 | ~50 行 | 简化为通用 JWKS 验签 |
| `auth_service.py` | 795 行 | ~150 行 | 砍掉 80%，只保留 token 管理和 user_id 提取 |
| `verification_store.py` | 374 行 | **0 行** | 删除，短信交给 IDaaS |
| `password_store.py` | 217 行 | **0 行** | 删除，密码交给 IDaaS |
| `api/v1/auth/*` （6 文件）| ~800 行 | ~100 行 | 只保留 `/callback` 和 `/me` |
| `user_rate_limiter.py` | 292 行 | ~200 行 | 改 Redis，保留业务逻辑 |
| `chat_service.py` | 897 行 | ~600 行 | 保留 SSE 代理 + 熔断，改 DeerFlow 格式 |
| `upload_service.py` | 396 行 | ~150 行 | 砍到只剩 presigned URL 签发 |
| `credentials.py` | 180 行 | **0 行** | 删除，AWS 专用 |
| `xray.py` + middleware | ~300 行 | **0 行** | 删除，SAE 原生 ARMS |
| `aws_clients.py` | 352 行 | ~100 行 | 只保留 OSS client（boto3 改 endpoint）|
| `sms_client.py` | 295 行 | 295 行 | **完全不动**（腾讯云 SMS 继续用，跨云合理）|
| **总计** | **5482 行** | **~1600 行** | **减少 70%** |

---

## Phase 概览

| Phase | 目标 | 预估工时 |
|---|---|---|
| Phase 0 | 阿里云基础设施 provisioning（含 IDaaS 托管登录页定制）| 2-3 天 |
| Phase 1 | user-backend 瘦身（**大量删除**，改 OIDC 接入）| 3-4 天 |
| Phase 2 | DeerFlow 集成（JWT 中间件 + SSE 代理改造 + OSS CSI 配置）| 3-4 天 |
| Phase 3 | 前端改造（OIDC 重定向 + presigned URL 直传）| 2 天 |
| Phase 4 | LLM 接入百炼 + 火山方舟验证 | 1-2 天 |
| Phase 5 | 上线前演练（压测/回滚/告警/灰度）| 2-3 天 |

**总预估**：13-18 人天。比原计划少 20%（因为删代码比改代码快）。

---

## Phase 0：基础设施 provisioning

> 把阿里云托管服务全部开出来。**不写业务代码**。

### Task 0.1：阿里云账号与 RAM 基线

**交付物**：
- 企业实名认证
- RAM 子账号 3 个：`dev` / `deploy` / `ops`（最小权限）
- deploy 子账号 AK/SK 存入 KMS
- MFA 配置

**验收**：`aliyun sts GetCallerIdentity` 返回正确身份。

---

### Task 0.2：网络基线（VPC + 安全组）

**交付物**：
- 1 VPC（`cn-hangzhou` 或 `cn-shanghai`）
- 2 vSwitch（跨可用区）
- 3 安全组：`sg-public`（HTTPS in）/ `sg-backend`（内部互通）/ `sg-data`（DB/Redis 专用）
- NAT 网关 + EIP

**验收**：跨组访问规则正确。

---

### Task 0.3：RDS PostgreSQL

**交付物**：
- RDS PostgreSQL 14+（主备 HA）
- pgvector 扩展
- 只允许 sg-backend 访问
- 自动备份 7 天
- 连接串存 KMS

**验收**：公网拒绝，内网 `psql` 通。

**设计决策**：RDS 只存**业务数据**（项目、会话元数据、用户偏好）。**不存用户账号/密码**（那是 IDaaS 的事）。

---

### Task 0.4：Redis 托管版

**交付物**：
- Redis 7.x 主备
- 密码认证
- 仅 sg-backend 访问
- 密码存 KMS

**验收**：公网拒绝，内网 `redis-cli` 通。

**用途**：**只用于用户级限流**（IDaaS 的限流粒度可能不够细）。短信验证码由 IDaaS 管，不需要我们存 Redis。

---

### Task 0.5：OSS Bucket + CSI 配置准备

**交付物**：
- 2 bucket：`ethoinsight-uploads`（私有）、`ethoinsight-outputs`（私有 + CDN）
- 版本控制 + SSE-KMS 加密
- CORS（允许前端域名直传）
- 生命周期规则
- **RAM 策略为 OSS CSI 准备好**：给 SAE 运行角色赋予 `oss:*` 权限仅限这两个 bucket

**验收**：
- boto3（`addressing_style='virtual'`）直传 300MB 文件 ✅
- Presigned PUT URL 前端能直传 ✅
- 记录好 bucket 名和 region，Phase 2 Task 2.4 要挂载

---

### Task 0.6：IDaaS CIAM 实例（含托管登录页定制）

**交付物**：
- 1 个 CIAM 实例
- 创建应用 `ethoinsight-web`（OIDC 协议）
- **登录页定制**：上传 logo、改主题色、配置文案（中文）
- **自助流程配置**：
  - 注册：允许邮箱+密码 / 手机号+短信验证码
  - 找回密码：短信或邮箱
  - MFA（TOTP）：默认可选，管理员强制
  - 改密码：内置页面
- 配置 redirect URI：`https://<前端域名>/auth/callback`、`http://localhost:3000/auth/callback`（dev）
- **配置短信 provider**（IDaaS 后台里配阿里云短信签名）
- 导出：discovery URL、JWKS URL、client_id、client_secret → KMS

**验收**：
- 浏览器访问 IDaaS 登录页，能完整跑"注册 → 登录 → 改密 → 绑定 MFA → 忘记密码"全流程 **无需写一行代码**
- `<discovery>/.well-known/openid-configuration` 返回标准 metadata
- Postman 走授权码流程拿到 JWT，JWT 带有 `sub`（user_id）、`email`、`phone_number` 等 claims

**关键**：这一步做完，我们的"用户系统"**在功能上就已经完整了**——剩下只是把它接到业务里。

---

### Task 0.7：阿里云短信签名与模板（给 IDaaS 用）

**交付物**：
- 申请签名（1-3 天审核，**提前启动**）
- 申请模板（注册/登录/改密/绑定）
- 在 IDaaS 后台里把这些签名和模板**配给 IDaaS 用**（不是我们的 user-backend 用）

**验收**：从 IDaaS 登录页触发短信注册，能收到短信。

**注意**：这里的短信是给**用户认证流程**用的，和 `sms_client.py`（腾讯云 SMS，用于业务通知）是两回事。

---

### Task 0.8：KMS + SLS + SAE + ACR

**交付物**：
- KMS CMK 创建，所有密钥/AK-SK 入库
- SLS project `ethoinsight` + logstore
- SAE 命名空间 `ethoinsight-prod`（绑定 VPC + vSwitch + 安全组）
- **SAE 启用 ARMS 和 SLS 集成**（勾选即可，无代码改动）
- ACR Enterprise：创建 `user-backend` 和 `deerflow-backend` 两个仓库

**验收**：SAE 命名空间可见，ACR 能推镜像。

---

## Phase 1：user-backend 瘦身

> 这一 phase 大量删代码，不是改代码。目标是把 user-backend 砍到只剩真正的业务层：OIDC 回调、用户资料、项目管理、限流。

### Task 1.1：依赖重构 + 删除 AWS 专属模块

**改动**：
- `pyproject.toml`：
  - 移除：`boto3` 的 cognito/ssm/dynamodb/xray 相关（OSS 还留 boto3）、`aws-xray-sdk`、`aioboto3`、`python-jose[cryptography]` 保留
  - 新增：`authlib`（标准 OIDC 客户端）、`redis[hiredis]`
- **删除文件**：
  - `app/core/cognito.py`（625 行）
  - `app/core/verification_store.py`（374 行）
  - `app/core/password_store.py`（217 行）
  - `app/core/xray.py` + `app/core/middleware/xray_middleware.py`
  - `app/api/v1/credentials.py`（180 行）
  - `app/core/aws_params.py`（92 行，改用环境变量 + KMS）
- **删除 API**：
  - `app/api/v1/auth/registration.py`（交给 IDaaS 登录页）
  - `app/api/v1/auth/password.py`（交给 IDaaS）
  - `app/api/v1/auth/sms.py`（交给 IDaaS）
  - `app/api/v1/auth/phone_binding.py` / `email_binding.py`（交给 IDaaS）
- **保留文件**（将在后续 Task 改造）：
  - `app/core/security.py` → 改为 JWKS 验签
  - `app/services/auth_service.py` → 砍到 150 行左右
  - `app/api/v1/auth/login.py` → 改为 OIDC 回调
  - `app/api/v1/auth/deps.py` → 改为 `get_current_user_from_jwt`

**验收**：代码编译通过（`ruff check` + `mypy`），相关单测删除或更新。

**Commit 粒度**：一次删一类，按 commit 分组（"删除 Cognito"、"删除 AWS X-Ray"、"删除 AWS 临时凭证"）。

---

### Task 1.2：OIDC 回调 + JWT 验签（authlib 接入）

**新增**：
- `app/core/oidc.py`（~60 行）：
  - 用 `authlib.integrations.starlette_client.OAuth` 注册 IDaaS provider
  - 配置 discovery URL / client_id / client_secret / scope

**重写**：
- `app/core/security.py`（233 行 → ~50 行）：
  - 只保留 `JWKSTokenVerifier`，从 IDaaS JWKS URL 拉公钥
  - 缓存 JWKS 1 小时
- `app/api/v1/auth/login.py`（~80 行）：
  - `GET /auth/login` → 302 到 IDaaS 登录页
  - `GET /auth/callback` → 换 token → 写 httponly cookie 或返回给前端 → 重定向
  - `POST /auth/refresh` → 用 refresh_token 换新 access_token
  - `POST /auth/logout` → 清 cookie + IDaaS `end_session_endpoint`
- `app/api/v1/auth/deps.py`（~30 行）：
  - `get_current_user(request)` → 从 cookie 或 header 取 JWT → 验签 → 返回 user_id
- `app/services/auth_service.py`（795 行 → ~150 行）：
  - 只保留：`exchange_code_for_token`、`refresh_token`、`get_user_info`（查 IDaaS UserInfo endpoint）
  - 删除：注册、改密、绑定手机、发短信验证码、校验短信验证码等所有自建流程

**验收**：
- 浏览器访问 `/auth/login` → 跳转 IDaaS → 登录 → 回到 `/auth/callback` → 拿到 JWT cookie
- 带 JWT 访问 `/api/v1/me` → 返回用户基本信息（user_id / email / phone）
- JWT 过期后 `/auth/refresh` 能续期

---

### Task 1.3：用户级限流改 Redis

**改造**：
- `app/core/user_rate_limiter.py`（292 行 → ~200 行）：
  - 把 DynamoDB 计数器换成 Redis `INCR` + `EXPIRE`
  - 用 `redis-py` async
  - API 不变

**验收**：1 秒 5 次/用户的限流测试通过。

---

### Task 1.4：配置管理改 SAE 原生

**改造**：
- `app/core/config.py`（359 行 → ~150 行）：
  - 敏感配置从环境变量读（SAE 注入时从 KMS 解密）
  - 删除所有 SSM Parameter Store 调用
  - `pydantic-settings` 不变

**验收**：启动时不调 AWS SSM，配置完全来自环境变量。

---

### Task 1.5：项目 / 会话元数据迁移到 PostgreSQL

**新增**：
- alembic 迁移脚本：
  - `users` 表：`user_id`（PK，来自 IDaaS sub）、`created_at`、`last_login_at`、`preferences` JSONB
  - `projects` 表：已有结构保留
  - `threads` 表：`thread_id`、`user_id`、`project_id`、`created_at`、`last_active_at`（Phase 2 Task 2.2 要用）

**改造**：
- `app/services/chat_service.py` 里从 DynamoDB 读的元数据改 PG（具体修改在 Phase 2 Task 2.3，这里先建表）

**验收**：`alembic upgrade head` 能成功；新表字段和索引完备。

---

### Task 1.6：Phase 1 集成测试

**新增**：`tests/integration/test_auth_flow.py`

**覆盖**：
- OIDC 完整流程（需要真实 IDaaS 实例，或用 IDaaS 的 staging tenant）
- JWT 验签正确性
- Refresh token 续期
- 限流触发
- 注销后 JWT 失效

**验收**：Happy path 全绿。

---

## Phase 2：DeerFlow 集成

> DeerFlow 本身几乎不改。user-backend 作为**授权网关 + 业务中间层**存在。文件存储走 OSS CSI 挂载。

### Task 2.1：确认 DeerFlow 的 thread 模型是否原生支持 user_id

**调研任务**（不写代码）：
- 读 `packages/agent/backend/app/gateway/routers/threads.py`、`thread_runs.py`
- 读 `packages/harness/deerflow/agents/checkpointer/async_provider.py`
- 确认 DeerFlow 2.0 的 thread 创建 API 是否接受 metadata（user_id）

**产出物**：决策文档（追加到本计划末尾）：
- 如果**支持** metadata → Task 2.2 的归属表可以**不建**，直接查 DeerFlow
- 如果**不支持** → 必须在 user-backend 建 `threads` 表（Phase 1 Task 1.5 已建）

---

### Task 2.2：Thread 归属校验（根据 Task 2.1 结果）

**情况 A（DeerFlow 支持 metadata）**：
- 创建 thread 时把 `user_id` 写进 metadata
- 中间件调 DeerFlow `GET /api/threads/{id}` 查 metadata 对比

**情况 B（DeerFlow 不支持）**：
- 用 Phase 1 Task 1.5 建的 `threads` 表
- `app/core/middleware/thread_ownership.py`（~50 行）：请求前查表

**验收**：越权访问返回 403。

---

### Task 2.3：Chat 流转发改造（AgentCore → DeerFlow）

**改造**：`app/services/chat_service.py`（897 行 → ~600 行）+ `app/api/v1/chat.py`

**关键差异**：
- **旧**：`POST {AGENTCORE_RUNTIME_URL}` + AgentCore session header
- **新**：`POST <deerflow>/api/threads/{thread_id}/runs/stream` + DeerFlow payload 格式

**保留的业务能力**（这就是"user-backend SSE 代理"的存在价值）：
- SSE 流式转发 + 30s 心跳（SAE SLB 的超时策略需确认，可能不再需要）
- 熔断 + 重试（`circuit_breaker.py` 可复用）
- `data_ready` 事件注入（如果 DeerFlow 的 artifact 通知机制不同需适配）
- **新增预留**：计费埋点、内容审计 hook（先留空接口）

**删除的能力**：
- AgentCore 的 runtime_session_id 管理

**验收**：
- 前端发消息 → user-backend → DeerFlow → SSE 流回 → 前端增量显示
- DeerFlow 挂掉时熔断生效

---

### Task 2.4：OSS CSI 挂载配置（DeerFlow 代码 0 改动）

**交付物**：
- SAE 应用 `deerflow-backend` 的卷挂载配置：
  - 类型：OSS CSI
  - bucket：`ethoinsight-uploads`（读写）+ `ethoinsight-outputs`（读写）
  - 挂载点：`/mnt/user-data/uploads/`、`/mnt/user-data/outputs/`
  - 子路径策略：按 `thread_id` 作为子目录前缀（避免所有 thread 共用一个挂载点）
- DeerFlow 的 `paths.py` 已经把文件都写到 `/mnt/user-data/threads/{thread_id}/...`，CSI 挂载后**自动落到 OSS**
- RAM 权限：SAE 运行角色只能访问这两个 bucket

**验收**：
- DeerFlow 在 SAE 里跑起来后，向 `/mnt/user-data/uploads/test.txt` 写文件 → OSS 控制台能看到
- 从 OSS 读的文件能被 DeerFlow 沙箱代码读到

**风险/注意**：
- OSS FUSE 的**小文件 IO 性能**远不如本地磁盘，DeerFlow 的 workspace（频繁读写中间状态）可能需要保留本地磁盘
- **建议分层**：`workspace/` 用本地 emptyDir 临时卷（agent 结束即删），`uploads/` 和 `outputs/` 挂 OSS
- 必须实测，不能假定

---

### Task 2.5：Presigned URL 上传接口（前端直传 OSS）

**改造**：`app/services/upload_service.py`（396 行 → ~150 行）+ `app/api/v1/upload.py`（142 行 → ~80 行）

**新流程**：
- `POST /api/v1/uploads/presign` → user-backend 生成 presigned PUT URL（有效期 15 分钟，绑定 `thread_id` 前缀路径）
- 前端拿 URL 直传 OSS，不经 user-backend
- 前端上传完成后 `POST /api/v1/uploads/confirm` → user-backend 在 PG 里登记元数据 + 通过 DeerFlow 内部 API 通知 thread 目录有新文件

**删除**：
- 所有"文件字节流经 user-backend"的代码
- 图片处理（Pillow / pillow-heif）如果有必要保留（大文件压缩），可以保留为独立 optional path

**验收**：
- 前端传 300MB 文件 < 30 秒（OSS 直传速度）
- 文件元数据在 PG `uploads` 表可查
- DeerFlow 能读到该文件

---

### Task 2.6：DeerFlow backend 容器化

**涉及**：
- `packages/agent/Dockerfile` 或 `packages/agent/backend/Dockerfile`
- 环境变量驱动（LLM API key、DeerFlow 配置 YAML 挂载为 ConfigMap）

**验收**：独立构建 DeerFlow 镜像，在 SAE 跑起来响应 `/api/threads`。

---

## Phase 3：前端改造 + 部署

### Task 3.1：前端登录流改造（OIDC 重定向）

**改动**：`packages/agent/frontend/`（Next.js）

- 点"登录"按钮 → 跳转 `<user-backend>/auth/login` → 302 到 IDaaS
- `/auth/callback` 页面 → 调 user-backend 接收 token → 存 httponly cookie
- 其他 API 调用自动带 cookie（或 Bearer token，看安全策略）
- 登出：`<user-backend>/auth/logout` → 清 cookie + IDaaS logout

**删除**：前端自建的注册 / 改密 / 找回密码表单（交给 IDaaS）

---

### Task 3.2：前端上传流改造（presigned URL 直传）

- 用户选文件 → 调 `/api/v1/uploads/presign` 拿 URL
- 用 `fetch` PUT 到 OSS（带进度条）
- 完成后调 `/api/v1/uploads/confirm`

---

### Task 3.3：前端部署到 OSS + CDN

- 构建 `out/` → 推独立 bucket（静态网站托管）
- CDN 加速域名 + HTTPS 证书

---

### Task 3.4：后端部署到 SAE

- 推镜像到 ACR
- SAE 创建应用：
  - `user-backend`：2 实例，公网 SLB，HTTPS 证书
  - `deerflow-backend`：2 实例，内网，**带 OSS CSI 挂载**
- 配置环境变量（从 KMS 读）
- 健康检查 endpoint 配置
- **勾选 ARMS + SLS 集成**（SAE 原生，无代码改动）

**验收**：域名可访问，`/health` 返回 200，日志进 SLS，trace 进 ARMS。

---

## Phase 4：LLM 接入

### Task 4.1：DeerFlow LLM 切到百炼

- 修改 `packages/agent/config.yaml`，使用百炼 OpenAI 兼容 endpoint
- Qwen-Max / Qwen-Plus 作为主力模型
- GLM 作为 fallback（通过 `litellm` 或 DeerFlow 原生多模型配置）

---

### Task 4.2：火山方舟 API 验证

- 用方舟 endpoint 跑一次推理
- 记录延迟、token 成本、输出质量对比

**目的**：为未来微调 Qwen3-8B 部署铺路。

---

## Phase 5：上线前演练

### Task 5.1：Staging 环境全链路测试（跑 golden-cases）

### Task 5.2：压测（登录 + 上传 + agent 任务）

### Task 5.3：回滚预案（文档 + 演练）

### Task 5.4：监控告警基线（SAE 原生 ARMS 告警规则）

### Task 5.5：灰度上线（白名单 → 早期用户 → 全量）

---

## 风险登记表

| 风险 | 影响 | 缓解 |
|---|---|---|
| OSS CSI 的小文件 IO 性能 | DeerFlow workspace 操作慢 | workspace 用 emptyDir 本地卷，只 uploads/outputs 挂 OSS |
| IDaaS CIAM 托管登录页的定制化上限 | UI 不够品牌化 | 先验收默认 UI 是否可接受，不行再评估工作量 |
| 阿里云短信签名审核（外部依赖）| Phase 0.7 阻塞 | 提前启动，Phase 1 用 IDaaS 测试签名 |
| DeerFlow SSE 事件格式 vs AgentCore | Task 2.3 前端需适配 | Phase 0 期间先搭 DeerFlow minimal demo |
| SAE 冷启动延迟 | 首次访问慢 | 最小实例数 ≥ 1，不 scale-to-zero |
| 百炼 API 速率限制 | 生产稳定性 | 配额提前申请，fallback 到 GLM |
| JWT 存 cookie vs localStorage | XSS 风险 | httponly cookie + SameSite=Strict，CSRF token 另算 |

---

## 关键并行机会

- Phase 0 所有 Task 大部分可并行（0.2 先跑）
- Phase 0.7 短信签名有 1-3 天 lead time，尽早启动
- Phase 1.1（大量删除）和 Phase 0 并行（删代码不需要阿里云资源）
- Phase 1.2（OIDC 接入）依赖 0.6（IDaaS 实例），串行
- Phase 2.1（DeerFlow 调研）可在 Phase 0 期间完成
- Phase 4 与 Phase 2/3 完全并行

---

## 与 [CLAUDE.md](../../CLAUDE.md) 规范的对接

- **TDD 强制**：每个 Task 必须带单测
- **Commit 规范**：中文 commit message，每个 Task 至少一次 commit
- **Handoff**：每个 Phase 结束写 `docs/handoffs/YYYY-MM-DD-phase-N-complete.md`
- **受保护文件**：Phase 2 改 DeerFlow 底层走 `scripts/sync-deerflow.sh` 流程（但我们目标是**不改** DeerFlow）

---

## 下一步建议

1. **先推进 Phase 0.1 / 0.2 / 0.7**（阿里云账号、VPC、短信签名审核）——这些是**人工 + 外部审核**，有 lead time
2. **并行启动 Phase 1.1**（大量删除 AWS 专属代码）——不依赖任何阿里云资源
3. **Phase 0.6**（IDaaS 实例 + 托管登录页定制）完成后，立刻进 Phase 1.2（OIDC 接入），这是关键路径

如需展开某个 Task 到 TDD 步骤级细节，可对该 Task 单独跑 `superpowers:writing-plans`。
