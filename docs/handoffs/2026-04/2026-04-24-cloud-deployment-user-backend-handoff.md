# 2026-04-24 云部署 + 用户后端改造评估 — 交接文档

> **给下一个 AI Agent：** 你无法访问本次会话上下文。这份文档让你快速理解我们在讨论什么，以及你需要做什么评估。
>
> **读取顺序**：本文档 → [CLAUDE.md](../../CLAUDE.md) → aws-user-backend 仓库（`/home/qiuyangwang/aws-user-backend/`）→ DeerFlow 后端（`packages/agent/backend/`）

---

## 1. 背景与决策上下文

EthoInsight 未来部署分两种模式：

| 模式 | 用户管理 | 部署目标 |
|---|---|---|
| **本地部署** | 不需要登录，单用户 | 客户自己的服务器/本机 |
| **在线部署** | 需要用户注册/登录/多租户隔离 | 云平台（阿里云 or 火山引擎） |

本次会话讨论了在线部署的架构方案，核心结论：

1. **DeerFlow 自身已有 thread 级隔离**（状态/文件系统/沙箱），但**没有认证层**——不知道"谁是谁"
2. **用户已有一个成熟的 AWS 用户后端**（`/home/qiuyangwang/aws-user-backend/`），需要改造复用
3. 需要评估**从 AWS 迁移到阿里云或火山引擎**的可行性和工作量

---

## 2. aws-user-backend 现有架构（必读）

**仓库位置**：`/home/qiuyangwang/aws-user-backend/`

**技术栈**：Python 3.10+ / FastAPI / SQLAlchemy 2.0 + Alembic / Docker + ECS Fargate

### 2.1 核心能力

| 能力 | 实现方式 | 关键文件 |
|---|---|---|
| **用户认证** | AWS Cognito（JWT RS256），邮箱+密码、短信验证码登录 | `app/core/cognito.py`（626 行）、`app/core/security.py`（234 行） |
| **用户管理** | 注册、登录、刷新 token、绑定手机/邮箱、改密码（28 个端点） | `app/services/auth_service.py`（796 行）、`app/api/v1/auth/` |
| **Agent 流式对接** | SSE 透传到 AgentCore Runtime，30s 心跳防 ALB 超时 | `app/api/v1/chat.py`、`app/services/chat_service.py` |
| **项目管理** | PostgreSQL 存项目，DynamoDB 存会话元数据 | `app/api/v1/projects.py`、`app/models/project.py` |
| **文件上传** | S3 存储（支持图片/音频/视频/文档），CDN 或 presigned URL | `app/api/v1/upload.py` |
| **限流** | 全局（slowapi）+ 用户级（DynamoDB）| `app/core/rate_limiter.py` |
| **熔断+重试** | AgentCore 3 次失败熔断 60s，指数退避 | `app/core/circuit_breaker.py` |
| **安全中间件** | CORS、HSTS、X-Frame-Options、请求大小限制 300MB | `app/core/middleware.py` |
| **可观测性** | 请求 ID、延迟百分位、X-Ray 分布式追踪 | 多个 middleware |

### 2.2 现有数据流（AgentCore 模式）

```
前端上传文件 → user-backend 存 S3 → 返回 CDN/presigned URL
前端发 chat  → user-backend 创建 project（PostgreSQL + DynamoDB）
             → 转发 {userId, projectId, prompt} 到 AgentCore Runtime
             → AgentCore 按需从 S3 拉文件
             → SSE 流式回传 → user-backend 透传给前端
             → user-backend 解析 tool_use 事件，注入 data_ready 通知
```

### 2.3 AWS 服务依赖清单

| AWS 服务 | 用途 | 迁移影响 |
|---|---|---|
| **Cognito User Pool** | 用户认证+管理 | **高**：20 个 boto3 调用，全部 Cognito 专有 |
| **Cognito Identity Pool** | JWT 换 AWS 临时凭证 | **中**：如果不用 AWS 服务可去掉 |
| **RDS (PostgreSQL)** | 项目表 | **低**：标准 PostgreSQL，任何云都有 |
| **DynamoDB** | 短信验证码、限流、会话元数据、项目元数据 | **中**：需要替代品 |
| **S3** | 文件存储（上传+产出物） | **中**：需要对象存储替代 |
| **ECS Fargate** | 容器部署 | **低**：容器化部署哪里都能跑 |
| **ECR** | Docker 镜像仓库 | **低**：替换为对应云的镜像服务 |
| **Parameter Store** | 配置管理（生产环境强制） | **中**：需要替代的配置中心 |
| **X-Ray** | 分布式追踪 | **低**：可选功能 |
| **SQS** | 日志队列（可选） | **低**：可选功能 |

---

## 3. DeerFlow 的多租户隔离现状

### 3.1 已有的隔离能力（无需改动）

| 层面 | 机制 | 关键文件 |
|---|---|---|
| **会话状态** | 每个 `thread_id` 独立 checkpoint | `packages/harness/deerflow/agents/checkpointer/async_provider.py` |
| **文件系统** | `.deer-flow/threads/{thread_id}/user-data/{workspace,uploads,outputs}` | `packages/harness/deerflow/config/paths.py` |
| **沙箱** | Docker 模式下每 thread 独立容器 | `packages/harness/deerflow/community/aio_sandbox/aio_sandbox_provider.py` |
| **路径安全** | thread_id 正则校验 + 路径穿越保护 | `paths.py :: _validate_thread_id()` |
| **文件上传** | `POST /api/threads/{thread_id}/uploads` → 存磁盘 + 自动转 markdown | `app/gateway/routers/uploads.py` |
| **文件注入** | UploadsMiddleware 把文件列表注入消息上下文 | `packages/harness/deerflow/agents/middlewares/uploads_middleware.py` |

### 3.2 缺失的（需要 user-backend 补）

- 没有认证——任何人知道 thread_id 就能操作
- 没有 user_id 概念
- 没有 thread 归属校验

---

## 4. 需要评估的问题（下一个 Agent 的任务）

### 4.1 云平台选择：阿里云 vs 火山引擎

用户倾向**最大程度保留 aws-user-backend 的架构和扩展性**。需要对比：

| 维度 | 阿里云 | 火山引擎 | 评估要点 |
|---|---|---|---|
| **认证服务** | IDaaS（标准 OIDC）| 无原生 IDaaS，需自建 Casdoor | Cognito 替代品的功能覆盖度 |
| **对象存储** | OSS（S3 兼容 API）| TOS（S3 兼容 API）| boto3 兼容层是否够用 |
| **NoSQL** | 表格存储 Tablestore / MongoDB | 需评估 | DynamoDB 替代品的 API 兼容度 |
| **PostgreSQL** | RDS PostgreSQL | RDS PostgreSQL | 几乎无差异 |
| **容器部署** | ACK（K8s）/ SAE（Serverless）| VKE（K8s）/ veFaaS | ECS Fargate 的替代 |
| **镜像仓库** | ACR | CR | ECR 替代 |
| **配置中心** | MSE Nacos / KMS | 需评估 | Parameter Store 替代 |
| **GPU/推理** | PAI / 灵积 | 方舟（火山方舟，豆包大模型） | GLM-5.1 / Qwen 部署选项 |
| **短信服务** | 短信服务（已有腾讯云短信集成） | 短信服务 | 现有用腾讯云 SMS SDK |

**用户特别关注**：
- S3 类对象存储**必须保留**——不能把用户文件全存容器本地磁盘，成本和效率都不合理
- DynamoDB 的替代品——用于限流、会话管理、项目元数据
- 整体迁移工作量

### 4.2 文件存储策略（用户已明确反对纯本地磁盘方案）

用户指出：把所有用户文件直接存到容器内 thread 目录的磁盘空间是不合理的——占用服务器空间，不如 S3 的存储效率和成本。

需要设计**混合方案**：

```
方案 A：user-backend 存 OSS/TOS → DeerFlow 运行时拉到本地 thread 目录
方案 B：user-backend 存 OSS/TOS → DeerFlow 直接挂载对象存储（FUSE/CSI）
方案 C：DeerFlow thread 目录用网络存储后端（NAS/NFS）而非本地磁盘
```

需要评估：
- DeerFlow 的 sandbox 能否配置为从对象存储读文件？
- 运行时拉取 vs 挂载 vs 网络存储的延迟/成本权衡
- agent 产出物（图表、报告）的持久化策略——是否也需要回传对象存储？

### 4.3 认证体系改造

本次会话已经分析了 Cognito 耦合范围（详见 §2.3），设计了 Provider 抽象层方案：

```python
class AuthProvider(ABC):
    async def sign_up(email, password) -> UserInfo
    async def sign_in(email, password) -> TokenSet
    async def refresh_tokens(refresh_token) -> TokenSet
    async def get_user(access_token) -> UserInfo
    async def sign_out(access_token)
    async def verify_token(token) -> Dict  # 标准化 claims
    # ... 等
```

**Cognito 耦合集中在 3 个文件**（共约 1600 行）：
- `app/core/cognito.py` — 20 个 boto3 SDK 调用（全替换）
- `app/core/security.py` — JWT 验证（改 JWKS/issuer URL 为可配置）
- `app/services/auth_service.py` — 业务编排（改为调 provider 接口）

替代方案选项：
- **Casdoor**（开源，自建，标准 OIDC，有 Python SDK）— 平台无关
- **阿里云 IDaaS**（托管，标准 OIDC）— 仅限阿里云
- **Authing**（国产 IDaaS SaaS）— 平台无关，收费

### 4.4 Chat 流转发改造

现在 user-backend 发给 AgentCore 的 payload：
```json
POST {AGENTCORE_RUNTIME_URL}
{"projectId": 12345, "userId": "cognito-sub-xxx", "prompt": "..."}
Headers: X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: {session_id}
```

需要改为 DeerFlow 的格式：
```json
POST /api/threads/{thread_id}/runs/stream
{
  "input": {"messages": [{"role": "user", "content": "..."}]}
}
```

评估要点：
- SSE 事件格式差异（AgentCore vs DeerFlow LangGraph）
- 心跳机制是否仍需要（DeerFlow 的流式是否有自己的 keep-alive）
- data_ready 事件注入逻辑是否还需要（DeerFlow 有 artifact URL 机制）

### 4.5 会话管理（DynamoDB 替代品）

当前 DynamoDB 用于：
1. **短信验证码**（TTL 5 分钟）→ 需要带 TTL 的 KV 存储
2. **用户限流**（每用户每端点计数）→ 需要原子递增操作
3. **项目/会话元数据**（runtime_session_id, example_urls 等）→ 可能迁移到 PostgreSQL
4. **用户密码哈希**（短信登录用）→ 可能迁移到 PostgreSQL

评估方向：
- 把 3、4 合并到 PostgreSQL（减少一个依赖）
- 1、2 用 Redis 替代（阿里云/火山引擎都有托管 Redis）
- 或者用阿里云 Tablestore（DynamoDB 风格 API）

---

## 5. 架构全景图（目标状态）

```
┌────────────────────────────────────────────────────────────────┐
│                     前端（Next.js, DeerFlow 原生）              │
└─────────┬──────────────────────────────────┬───────────────────┘
          │ 认证相关请求                      │ Agent 交互请求
          ▼                                  ▼
┌─────────────────────┐         ┌────────────────────────────────┐
│   user-backend      │         │   DeerFlow LangGraph API       │
│   (改造后)           │         │   (不改动)                      │
│                     │         │                                │
│ - AuthProvider 接口  │ thread  │ - /api/threads/{id}/runs       │
│ - JWT 验证          │ 归属校验 │ - /api/threads/{id}/uploads    │
│ - 用户管理 CRUD     ├────────►│ - Thread 级状态/文件/沙箱隔离   │
│ - Thread 归属表     │         │ - SSE 流式响应                  │
│ - 限流/熔断         │         │                                │
│ - 文件元数据管理     │         └──────────┬─────────────────────┘
│                     │                    │ 运行时文件
└────────┬────────────┘                    ▼
         │                    ┌──────────────────────┐
         │ 文件持久化          │ /mnt/user-data/       │
         ▼                    │   uploads/ (输入)      │
┌─────────────────────┐      │   outputs/ (产出)      │
│ 对象存储             │      │   workspace/ (临时)    │
│ (OSS / TOS / S3)    │◄─────┤                        │
│                     │ 持久化 └──────────────────────┘
│ - 用户上传文件       │
│ - Agent 产出物归档   │
│ - CDN 分发           │
└─────────────────────┘

┌─────────────────────┐   ┌──────────────────┐   ┌───────────────┐
│ PostgreSQL           │   │ Redis            │   │ AuthProvider   │
│ - users/threads 表   │   │ - 验证码 TTL     │   │ - Casdoor      │
│ - projects 表        │   │ - 限流计数器     │   │ - 阿里云 IDaaS  │
│ - 会话元数据         │   │                  │   │ - Cognito      │
└─────────────────────┘   └──────────────────┘   └───────────────┘
```

---

## 6. 下一个 Agent 的任务

### 6.1 首要任务：云平台选型评估报告

**产出物**：一份对比文档，覆盖阿里云 vs 火山引擎在以下维度的评估：
1. AWS 服务对应替代品的功能覆盖度和 API 兼容性
2. GPU/推理服务对 GLM-5.1 和 Qwen3-8B 的支持情况
3. 成本估算（按在线部署预期用户量）
4. 迁移工作量（按 AWS 服务依赖逐项评估）

### 6.2 次要任务：文件存储混合方案设计

**核心约束**（用户已明确）：
- **必须有对象存储**——不能把用户文件全存容器本地磁盘
- 但 DeerFlow 的 agent 运行时需要从 `/mnt/user-data/uploads/` 读文件

需要设计文件在对象存储和 DeerFlow thread 目录之间的同步/挂载方案。

### 6.3 延伸任务（如果还有 token）

- 认证 Provider 抽象层的接口设计
- DynamoDB → Redis/PostgreSQL 的迁移方案
- Chat 流转发的具体改造点

### 安全检查清单（第一步前必做）

- [ ] 读 CLAUDE.md 了解项目全貌
- [ ] 读 `/home/qiuyangwang/aws-user-backend/` 的代码结构（特别是 `app/core/` 和 `app/services/`）
- [ ] 确认 DeerFlow 的沙箱配置（`packages/harness/deerflow/community/aio_sandbox/`）
- [ ] 不要在 DeerFlow 代码里加认证逻辑——认证由 user-backend 负责

---

## 7. 关键文件索引

### aws-user-backend（待改造）
- `app/core/cognito.py` — Cognito SDK 封装（626 行，20 个 boto3 方法）
- `app/core/security.py` — JWT 验证（CognitoTokenVerifier，234 行）
- `app/services/auth_service.py` — 认证业务逻辑（796 行）
- `app/api/v1/chat.py` — Chat 流式端点
- `app/services/chat_service.py` — Chat 业务逻辑（SSE 转发、心跳、事件解析）
- `app/api/v1/upload.py` — 文件上传（S3）
- `app/api/v1/credentials.py` — AWS 临时凭证交换（可能去掉）
- `app/core/config.py` — 配置管理（含 Parameter Store 加载）

### DeerFlow（不需要改动）
- `packages/harness/deerflow/config/paths.py` — 文件路径管理
- `packages/harness/deerflow/agents/middlewares/uploads_middleware.py` — 文件注入
- `packages/harness/deerflow/agents/middlewares/thread_data_middleware.py` — Thread 目录创建
- `packages/harness/deerflow/community/aio_sandbox/aio_sandbox_provider.py` — Docker 沙箱隔离
- `app/gateway/routers/uploads.py` — DeerFlow 上传 API
- `app/gateway/routers/threads.py` — Thread CRUD
- `app/gateway/routers/thread_runs.py` — Run 管理

### 项目文档
- [CLAUDE.md](../../CLAUDE.md) — 项目总览
- [docs/sop/training-data-flywheel-sop.md](../sop/training-data-flywheel-sop.md) — 飞轮 SOP
- [docs/sop/golden-case-sop.md](../sop/golden-case-sop.md) — Golden-case SOP

---

## 8. 本次会话的关键结论（不要推翻）

1. **DeerFlow 不需要改动**——认证和用户管理由 user-backend 负责
2. **user-backend 的架构值得复用**——认证、限流、熔断、可观测性都是生产级的
3. **对象存储必须保留**——用户明确反对纯本地磁盘方案
4. **认证体系需要 Provider 抽象**——支持 Cognito/Casdoor/IDaaS 可插拔
5. **DynamoDB 的部分职责可以迁移到 PostgreSQL + Redis**——减少云平台专有依赖
6. **本地部署和在线部署是两种模式**——本地不需要登录，在线需要完整用户系统
