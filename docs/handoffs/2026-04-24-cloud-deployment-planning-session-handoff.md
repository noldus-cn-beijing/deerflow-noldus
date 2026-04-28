# 2026-04-24 云部署选型与实施计划 — 会话交接

> **给接手的 Agent**：这是一次完整的"云平台选型 + 在线部署规划"会话的产出交接。你进入时没有本次会话上下文，请**严格按这份文档恢复工作状态**。
>
> **读取顺序**：本文档 → [项目 memory](../../../.claude/projects/-home-qiuyangwang/memory/MEMORY.md) → [需求书](./2026-04-24-cloud-selection-requirements.md) → [最终计划](../plans/2026-04-24-online-deployment-implementation-plan.md) → [Phase 0 清单](../plans/2026-04-24-phase-0-aliyun-setup-checklist.md)

---

## 1. 当前任务目标

**核心任务**：把 EthoInsight（Noldus 行为学 AI 分析助手）从"本地可用"推向"在线多租户部署"。

**本次会话解决的问题**：
1. 云厂商选型（AWS 之外的选择）
2. user-backend 迁移路径（aws-user-backend 5482 行代码）
3. 在线部署的端到端实施计划

**本次会话**未**覆盖的问题**（留给后续）：
- 实际代码改造（只写了计划，没改代码）
- 阿里云资源实际 provisioning（用户要自己去控制台操作）
- Phase 1+ 的任何实施

---

## 2. 当前进展

### ✅ 已完成

#### 2.1 云平台选型决策锁定
**结论**：**阿里云全栈 + 火山引擎做 LLM 推理**。已写入项目 memory：[project_cloud_platform_decision.md](../../../.claude/projects/-home-qiuyangwang/memory/project_cloud_platform_decision.md)

| 组件 | 选定产品 |
|---|---|
| 用户认证 | 阿里云 IDaaS CIAM（**不是 EIAM**）|
| 对象存储 | 阿里云 OSS（boto3 改 endpoint + `addressing_style='virtual'`）|
| 关系数据库 | 阿里云 RDS PostgreSQL（含 pgvector）|
| 缓存/TTL KV | 阿里云 Redis 托管版 |
| 容器部署 | 阿里云 SAE（Serverless 应用引擎）|
| 短信（认证用）| 阿里云短信服务 → 配给 IDaaS |
| 短信（业务通知）| **保留腾讯云 SMS**（现有 `sms_client.py` 用 `qcloudsms_py`，跨云合理）|
| LLM 主力 | 阿里云百炼（Qwen / GLM / DeepSeek 统一入口）|
| LLM 备选 | 火山方舟（未来跑 Qwen3-8B 微调）|
| 配置 | 阿里云 KMS + SAE 配置管理 |
| 日志/APM | 阿里云 SLS + ARMS（SAE 原生集成，勾选即可）|

**已否决方案**：
- ❌ 火山引擎做主栈（无 Cognito 对位产品，实名核验不是账号体系）
- ❌ 腾讯云做主栈（CIAM 是 C 端基因贴切，但 OSS S3 兼容性和 LLM 聚合不如阿里云）
- ❌ Casdoor 自建（违反"全托管"原则）
- ❌ AWS 中国区（本次迁移目标就是脱离 AWS）

#### 2.2 核心原则确认
用户明确："**除了自己必须做的（agent）外，都托付给云厂商**。**价钱不是问题**。"

这条是本次会话最核心的取舍尺——写入 [memory](../../../.claude/projects/-home-qiuyangwang/memory/project_cloud_platform_decision.md)。

#### 2.3 需求书
[docs/handoffs/2026-04-24-cloud-selection-requirements.md](./2026-04-24-cloud-selection-requirements.md) —— 完整需求规范。

#### 2.4 实施计划（v2，已按"不造轮子"原则重写）
[docs/plans/2026-04-24-online-deployment-implementation-plan.md](../plans/2026-04-24-online-deployment-implementation-plan.md)

**关键改造决策（用户拍板）**：

| 决策点 | 选择 |
|---|---|
| SSE 流式转发 | **保留 user-backend 代理**（为将来计费/审计/内容过滤预留 hook）|
| 登录页 | **用 IDaaS 托管登录页**（OIDC 授权码重定向，不自建登录 UI）|
| 文件存储 | **OSS CSI 挂载**（SAE 部署 DeerFlow 时把 OSS bucket 挂成 /mnt/user-data/，DeerFlow 代码零改动）|
| AuthProvider 抽象层 | **不做**（authlib 已是 OIDC 标准抽象，YAGNI）|

**代码规模变化**：5482 行 → **~1600 行**（减少 70%，大量是**删代码**而不是改代码）。

**6 个 Phase**：
- Phase 0：阿里云基础设施 provisioning（2-3 天）
- Phase 1：user-backend 瘦身（3-4 天，大量删除 AWS 专属模块）
- Phase 2：DeerFlow 集成（3-4 天，JWT 中间件 + SSE 代理改造 + OSS CSI 配置）
- Phase 3：前端改造 + SAE 部署（2 天）
- Phase 4：LLM 接入百炼 + 火山方舟验证（1-2 天）
- Phase 5：上线前演练（2-3 天）

**总预估**：13-18 人天。

#### 2.5 Phase 0 人工操作清单
[docs/plans/2026-04-24-phase-0-aliyun-setup-checklist.md](../plans/2026-04-24-phase-0-aliyun-setup-checklist.md)

9 个步骤（0.A ~ 0.I），每步都有具体字段值 + 验收打钩项。**特别注意 0.G 短信签名审核 1-3 天 lead time，必须最早启动**。

---

## 3. 关键上下文

### 3.1 项目定位
- **EthoInsight**：Noldus 给行为学研究员的 AI 分析助手
- 基于 **DeerFlow**（`bytedance/deer-flow` fork，作为 subtree 引入 `packages/agent/`）
- 当前状态：端到端流水线可用，`shoaling` 范式完整
- 详见 [CLAUDE.md](../../CLAUDE.md)

### 3.2 aws-user-backend 现状
**路径**：`/home/qiuyangwang/aws-user-backend/`
- FastAPI + SQLAlchemy + Cognito + S3 + DynamoDB + ECS Fargate
- **5482 行核心代码**，已实测文件行数
- 重点文件：
  - `app/core/cognito.py` 625 行（20 个 boto3 调用）
  - `app/core/security.py` 233 行（JWT 验证）
  - `app/services/auth_service.py` 795 行（业务编排）
  - `app/services/chat_service.py` 897 行（AgentCore SSE 转发，最大的文件）
  - `app/core/sms_client.py` **已经是腾讯云 SMS**，不是 AWS SNS（`qcloudsms_py`）

### 3.3 模型路由已存在的关键决策
刚刚（本次会话之前或同期）加入了 [project_model_routing.md](../../../.claude/projects/-home-qiuyangwang/memory/project_model_routing.md)：
- **所有 LLM 通过 NewAPI 网关统一走 Anthropic 协议**
- `config.yaml` 所有模型用 `use: langchain_anthropic:ChatAnthropic`
- endpoint：`anthropic_api_url: https://newapi.noldusapi.com`
- **Phase 4 LLM 接入要遵守这个**——不是直连百炼，是通过 NewAPI

这条会影响 Phase 4 Task 4.1/4.2 的实施——**接手 Agent 不要按原计划里"直连百炼"改 config.yaml**，要通过 NewAPI。

### 3.4 项目其他关键约束（摘自 CLAUDE.md）
- TDD 强制（每个功能带单测）
- commit message 中文
- v0.1 硬截止：**2026 年 9 月**
- skills/custom/ 4 个定制 skill **在 git 里**，不是 gitignored
- `noldus-kb` MCP 当前禁用（`extensions_config.json` `enabled: false`）
- 受保护文件改动要走 `scripts/sync-deerflow.sh` 流程

---

## 4. 关键发现

### 4.1 火山引擎的"身份认证"不是账号体系
反直觉：火山引擎控制台有个叫"身份认证"的产品，但它是**实名核验**（OCR + 活体 + 人证比对），**不是 Cognito 类 CIAM**。所以火山引擎不能单独做用户后端主栈。

### 4.2 阿里云 IDaaS 有两个 Edition，别选错
- **EIAM**（企业身份）：给员工 SSO 用，B 端
- **CIAM**（客户身份）：给 C 端终端用户，**这是我们要的**

一旦创错实例就换不回来。Phase 0.F 专门强调了。

### 4.3 DeerFlow 是 ByteDance 官方开源
- 仓库：[bytedance/deer-flow](https://github.com/bytedance/deer-flow)
- 47.3k stars
- **和火山引擎深度官方集成**：FaaS 一键部署、方舟 API、AIO Sandbox、火山 TTS
- 这意味着未来火山对 DeerFlow 的支持只会更好，不会变差
- `config.example.yaml` 示例就是方舟 endpoint

### 4.4 sms_client.py 已经用的是腾讯云 SMS
这是我在读代码时才发现的——`qcloudsms_py` 依赖在 `pyproject.toml` 里。所以：
- 认证短信（注册/登录/改密验证码）→ **IDaaS CIAM 内置**（配阿里云短信给 IDaaS）
- 业务通知短信 → **保留腾讯云 SMS**（跨云合理，不违反"全托管"）

### 4.5 原计划有 3 处"造轮子"被用户 review 纠正
原计划 v1 违反"不造轮子"的地方：
1. Task 2.3：改造 chat_service.py 897 行做 SSE 转发 → **部分砍**（用户选择保留为业务中间层做计费/审计）
2. Task 2.4：自建文件同步服务 → **完全砍**，改用 OSS CSI 挂载
3. Task 1.3：自建密码哈希表 → **完全砍**，IDaaS 管密码
4. Task 1.2：建 AuthProvider 抽象层 → **完全砍**（authlib 已是抽象）

最终大幅简化到 ~1600 行代码。

### 4.6 OSS CSI 挂载的性能坑
OSS FUSE 对小文件 IO 性能远不如本地磁盘。DeerFlow 的 `workspace/`（agent 频繁读写中间状态）**不能**挂 OSS，要用 emptyDir 本地卷。只有 `uploads/` 和 `outputs/`（大文件、低频）挂 OSS。

---

## 5. 未完成事项（按优先级排序）

### 🔴 高优先级（立刻做）

#### U1. 推进 Phase 0.G 短信签名审核
**Why**：1-3 天外部审核 lead time，不能后做
**What**：用户登录阿里云控制台，按 [Phase 0 清单 0.G](../plans/2026-04-24-phase-0-aliyun-setup-checklist.md#0g-短信签名与模板-⚠️最早启动) 提交
**谁做**：**用户本人**（涉及企业资质上传）

#### U2. 推进 Phase 0.A/B 基础账号和网络
**Why**：后续所有 Task 都依赖这个
**What**：按 [Phase 0 清单 0.A ~ 0.B](../plans/2026-04-24-phase-0-aliyun-setup-checklist.md#0a-账号--ram-子账号) 操作
**谁做**：**用户本人**（控制台操作）

#### U3. Phase 1.1 大量删除 AWS 专属代码
**Why**：不依赖阿里云资源，可以和 Phase 0 并行
**What**：按 [实施计划 Phase 1 Task 1.1](../plans/2026-04-24-online-deployment-implementation-plan.md#task-11依赖重构--删除-aws-专属模块) 执行
**谁做**：**Agent 可执行**（但要用户确认可以动手）

### 🟡 中优先级（Phase 0 完成后）

#### U4. Phase 1.2 OIDC 接入（authlib）
**依赖**：Phase 0.F（IDaaS CIAM 实例就绪，拿到 discovery URL + client_id/secret）
**What**：按 [实施计划 Phase 1 Task 1.2](../plans/2026-04-24-online-deployment-implementation-plan.md#task-12oidc-回调--jwt-验签authlib-接入) 执行

#### U5. Phase 2.1 DeerFlow thread 模型调研
**重要**：这是个**调研任务**，不写代码。决定 Phase 2.2 是否需要自建归属表
**What**：读 `packages/agent/backend/app/gateway/routers/threads.py`、`thread_runs.py`，确认 DeerFlow 2.0 thread 是否原生支持 user_id metadata
**产出**：追加决策到实施计划末尾

### 🟢 低优先级（后期）

#### U6. Phase 4 LLM 接入计划需按 NewAPI 架构修订
**Why**：[project_model_routing.md](../../../.claude/projects/-home-qiuyangwang/memory/project_model_routing.md) 规定所有 LLM 走 NewAPI Anthropic 协议，不是直连百炼
**What**：修订实施计划 Task 4.1 的描述，改为"通过 NewAPI 配置百炼渠道走 Anthropic 协议"
**谁做**：**Agent 可执行**（不急，Phase 4 开始前做即可）

---

## 6. 建议接手路径

### 6.1 接手第一步
1. **读 [MEMORY.md](../../../.claude/projects/-home-qiuyangwang/memory/MEMORY.md)** 索引全部记忆
2. **打开 [project_cloud_platform_decision.md](../../../.claude/projects/-home-qiuyangwang/memory/project_cloud_platform_decision.md)**——这是所有后续决策的锚点
3. **扫 [实施计划](../plans/2026-04-24-online-deployment-implementation-plan.md) 的"代码改动规模对比"表**——理解哪些是删代码、哪些是改代码
4. **问用户"Phase 0 进展到哪了？"** —— 基础设施还没就绪的话，所有代码 Task 都无法执行

### 6.2 关键文件路径
- 项目根：`/home/qiuyangwang/noldus-insight/`
- user-backend：`/home/qiuyangwang/aws-user-backend/`（**不在 noldus-insight 内**，是独立仓库）
- DeerFlow subtree：`packages/agent/`
- 本次会话产出：
  - `docs/handoffs/2026-04-24-cloud-deployment-user-backend-handoff.md`（背景）
  - `docs/handoffs/2026-04-24-cloud-selection-requirements.md`（需求书）
  - `docs/handoffs/2026-04-24-cloud-selection-decision.md`（决策锁定）
  - `docs/plans/2026-04-24-online-deployment-implementation-plan.md`（实施计划）
  - `docs/plans/2026-04-24-phase-0-aliyun-setup-checklist.md`（人工操作清单）

### 6.3 常用命令
```bash
# 查看 user-backend 代码结构
ls /home/qiuyangwang/aws-user-backend/app/

# 读关键文件
wc -l /home/qiuyangwang/aws-user-backend/app/core/cognito.py
wc -l /home/qiuyangwang/aws-user-backend/app/services/chat_service.py

# 项目测试
cd /home/qiuyangwang/noldus-insight/packages/agent/backend && make test

# DeerFlow subtree 同步
cd /home/qiuyangwang/noldus-insight && ./scripts/sync-deerflow.sh --dry-run
```

---

## 7. 风险与注意事项

### 7.1 容易混淆/踩坑的点

| 陷阱 | 避免方式 |
|---|---|
| IDaaS EIAM vs CIAM | 只选 CIAM，EIAM 是给员工 SSO 用的 |
| 火山引擎"身份认证" | 不是账号体系，是实名核验，别当 Cognito 用 |
| 阿里云短信签名 | **1-3 天外部审核**，所有人工操作里第一个启动 |
| OSS CSI 性能 | 小文件 IO 差，workspace/ 不能挂，只 uploads/outputs/ 挂 |
| RDS 白名单默认 0.0.0.0/0 | 创建实例后**必须删**，只留安全组白名单 |
| KMS 凭据 vs KMS 密钥 | 凭据管理存 key-value；密钥是加密 key。别混 |
| sms_client.py 是腾讯云不是 AWS | 不要迁移到阿里云短信，保留原有跨云合理 |
| NewAPI 取代直连百炼 | Phase 4 不要让 DeerFlow 直连百炼，走 NewAPI Anthropic 协议 |

### 7.2 不建议的方向（已被否决）

- ❌ 自建 Casdoor/Keycloak 做认证
- ❌ 自建 AuthProvider 抽象层
- ❌ user-backend 做文件同步服务
- ❌ 自建登录表单调 IDaaS API（要用 IDaaS 托管登录页）
- ❌ 直连百炼（必须走 NewAPI）
- ❌ 把所有文件经 user-backend 中转（前端直传 OSS）
- ❌ 自建 ARMS Python Agent（SAE 勾选即可）

### 7.3 争议点的最终决策

- **SSE 代理是否保留**：保留（用户决定）。理由：为将来计费/审计/内容过滤做业务中间层，不是纯转发
- **AuthProvider 抽象层**：不做。authlib 已是抽象，YAGNI
- **腾讯云 SMS 是否迁走**：不迁，跨云合理

---

## 8. 下一位 Agent 的第一步建议

### 情景 A：用户刚回来，询问"Phase 0 做到哪了"

**回复模板**：
> 根据交接文档，Phase 0 的 9 个步骤中，最紧急的是 **0.G 短信签名审核（1-3 天 lead time）**。你那边做到哪一步了？具体来说：
> - [ ] 0.A 企业实名 + 3 个 RAM 子账号
> - [ ] 0.G 短信签名提交审核
> - [ ] 0.F IDaaS CIAM 实例创建 + 登录页定制

### 情景 B：用户说"Phase 0 已经就绪，开始代码改造"

**第一步**：用户提供这些 Phase 0 凭据（不是凭据本身，是名字/URL/ID）：
- IDaaS Discovery URL
- IDaaS Client ID / Client Secret 的 KMS 引用
- RDS PostgreSQL endpoint（内网）
- Redis endpoint（内网）
- OSS bucket 名（uploads + outputs）
- SAE 命名空间名
- ACR 仓库地址

**第二步**：开始 **Phase 1.1 大量删除 AWS 专属代码**。按实施计划 Task 1.1 的清单逐项删除，每类一个 commit：
1. commit 1：删除 Cognito（`app/core/cognito.py`）
2. commit 2：删除短信/密码存储（`verification_store.py` + `password_store.py`）
3. commit 3：删除 X-Ray
4. commit 4：删除 AWS 临时凭证（`credentials.py`）
5. commit 5：删除 SSM Parameter Store（`aws_params.py`）
6. commit 6：删除 auth 下的注册/改密/绑定等 API

然后跑 `ruff check` + `mypy --strict` 确保编译通过（测试会大量失败，正常，Phase 1.2 恢复）。

### 情景 C：用户想先调整计划

**第一步**：问清楚具体哪里要改。可能的调整方向：
- Phase 4 LLM 接入改为 NewAPI 架构（已知待办 U6）
- DeerFlow thread 归属模型调研（U5）——可以先做这个不依赖基础设施
- 具体 Task 的 TDD 步骤级展开（用 `superpowers:writing-plans` 对该 Task 再跑一次）

### 情景 D：用户问别的问题（和云部署无关）

**识别**：如果用户问 EthoInsight 日常开发（agent 调试、ethoinsight 库、golden-cases 等），本次会话的云部署上下文**不相关**，按 CLAUDE.md 的常规流程处理即可，不要强行关联到云部署。

---

## 9. 本次会话使用的 skills

- `superpowers:writing-plans` —— 生成实施计划
- `superpowers:handoff` —— 本文档

下一次会话如果要继续规划，建议用：
- `superpowers:writing-plans`（展开具体 Task 到 TDD 步骤）
- `superpowers:executing-plans`（实际跑 Task）
- `superpowers:subagent-driven-development`（派子 agent 跑 Task，每次 review）

---

## 10. 本次会话的关键结论（不要推翻）

1. **阿里云全栈 + 火山 LLM** 已锁定，不重新对比云厂商
2. **全托管优先** 是最高原则，价格不是考量
3. **DeerFlow 不改认证**，user-backend 做授权网关
4. **IDaaS 托管登录页** 是登录 UI 的最终选择
5. **OSS CSI 挂载** 是文件存储的最终选择
6. **authlib 直用** 不做 AuthProvider 抽象
7. **腾讯云 SMS 保留**（业务通知短信）
8. **Phase 0 必须先做**，特别是 0.G 短信签名
9. **所有 LLM 走 NewAPI Anthropic 协议**（见 project_model_routing.md）

---

**交接日期**：2026-04-24
**交接给**：下一次会话的 AI Agent（人类用户 qiuyangwang）
**状态**：规划阶段完成，实施阶段未开始，等待 Phase 0 人工操作完成
