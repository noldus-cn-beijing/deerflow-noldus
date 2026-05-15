# EthoInsight 在线部署 — 云平台选型需求书

> **给调研 Agent 的说明**：这是一份**需求文档**，不是方案文档。请基于需求独立调研并给出你的推荐（可以推翻之前讨论过的任何倾向）。产出物见文末 §6。
>
> **前置阅读**：[2026-04-24-cloud-deployment-user-backend-handoff.md](./2026-04-24-cloud-deployment-user-backend-handoff.md)（背景上下文、现有 aws-user-backend 架构详情）

---

## 1. 核心原则（最高优先级，不可妥协）

**全托管优先，能用 SaaS 就不自建。**

我们是小团队，人力有限。唯一应该由我们团队维护的代码是：

1. `user-backend` 业务代码（FastAPI）
2. `DeerFlow` + `ethoinsight` skill（agent 本体）

**除此之外的一切基础设施，必须是托管服务（SaaS/PaaS）**。出问题找云厂商工单，而不是 SSH 上机器 debug。

### 判定规则

| 这是托管服务吗？ | 判定 |
|---|---|
| 云厂商 SLA 保证、自动扩容、自动升级 | ✅ 可选 |
| 我们自己跑 Docker/K8s/二进制 | ❌ 不可选 |
| 云厂商卖的 VPS/GPU 服务器上自己装软件 | ❌ 不可选 |
| 云厂商的"应用市场一键部署"但之后要自己运维 | ❌ 不可选 |

**价格不是主要考量**（小团队的规模下，托管服务的成本远低于运维工时成本）。

---

## 2. 部署模式

EthoInsight 有两种部署模式，本文档**只关注在线部署**：

| 模式 | 用户 | 目标环境 | 本文档覆盖 |
|---|---|---|---|
| 本地部署 | 单用户，无需登录 | 客户自有服务器 | ❌ |
| **在线部署** | **多租户，需登录** | **公有云** | ✅ |

---

## 3. 功能需求

### 3.1 用户认证与管理

需要的能力（现 aws-user-backend 在 AWS Cognito 上已实现）：

| 能力 | 说明 |
|---|---|
| 邮箱 + 密码注册/登录 | 标准流程 |
| 短信验证码登录 | 国内用户主流方式 |
| JWT token（RS256 签名）| 前端保存，后端验签 |
| Refresh token 机制 | 长会话 |
| 改密码 / 重置密码 | 标准流程 |
| 绑定手机号 / 邮箱 | 用户资料管理 |
| MFA（可选，未来可能需要）| 管理员账号加强安全 |
| 多租户隔离 | 每个用户有 `user_id`，用于 DeerFlow thread 归属校验 |

**非功能要求**：
- **必须是托管 SaaS**（不自建 Keycloak/Casdoor/Authing 社区版）
- 面向 **C 端公众用户**的账号体系（不是企业员工 SSO 场景）
- 支持**亿级用户规模**的验证（未来增长考虑）
- 标准 OIDC 协议（便于将来替换或联邦）

**未来可能的扩展**（不是硬需求，但加分项）：
- 微信/QQ/第三方社交登录
- 企业客户定制域名的登录页

### 3.2 对象存储

研究员上传 EthoVision XT 导出的轨迹数据（`.xlsx` / `.csv` / `.txt`），agent 产出图表（`.png`）和报告（`.md` / `.pdf`）。

**需求**：

| 能力 | 说明 |
|---|---|
| 上传原始数据 | 最大单文件 300MB，多文件批量 |
| 存储 agent 产出物 | 图表、报告、中间分析结果 |
| CDN 或 presigned URL 分发 | 前端直接下载 |
| 生命周期管理 | 过期数据自动清理（可选）|
| 必须是托管对象存储 | ❌ 不接受"容器本地磁盘 + NFS"方案 |
| **强烈偏好 S3 兼容 API** | 现有代码用 boto3，改 endpoint 就能用 |

**关键约束**（用户明确过）：
不能把用户文件全部存在容器本地磁盘——占用服务器空间、成本不合理、扩容困难。

### 3.3 关系数据库

用户表、项目表、会话元数据、agent 运行历史。

| 能力 | 说明 |
|---|---|
| PostgreSQL 托管服务 | 标准需求 |
| 自动备份 / 时间点恢复 | 生产级 |
| 高可用（主备）| 生产级 |

### 3.4 缓存 / TTL KV 存储

当前 AWS 用 DynamoDB 承担：

1. **短信验证码**（5 分钟 TTL）
2. **用户限流**（原子计数器 + 过期）
3. 项目/会话元数据（低频）
4. 密码哈希（短信登录场景）

我们已经决定把 **(3) 和 (4) 迁到 PostgreSQL**，所以真正需要的是：

| 能力 | 说明 |
|---|---|
| TTL KV 存储（短信验证码）| 典型 Redis `SETEX` 场景 |
| 原子计数器（限流）| 典型 Redis `INCR` + 过期 |
| **托管 Redis 即可**（不需要 DynamoDB 替代品） | 降低技术栈复杂度 |

### 3.5 容器化部署

user-backend（FastAPI）和 DeerFlow backend（LangGraph）需要容器化运行。

| 能力 | 说明 |
|---|---|
| 容器运行时托管服务 | Fargate / Cloud Run / SAE / Knative 之类 |
| **Serverless 容器优先** | 按实际负载计费，零运维 |
| 如果 Serverless 不行，**托管 K8s** 也可接受 | 但不接受自建 K8s |
| HTTPS 入口 / 负载均衡 | 托管 LB |
| 滚动更新 / 蓝绿部署 | 托管能力 |

### 3.6 短信服务

验证码短信发送。托管服务，支持国内三网。

### 3.7 LLM 推理

这一块情况特殊——**LLM 本身就是"买 API"模式**，我们不自建推理集群：

**当前方案**：
- **GLM-5.1**（智谱）—— 已验证，走官方 API
- **未来可能切换到 Qwen3-8B** —— 参见 [project_finetune_direction.md](../../../.claude/projects/-home-qiuyangwang/memory/project_finetune_direction.md)

**核心需求**：
- 能调用 Qwen、GLM 这类国产大模型的**托管 API**
- 将来做 SFT 微调时，希望有**托管的模型精调服务**（不自己买 GPU 卡）
- 不排除从不同云厂商调不同模型（LLM API 可以跨云）

### 3.8 配置管理

API keys、数据库连接串等敏感配置。

| 能力 | 说明 |
|---|---|
| 托管配置中心 / Secret Manager | 不接受配置文件明文 |
| 支持按环境（dev/staging/prod）隔离 | |
| 支持 RBAC | |

### 3.9 日志与可观测性

| 能力 | 说明 |
|---|---|
| 托管日志服务（类 CloudWatch Logs）| 容器 stdout 自动收集 |
| APM / 分布式追踪（可选）| |
| 指标监控 + 告警 | CPU/内存/延迟 P99 |

---

## 4. 架构约束

### 4.1 DeerFlow 不改动

DeerFlow 是字节跳动开源项目（`bytedance/deer-flow`），作为 subtree 引入。我们不往 DeerFlow 里加认证逻辑——**认证由 user-backend 独立承担**，user-backend 通过 thread 归属表做授权检查。

### 4.2 user-backend 需要解耦云厂商

现有 aws-user-backend 对 Cognito / S3 / DynamoDB / Parameter Store 有硬编码依赖。迁移过程中需要引入 **Provider 抽象层**，让同一份代码能切换云厂商（至少为 Cognito 替换出口做好准备）。

### 4.3 网络拓扑

user-backend 和 DeerFlow backend 之间是**高频 SSE 流式通信**，必须**同 VPC 内网**。LLM API 调用可以跨公网（反正是 HTTPS，延迟几 ms 对 LLM 推理时长来说可忽略）。

### 4.4 数据合规

- 主要用户是**国内高校/研究所的行为学研究员**
- 数据在境内处理即可（暂无出海需求）
- 研究数据属于科研数据，敏感度中等（非医疗 PHI、非金融 PII）

---

## 5. 调研需要回答的问题

### 5.1 必答问题

1. **有没有完全符合原则（§1）的云厂商组合方案？**
   - 每一个组件（§3.1 ~ §3.9）都必须是托管 SaaS
   - 如果某个组件必须自建，请明确说明并论证为什么不得不如此

2. **认证服务选哪个？**
   - 候选：AWS Cognito / 阿里云 IDaaS / 腾讯云 CIAM / Authing / 其他
   - 必须是 **C 端用户基因**的产品（不是企业员工 SSO）
   - 必须托管（不是 Casdoor/Keycloak 自建）

3. **对象存储和 DynamoDB 的替代方案**
   - 对象存储选哪家？boto3 兼容性如何？
   - DynamoDB 的职责拆到 Redis + PostgreSQL 是否可行？有没有被遗漏的场景？

4. **LLM 推理 API 策略**
   - 基础方案：直接调 GLM/智谱官方 API + 未来调火山方舟 API（跑 Qwen）
   - 是否有更好的国产 LLM API 聚合平台？
   - 如果做 SFT 微调，哪个托管精调服务最合适（Fireworks.ai / 阿里云 PAI / 火山方舟精调 / 腾讯云 TI）？

5. **整体工作量估算**
   - 从 AWS 架构迁到推荐方案，大致多少人天？
   - 有哪些风险点？

### 5.2 加分问题

6. **有没有我们没想到的好组件？**
   - 比如某家有"一键部署 LangGraph 应用"的托管能力？
   - 比如某家的 PostgreSQL 自带向量检索（pgvector 托管）？
   - 比如某家的日志服务特别便宜/好用？

7. **多云拆分是否值得？**
   - 比如 user-backend 在 A 云，LLM 推理在 B 云
   - 跨云带宽成本、运维复杂度、数据合规影响
   - 一般推荐单云还是多云？

8. **未来扩展的前瞻性**
   - 如果 1 年后用户量从 100 涨到 10 万，选型是否会成为瓶颈？
   - 如果要做 GPU 微调 / 向量数据库 / 消息队列，候选方案是否都有托管产品？

### 5.3 禁止做的事

- **不要推荐 Casdoor / Keycloak / Authing 开源版 / 自建 Supabase** —— 违反§1
- **不要推荐"买 GPU 服务器自己跑模型"** —— 违反§1
- **不要推荐 AWS 中国区**作为主方案 —— 我们要脱离 AWS
- **不要推荐混合方案**（"认证用 SaaS，但限流自己搭 Redis Cluster"这种）—— 要么全托管，要么别选

---

## 6. 产出物

请产出一份 **`docs/plans/2026-04-XX-cloud-selection-recommendation.md`**，包含：

### 6.1 结论部分（先给结论）
- 推荐的云厂商组合（1-2 个候选方案，不要列 4 个让人选）
- 每个组件的具体托管产品名
- 为什么不选其他方案（一句话说明）

### 6.2 对比表
- 横向：候选云厂商
- 纵向：§3 的每一项需求
- 格子内：具体产品名 + 是否托管 + 关键注意事项

### 6.3 风险清单
- 已知的坑（比如某家 S3 兼容 API 有什么行为差异）
- 成本在什么规模会失控
- vendor lock-in 程度

### 6.4 迁移工作量估算
- 按文件/模块列
- 每项大致人天
- 关键路径和并行机会

### 6.5 需要用户决策的开放问题
- 不要什么都替用户决定
- 明确列出"这几个点我需要你拍板"

---

## 7. 参考资料

### 现有代码
- `/home/qiuyangwang/aws-user-backend/` — 现有 user-backend（AWS 上线）
  - 特别关注：`app/core/cognito.py`、`app/core/security.py`、`app/services/auth_service.py`
  - 详细架构分析见 [2026-04-24-cloud-deployment-user-backend-handoff.md §2](./2026-04-24-cloud-deployment-user-backend-handoff.md)

### DeerFlow
- 官方仓库：[bytedance/deer-flow](https://github.com/bytedance/deer-flow)
- 本地 fork：`packages/agent/`

### 项目背景
- [CLAUDE.md](../../CLAUDE.md) — 项目全貌
- [docs/roadmap.md](../roadmap.md) — 12 个月规划
- [docs/specs/llm-finetuning-strategy.md](../specs/llm-finetuning-strategy.md) — 微调策略（Qwen3-8B 方向）

### 本次会话已经初步调研过的信息（供参考，但请独立验证）
- 火山引擎**没有** Cognito 对位的托管 CIAM 产品（只有实名核验服务）
- 腾讯云 CIAM 是 C 端基因，阿里云 IDaaS/EIAM 是 B 端基因
- 腾讯云 COS、阿里云 OSS、火山引擎 TOS 都支持 S3 兼容 API（boto3 改 endpoint 即可）
- Casdoor 功能够用但违反"全托管"原则，已否决
- DeerFlow 是字节开源，和火山引擎有官方深度集成（FaaS 一键部署、方舟 API、AIO Sandbox）

---

## 8. 不变的决策（不要推翻）

这些是本次会话之前或本次会话中已经和用户确认的，不在本次调研范围内：

1. DeerFlow 不做认证改造
2. 对象存储是硬需求，不接受纯本地磁盘
3. DynamoDB 的职责部分迁到 PostgreSQL，部分迁到 Redis
4. 本地部署（单用户无登录）和在线部署（多租户）是两套模式
5. 现有 AWS user-backend 的架构本身是值得复用的（只是换云厂商）
6. LLM 推理走 API，不自建 GPU 集群
