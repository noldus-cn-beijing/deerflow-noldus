# 2026-04-24 云平台选型最终决策 — 交接文档

> **状态**：✅ 已锁定，不再开放讨论
> **上下文链**：[需求书](./2026-04-24-cloud-selection-requirements.md) → [背景 handoff](./2026-04-24-cloud-deployment-user-backend-handoff.md) → 本文档

---

## 最终决策

**在线部署采用阿里云全栈 + 火山引擎做 LLM 推理。**

所有基础设施组件均为托管 SaaS/PaaS，零自建。团队只维护两件东西：
1. `user-backend` 业务代码（FastAPI）
2. `DeerFlow` + `ethoinsight` skill（agent 本体）

---

## 组件清单

| §  | 需求 | 推荐产品 | 托管级别 |
|---|---|---|---|
| 3.1 | 用户认证 | **阿里云 IDaaS CIAM** | ✅ 全托管 SaaS |
| 3.2 | 对象存储 | **阿里云 OSS** | ✅ 全托管，S3 兼容性最佳 |
| 3.3 | 关系数据库 | **阿里云 RDS PostgreSQL** | ✅ 全托管，支持 pgvector |
| 3.4 | 缓存/TTL KV | **阿里云 Redis 托管版** | ✅ 全托管 |
| 3.5 | 容器化部署 | **阿里云 SAE**（Serverless 应用引擎） | ✅ 全托管 Serverless |
| 3.6 | 短信服务 | **阿里云短信服务** | ✅ 全托管，三网 |
| 3.7 | LLM 推理 | **阿里云百炼**（Qwen3 / GLM / DeepSeek 统一入口）+ 智谱官方 API fallback；**未来产品级模型可走火山引擎方舟** | ✅ 全托管 MaaS |
| 3.8 | 配置管理 | **阿里云 KMS + SAE 配置管理** | ✅ 全托管 |
| 3.9 | 日志/APM | **阿里云 SLS + ARMS** | ✅ 全托管 |

---

## 决策依据

### 为什么阿里云做主栈（4 个理由）

1. **单云闭环** — 9 项组件在同一家云内完成。网络链路最短，排障链路最清晰，账单统一管理。对比多云方案减少 2-4 家供应商对接成本。

2. **OSS 的 S3 兼容性是国产云里最好的** — 阿里云有专门的「使用 AWS SDK 访问 OSS」官方文档。`boto3` 改 `endpoint_url` + `addressing_style='virtual'` 即可，presigned URL 完全兼容。

3. **百炼 LLM API 聚合最全** — 统一入口调用 Qwen3、GLM、DeepSeek 等主流国产模型，未来切换模型只改 `model_id`，不改代码。

4. **IDaaS CIAM 覆盖 C 端需求** — 支持 JWT RS256、标准 OIDC 协议（授权码/刷新 token/客户端模式）、短信验证码登录、MFA。完全托管 SaaS，获 EAL3+ 认证。

### 为什么 LLM 推理可切火山引擎

- **DeerFlow 是字节跳动官方开源**，和火山方舟深度集成（`config.example.yaml` 示例就是方舟 endpoint）
- 未来 Qwen3-8B SFT 微调时，**火山方舟精调服务**是重点候选（对比 Fireworks.ai / 阿里云 PAI）
- LLM API 跨云调用延迟 < 20ms，对 agent 无感（LLM 推理本身就要几秒）
- **只消费火山的 API，不运维火山的任何东西** —— 仍符合"全托管"原则

---

## 已否决方案（不要重开讨论）

| 方案 | 否决原因 |
|---|---|
| 火山引擎做主栈 | 缺 Cognito 对位产品（只有实名核验），认证层要自建 Casdoor |
| 腾讯云做主栈 | CIAM C 端基因贴切，但 OSS S3 兼容性和 LLM 聚合不如阿里云 |
| Casdoor 自建 | 违反"全托管优先"原则 |
| AWS 中国区 | 本次迁移目标就是脱离 AWS |
| 混合方案（部分托管 + 部分自建）| 违反原则 |

---

## 下一步工作

### 首要任务：user-backend 迁移工作量清单

基于 `/home/qiuyangwang/aws-user-backend/` 现有代码，逐文件评估迁移到阿里云的工作量：

| 改造项 | 影响文件 | 预估 |
|---|---|---|
| Cognito → IDaaS CIAM（通过 AuthProvider 抽象层） | `app/core/cognito.py` + `security.py` + `auth_service.py` (~1600 行) | 中，2-3 天 |
| S3 → OSS | `app/api/v1/upload.py` + 所有 boto3 S3 调用 | 小，0.5 天（改 endpoint + addressing_style）|
| DynamoDB → Redis + PostgreSQL | 短信验证码/限流/元数据 4 处调用点 | 中，1-2 天 |
| Parameter Store → 阿里云 KMS | `app/core/config.py` | 小，0.5 天 |
| ECS Fargate → SAE | CI/CD + Dockerfile | 小，0.5 天 |
| X-Ray → ARMS | middleware | 小，可选 |

**总体预估**：一周上下能完成 user-backend 的云平台迁移，前提是 **AuthProvider 抽象层**先设计好。

### 次要任务：AuthProvider 接口设计

抽象层设计原则：
- 支持多 Provider 切换（AWS Cognito / 阿里云 IDaaS CIAM / 未来可能的本地部署 Provider）
- 标准化 JWT claims、用户属性、token 生命周期
- Cognito 耦合集中在 3 个文件，全部走 Provider 接口后业务代码不再知道具体实现

### 延伸任务：文件存储混合方案

DeerFlow 的 agent 运行时需要读写 `/mnt/user-data/{uploads,outputs,workspace}/`，而对象存储在 OSS 上。需要设计同步/挂载方案：

- 方案 A：user-backend 存 OSS → 运行时拉到 DeerFlow thread 目录
- 方案 B：user-backend 存 OSS → DeerFlow 用 ossfs 挂载
- 方案 C：DeerFlow thread 目录用 NAS 后端

这个不在本文档决策范围内，留给后续设计。

---

## 参考文档

- **本次决策的需求书**：[2026-04-24-cloud-selection-requirements.md](./2026-04-24-cloud-selection-requirements.md)
- **背景与现状**：[2026-04-24-cloud-deployment-user-backend-handoff.md](./2026-04-24-cloud-deployment-user-backend-handoff.md)
- **现有 AWS 架构**：`/home/qiuyangwang/aws-user-backend/`
- **DeerFlow 官方仓库**：[bytedance/deer-flow](https://github.com/bytedance/deer-flow)
- **项目全貌**：[CLAUDE.md](../../CLAUDE.md)
