# 多用户部署问题分析与计划

> 写给执行此计划的 AI Agent。假设你无法访问之前的对话上下文。

## 1. 背景

EthoInsight Agent（基于 DeerFlow 2.0）目前以单用户模式运行，所有服务通过 Docker Compose 一键启动：

```
Nginx (:2026) → Frontend (:3000) + Gateway (:8001) + LangGraph (:2024)
```

目标：部署到服务器上供 **10 名内部测试人员**同时使用。

## 2. 服务器配置建议

由于 LLM 调用走智谱 GLM-5 云端 API（非本地模型），**不需要 GPU**。

| 组件 | 最低 | 推荐 |
|------|------|------|
| CPU | 4 核 | 8 核 |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 40 GB SSD | 80 GB SSD |
| 网络 | 5 Mbps | 10 Mbps+ |

各服务内存估算：

| 服务 | 占用 |
|------|------|
| Nginx | ~50 MB |
| Frontend (Next.js) | ~300-500 MB |
| Gateway (FastAPI × 4 workers) | ~1-2 GB |
| LangGraph (10 jobs/worker) | ~2-4 GB |
| 用户上传文件临时存储 | ~1-2 GB |
| **合计** | **~4-7 GB（峰值更高）** |

## 3. 多用户隔离问题（核心阻塞项）

当前系统**不支持多用户**，存在三个致命问题：

### 3.1 Thread 列表不隔离

- 前端 `useThreads()` 调用 `apiClient.threads.search()` 时**不传 user_id**
- 后端 `POST /api/threads/search` 返回**所有用户的 thread**
- 结果：用户 A 能看到并进入用户 B 的对话，互相干扰

涉及代码：
- `frontend/src/core/threads/hooks.ts` — `useThreads()` 搜索逻辑
- `backend/app/gateway/routers/threads.py` — `search_threads()` 无 user 过滤

### 3.2 Memory 不隔离

- Memory 存储在**单个** `memory.json` 文件中（路径见 `config.yaml` → `memory.storage_path`）
- 所有用户共享同一份记忆数据
- 结果：用户 A 的对话记忆会被注入到用户 B 的 system prompt 中

涉及代码：
- `packages/harness/deerflow/agents/middlewares/memory_middleware.py`
- `packages/harness/deerflow/agents/memory/updater.py`
- `app/gateway/routers/memory.py`

### 3.3 认证系统未贯通

虽然前端集成了 `better-auth`（邮箱密码登录），但后端完全没有接入：

- `frontend/src/server/better-auth/config.ts` — 已配置 `emailAndPassword: enabled: true`
- `frontend/src/app/api/auth/[...all]/route.ts` — 已注册 Next.js API 路由
- **但是**：Gateway 和 LangGraph Server 没有任何鉴权中间件，thread/memory 操作不携带 user_id

## 4. 解决方案

### 方案 A：完整多用户支持（推荐，3-5 天）

#### 步骤 1：认证贯通

1. Gateway 加 session 校验中间件（从 better-auth 的 cookie/header 中提取 user_id）
2. LangGraph Server 通过 nginx 转发时携带 user_id header（或信任 Gateway 的内部调用）
3. 所有 API 端点强制要求已认证用户

#### 步骤 2：Thread 按 user_id 隔离

1. 创建 thread 时在 metadata 中写入 `user_id`
2. 搜索 thread 时按当前用户的 `user_id` 过滤
3. 前端 `useThreads()` 附带用户身份信息

改动点：
- `frontend/src/core/threads/hooks.ts` — 创建/搜索附带 user_id
- `backend/app/gateway/routers/threads.py` — 从 session 提取 user_id 并过滤
- LangGraph 的 thread metadata 需要包含 user_id

#### 步骤 3：Memory 按 user_id 隔离

1. 每个用户独立的 `memory-{user_id}.json`
2. 修改 `storage_path` 逻辑为 `memory/{user_id}.json`
3. Memory 注入时只读取当前用户的记忆

改动点：
- `packages/harness/deerflow/agents/memory/updater.py` — 按用户路径读写
- `packages/harness/deerflow/agents/middlewares/memory_middleware.py` — 传递 user_id
- `app/gateway/routers/memory.py` — 按用户查询

### 方案 B：快速绕过（1 天，仅限内测）

如果时间紧迫且 10 人之间互相信任：

1. **加 nginx basic auth** — 统一密码，至少阻止外部访问
2. **前端 localStorage 存 user_id** — 不做后端认证，纯前端过滤 thread 列表
3. **接受 memory 共享** — 内测阶段所有用户共享记忆（可能需要关闭 memory 功能）

优点：1 天搞定，能跑起来
缺点：没有真正的隔离，不适合正式环境

### 方案 C：多实例部署（最快但最浪费）

每人一个 Docker Compose 实例，不同端口：

```
用户 1 → :2001    用户 2 → :2002    ...    用户 10 → :2010
```

8C16G 服务器最多跑 3-4 个实例，需要更高配置或分散到多台服务器。

## 5. 建议

对于 10 人内测，**推荐方案 A**。理由：

1. DeerFlow 的 better-auth 已经搭好了一半，后端接入不算复杂
2. Thread 隔离改动量可控（metadata 加字段 + 查询过滤）
3. Memory 隔离改动最小（改个文件路径逻辑）
4. 做完后直接为将来的正式部署打好基础

## 6. 相关文件清单

| 类别 | 文件路径 |
|------|---------|
| 认证（前端） | `frontend/src/server/better-auth/config.ts` |
| 认证（前端路由） | `frontend/src/app/api/auth/[...all]/route.ts` |
| Thread 管理 | `frontend/src/core/threads/hooks.ts` |
| Thread 后端 | `backend/app/gateway/routers/threads.py` |
| Memory 中间件 | `packages/harness/deerflow/agents/middlewares/memory_middleware.py` |
| Memory 更新器 | `packages/harness/deerflow/agents/memory/updater.py` |
| Memory API | `backend/app/gateway/routers/memory.py` |
| 主配置 | `config.yaml`（memory、checkpointer 部分） |
| Docker 部署 | `docker/docker-compose.yaml` |
