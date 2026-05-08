# EthoInsight Multi-User 部署 SOP

**版本**: v0.1（2026-05-08）
**适用**: better-auth + sqlite/postgres + LangGraph Server + 单机 Docker Compose

---

## 0. 概述

EthoInsight v0.1 多用户研究助手在单机 Docker Compose 中部署，10 人内并发：

```
Internet
    │
    ▼
nginx (TLS, 2026 → 80/443)
    │
    ├─ /api/v1/auth/*       → Gateway (better-auth: register/login/setup)
    ├─ /api/threads/*       → Gateway (per-user thread metadata)
    ├─ /api/langgraph/*     → LangGraph Server (agent runtime)
    └─ /                    → Frontend (Next.js)
                  │
                  └─→ PostgreSQL RDS (用户/线程元数据/反馈, 阿里云)
                  └─→ 共享文件系统 .deer-flow/users/{user_id}/threads/{tid}/...
```

**用户隔离层级**:
1. **行级**: SQL 查询通过 `threads_meta.user_id` 过滤（auth contextvar 自动注入）
2. **文件系统**: `.deer-flow/users/{user_id}/threads/{tid}/user-data/` 物理目录隔离
3. **Sandbox**（轮 4 切 AioSandboxProvider 后）: 每用户容器隔离

---

## 1. 依赖

| 组件 | 版本 | 用途 |
|---|---|---|
| Docker + Docker Compose | 24.x+ | 容器编排 |
| 阿里云 PostgreSQL RDS | 14+ | 用户/线程元数据存储 |
| nginx | 1.18+（容器内）| TLS + 反向代理 + cookie 透传 |
| Let's Encrypt certbot | latest | 自动续签 TLS 证书 |
| Python 3.12+ | (容器内) | Backend runtime |
| Node.js 22+ | (容器内) | Frontend runtime |

**最低硬件**:
- 4 vCPU / 8GB RAM / 100GB SSD（10 人内并发）
- 磁盘需要为 `.deer-flow/users/` 预留足够空间（每用户 ≈ 1-5GB 训练数据 + artifact）

---

## 2. 阿里云 Postgres 准备

### 2.1 创建实例

阿里云控制台 → 云数据库 RDS → 创建 PostgreSQL 实例：
- 版本: PostgreSQL 14 或 15
- 规格: 推荐 2 核 4GB（pg.mysql.x4.medium 或类似）
- 存储: 100GB SSD
- 网络: 与 ECS 同 VPC（避免公网走流量），开启 SSL

### 2.2 创建数据库与用户

```sql
-- 通过 RDS 控制台「账号管理」创建账号 ethoinsight 并授权
-- 然后通过「数据库管理」创建数据库 ethoinsight
-- 或通过 psql 客户端：

CREATE DATABASE ethoinsight ENCODING 'UTF8' LC_COLLATE 'zh_CN.UTF-8' LC_CTYPE 'zh_CN.UTF-8' TEMPLATE template0;
CREATE USER ethoinsight WITH PASSWORD '<strong-random-password>';
GRANT ALL PRIVILEGES ON DATABASE ethoinsight TO ethoinsight;
\c ethoinsight
GRANT ALL ON SCHEMA public TO ethoinsight;
```

### 2.3 防火墙白名单

RDS 实例 → 数据安全性 → 白名单 → 添加 ECS 内网 IP 段（VPC 内网通信）。

### 2.4 SSL 证书

阿里云 PG 强制 SSL，asyncpg 需 sslmode：
```
DATABASE_URL=postgresql+asyncpg://ethoinsight:<pass>@rm-xxxxx.pg.rds.aliyuncs.com:5432/ethoinsight?sslmode=require
```

---

## 3. 首次部署步骤

### 3.1 准备 .env

在项目根目录创建 `.env`（chmod 0600）：

```bash
# ─── Auth ──────────────────────────────────────────────────────────
# 必须 ≥ 32 字符；用 python -c "import secrets; print(secrets.token_urlsafe(48))" 生成
AUTH_JWT_SECRET=<48-char-random-string>

# ─── Database ──────────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://ethoinsight:<pass>@rm-xxxxx.pg.rds.aliyuncs.com:5432/ethoinsight?sslmode=require

# ─── Network ───────────────────────────────────────────────────────
DEER_FLOW_HOST=ethoinsight.example.com  # 给 nginx 用

# ─── 模型 API keys（按需）─────────────────────────────────────────
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...
# ... 其他模型 keys
```

### 3.2 改 config.yaml 切 postgres

在 `packages/agent/config.yaml`：

```yaml
database:
  backend: postgres
  postgres_url: $DATABASE_URL
  # sqlite_dir 不需要

run_events:
  backend: db   # 用 PG 持久化运行事件
```

### 3.3 启动

```bash
cd /opt/ethoinsight
docker compose up -d

# 等待 60s
sleep 60

# 检查
curl -s https://ethoinsight.example.com/api/v1/auth/setup-status
# 期望: {"needs_setup": true}

curl -s https://ethoinsight.example.com/health
# 期望: {"status": "healthy"}
```

### 3.4 创建 admin（首次访问 setup wizard）

浏览器访问 `https://ethoinsight.example.com/setup`：
- 输入 admin 邮箱（如 `admin@yourorg.com`）
- 设置 admin 密码（≥ 8 字符，建议用密码管理器生成）
- 点击「创建管理员账户」

后续访问 `/login` 用此凭证登录。

---

## 4. 运维

### 4.1 加用户

**方法一（推荐）**: 通过前端注册页 `/login` → 切到「注册」标签页 → 填写邮箱密码注册。

**方法二（CLI）**:
```bash
docker compose exec gateway uv run python -m app.gateway.auth.reset_admin --create-user user@example.com
# 会输出临时密码，让用户首次登录后改密
```

### 4.2 重置密码

用户忘记密码：

```bash
docker compose exec gateway uv run python -m app.gateway.auth.reset_admin user@example.com
# 生成新临时密码，写入 /etc/.../admin_initial_credentials.txt（0600 权限）
# 通过安全渠道发送给用户
```

### 4.3 备份

**数据库**:
```bash
# 阿里云控制台启用自动备份（每天）
# 或手动 pg_dump:
pg_dump -h rm-xxxxx.pg.rds.aliyuncs.com -U ethoinsight -d ethoinsight \
        --format=custom > /backup/ethoinsight-$(date +%Y%m%d).dump
```

**文件系统**:
```bash
# .deer-flow 用户数据 + 训练数据
tar -czf /backup/deerflow-data-$(date +%Y%m%d).tar.gz \
        /opt/ethoinsight/.deer-flow
```

### 4.4 监控

| 指标 | 告警阈值 | 来源 |
|---|---|---|
| Gateway HTTP 5xx 率 | > 1% | nginx access.log |
| 登录响应时间 | > 2s | 应用日志 / Prometheus |
| PG 连接数 | > 70%上限 | 阿里云监控 |
| 磁盘使用 | > 80% | 系统监控 |
| `.deer-flow/users/` 目录大小 | > 50GB | 定时任务 du |

### 4.5 性能注意事项

- bcrypt 默认 12 rounds，每次登录耗 ~300ms。10 人并发不是瓶颈，但若用户增加到 50+ 并经常登录，考虑把 cookie 有效期延长（`AuthConfig.token_expiry_days`）减少登录次数。
- 阿里云 PG 默认连接池上限 60，asyncpg 默认每个 Gateway 进程 8-10 个连接，单机部署足够。

---

## 5. 排障

### 5.1 401 Authentication Required

| 现象 | 原因 | 解决 |
|---|---|---|
| 登录页所有请求 401 | AuthMiddleware 注册顺序错 | 检查 `app.py` middleware 顺序: CORS → Auth → CSRF |
| 登录后 API 仍 401 | cookie 没正确写入 | 检查 nginx `proxy_pass_header Set-Cookie` 配置 |
| 老 thread 列表为空 | 未 migrate | 重启 Gateway，lifespan 会自动 migrate orphan threads |

### 5.2 CSRF Token Mismatch

Channels 调用 LangGraph SDK 报 CSRF 错：
- 检查 `channels/manager.py` 是否生成 `_csrf_token = generate_csrf_token()`
- 检查 SDK headers 是否包含 `CSRF_HEADER_NAME` + `Cookie: CSRF_COOKIE_NAME=...`

### 5.3 PG 连接失败

```
asyncpg.exceptions._base.PostgresError: SSL connection has been closed unexpectedly
```

- 检查 DATABASE_URL 含 `?sslmode=require`
- 检查 ECS IP 在 RDS 白名单
- 检查 RDS 实例运行状态（控制台监控页）

### 5.4 Orphan thread

如果用户登录后看不到老 thread：
```bash
# 查看 LangGraph store 中 owner_id 为空的 thread
docker compose exec gateway uv run python -c "
import asyncio
from app.gateway.deps import langgraph_runtime
# ... orphan migration 在 lifespan 启动时自动跑, 一般不需要手动
"
```

如果 lifespan migration 遗漏：手动重启 Gateway 即可（idempotent）。

---

## 6. 数据隔离保证

| 保证 | 实现 |
|---|---|
| **行级**: 用户只能看自己的 thread/run/feedback | SQL 查询 WHERE user_id = current_user.id（authz.py 装饰器自动注入）|
| **文件系统**: 用户上传/分析数据物理隔离 | `paths.thread_dir(tid, user_id=...)` → `{base}/users/{user_id}/threads/{tid}/` |
| **JWT 防伪**: token 不可伪造 | bcrypt + SHA-256 预哈希签名，token_version 字段防 replay |
| **CSRF 防御**: state-changing 请求需 CSRF token | Double Submit Cookie pattern（`csrf_middleware.py`）|
| **API 路径**: server-controlled metadata 不可被客户端覆盖 | threads.py 中 `_strip_reserved_metadata` 移除客户端注入的 owner_id |

---

## 7. 升级到轮 4（轮 3 不实施，留参考）

轮 4 计划：
1. **LangGraph PostgresSaver**: thread state 也走 PG 持久化（重启不丢对话）
2. **AioSandboxProvider**: 每用户独立 Docker 容器，sandbox 物理隔离
3. **Multi-user 性能压测**: 10 / 30 / 50 人并发基准

详见 `docs/superpowers/specs/2026-05-07-deerflow-tier234-round3-design.md §6`。

---

## 8. 相关文档

- 设计 spec: `docs/superpowers/specs/2026-05-07-deerflow-tier234-round3-design.md`
- 实施 plan: `docs/superpowers/plans/2026-05-07-deerflow-tier234-round3-plan.md`
- 完成 handoff: `docs/handoffs/2026-05-08-deerflow-tier234-round3-completed-handoff.md`
- 弃用的旧计划: `docs/plans/2026-04-23-multi-user-deployment.md`
- DeerFlow 同步 SOP: `docs/sop/deerflow-sync-sop.md`
