# EthoInsight 阿里云部署指南

**日期**: 2026-05-09（最后更新 2026-05-26）
**范围**: v0.1 单机 Docker Compose，10 人内并发
**预计时间**: 首次部署 ~90 分钟（含 PG 创建等待）

> ⚠️ **当前实际部署方式已变更**（2026-05-26）
>
> ACR 权限暂未到位，当前走「本地 build → 镜像 tar 推送 ECS」方案，不用 ACR / watchtower。
> **新部署 SOP 见**: [deploy-via-tar-sop.md](deploy-via-tar-sop.md)
> **一句话操作**: `cd packages/agent && make deploy-tar`
>
> 本文档保留作为未来切回 ACR pipeline 时的参考，以及 ECS / 安全组 / 域名等基础设施配置的参考。

---

## 架构（部署后）

```
浏览器 → HTTPS → ECS (公网 IP)
                    │
                    ├─ nginx:2026 (TLS)
                    ├─ frontend:3000 (Next.js)
                    ├─ gateway:8001 (FastAPI + better-auth)
                    └─ langgraph:2024 (agent runtime)
                         │
                         └─→ 阿里云 RDS PostgreSQL (用户/线程/反馈)
```

只用了 2 个阿里云产品：**ECS + RDS PostgreSQL**。

---

## 第 1 步：开阿里云资源（控制台操作，~20 分钟）

### 1.1 ECS 实例

- **地域**: 离用户最近的（如华东 2 上海）
- **规格**: 8 vCPU / 16 GB 内存（ecs.g7.2xlarge 或类似）
- **系统盘**: 100 GB ESSD PL1
- **OS**: Ubuntu 22.04 LTS
- **网络**: 分配公网 IP，安全组放行 22 (SSH) + 443 (HTTPS) + 80 (HTTP)

### 1.2 RDS PostgreSQL

- **版本**: PostgreSQL 15
- **规格**: 2 核 4 GB（`pg.n2.small.2` 或类似）
- **存储**: 100 GB SSD
- **网络**: 与 ECS **同一 VPC**（内网访问，不走公网）
- **白名单**: 添加 ECS 内网 IP 段

### 1.3 域名（可选，v0.1 可以先用 IP）

如果有域名，添加 A 记录指向 ECS 公网 IP。

---

## 第 2 步：初始化 RDS（在 ECS 上通过 psql 操作，~5 分钟）

SSH 到 ECS：

```bash
ssh root@<ECS公网IP>

# 安装 psql 客户端
apt-get update && apt-get install -y postgresql-client

# 连接 RDS（地址从控制台获取）
psql -h rm-xxxxx.pg.rds.aliyuncs.com -U <root账号> -d postgres
```

在 psql 中执行：

```sql
CREATE DATABASE ethoinsight;
CREATE USER ethoinsight WITH PASSWORD '<生成强密码>';
GRANT ALL PRIVILEGES ON DATABASE ethoinsight TO ethoinsight;
\c ethoinsight
GRANT ALL ON SCHEMA public TO ethoinsight;
\q
```

记下连接串（后面要用）：

```
DATABASE_URL=postgresql+asyncpg://ethoinsight:<密码>@rm-xxxxx.pg.rds.aliyuncs.com:5432/ethoinsight?sslmode=require
```

---

## 第 3 步：ECS 环境准备（SSH 操作，~20 分钟）

```bash
# ─── 3.1 基础工具 ────────────────────────────────────────────────
apt-get update && apt-get install -y \
  docker.io docker-compose-v2 git curl openssl nginx certbot python3-certbot-nginx

systemctl enable docker && systemctl start docker

# ─── 3.2 拉代码 ──────────────────────────────────────────────────
mkdir -p /opt/ethoinsight
cd /opt
git clone <你的仓库地址> ethoinsight
cd ethoinsight
git checkout dev  # 或你部署的分支

# ─── 3.3 创建数据目录 ────────────────────────────────────────────
mkdir -p /opt/ethoinsight/data/deer-flow
```

---

## 第 4 步：配置（在 ECS 上，~10 分钟）

### 4.1 生成密钥

```bash
cd /opt/ethoinsight

# 生成 JWT secret（≥32 字符）
echo "AUTH_JWT_SECRET=$(openssl rand -hex 48)" > packages/agent/.env
chmod 600 packages/agent/.env

# 追加数据库连接
cat >> packages/agent/.env <<'EOF'
DATABASE_URL=postgresql+asyncpg://ethoinsight:<密码>@rm-xxxxx.pg.rds.aliyuncs.com:5432/ethoinsight?sslmode=require
EOF
```

### 4.2 修改 config.yaml

编辑 `packages/agent/config.yaml`，改三个地方：

```yaml
# 1. database 段 — 从 sqlite 切 postgres
database:
  backend: postgres
  postgres_url: $DATABASE_URL

# 2. run_events 段 — 用 PG 持久化
run_events:
  backend: db
```

其他保持不变（sandbox 暂时用 `LocalSandboxProvider`，v0.1 单机够用）。

### 4.3 构建 Docker 镜像

```bash
cd /opt/ethoinsight/packages/agent

# 构建所有服务镜像（首次 ~10 分钟）
docker compose -f docker/docker-compose.yaml build
```

---

## 第 5 步：启动（~2 分钟）

```bash
cd /opt/ethoinsight/packages/agent

# 设置环境变量（docker-compose 需要）
export DEER_FLOW_HOME=/opt/ethoinsight/data/deer-flow
export DEER_FLOW_REPO_ROOT=/opt/ethoinsight/packages/agent
export DEER_FLOW_CONFIG_PATH=/opt/ethoinsight/packages/agent/config.yaml
export DEER_FLOW_EXTENSIONS_CONFIG_PATH=/opt/ethoinsight/packages/agent/extensions_config.json
export DEER_FLOW_DOCKER_SOCKET=/var/run/docker.sock
export HOME=/root

# 启动
docker compose -f docker/docker-compose.yaml up -d

# 等 60 秒让服务就绪
sleep 60

# 验证
curl -s http://localhost:2026/api/v1/auth/setup-status
# 期望: {"needs_setup":true}

curl -s http://localhost:2026/health
# 期望: {"status":"healthy"}

docker compose -f docker/docker-compose.yaml ps
# 期望: nginx/frontend/gateway/langgraph 四个都是 Up
```

---

## 第 6 步：配置域名 + TLS（~15 分钟）

### 选项 A：用域名 + Let's Encrypt（推荐）

```bash
# 编辑 nginx 站点
cat > /etc/nginx/sites-available/ethoinsight <<'EOF'
server {
    listen 80;
    server_name ethoinsight.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:2026;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_buffering off;
        proxy_read_timeout 600s;
    }
}
EOF

ln -s /etc/nginx/sites-available/ethoinsight /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# 签发证书
certbot --nginx -d ethoinsight.yourdomain.com

# 验证
curl -s https://ethoinsight.yourdomain.com/api/v1/auth/setup-status
```

### 选项 B：直接用 IP（v0.1 临时方案）

把 ECS 安全组从 443 改成开放 2026 端口，然后直接访问 `http://<ECS公网IP>:2026`。

> 注意：这种模式下 cookie `secure` 标志为 false，仅限内部测试。

---

## 第 7 步：初始化系统（浏览器操作，~3 分钟）

1. 浏览器访问 `https://ethoinsight.yourdomain.com`（或 `http://<IP>:2026`）
2. 自动跳转到 `/setup` — 首次引导页
3. 输入 admin 邮箱和密码 → 点击创建
4. 成功后自动登录，进入 workspace

---

## 第 8 步：注册用户

**方式一（推荐）**: 让用户访问 `/login` → 切到注册标签页 → 填写邮箱密码。

**方式二（CLI）**: 管理员在 ECS 上执行：

```bash
cd /opt/ethoinsight/packages/agent
docker compose -f docker/docker-compose.yaml exec gateway \
  uv run python -m app.gateway.auth.reset_admin --create-user zhangsan@noldus.com
# 输出临时密码，通过私密渠道发给用户
```

---

## 运维速查

```bash
# 查看日志
cd /opt/ethoinsight/packages/agent
docker compose -f docker/docker-compose.yaml logs -f --tail=100 gateway

# 重启某个服务
docker compose -f docker/docker-compose.yaml restart gateway

# 全栈重启
docker compose -f docker/docker-compose.yaml down && docker compose -f docker/docker-compose.yaml up -d

# 备份数据库（RDS 控制台已自动每天备份，这是手动保险）
pg_dump -h rm-xxxxx.pg.rds.aliyuncs.com -U ethoinsight -d ethoinsight \
  --format=custom > /backup/ethoinsight-$(date +%Y%m%d).dump

# 备份文件数据
tar -czf /backup/deerflow-$(date +%Y%m%d).tar.gz /opt/ethoinsight/data/deer-flow

# 磁盘监控
df -h /opt/ethoinsight/data
```

---

## 阿里云费用预估（月）

| 资源 | 规格 | 月费（约） |
|---|---|---|
| ECS | 8C16G | ¥600-800 |
| RDS PostgreSQL | 2C4G 100GB | ¥400-500 |
| **合计** | | **¥1000-1300** |

v0.1 10 人内部使用够用。后续加 OSS 存储约 ¥50-100/月。

---

## 相比之前 aws-user-backend 方案，省掉了什么

| 之前需要的 | 现在 | 原因 |
|---|---|---|
| AWS Cognito / 阿里云 IDaaS | ❌ 不需要 | better-auth 嵌入 Gateway，本地 JWT |
| DynamoDB / Tablestore | ❌ 不需要 | 验证码/会话/限流都用 PG |
| 独立 user-backend 服务 | ❌ 不需要 | auth 逻辑在 Gateway 进程内 |
| Redis | ❌ 不需要 | v0.1 10 人，内存限流够用 |
| OSS 对象存储 | ⚠️ 暂不需要 | v0.1 单机磁盘够用，后续可加 |
| **实际上只需要** | **ECS + RDS PG** | 两个产品 |

---

## 常见问题

**Q: 上传的文件存哪里？**
A: ECS 本地磁盘 `/opt/ethoinsight/data/deer-flow/threads/{id}/user-data/uploads/`，定期备份到 OSS 或本地 tar。轮 4 会加 OSS 集成。

**Q: 怎么更新代码？**
```bash
cd /opt/ethoinsight
git pull
cd packages/agent
docker compose -f docker/docker-compose.yaml build
docker compose -f docker/docker-compose.yaml up -d
```

**Q: 怎么监控？**
初期手动：`docker compose logs -f gateway | grep ERROR`，RDS 控制台看连接数/IOPS。轮 4 加 Prometheus。
