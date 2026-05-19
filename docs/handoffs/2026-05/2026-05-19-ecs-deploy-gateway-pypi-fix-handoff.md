# Handoff: EthoInsight ECS 部署 — gateway 依赖安装失败

**日期**：2026-05-19
**ECS**：i-2ze446h4ats7y8bgjn4w, 公网 IP 39.105.231.16, 华北2 (cn-beijing)
**项目**：`/root/noldus-insight/` (dev 分支)

## 现状

4 个 Docker 容器已启动，但 **gateway 容器不可用**导致注册页 502：

| 容器 | 状态 |
|------|------|
| deer-flow-nginx | running (已加 `/api/v1/` → gateway 的路由) |
| deer-flow-frontend | running |
| deer-flow-gateway | **running 但没在监听 8001**（.venv 里缺依赖） |
| deer-flow-langgraph | running |

## 根因

Docker build 阶段 `uv sync` 需要从 PyPI 下载 Python 包（fastapi、sqlalchemy 等），但 ECS 在国内，pypi.org 超时，导致 `.venv` 为空。

已在容器内手动验证：`docker exec deer-flow-gateway python -c "import fastapi"` → `ModuleNotFoundError`。

## 已尝试的修复

1. `export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/` + make up — env var 没透传到 build
2. sed 改 Dockerfile 第 47 行为 aliyun 镜像 — 确认改好了但重建后仍 502
3. sed 改 docker-compose.yaml 的 build arg 默认值 — 同上
4. `--no-cache` rebuild — 镜像时间戳仍是旧的，怀疑 docker compose 没正确读到环境变量

## 正确修复方向

需要彻底确保 `uv sync` 走阿里云镜像。两个文件（Dockerfile + docker-compose.yaml）都已 sed 修改但可能缓存仍没清干净。建议：

1. 确认两个文件都硬编码了镜像（不用变量默认值）：
   - Dockerfile L47: `sh -c "cd backend && UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ uv sync"`
   - docker-compose.yaml: `UV_INDEX_URL: https://mirrors.aliyun.com/pypi/simple/`

2. 彻底清理后重建：
```bash
docker builder prune -af
docker system prune -af
cd /root/noldus-insight/packages/agent && make down
# 带全量 env 重建
DEER_FLOW_HOME=/root/noldus-insight/packages/agent/backend/.deer-flow \
DEER_FLOW_CONFIG_PATH=/root/noldus-insight/packages/agent/config.yaml \
DEER_FLOW_EXTENSIONS_CONFIG_PATH=/root/noldus-insight/packages/agent/extensions_config.json \
DEER_FLOW_DOCKER_SOCKET=/var/run/docker.sock \
DEER_FLOW_REPO_ROOT=/root/noldus-insight/packages/agent \
make up
```

3. 验证：`docker exec deer-flow-gateway python -c "import fastapi; print('ok')"`

## nginx 修复（已完成）

`docker/nginx/nginx.conf` 缺少 `/api/v1/` 路由，已在 `/api/agents` 前追加了：
```
location /api/v1/ {
    proxy_pass http://gateway;
    proxy_http_version 1.1;
    proxy_set_header Host $http_host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```
