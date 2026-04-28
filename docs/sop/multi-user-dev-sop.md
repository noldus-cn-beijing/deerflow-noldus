# 多人共享开发机指南

> 适用场景：多个开发者在同一台塔式工作站上各自 clone 代码、并行开发 noldus-insight。

## 问题

`make dev` 默认占用 5 个固定端口（2026/3000/2024/8001/8002），同一台机器上**不能两个人同时跑**——后启动者会撞 "Address already in use"。

## 解决方案：每人一份端口偏移

启动时设 `PORT_OFFSET` 环境变量，所有端口自动加上这个数。

### 端口分配约定

| 开发者 | 偏移 | Nginx 入口（浏览器访问） | 备注 |
|---|---|---|---|
| 默认 / 王秋阳 | 0 | http://localhost:**2026** | 不加 PORT_OFFSET 即可 |
| 同事 A | 10000 | http://localhost:**12026** | `PORT_OFFSET=10000` |
| 同事 B | 20000 | http://localhost:**22026** | `PORT_OFFSET=20000` |

> **新人加入**：选一个还没被占的偏移量（30000、40000…），把自己加到上表。

### 操作步骤

```bash
# 1. 在自己 home 拉代码（一次性）
cd ~
git clone <repo-url> noldus-insight
cd noldus-insight/packages/agent
make install

# 2. 起服务（每次开发）
PORT_OFFSET=10000 make dev   # 同事 A
# 或
PORT_OFFSET=20000 make dev   # 同事 B

# 3. 浏览器访问对应入口
#    同事 A → http://localhost:12026
#    同事 B → http://localhost:22026

# 4. 关服务（Ctrl+C，或新终端跑 make stop）
make stop
```

### 永久写到 shell 配置（推荐）

每次都打 `PORT_OFFSET=10000` 麻烦，写到 `~/.bashrc` 或 `~/.zshrc`：

```bash
echo 'export PORT_OFFSET=10000' >> ~/.bashrc
source ~/.bashrc
```

之后直接 `make dev` 就用你的偏移端口，跟单人开发体验一致。

## 常见问题

### Q：浏览器在哪？我用 VS Code Remote-SSH 的话

VS Code Remote-SSH 会**自动检测**工作站上 listen 的端口，转发到你笔记本的 localhost。

- 同事 A 工作站启动 12026 → 同事 A 笔记本浏览器访问 `http://localhost:12026`
- 你笔记本看不到同事 A 工作站的端口（VS Code 按 SSH session 隔离）

VS Code 不能解决端口冲突，只能解决"如何把远程端口送到我笔记本"。

### Q：`make stop` 会不会杀掉同事的服务？

不会。已修改为 `pkill -u $USER`，**只杀自己的进程**，不影响同事。

### Q：能不能两个人同时调一个 thread？

技术上能（数据按 thread_id 隔离），但实际上你们各自跑各自的服务，**数据目录都在自己 home 下**（`~/noldus-insight/.deer-flow/`），互不可见——除非主动共享 thread_id 才能在对方机器上"重放"那个对话。

### Q：怎么看自己的端口被谁占了？

```bash
ss -tlnp | grep <端口号>     # 看是哪个进程
ps -ef | grep <进程ID>       # 看进程详情
```

如果发现自己的偏移端口被占（异常残留进程），跑 `make stop` 清理。

### Q：单独覆盖某个端口可以吗？

可以。`PORT_OFFSET` 给所有端口统一加偏移，但 `NGINX_PORT` / `LANGGRAPH_PORT` / `GATEWAY_PORT` / `FRONTEND_PORT` 任意一个单独设也会生效（覆盖偏移量）：

```bash
# 用偏移 10000 但 nginx 强制走 8888
PORT_OFFSET=10000 NGINX_PORT=8888 make dev
```

## 端口分配登记表

> 新人加入时编辑此文件添加自己。

| PORT_OFFSET | 开发者 | 加入日期 |
|---|---|---|
| 0 | 王秋阳 | 2026-04-28 |
| 10000 | _待分配_ | |
| 20000 | _待分配_ | |
