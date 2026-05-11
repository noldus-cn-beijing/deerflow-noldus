# 2026-05-08 nginx 大响应被截断 (ChunkLoadError) 修复 — 完成

## TL;DR

浏览器访问 `192.168.1.116:2026` 报 `ChunkLoadError: Failed to load chunk` + `ERR_INCOMPLETE_CHUNKED_ENCODING` + `ERR_CONTENT_LENGTH_MISMATCH`，调用栈定格在 `WorkspaceLayout` 的 `<AuthProvider>`（**误导性栈帧** — 真因不在应用代码）。

根因：**nginx worker（wangqiuyang 用户）无法写系统 temp 目录 `/var/lib/nginx/proxy/`（属主 `www-data:root`，权限 0700）**。`location /` 默认 `proxy_buffering on`，大响应（3.7MB chunk JS / 中文字体 woff2 / 图片）超过默认 32KB 缓冲区时溢出到磁盘 → EACCES → 响应被截断。

修复：本地 + Docker 两份 nginx conf 的 `location /` 块加 `proxy_buffering off`（与 `/api/langgraph/` 块保持一致）+ `http {}` 顶部加独立 `*_temp_path` 作深度防御（避免依赖系统目录）。

**改动范围**：2 个文件 / 共 ~30 行净增。**验证**：3.7MB chunk 完整下载、字体完整下载、nginx error log reload 后零 Permission denied 错误。

## 改动清单

| 文件 | 改动 |
|---|---|
| `packages/agent/docker/nginx/nginx.local.conf` | `location /` 加 `proxy_buffering off; proxy_cache off;`；`http {}` 顶部加 5 个 `*_temp_path` 指向 `logs/`（项目自有目录） |
| `packages/agent/docker/nginx/nginx.conf` | 同上同步；`*_temp_path` 指向 `/tmp/`（容器内 nginx 用户可写） |

## 根因诊断（Phase 1: 收集证据）

### 错误链反向追踪

```
浏览器：ChunkLoadError @ WorkspaceLayout (误导性栈帧)
    ↓
浏览器：ERR_INCOMPLETE_CHUNKED_ENCODING / ERR_CONTENT_LENGTH_MISMATCH
    ↓
nginx 错误日志(logs/nginx-error.log)：持续 4 月 28 起
    [crit] open() "/var/lib/nginx/proxy/X/YY/000000ZZZZ" failed (13: Permission denied)
    while reading upstream
    ↓
权限层：/var/lib/nginx/proxy 属主 www-data:root, 模式 0700
    nginx worker 由 wangqiuyang 用户启动 (make dev) → EACCES
    ↓
配置层：location / 没有显式 proxy_buffering off，默认 on
    ↓
架构层：项目用 wangqiuyang 启 /usr/sbin/nginx (系统二进制 + 编译时硬编码 temp 路径)
    + 本地 conf 没声明独立 proxy_temp_path
    = 借用系统 nginx 但没接管它的所有路径
```

### 多 nginx 实例确认

```
PID 153830 (root + www-data workers, 4月28起)  ← 系统 systemctl 启的，没监听 2026
PID 3391114/3391115 (wangqiuyang, 17:07起, listening on 2026)  ← make dev 启的，触发本 bug
```

`lsof -iTCP:2026 -sTCP:LISTEN` 直接证明 2026 端口由 wangqiuyang 进程持有。

### 为什么栈帧是 WorkspaceLayout

react-server-dom-turbopack chunk 加载失败 → React 构造组件树时找不到导出 → 栈定格在第一个 `'use client'` 边界（恰好是 `<AuthProvider>`）。**这只是症状的最浅层暴露点，不是 bug 位置**。

## Phase 2-3: 假设与验证

**假设**：`location /` 加 `proxy_buffering off` 后，nginx 不再写 temp 文件，权限问题不影响响应完整性。

**验证步骤**：
1. 修改本地 conf → 重新渲染 → reload nginx
2. `curl -v http://192.168.1.116:2026/_next/static/chunks/8767c__pnpm_2122bd35._.js`
3. 比对 reload 前后 nginx error log

**结果**：
- 修复前：响应被截断，error log 持续 `Permission denied`
- 修复后：HTTP 200，size=3,744,589（与 Content-Length 头一致），文件以 `//# sourceMappingURL=...` 完整结尾，error log 自 reload 后零错误

## Phase 4: 实施（两层防御）

### 第一层：禁用 proxy_buffering

`location /` 块对齐 `/api/langgraph/` 的处理（后者一直显式 off，所以 SSE/流式 API 没碰到这问题）：

```nginx
location / {
    # ... existing config ...

    # 禁用 proxy 缓冲：dev mode 大 chunk 直接流给客户端，避免写盘
    proxy_buffering off;
    proxy_cache off;
}
```

效果：根本上不写 temp 文件，权限问题/磁盘 IO 问题都不存在。

### 第二层：独立 *_temp_path（防御纵深）

`http {}` 顶部声明：

```nginx
# nginx.local.conf：
proxy_temp_path logs/nginx-proxy-temp;
client_body_temp_path logs/nginx-client-body-temp;
fastcgi_temp_path logs/nginx-fastcgi-temp;
uwsgi_temp_path logs/nginx-uwsgi-temp;
scgi_temp_path logs/nginx-scgi-temp;

# nginx.conf (Docker)：路径换成 /tmp/nginx-*-temp
```

效果：将来若有人在某个新 location 又默认开了 buffering，temp 文件会写到项目自己拥有的目录而不是系统目录，不会再撞权限。

## 为什么本地 conf 用 `logs/`、Docker conf 用 `/tmp/`

| 部署 | nginx 启动方式 | 当前用户 | temp 路径选择 |
|---|---|---|---|
| **本地 dev** | `nginx -p $REPO_ROOT` | wangqiuyang（启动者） | `logs/`（相对项目根，nginx 自动建子目录，0700 wangqiuyang:developers） |
| **Docker 容器** | `nginx:alpine` 默认入口 | 容器内 nginx 用户 | `/tmp/`（容器内 nginx 用户可写；某些 distroless 镜像不带 `/var/cache/nginx/`） |
| **K8s Deployment** | 同 Docker | 容器内 nginx 用户 | `/tmp/`（emptyDir 通常挂这里，由 sizeLimit 控制） |

## 上云风险评估（用户问的问题）

| 部署形态 | 这个具体 bug | 同类（权限/路径冲突）风险 |
|---|---|---|
| 本地 dev（已修） | ✅ 触发过 | 低（修复后） |
| Docker 容器（已对齐） | ❌ 不会（镜像里 nginx 用户拥有 /var/cache/nginx/） | 低 |
| K8s | ❌ 不会 | 低（buffering 关了 + temp 路径独立） |
| 传统 VM 裸装 nginx + systemd | ❌ 不会（root master + www-data worker，能写自家目录） | 中（SELinux/AppArmor 可能拦截） |
| Serverless / Vercel + CloudFront | 不存在 nginx | 不适用 |

**结论**：当前修复对所有部署形态都安全；不依赖任何特定部署模式的特殊权限。

## 验证

### 修复前
```
GET /_next/static/chunks/8767c__pnpm_2122bd35._.js HTTP/1.1
→ ERR_INCOMPLETE_CHUNKED_ENCODING（响应中途被关）

nginx error log:
  [crit] open() "/var/lib/nginx/proxy/X/YY/..." failed (13: Permission denied)
  while reading upstream, ...
```

### 修复后
```
$ curl -sS -o /tmp/chunk_retest.js -w "HTTP %{http_code} | size=%{size_download}\n" \
    http://192.168.1.116:2026/_next/static/chunks/8767c__pnpm_2122bd35._.js
HTTP 200 | size=3744589

$ tail -c 50 /tmp/chunk_retest.js | od -c | tail -3
0000040   _   2   1   2   2   b   d   3   5   .   _   .   j   s   .   m
0000060   a   p
0000062

$ ls -ld packages/agent/logs/nginx-*-temp
drwx------ 2 wangqiuyang developers 4096 5月 8 18:17 logs/nginx-client-body-temp
drwx------ 2 wangqiuyang developers 4096 5月 8 18:17 logs/nginx-fastcgi-temp
drwx------ 2 wangqiuyang developers 4096 5月 8 18:17 logs/nginx-proxy-temp
drwx------ 2 wangqiuyang developers 4096 5月 8 18:17 logs/nginx-scgi-temp
drwx------ 2 wangqiuyang developers 4096 5月 8 18:17 logs/nginx-uwsgi-temp

# nginx error log（reload 后 18:09 起零 Permission denied）
$ awk '$1>="2026/05/08" && $2>="18:09:00"' packages/agent/logs/nginx-error.log
(空)
```

字体下载验证：
```
$ curl -sS -o /tmp/font_test.woff2 -w "HTTP %{http_code} | size=%{size_download}\n" \
    http://192.168.1.116:2026/fonts/OPPOSans-zh.woff2
HTTP 200 | size=3706784
```

## 经验沉淀

1. **栈帧是症状，不是根因** — `ChunkLoadError @ WorkspaceLayout` 看起来像是 React 应用 bug，实际是传输层问题。规律：**`ERR_INCOMPLETE_CHUNKED_ENCODING` + `ERR_CONTENT_LENGTH_MISMATCH` 同时出现 = 反向代理 / 上游响应被截断的典型签名**，立刻去查 nginx error log，不要在 React 代码里找。

2. **借用系统软件但不接管所有路径是隐性技术债** — `make dev` 用 `/usr/sbin/nginx`（系统二进制）但跑在自己的 conf 下，编译时硬编码的 `/var/lib/nginx/proxy/` 仍生效。**今后凡是借用系统软件的本地启动脚本，都应显式声明所有路径**（temp、log、pid 等），避免类似的隐性依赖。

3. **`proxy_buffering off` 应该是反向代理 dev/前端流量的默认选择** — 除了少数缓存场景，dev mode 大 chunk + 静态资源 + SSE 流式响应都不应该被 nginx 缓冲。**`/api/langgraph/` 块从一开始就 off，但 `location /` 漏了**——这是历史不一致。

4. **深度防御 = 同一类问题在多个层都堵住** — 即使单层失效（比如有人改回 `proxy_buffering on`），独立的 `*_temp_path` 仍能避免撞系统目录权限问题。这种"约束分散到多处"的思路也是 EV19 模板地基设计（[2026-05-08-ev19-template-skill-foundation-design.md](../superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md) §6 鲁棒性分层）的核心模式。

## 上游归属判定

**两份 nginx conf 都是 noldus 本地 fork 的**（含 langgraph upstream 块、特定 location 配置等）。本次改动**不需要给上游 deerflow 提 issue**：

- `proxy_buffering off` 是合理的反向代理默认，但上游可能有理由保留（如生产场景某些 location 需要缓冲）
- `*_temp_path` 配置是部署环境强相关，不是普适修复
- 下次 sync 上游 nginx conf 时，**记得检查这两处定制是否还在**：
  - `location / { ... proxy_buffering off; proxy_cache off; ... }`
  - `http { ... proxy_temp_path ... client_body_temp_path ... }`

## 后续 / 不在本次范围

- **`location /` 还有一个潜在问题**：`proxy_set_header Connection 'upgrade'` 是无条件设的，所有对 frontend 的请求都被打上 `Connection: upgrade` 头（不仅 WebSocket）。这通常只在某些边缘情况下导致 keep-alive 行为异常。当前未观察到问题，**先不动，避免 "while I'm here" 改动**——如果将来发现 HMR / keep-alive 异常再处理。

- **系统级 nginx（PID 153830，4 月 28 起）仍在跑但不监听 2026** — 不是本次修复目标，但建议运维 / 用户后续 `sudo systemctl disable nginx` 或弄清楚它在干啥（可能是某次 apt 升级遗留）。**当前不影响 make dev 本地开发，只占内存**。

## 待办

- [ ] git add + commit（用户决定）— 改动文件：
  - `packages/agent/docker/nginx/nginx.local.conf`
  - `packages/agent/docker/nginx/nginx.conf`
- [ ] 浏览器硬刷新（Ctrl+Shift+R / Cmd+Shift+R）清掉之前的不完整缓存

## 相关文档

- 上次会话总交接：[2026-05-08-multi-user-path-csrf-and-resources-fix-handoff.md](2026-05-08-multi-user-path-csrf-and-resources-fix-handoff.md)
- 同日 EV19 模板地基设计：[2026-05-08-ev19-template-skill-foundation-design.md](../superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md)
- 同日 EV19 模板地基实施计划：[2026-05-08-ev19-template-skill-foundation-plan.md](../superpowers/plans/2026-05-08-ev19-template-skill-foundation-plan.md)
