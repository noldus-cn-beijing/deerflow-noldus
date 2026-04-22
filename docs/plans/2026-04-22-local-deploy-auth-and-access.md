# 本地部署授权与多终端访问方案

> **创建日期**: 2026-04-22
> **状态**: 设计完成，待 v0.1+ 实施
> **作者**: Qiuyang + Claude Opus 4.7
> **适用场景**: 实验室本地部署（Docker），Web 多终端访问

---

## 1. 部署形态

```
实验室服务器（Linux，Docker）
  └── Docker Compose 一键启动
        ├── Nginx (:2026)
        ├── Frontend (Next.js)
        ├── Gateway (FastAPI)
        └── LangGraph (Agent 后端)

实验室里所有终端（笔记本 / 台式机 / iPad）
  └── 浏览器访问 http://服务器IP:2026 → 直接用
```

不搞客户端安装，不搞 Electron 打包分发。Web 天然跨平台、多终端。

---

## 2. 授权机制：签名 License + 服务器绑定

### 2.1 核心思路

授权绑的是**实验室那台服务器**，不是任何终端设备。我们持有一对 RSA 密钥，给每个客户签发一个签名过的 license 文件，绑定其服务器机器指纹。

```
Noldus（我们）                    实验室（客户）

持有 RSA 私钥                     收到部署包
签发 license.lic                  ├── Docker 镜像（内嵌公钥）
     │                            ├── license.lic（我们签发的）
     └── 发给客户 ─────────────→  └── 部署到服务器上
```

### 2.2 License 文件结构

```json
{
  "customer": "北京大学行为学实验室",
  "machine_id": "a3f7c2e1b9d4...",
  "issued_at": "2026-09-01",
  "expires_at": "2027-09-01",
  "max_concurrent_users": 10,
  "paradigms": "all",
  "features": ["analysis", "knowledge_qa"]
}
```

整个 JSON 用 RSA 私钥签名，生成 `license.lic`。客户改任何一个字段签名就对不上。

### 2.3 机器指纹采集

容器内取不到真实硬件信息，指纹在**宿主机**上采集：

**部署脚本**（宿主机上执行）：

```bash
# get-machine-id.sh — 在宿主机上运行
MACHINE_ID=$(sha256sum <<EOF
$(cat /proc/cpuinfo | grep -m1 "serial" | awk '{print $3}')
$(lsblk -n -o SERIAL /dev/sda 2>/dev/null || cat /etc/machine-id)
$(ip link show | grep ether | head -1 | awk '{print $2}')
EOF
)

echo "MACHINE_ID=${MACHINE_ID}"
echo "请将此 ID 发送给 Noldus 获取授权文件"
```

结果写入 `.env`：

```env
MACHINE_ID=a3f7c2e1b9d4...
```

Docker Compose 将 `MACHINE_ID` 传入容器，Gateway 中间件用它与 license 中的 `machine_id` 比对。

### 2.4 Gateway 校验流程

每次请求经过中间件：

```
请求进来
  ↓
license.lic 存在吗？      → 否 → 403 + 引导页（显示机器 ID + 复制按钮）
  ↓ 是
签名合法吗？               → 否 → 403 "授权文件无效"
  ↓ 是
machine_id 匹配吗？        → 否 → 403 "授权文件与本机不匹配"
  ↓ 是
过期了吗？                 → 是 → 403 "授权已过期，请联系 Noldus 续期"
  ↓ 否
max_concurrent_users 超了吗？ → 是 → 429 "当前在线人数已达上限"
  ↓ 否
放行
```

### 2.5 部署流程（实验室管理员，5 分钟）

```
1. docker compose up
2. 首次启动 → 检测无 license → 页面显示引导：
   ┌────────────────────────────────────────┐
   │  EthoInsight 首次使用                   │
   │                                        │
   │  请将以下机器 ID 发送给 Noldus 获取授权    │
   │  [ a3f7c2e1b9d4... ]     [复制]        │
   └────────────────────────────────────────┘
3. 管理员把 ID 发给我们（邮件/微信）
4. 我们签发 license.lic 发回
5. 管理员放到 docker 挂载目录
6. 服务自动检测 → 生效（或 docker compose restart）
```

### 2.6 续期

发新的 `license.lic` 覆盖旧文件，重启服务即可。邮件、U 盘、网盘传都行。

---

## 3. 数据隔离：自报名（非登录）

**不搞登录系统。** 本地部署场景下登录注册体验差，没有必要。

### 3.1 机制

```
第一次打开浏览器：
  ┌──────────────────────────────────┐
  │  请输入你的名字                    │
  │  （仅用于区分不同研究者的数据）       │
  │  [ 张三________________ ]  [确认]  │
  └──────────────────────────────────┘

之后每次打开：自动以"张三"身份进入，不再输入。
换一台终端打开，输入同一个名字 → 看到自己的数据。
```

- 名字存浏览器 localStorage
- 后续所有请求自动带 user_id（名字的哈希）
- Thread（对话记录）按 user_id 隔离
- Memory（AI 记忆）按 user_id 隔离
- A 看不到 B 的分析，AI 不会把 A 的上下文混给 B

### 3.2 这不是安全机制

安全靠 license（只有这个实验室能用）。自报名只管数据不串。

实验室内部互相信任的前提下，自报名足够。如果某天需要更强的用户认证（比如线上 SaaS），再引入 better-auth 登录系统。

---

## 4. 整体分层

```
授权层（Noldus → 实验室）
┌─────────────────────────┐
│ license.lic 签名验证      │
│ 服务器机器指纹绑定         │
│ 到期日 + 并发用户数检查     │
└─────────────────────────┘
            │
            ▼
网络层（局域网）
┌─────────────────────────┐
│ 实验室内部网络            │
│ 任何终端浏览器访问         │
│ 天然多终端、跨平台         │
└─────────────────────────┘
            │
            ▼
隔离层（浏览器 localStorage）
┌─────────────────────────┐
│ 自报名 → user_id         │
│ Thread 按用户隔离         │
│ Memory 按用户隔离         │
└─────────────────────────┘
```

---

## 5. 方案对比（为什么不选其他方案）

| 方案 | 为什么不选 |
|---|---|
| **better-auth 登录系统** | 本地部署场景下要求研究员注册登录体验差，增加部署复杂度（数据库、session 管理） |
| **API Key in .env** | 能防外部访问但绑不了机器，客户可以把整套 Docker 拷到别处跑 |
| **硬件加密狗** | 上世纪方案，和"新时代 AI 产品"定位冲突，Web 多终端场景不适配，增加故障点 |
| **Keygen EE** | 需要自托管一套额外的许可证管理服务（PostgreSQL + Redis + Go），架构过重。客户数量有限（几十到几百个实验室）时自建更轻 |
| **网络白名单** | 只能防外部，防不了实验室内部的人把 Docker 拷走 |

---

## 6. 实现量估算

| 组件 | 工作量 | 说明 |
|---|---|---|
| Gateway License 校验中间件 | ~100 行 Python | 读 lic → 验签 → 验指纹 → 验到期 → 验并发 |
| 机器指纹采集脚本 | ~30 行 Bash | 取宿主机硬件特征 → SHA256 |
| License 签发工具（Noldus 内部） | ~50 行 Python | 输入客户信息 → RSA 签名 → 输出 .lic |
| 前端首次引导页 | ~80 行 TSX | 无 license 时显示机器 ID + 复制按钮 |
| 前端自报名组件 | ~50 行 TSX | 首次访问输入名字，存 localStorage |
| Thread/Memory 按用户隔离 | 修改 hooks.ts + updater.py | 请求带 user_id，查询过滤 |
| **合计** | **~300 行 + 测试** | 不动 LangGraph 核心逻辑 |

---

## 7. 实施节奏

| 阶段 | 时间 | 做什么 |
|---|---|---|
| **v0.1**（2026-09） | 不做 | 内部验证阶段，没有外部客户 |
| **v0.1+ 首批客户部署** | v0.1 后 | 实装全部：License 签发 + 校验 + 自报名隔离 |
| **商业化** | 规模化阶段 | Web 管理后台批量签发 license + 按范式模块授权 + 可选 SaaS 模式切 better-auth |

---

## 8. 一句话总结

License 文件绑服务器（防盗用）+ 浏览器自报名隔离数据（防串数据）+ 所有终端打开就用（零摩擦）。不搞登录、不搞加密狗、不搞客户端安装。

---

## 附录：相关文档

- [docs/plans/2026-04-08-multi-user-deployment.md](2026-04-08-multi-user-deployment.md) — 多用户部署分析（better-auth 方案，v0.1+ 可参考其 Thread/Memory 隔离的代码改动点）
- [docs/roadmap.md](../roadmap.md) Phase 5 — 部署与规模化
