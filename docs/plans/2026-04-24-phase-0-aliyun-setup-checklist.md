# Phase 0：阿里云基础设施 setup checklist（人工操作清单）

> **给你的说明**：这是 Phase 0 所有**需要人工到控制台操作**的清单。按顺序做，每一步都有"验收"，确认通过再进下一步。做完这份，基础设施就绪，后续 Phase 1-5 的代码改造 agent 可以接手了。
>
> **预计耗时**：纯操作 2-3 小时，但含"等阿里云审核"（短信签名 1-3 天）。
>
> **需要准备**：
> - 企业营业执照（实名认证用）
> - 可收短信的手机号（签名备案人）
> - 一张信用卡或支付宝余额（开通服务）
> - 一个用于签名的品牌名/公司名（短信签名用）

---

## 检查清单总览

| 步骤 | 阿里云产品 | 人工耗时 | 外部审核耗时 |
|---|---|---|---|
| ✅ 0.A 账号 + RAM | 访问控制 | 20 分钟 | 0 |
| ✅ 0.B VPC 网络 | 专有网络 VPC | 30 分钟 | 0 |
| ✅ 0.C RDS PostgreSQL | 云数据库 RDS | 20 分钟 | 实例创建 10 分钟 |
| ✅ 0.D Redis | 云数据库 Redis | 10 分钟 | 实例创建 5 分钟 |
| ✅ 0.E OSS | 对象存储 OSS | 15 分钟 | 0 |
| ✅ 0.F IDaaS CIAM | 应用身份服务 | 30 分钟 | 0 |
| ✅ 0.G 短信签名 | 短信服务 | 15 分钟 | **1-3 天（最早启动）** |
| ✅ 0.H KMS + SLS | 密钥管理 + 日志 | 10 分钟 | 0 |
| ✅ 0.I SAE + ACR | Serverless 应用引擎 + 容器镜像 | 15 分钟 | 0 |

**执行顺序建议**：
- **Day 1 上午**：立刻做 0.A → 0.G（短信签名提交审核，启动外部时钟）
- **Day 1 下午**：做 0.B → 0.F（基础设施全开出来）
- **Day 1 晚上～Day 2**：等短信签名审核，做 0.H → 0.I

---

## 0.A 账号 + RAM 子账号

### A.1 主账号准备
1. 登录阿里云，**必须完成企业实名认证**（不是个人实名）
   - 控制台 → 账号中心 → 实名认证 → 企业认证
   - 上传营业执照，法人对私打款验证
2. 绑定支付方式（信用卡或余额）
3. **强烈建议**：主账号绑定硬件 MFA 或手机 MFA

### A.2 创建 RAM 子账号
控制台 → 访问控制 RAM → 用户 → 创建用户

创建 3 个：

| 用户名 | 用途 | 权限 |
|---|---|---|
| `ethoinsight-dev` | 开发调试 | `ReadOnlyAccess` + 测试环境相关写权限 |
| `ethoinsight-deploy` | CI/CD 部署 | 最小化：`AliyunSAEFullAccess` + `AliyunContainerRegistryFullAccess` + OSS 写 |
| `ethoinsight-ops` | 运维应急 | `AdministratorAccess` + **强制 MFA** |

每个用户：
- 勾选"**OpenAPI 调用访问**"（用于 AccessKey）
- **不勾选**"控制台访问"，除非需要（ops 账号需要）
- 创建后立刻下载 AccessKey CSV，**主账号 AK/SK 不要使用**

### A.3 给子账号绑权限策略
用户详情 → 权限管理 → 新增授权 → 选系统策略

### ✅ 验收
- [ ] 3 个子账号都已创建
- [ ] 3 份 AccessKey CSV 已下载并**安全保存**（1Password / Bitwarden）
- [ ] ops 账号已开启 MFA
- [ ] 主账号 AK/SK **从未使用过**（使用了的话立即轮换）

---

## 0.B VPC 网络基线

### B.1 创建 VPC
控制台 → 专有网络 VPC → 创建 VPC

| 字段 | 值 |
|---|---|
| 地域 | **华东 1（杭州）** 或 **华东 2（上海）**（就近用户）|
| VPC 名称 | `ethoinsight-prod-vpc` |
| IPv4 网段 | `10.0.0.0/16` |

### B.2 创建 2 个 vSwitch（跨可用区）

同一 VPC 下创建：

| 名称 | 可用区 | 网段 |
|---|---|---|
| `vsw-prod-az-h` | 可用区 H | `10.0.1.0/24` |
| `vsw-prod-az-i` | 可用区 I | `10.0.2.0/24` |

### B.3 创建 3 个安全组
控制台 → ECS → 网络与安全 → 安全组 → 创建安全组

| 安全组名称 | 用途 | 入方向规则 |
|---|---|---|
| `sg-ethoinsight-public` | 公网入口（给 SAE SLB 用）| 允许 0.0.0.0/0 的 443 |
| `sg-ethoinsight-backend` | 内部服务互通 | 允许 `sg-ethoinsight-public` 所有端口 |
| `sg-ethoinsight-data` | 数据层（RDS/Redis 挂这个）| 允许 `sg-ethoinsight-backend` 5432/6379 |

**关键原则**：`sg-data` **只**允许 `sg-backend` 访问，不开公网任何端口。

### B.4 NAT 网关 + EIP
控制台 → NAT 网关 → 创建

- 创建 NAT 网关挂在 `vsw-prod-az-h`
- 创建 EIP 绑定到 NAT
- 创建 SNAT 规则：`10.0.0.0/16` → EIP

用途：backend 容器访问外网（调 LLM API、拉镜像等）。

### ✅ 验收
- [ ] VPC 创建完成
- [ ] 2 个 vSwitch 跨可用区
- [ ] 3 个安全组规则正确
- [ ] NAT 网关可用

---

## 0.C RDS PostgreSQL

### C.1 创建实例
控制台 → RDS → 实例列表 → 创建实例

| 字段 | 值 |
|---|---|
| 引擎 | **PostgreSQL 14** 或更高 |
| 系列 | **高可用版**（主备双机热备）|
| 规格 | `pg.n2.small.2c`（2C 4G，起步够用）|
| 存储 | ESSD 100GB（可扩）|
| VPC | 选 Phase 0.B 的 VPC |
| vSwitch | `vsw-prod-az-h` |
| 购买时长 | 包年包月（正式）或按量付费（测试）|

### C.2 配置白名单
实例详情 → 数据库连接 → 修改白名单
- **删除默认的 `127.0.0.1`**
- 添加安全组 `sg-ethoinsight-data`
- **不要**添加 `0.0.0.0/0`

### C.3 创建数据库和账号
实例详情 → 账号管理 → 创建账号

| 账号 | 权限 | 密码管理 |
|---|---|---|
| `app` | 读写 `ethoinsight` 库 | 强密码，存 KMS |
| `admin` | DBA 权限 | 强密码，只本人掌握 |

数据库管理 → 创建数据库：
- 名称：`ethoinsight`
- 字符集：`UTF8`
- Owner：`admin`

### C.4 启用 pgvector（为未来向量检索）
参数设置 → 搜索 `shared_preload_libraries` → 加入 `vector`
（RDS PG 默认支持，无需额外操作，建表时 `CREATE EXTENSION vector;` 即可）

### C.5 开启自动备份
备份恢复 → 备份策略 → 设置保留 7 天，每日凌晨 3 点

### ✅ 验收
- [ ] 实例运行中
- [ ] 从 VPC 内跳板机 `psql` 能连接成功
- [ ] 公网直连**失败**（说明白名单正确）
- [ ] `CREATE EXTENSION vector;` 成功
- [ ] 连接串已存到 KMS（0.H 完成后）

---

## 0.D Redis 托管版

### D.1 创建实例
控制台 → 云数据库 Redis → 创建实例

| 字段 | 值 |
|---|---|
| 版本 | **Redis 7.0 社区版** |
| 架构 | **标准版-双副本**（主备 HA）|
| 规格 | 1GB 起步 |
| VPC | Phase 0.B 的 VPC |
| vSwitch | `vsw-prod-az-h` |

### D.2 白名单
- 修改白名单：**删除默认**，添加 `sg-ethoinsight-data`
- 不开公网

### D.3 设置密码
实例 → 账号管理 → 重置密码 → 强密码，存 KMS

### ✅ 验收
- [ ] 从 VPC 内 `redis-cli -h xxx -a xxx` 能连接
- [ ] 公网直连失败
- [ ] 密码已存 KMS

---

## 0.E OSS

### E.1 创建 Bucket

| Bucket 名称 | 用途 | 读写权限 | 地域 |
|---|---|---|---|
| `ethoinsight-uploads-<随机后缀>` | 用户上传 | **私有** | 同 VPC 地域 |
| `ethoinsight-outputs-<随机后缀>` | Agent 产出 | **私有** | 同 VPC 地域 |

**注意**：bucket 名全球唯一，加随机后缀避免冲突。例如 `ethoinsight-uploads-20260424`。

### E.2 基础配置（每个 bucket 都做）

**版本控制**：Bucket 设置 → 版本控制 → 开启

**服务端加密**：Bucket 设置 → 服务端加密 → SSE-KMS（选 0.H 里创建的 CMK）

**CORS**（允许前端直传）：
- Bucket 设置 → 跨域设置 → 创建规则
- 来源：你的前端域名（如 `https://app.ethoinsight.com`）+ `http://localhost:3000`（dev）
- 允许方法：GET, PUT, POST, HEAD
- 允许 Headers：`*`
- 暴露 Headers：`ETag`
- 缓存：`3600`

### E.3 生命周期规则

`uploads`：90 天转低频存储，365 天转归档
`outputs`：180 天转低频，730 天转归档

### E.4 CDN 加速（仅 outputs）
OSS 控制台 → `outputs` → 传输加速 → 开启 CDN

- 加速域名：`cdn-outputs.ethoinsight.com`（备案过的域名）
- 回源：OSS 内网
- HTTPS：申请免费证书

### E.5 为 SAE OSS CSI 挂载准备 RAM 角色
RAM → 角色 → 创建角色 → 阿里云服务 → 选 "SAE"

绑定策略（只能访问这两个 bucket）：
```json
{
  "Version": "1",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["oss:*"],
    "Resource": [
      "acs:oss:*:*:ethoinsight-uploads-<后缀>",
      "acs:oss:*:*:ethoinsight-uploads-<后缀>/*",
      "acs:oss:*:*:ethoinsight-outputs-<后缀>",
      "acs:oss:*:*:ethoinsight-outputs-<后缀>/*"
    ]
  }]
}
```

记下角色 ARN，SAE 部署时用。

### ✅ 验收
- [ ] 2 bucket 私有状态
- [ ] CORS 配置能让前端直传
- [ ] CDN 域名可访问（outputs 侧）
- [ ] SAE RAM 角色 ARN 已记录

---

## 0.F IDaaS CIAM 实例（最核心的一步）

> **这一步做完，你的"用户系统"功能上就完整了**——注册、登录、改密、找回、MFA、短信验证码全有，不用写一行代码。

### F.1 开通 CIAM
控制台 → 应用身份服务 IDaaS → **选择 CIAM（客户身份和访问管理）**，不是 EIAM

**关键**：IDaaS 有两种 Edition——EIAM 是给企业员工 SSO 用的，CIAM 才是我们要的 C 端用户体系。确保你选的是 **CIAM** 实例。

### F.2 创建"实例"
- 实例名：`ethoinsight-ciam`
- 规格：按 MAU 计费的最小档起步

### F.3 创建"应用"
实例 → 应用管理 → 添加应用 → 自研应用 → 选择 **OIDC** 协议

| 字段 | 值 |
|---|---|
| 应用名称 | `ethoinsight-web` |
| 应用图标 | 上传 logo |
| 应用主页 | `https://app.ethoinsight.com` |
| 授权模式 | ✅ 授权码模式、✅ 刷新 Token |
| Redirect URI | `https://app.ethoinsight.com/auth/callback`<br>`http://localhost:3000/auth/callback` |
| Logout Redirect URI | `https://app.ethoinsight.com/`<br>`http://localhost:3000/` |
| Scope | `openid profile email phone` |
| 返回 Claims | 勾选：`sub`、`email`、`phone_number`、`name` |

创建后保存：
- **client_id**
- **client_secret**
- **Issuer URL**
- **Discovery URL**（形如 `https://<实例>.idaas.aliyuncs.com/oauth2/.well-known/openid-configuration`）
- **JWKS URI**

全部存 KMS。

### F.4 定制登录页（零代码）
实例 → 登录定制 → 主题配置

- 上传 logo
- 改品牌色
- 中文文案
- 添加用户协议/隐私政策 URL
- 设置支持的登录方式：✅ 账号密码、✅ 短信验证码、✅ 邮箱验证码

### F.5 开启自助流程
实例 → 用户管理 → 自助服务

- ✅ 允许用户自助注册（**关键**）
- 注册字段必填：邮箱 或 手机号（至少一个）
- ✅ 允许忘记密码（短信/邮箱重置）
- ✅ 允许改密码
- ✅ 允许修改个人信息

### F.6 开启 MFA（可选但建议）
实例 → 安全策略 → 多因子认证

- 默认策略：**可选**（用户自行在个人中心开启）
- 管理员用户：**强制**

### F.7 配置短信 Provider（等 0.G 完成后回来做）
实例 → 渠道配置 → 短信服务
- 选阿里云短信
- 填入 0.G 申请的签名 + 模板 Code

### ✅ 验收
- [ ] 浏览器打开 IDaaS 登录 URL（`https://<实例>.login.aliyunidaas.com`），能看到定制过的登录页
- [ ] 能**完整走通**："注册 → 收到短信 → 登录 → 改密码 → 忘记密码 → 重置" 全流程，**没写一行代码**
- [ ] 用 Postman 模拟 OIDC 授权码流程能拿到 JWT
- [ ] JWT 里 `sub` 字段就是 user_id，`email`/`phone_number` 字段可见

---

## 0.G 短信签名与模板 ⚠️最早启动

> **审核 1-3 天**，**所有操作里第一个要做的**！

### G.1 实名和入驻
控制台 → 短信服务 → 国内消息 → 签名管理
（如果没入驻，会引导企业认证 + 选择资质类型）

### G.2 申请签名

签名管理 → 添加签名

| 字段 | 值 |
|---|---|
| 签名名称 | 品牌名，例如 `EthoInsight` 或 `行为洞察` |
| 签名来源 | 企业名称 |
| 签名用途 | "验证码"类 |
| 证明文件 | 上传营业执照 |

⏱️ **提交后 1-3 天审核**。

### G.3 申请模板

审核通过后，创建 4 个模板：

| 模板名 | 模板内容 |
|---|---|
| 注册验证码 | `您的注册验证码是 ${code}，5 分钟内有效。` |
| 登录验证码 | `您的登录验证码是 ${code}，5 分钟内有效。` |
| 改密验证码 | `您的改密验证码是 ${code}，5 分钟内有效。` |
| 绑定验证码 | `您的绑定验证码是 ${code}，5 分钟内有效。` |

每个模板也要审核（通常 2 小时内）。

### G.4 生成 AK/SK for IDaaS
- RAM → 为 IDaaS 专用的子账号 `ethoinsight-idaas-sms` 创建
- 只授权 `AliyunDysmsFullAccess`
- AK/SK 存 KMS，Phase 0.F.7 时配给 IDaaS

### ✅ 验收
- [ ] 签名状态"已通过"
- [ ] 4 个模板状态"已通过"
- [ ] IDaaS 子账号 AK/SK 已存 KMS
- [ ] 用 aliyun CLI 能发一条测试短信到本人手机

---

## 0.H KMS + SLS

### H.1 KMS
控制台 → 密钥管理服务 KMS → 创建密钥

| 字段 | 值 |
|---|---|
| 密钥用途 | 加密/解密 |
| 密钥类型 | 软件密钥 |
| 别名 | `ethoinsight-cmk` |

**入库所有密钥**（KMS → 凭据管理 → 创建凭据）：

- `rds/app-password`
- `rds/admin-password`
- `redis/password`
- `idaas/client-secret`
- `idaas/sms-ak-sk`
- `oss/deploy-ak-sk`
- `tencent-sms/ak-sk`（腾讯云 SMS 已有的）

### H.2 SLS
控制台 → 日志服务 SLS → 创建 Project

| 字段 | 值 |
|---|---|
| Project 名称 | `ethoinsight-prod` |
| 地域 | 同 VPC 地域 |

创建 Logstore：
- `backend`（user-backend 日志）
- `deerflow`（DeerFlow 日志）
- `access`（SLB 访问日志）

保留时长：30 天（可调）。

### ✅ 验收
- [ ] KMS CMK 创建完成，所有凭据入库
- [ ] SLS Project + 3 Logstore 可见
- [ ] 用 aliyun CLI `log list-projects` 返回正确

---

## 0.I SAE + ACR

### I.1 开通 SAE
控制台 → Serverless 应用引擎 SAE

### I.2 创建命名空间
命名空间 → 创建

| 字段 | 值 |
|---|---|
| 命名空间名 | `ethoinsight-prod` |
| 地域 | 同 VPC 地域 |
| VPC | Phase 0.B 的 VPC |
| vSwitch | `vsw-prod-az-h` + `vsw-prod-az-i` |
| 安全组 | `sg-ethoinsight-backend` |

### I.3 开通 ACR Enterprise
控制台 → 容器镜像服务 ACR

- 选 **企业版** （个人版也行但功能有限）
- 创建命名空间：`ethoinsight`
- 创建仓库：
  - `ethoinsight/user-backend`
  - `ethoinsight/deerflow-backend`
  - `ethoinsight/frontend`
- 仓库属性：**私有**

### I.4 SAE 绑定 ARMS + SLS
SAE → 应用高级配置（应用部署时勾选，此时预配置）

- 预先在 ARMS 控制台创建应用接入：`ethoinsight-user-backend`、`ethoinsight-deerflow-backend`
- 记录 License Key，部署时填

### ✅ 验收
- [ ] SAE 命名空间可见
- [ ] ACR 能用 `docker login` + `docker push` 推测试镜像
- [ ] ARMS 应用接入凭证已记录

---

## 总验收（Phase 0 Done 标准）

全部打钩后，Phase 0 完成：

### 账号 & 网络
- [ ] 企业实名认证通过
- [ ] 3 个 RAM 子账号 + MFA
- [ ] VPC + 2 vSwitch + 3 安全组 + NAT + EIP 就绪

### 数据层
- [ ] RDS PostgreSQL 运行，`pgvector` 可用
- [ ] Redis 运行
- [ ] OSS 2 个 bucket 配好 CORS + 生命周期 + SAE RAM 角色

### 身份层
- [ ] IDaaS CIAM 实例 + `ethoinsight-web` OIDC 应用
- [ ] 登录页定制完成
- [ ] 浏览器能完整跑一遍"注册 → 登录 → 改密"**零代码**
- [ ] 短信签名 + 模板全部审核通过
- [ ] IDaaS 配好短信 provider，能发真实短信

### 运维层
- [ ] KMS 所有凭据入库
- [ ] SLS Project + Logstore
- [ ] SAE 命名空间 + ACR 仓库
- [ ] ARMS 接入凭证

### 密钥清单（**保管好**）
所有敏感凭据应存在 KMS 凭据管理，**不要**出现在任何 git、Slack、微信里。清单：

| 凭据 | 用途 |
|---|---|
| IDaaS client_secret | user-backend OIDC 换 token |
| IDaaS Issuer/JWKS URL | user-backend JWT 验签 |
| RDS `app` 密码 | user-backend 连 DB |
| Redis 密码 | user-backend 连 Redis |
| OSS deploy AK/SK | CI/CD 推前端静态产物 |
| ACR 用户名密码 | CI/CD 推镜像 |
| 腾讯云 SMS AK/SK | 业务通知短信（不是认证短信）|

---

## 做完后下一步

**Phase 0 DONE** 后，可以：

1. 写一份 handoff：`docs/handoffs/YYYY-MM-DD-phase-0-complete.md`，列明所有资源 ID / Endpoint / 凭据引用（不是凭据本身）
2. 启动 **Phase 1.1**（大量删除 AWS 专属代码）—— agent 可以执行
3. 启动 **Phase 1.2**（OIDC 接入 authlib）—— agent 可以执行，需要 Phase 0.F 的 IDaaS discovery URL

---

## 常见坑

1. **实名认证选错类型**：必须是**企业**认证，个人认证很多 2C 功能用不了（比如短信签名）
2. **VPC 地域选错**：所有资源必须**同地域**，否则无法互通。杭州和上海不能混
3. **IDaaS 选错 Edition**：EIAM 是员工 SSO，CIAM 才是 C 端。一旦创错实例就换不回来
4. **短信签名太晚提交**：这是**最容易被忽略的 blocker**。先交！
5. **忘记给 SAE RAM 角色加 OSS 权限**：CSI 挂载时会 fail，排查半天
6. **KMS 凭据 vs KMS 密钥混淆**：凭据管理是专门存 "key-value" 敏感配置的，密钥是真正的加密 key。别混
7. **RDS 白名单留了 0.0.0.0/0**：默认有时会带这个，**必须删**

---

## 参考文档

- [阿里云 IDaaS CIAM 官方文档](https://help.aliyun.com/zh/idaas/ciam/)
- [SAE + OSS CSI 挂载文档](https://help.aliyun.com/zh/sae/user-guide/mount-an-oss-volume)
- [阿里云短信签名审核规范](https://help.aliyun.com/zh/sms/user-guide/signature-specifications)
- [OSS boto3 兼容性](https://help.aliyun.com/zh/oss/developer-reference/use-amazon-s3-sdks-to-access-oss)
