# 探索：Electron 桌面化踩坑 —— 面向不懂技术研究员的一键安装包

> 状态：**探索文档（非实施 spec、非立项）**。摸清坑、给倾向，不写实现代码。
> 日期：2026-06-26
> 代码基线：dev HEAD（写时 `0f7f9787`）
> 前提（用户拍板）：① 桌面版**给不懂技术的行为学研究员**；② 用 **Electron 不用 Tauri**（前端 SSR 重度依赖，已记 memory `feedback_desktop_packaging_electron_over_tauri_ssr_dependency`）；③ **装时弹管理员权限/重启在 Windows 装软件里是正常步骤、可接受**；④ 目标是**一键装好一切，用户不用懂 Docker**。

---

## 〇、一句话结论（先给倾向，依据在下文）

两条后端打包路线各有命门，**没有干净的赢家**：
- **路线 A（捆绑容器运行时 + 复用现成 compose 镜像）**：省掉 Python 科学栈冻结的全部痛苦（这是最大优点），但继承容器栈的体积（~2GB）、首次拉/载镜像耗时、Windows 虚拟化前提、容器↔宿主文件路径映射。
- **路线 B（sidecar 直起进程 + 冻结 Python）**：体积小、启动快、离线即用、无虚拟化依赖（成熟桌面 AI 应用的主流做法），但 **ethoinsight 的 scipy/matplotlib/numpy 科学栈冻结是真实高风险坑**，且跨平台编译 + 三处代码签名成本高。

**倾向**：**先用路线 A 出可用桌面版**（因为它把"科学栈能不能跑"这个最大不确定性用现成已验证镜像消掉了，符合"先用 DeerFlow infra"原则——compose 是 DeerFlow 现成产物）；**B 作为体积/体验优化的后续路线**，仅当 A 的体积/首启慢成为真实用户投诉时再投入冻结。**但 A 有一个必须先验证的硬前提**（见 §四 路线 A 命门）。

---

## 一、本项目后端的真实形态（决定打包难度的根本）

### 1.1 服务栈（生产 compose，`packages/agent/docker/docker-compose.yaml`）

`make up` → `scripts/deploy.sh` 起 4 个容器：
- **nginx**（`:2026` 对外）→ 反代到 gateway/frontend
- **gateway**（`:8001`）：`uvicorn app.gateway.app:app`，**嵌入式 LangGraph runtime**（无独立 langgraph 进程）——agent 编排、subagent、sandbox 全在这个进程
- **frontend**（`:3000`）：`pnpm start`，**真 Next.js Node server**（非静态导出）
- **provisioner**：仅 K8s 模式用，桌面版跳过

这套 compose **自洽**（depends_on + env_file + 数据卷齐全），用户机器有容器运行时即可 `up`。**这是路线 A 的基础——DeerFlow 已经把整套打成镜像了。**

### 1.2 后端的进程外重依赖（决定路线 B 难度的根本）

Gateway 起来后，agent 执行时 **sandbox 在同机起 bash/python 子进程**跑 ethoinsight 脚本（`sandbox/local/local_sandbox.py`）。这意味着后端环境里**必须有完整 Python venv + ethoinsight 全科学栈**：
- `pandas / numpy / scipy`（统计决策树 Shapiro-Wilk/t-test）
- `matplotlib / seaborn`（**发表级图表——用户可见的核心产出**）
- `openpyxl / python-calamine`（Excel 解析）、`duckdb`、`markitdown`

**这就是 B 路线的命门**：PyInstaller/嵌入式 Python 冻结 numpy/scipy/matplotlib 这类带原生扩展（BLAS/LAPACK、动态后端）的科学栈，**业界公认易碎**（hidden imports 漏、运行期找不到 backend、子进程 PYTHONHOME 错乱）。而 A 路线这些已在容器里验证过，零冻结风险。

### 1.3 数据落盘根（`config/paths.py` 的 base_dir / `DEER_FLOW_HOME`）

thread 文件树（`users/{uid}/threads/{tid}/{user-data,shared}`）落在 `DEER_FLOW_HOME`。桌面版应指向 OS 用户数据目录（Win `%LOCALAPPDATA%\EthoInsight`、mac `~/Library/Application Support/EthoInsight`、Linux `~/.local/share/ethoinsight`），避免写程序目录的权限问题。

> ⚠️ **路线 A 的文件映射坑**：容器内 agent 产物落 `/mnt/...`，用户上传的数据要进容器、产物要出来给前端看——靠 volume mount。Windows 上 WSL2 路径转换（`C:\` ↔ `/mnt/c`）是历史踩坑高发区。这条与本项目"文件路径可靠性"主题正交但叠加（见 `2026-06-26-file-path-reliability-loadbearing-convergence-spec.md`）。

---

## 二、前端 SSR 对桌面的约束

`workspace/layout.tsx` `force-dynamic` + `getServerSideUser()` SSR fetch 认证 → **桌面版必须真跑一个 Next.js Node server 进程，不能 load 静态文件**。两条路线都得带这个 Node 进程（A 在 frontend 容器里、B 用 Node standalone sidecar）。这也是当初否掉 Tauri 的同一事实。

---

## 三、认证简化（桌面单用户）

现状是 JWT + HttpOnly cookie + CSRF double-submit（`fetcher.ts` 注入 `X-CSRF-Token`，后端 CSRFMiddleware 校验；历史有 csrf 403 = session cookie 生命周期坑，memory `feedback_csrf_403_root_cause_is_session_cookie_lifetime_asymmetry`）。

桌面单用户**应绕过登录页**：启动时后端检测桌面模式 → 自动建/取本地用户 + 注入 session → 前端无缝 authenticated。**localhost 下 CSRF 需配 SameSite=Lax + 本地 origin 白名单**，否则跨进程 cookie 可能被拦。这是要专门验证的点，不是改改就行。

---

## 四、路线 A vs B 逐坑对比（带"管理员权限/重启可接受"修正前提）

> **诚实标注**：逐坑数 B 占优项更多（体积/启动/离线/路径/生命周期/无虚拟化依赖），A 占优在"科学栈零冻结风险/跨平台省事/开发生产一致/签名成本低/运维成熟"。但**占优项的"权重"不对等**——A 占优的几条恰好压在本项目最大的不确定性（科学栈能不能跑）和最大的工程量（三平台冻结+签名）上。"管理员权限/重启可接受"这个前提**显著抬高 A 的可行性**（A 最大减分项被判可接受），但**不改变 A 的体积/首启慢**。

| 坑 | 路线 A（容器） | 路线 B（sidecar 冻结） | 谁占优 |
|---|---|---|---|
| **ethoinsight 科学栈** | 镜像已验证，零冻结风险 | scipy/matplotlib/numpy 冻结**高风险** | **A（关键）** |
| **sandbox 子进程** | 容器内已工作 | 冻结环境里子进程找 venv 易错（PYTHONHOME） | **A** |
| **开发=生产一致** | compose 即生产 | 源码 venv vs 冻结 exe 两套环境 | **A** |
| **跨平台编译** | 一份镜像 + 三个 Electron 壳 | 三平台各冻结一次（原生扩展） | **A** |
| **代码签名成本** | 只签 Electron 壳 | 签 Python exe + Node exe + 壳（3×） | **A** |
| **包体积** | 镜像 ~2GB | ~600MB | **B** |
| **首次启动** | 拉/载镜像数分钟 | 解包 <2s | **B** |
| **离线即用** | 首次依赖镜像可达（除非预装 tar） | 装完离线可用 | **B** |
| **虚拟化前提** | Win 要 WSL2/HyperV（**用户判定可接受**） | 无 | **B（但 A 减分被前提抵消）** |
| **文件路径映射** | 容器↔宿主 volume，Win WSL2 路径转换坑 | 进程直接访问宿主 FS | **B** |
| **容器商业授权** | Docker Desktop 企业付费墙 → **改用 Podman 规避** | 无 | 平（A 用 Podman） |
| **生命周期/孤儿** | 崩溃留孤儿容器，需清理逻辑 | Electron spawn/kill 自然管理 | **B** |
| **进程异常重启** | `restart: unless-stopped` 自愈 | 需自写 watchdog | **A** |

---

## 五、路线 A 的"命门"——必须先验证的硬前提

A 路线最大风险**不在容器本身**，而在 **DooD（Docker-out-of-Docker）/ sandbox 在用户机器上能不能跑**：
- 生产 compose 挂了 `/var/run/docker.sock`（AioSandboxProvider 用），且 sandbox 默认在 gateway 容器内起子进程跑 ethoinsight。
- **必须先实测**：在一台干净的目标用户机器（尤其 Windows + WSL2）上 `docker compose up`，走一遍真实 dogfood（上传数据→agent 分析→出图→报告），确认 sandbox 子进程、文件映射、出图全链路通。
- 如果这条不通，A 路线的"复用现成镜像"优势就站不住——这是 A 立项前的 **go/no-go 验证**。

---

## 六、通用 Electron 坑（两路线共有）

1. **代码签名**：mac notarization（要 Apple Developer 账户）+ Win Authenticode（年费），不签会被 Gatekeeper/SmartScreen/杀毒拦，**对"不懂技术研究员"是致命体验**（弹"未知发布者/可能有害"会直接吓退）。**必做项**。
2. **窗口加载时序**：后端没起好就 `loadURL` → 白屏。必须 health check 轮询 `:2026/health` 通过再加载，加载期显进度条。
3. **自动更新**：electron-updater + 签名 + 托管。首版可不做（手工下载新包），但要预留。
4. **主↔渲染 IPC**：管理后端生命周期、健康状态、优雅关闭；Error 对象要转 JSON 传。
5. **生产关 DevTools**、SSE 流式经本地代理别被吃事件（历史有 async IO 阻塞 event loop 坑，memory `feedback_async_io_blocks_event_loop`）。

---

## 七、业界参照（WebSearch 坐实）

- **Ollama**：Electron + Go sidecar 单二进制，无容器，~200-300MB，启动 <5s。**但 Go 单文件无科学栈痛苦，不直接适用我们**。
- **LM Studio**：Electron + C++ 推理引擎嵌入，无 Docker。同样原生编译优势我们（Python）拿不到。
- **Datasette App**：Electron + PyInstaller 冻结 Python + SQLite，~150MB。**但 datasette 无 matplotlib/scipy，冻结容易——恰好印证我们 B 路线的科学栈冻结才是真坑**。

**结论**：业界 Python 系桌面应用之所以敢 sidecar 冻结，多是因为它们**没有重科学计算栈**。我们有 scipy/matplotlib，这把 B 路线的难度推到与众不同的高度——这正是倾向先走 A 的实证理由。

---

## 八、倾向与后续（非立项，待用户拍）

**倾向方案：路线 A（Podman/容器 + 复用 compose 镜像 + Electron 壳）先行**，理由：
1. 把本项目最大不确定性（科学栈能否跑）用**现成已验证镜像**消除——符合"先用 DeerFlow infra"原则。
2. 用户已判定"管理员权限/重启可接受"，A 的最大减分项被抵消。
3. 跨平台/签名工程量比 B 小（只签壳、一份镜像）。

**A 的代价（要接受）**：包体积大、首启拉镜像慢、Win WSL2 前提、文件映射要仔细做。

**B 留作后续优化**：当体积/首启成为真实研究员投诉时，再投入科学栈冻结（届时 ethoinsight 的依赖也更稳定）。

**真正开工前的 3 个 go/no-go**：
1. **§五 命门验证**：干净 Win+WSL2 机器上 compose 全链路 dogfood 通过。
2. **认证简化**：桌面单用户绕登录 + localhost CSRF 配通。
3. **签名通道**：拿到 Apple Developer + Win 代码签名证书（否则"不懂技术用户"会被系统拦死）。

三个都过，再立项写实施 spec（建 `packages/electron/` 壳 + backend 生命周期管理 + CI 三平台打包）。

---

*依据：实读 `docker-compose.yaml`（4 服务自洽、Gateway 嵌入式 runtime）、`local_sandbox.py`（同机起子进程跑 ethoinsight 科学栈）、前端 SSR 约束、CSRF/auth 现状；WebSearch 坐实业界 Python 桌面应用多无重科学栈故敢冻结。倾向先 A（复用 DeerFlow 现成镜像、消除科学栈冻结风险），B 作后续优化。A 立项前必过 §五 DooD/sandbox 在 Windows 实跑验证。*
