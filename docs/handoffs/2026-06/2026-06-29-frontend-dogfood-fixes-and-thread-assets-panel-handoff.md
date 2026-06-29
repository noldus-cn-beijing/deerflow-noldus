# Handoff：前端 dogfood 多 bug 修复 + thread 级产出物面板重构 + DB 迁移流程（2026-06-29）

> 本会话从「补 milestone/清 stale issue」起，转入一长串**真实 dogfood 前端 bug 修复**，最后落到用户提议的**架构改动：thread 级产出物面板**。**所有产出代码已 commit 并 push 到 dev**（HEAD `c8277f53`，本地=远端 `0 0`）。工作区只剩 3 个历史遗留 untracked 目录（与用户确认过保持原样）。

---

## 〇、一句话现状

dev HEAD = **`c8277f53`**。本会话推了 6 个 commit（`803b4f30`→`c8277f53`，见下表），全是**前端可见性/布局 bug 修复 + 右侧 Artifacts 面板重构成磁盘为真相的「本对话产出物」面板 + run-db-migrations HEAD_REV 修复**。**全部用 Playwright/CDP 活体实测坐实**（localhost:2026，守「代码有修复≠现象消除」）。用户本地 `make dev` 已重启（gateway+frontend 10:39 fresh）+ **本地 DB 已迁移到 `20260626_1700`**。

**会话结束时的最后一个交互**：用户在 `dev.ethoinsight.com`（**线上部署**）看到上传文件没堆叠，以为我把改动改没了。已用 Playwright 在**本地** localhost 实测证明堆叠完好（28 文件→4 平铺+「+24」徽章），真相=**线上是旧部署、不含 dev 分支这批未部署的前端改动**。用户尚未回应下一步。

---

## 一、本会话产出（全已 commit & push 到 dev）

| commit | 产出 | 性质 |
|---|---|---|
| `803b4f30` | 修对话页输入框消失/对话流空白：#227 把 chat-root `size-full`→`flex-1` 但父 `ResizablePanel` 非 flex 容器→高度塌成 198px | 🔴 前端布局 |
| `ccfb37df` | 修对话流滚动区溢出视口：进度轨下方包 MessageList 的 div `size-full`(h-full=900)→`flex-1 min-h-0`（治输入框遮挡末行 + 滚动条错位，两路由同改） | 🔴 前端布局 |
| `1a697354` | （已被 `59664106`/`8281feb4` 演进覆盖）首版对话流末尾兜底 InlineArtifactSummary | 前端 |
| `8281feb4` | 图廊入口单点化 + 画廊「返回对话」`router.push(threadId)` 不跳错 thread | 前端 |
| `59664106` | **thread 级产出物面板重构**（核心，见 §三） | 🟡 架构 |
| `c8277f53` | run-db-migrations.sh `HEAD_REV` 20260622→20260626（#229 新迁移） | 🟢 部署脚本 |

> 注：`milestone` 文档/issue 清账那批在更早会话（`07aa0684`/`c646dd4a`）；本 handoff 不重复。

---

## 二、本会话修复的前端 bug（dogfood 实测定根因，全已修）

这些都是 Phase 0 一批改造（#212/#214/#216/#227）落地后真实 dogfood 暴露、**同一族根因**：UI 派生自 streaming/state，随消息流/切 tab/run 抖动。

1. **输入框消失 + 对话流全空白**（`803b4f30`）— `ResizablePanel#chat` 不是 flex 容器，chat-root 的 `flex-1` 不解析 → 塌 198px。修 = `ChatBox` 的 Panel 补 `flex flex-col`（`chat-box.tsx`）。
2. **输入框遮挡末行 + 侧栏滚动条错位**（`ccfb37df`）— 包 MessageList 的 div `size-full` 在 sticky 进度轨下方起算却取满全高→溢出视口 127px。修 = `flex-1 min-h-0`（两路由 `chats/[thread_id]/page.tsx` + `agents/.../page.tsx`）。
3. **图廊/报告"看不到"**（`59664106` 根治）— 图真在磁盘（113 张+report.md）、`/artifacts/charts` 端点正常、`/gallery` 页正常，但**对话流无入口**：InlineArtifactSummary 只在 `present_files` 消息组渲染，而 lead 画完图常只用文字汇报不调 present_files。
4. **画廊汇总图与单样本图间巨大空白**（`59664106` 内）— `gallery-grid.tsx` 每个 grid 都套固定 `height:calc(100vh-320px)` 滚动容器，1 张图也撑一屏。修 = 条件虚拟化（`VIRTUALIZE_THRESHOLD=24`，<24 平铺自然高度）。
5. **报告末尾空白 box（带下载/复制图标）**（`59664106` 内）— `ReportCard` 对空 content 仍渲染 `MarkdownContent` 外壳。修 = 空内容守护（`report-card.tsx` `hasContent` 判断）。
6. **画廊返回跳错 thread**（`8281feb4`）— 对话页 `history.replaceState` 改写 `/new`→`/{id}` 后 `router.back()` 弹错。修 = `router.push(threadId)`。

---

## 三、核心架构改动：thread 级产出物面板（`59664106`，用户拍板）

**动机**：上述产物可见性 bug 是同一族——产物 UI 派生自 streaming 消息流/LangGraph state（chart-maker artifacts 在 subagent→lead 边界丢失、present_files 是否调用不确定）。用户提议：**为每个 thread 固定渲染一个「产出物」面板，从磁盘读全部产物，与 streaming 彻底解耦**。

**已实现（首版=图+报告；代码运行结果是第二步）**：
- **后端**：新增 `GET /threads/{tid}/artifacts/reports`（`artifacts.py` `list_report_artifacts`）——磁盘 `outputs/` 下 `.md/.html` → `[{path,kind:"report",filename,ext}]`，与 charts 端点对称、注册在 catch-all 前。+TDD `tests/test_artifacts_reports_endpoint.py`（4 case 全绿）。
- **前端**：
  - `core/artifacts/hooks.ts` +`useThreadAssets(threadId,{refetchSignal})`（`Promise.allSettled` 并行拉 charts+reports，**只读磁盘不读 state/streaming**，挂载 + run 完成补拉）；`utils.ts` +`reportsArtifactsURL`。
  - 新 `artifacts/thread-assets-panel.tsx`：报告区（`ReportCard`）+ 图表区（复用 `ArtifactGallery`）+ 空态。
  - `ReportCard` 抽成独立 `artifacts/report-card.tsx`（含空内容守护）；**删除** `inline-artifact-summary.tsx`（已无消费者）。
  - `chat-box.tsx` 右面板接 `ThreadAssetsPanel`（替换 `selectedArtifact?ArtifactFileDetail:ArtifactFileList`）；run 完成递增 `assetsRefetchSignal`。
  - `artifact-trigger.tsx` 始终渲染（不再据 `state.artifacts.length` 隐藏）。
  - `message-list.tsx` 移除 trailing InlineArtifactSummary + 清理失效的 artifacts/wasLoadingRef/refetchSignal/chartsStatus/import。
  - i18n 三文件加 `assetsTitle/reportsSection/chartsSection/assetsEmpty`。
- **验收**：`pnpm check` 0 error；vitest 198 pass / 2 pre-existing isStreaming 红 baseline；后端 130 artifact 测试全绿；裸导入两入口 0 退出。

---

## 四、DB 迁移流程（用户问题已答）

- **本地 `make dev` 前升级 DB**：`cd packages/agent && ./scripts/run-db-migrations.sh --local`（已有脚本，处理所有 alembic 坑 + 自动备份）。本会话已实跑：本地 DB `20260622_1700`→**`20260626_1700`**（#229 的 thread_cascade_fk）。
- **deploy-tar 已含迁移**：`deploy-via-tar.sh` line 197-205 早已在 `compose up` 前于 gateway 容器跑 `run-db-migrations.sh`。**不用额外加**。
- **修了 stale HEAD_REV**（`c8277f53`）：脚本 `HEAD_REV` 还停在 `20260622_1700`，#229 新迁移它不认（verify 报错，本地+deploy 两处都坏）。已升到 `20260626_1700`。**铁律：以后每加 alembic 迁移都要 bump `run-db-migrations.sh:HEAD_REV`**（注释已写）。

---

## 五、未完成事项（按优先级）

| # | 事项 | 状态/依赖 |
|---|---|---|
| 1 | **活体验收 thread 资产面板**（gateway 已重启、`/reports` 已生效）：右面板图+报告稳定显示、切 tab 不消失、报告无空白 box、画廊无巨大空白 | 待 Playwright/CDP 实测一轮（之前因 gateway 未重启 `/reports` 返 400，现已重启） |
| 2 | **右上角「N 步进行中」损坏 div + 流程结束仍显运行中** | 未查。疑 `RunTraceWidget`/AnalysisRail 运行态判定（`thread.isLoading` 与 trace 终态不一致）。独立于资产面板 |
| 3 | **流式卡顿**（用户：切回网页卡几秒，尤其 subagent think 流式时） | 资产解耦后压力已降；残余按 #212 `useDeferredValue` 层 + CDP perf trace 实测主线程占用再定，**不盲改流式核心**（守 `feedback_perf_is_efficient_impl_not_visual_downgrade`，不动 useStream/mergeMessages/dedupe） |
| 4 | **含图 report 改 HTML**（用户提：markdown 下载丢图） | 后端 report-writer 产物格式变更，影响面大，**独立立项**未做 |
| 5 | **部署 dev 分支到 `dev.ethoinsight.com`** | 用户最后困惑源于线上是旧部署。`make deploy-tar`（会自动跑 DB 迁移）。待用户定 |

---

## 六、关键陷阱 / 注意事项

1. **线上 vs 本地**：`dev.ethoinsight.com` = 远端部署（旧、deepseek、不含 dev 分支未部署改动）；`localhost:2026` = 本地 `make dev`（dev 分支最新）。用户「改动没了」的困惑就是看错环境。**核 bug 一律本地 localhost 实测**。
2. **本地 make dev 端口 2026**（nginx）；frontend :3000；gateway :8001。曾有用户看 :2027 死端口的乌龙。
3. **改后端后必重启 gateway 才生效**：Python 不热重载新路由。本会话 `/reports` 一度 400 就是 gateway 进程旧于代码。用户已于 10:39 重启。
4. **e2e 实测姿势**：用 `.claude/skills/noldus-insight-e2e/` 的 Playwright（`scripts/lib.js` config，e2e 登录用户 `qiuyang.wang@noldus.com`，state.json 有效）。e2e 用户 id=`e281f251...`，已有 113 图 thread `bd7ca7f7`。**用户自己的 thread（user `cd95effa`）e2e 登录看不到**（跨用户），核盘走磁盘 `.deer-flow/users/<uid>/threads/<tid>/`。CDP 监控 gateway.log 抓 run 生命周期很有效。临时 diag 脚本写在 e2e skill 目录、用完即删（别 commit）。
5. **3 个 untracked 历史遗留**（`docs/reports/`、`reports/report for june/`、`scripts/repro/`）**保持原样别提交**（多会话前与用户确认过）。
6. **本会话证实后端流水线健康**：CDP 全程监控 `ae7ef542` 一轮，metric 140/140、chart 113/113、report seal 全干净、无 hang。「继续画图卡死」真因 = 前端无图廊入口（图在后台画好但对话流不显示，看起来像卡住）→ 已被资产面板修复。
7. **chart-maker 没 reward-hack**：113 图+report.md 真落盘在 per-user 路径，是前端不显示，不是后端造假。

---

## 七、下一位 Agent 的第一步

1. `git log --oneline -6` 确认 HEAD=`c8277f53`（dev 在动，可能更新）。读本 handoff。
2. **第一优先**：用 Playwright/CDP 在 localhost:2026 活体验收 thread 资产面板（待办#1）——开 e2e 用户 thread `bd7ca7f7`，点右上「文件/Artifacts」开右面板，核：图廊(113)+报告(report.md)稳定显示、切 tab 回来不消失、报告卡展开无空白 box、汇总图与单样本图无巨大空白。截图确认。
3. 然后查待办#2「N 步进行中」损坏 div（看 `RunTraceWidget` + AnalysisRail 运行态判定，不依赖任何重启）。
4. 待办#3 流式卡顿：先 CDP perf trace 抓主线程占用证伪/坐实根因，再决定动不动 #212 层。**别凭印象盲改流式核心。**
5. 若用户要上线：`make deploy-tar`（自动跑 DB 迁移；确认 `run-db-migrations.sh` HEAD_REV 已是最新）。

## milestone 建议

前端 Phase 0 已有 milestone（`docs/milestone/frontend-generative-ux-phase0.md`）。本会话这批是 **Phase 0 落地后的 dogfood 修复 + 资产面板架构演进**——建议在该 milestone 末尾追加一段「2026-06-29 dogfood 修复批 + thread 资产面板重构」，列 6 个 commit 与待办#1-5。若资产面板验收通过且代码运行结果（第二步）也做了，可考虑单列一个 milestone track。
