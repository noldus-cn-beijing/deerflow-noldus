# Handoff：Playwright/CDP 从头跑 dogfood E2E，复现并坐实前端 bug（2026-06-26）

> 给下一个 agent：用 **Playwright（或裸 CDP）驱动真实 Chromium，从登录到分析完成跑一次完整 dogfood**，复现并坐实本会话已定位的一批前端 bug——它们大多需要**真实交互 + 进行中的 SSE 流**才暴露，静态/单点测不到。你的产出 = 每个 bug 的"复现确认 + 火焰图/数据证据"，回填到对应 spec 的"验证"节，并捕捉新问题。
> dev HEAD：`07838ae6`。核实代码用 `git show HEAD:`，别用工作树 grep。

---

## 〇、为什么要这个（背景）

本会话用 checkpoints 解码 + 读码定位了一批 bug，写了 spec。但有几个 bug **headless 单点测不到、必须真实 E2E 才能坐实**：
- **切回卡顿**（切到别的 tab→切回卡十几秒）——依赖真实后台 tab 节流 + 切走时 SSE 还在到达。我用 headless 模拟 visibilitychange **测不到**（静态切回 long task=0），但代码证据（SSE setMessages 零节流 + 贴底狂滚）+ 排除法把根因收紧到"后台 SSE 积压回灌"。**需真机火焰图最终坐实。**
- **重进 thread 历史乱序**（input 跑中间/输出消失）——需真实多轮历史 + 合并。
- **输入框遮挡底部**（决策卡/按钮被盖）——需真实决策卡在最底。
- **画廊只显示 1 张图 / 报告看不到 / 返回卡顿**——需真实跑完 113 图的分析。

**一次完整 E2E 能同时复现这一串**，比逐个单测高效，且能抓到 spec 没预料的新问题。

---

## 一、环境准备（关键前提，先做）

1. **Chromium 二进制已有**：`~/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome`。
2. **Playwright 包未装**（frontend `package.json` 无依赖）。三选一：
   - `cd packages/agent/frontend && pnpm add -D @playwright/test`（最顺，但改 package.json——临时测试用，别提交）。
   - 或裸 CDP：node 22 有内置 `WebSocket`，可零依赖驱动 chromium（本会话用过此法，脚本范式见 §四）。
   - 或 **chrome-devtools MCP**（缓存目录有 `mcp-chrome`，若 MCP 工具可用，优先——用户明确提到"chrome dev tool"）。
3. **dev 服务在跑**：`make dev` 起的 `next dev`(:3000) + nginx(:2026)。**用 :2026**（带后端代理）。注意用户浏览器可能用 :2027（另一实例）——以实际监听端口为准（`ss -ltnp | grep -E ":2026|:2027"`）。
4. **登录账号**（owner，本地 dev）：`qiuyang.wang@noldus.com.cn` / `19961031`（**注意 `.com.cn`**，`.com` 是另一个账号会因用户隔离看不到 thread）。登录端点 `POST /api/v1/auth/login/local`（form-urlencoded：`username`=email + `password`），token 走 HttpOnly cookie。
5. **⚠️ 测试数据需用户提供**：dogfood 用的 28 个 `Raw data-EPM-Xuhui-Trial 1-28.xlsx` **不在 repo**（demo-data 里没有）。**开跑前问用户要这 28 个 EPM XLSX 的本地路径**，否则 E2E 走不到分析。

---

## 二、E2E 剧本（从头到尾一次跑通，逐步验证）

按真实 dogfood 流程走，每步**挂上 PerformanceObserver 收集 long task** + 截图/DOM 断言：

### Step 1 — 登录 + 进 workspace
- 登录 → `/workspace`。确认无 `Failed to load memory context` 类 ERROR（gateway.log）——顺带验 **spec A memory UUID**（若未修，log 会有 `got 'UUID'`）。

### Step 2 — 上传 28 个 XLSX（验输入框堆叠 + 遮挡前置）
- 上传 28 文件 → 确认输入框**堆叠**（#8 已合，≤5 平铺、>5 堆叠 "+N"，hover 扇开）。
- 发送 → 确认消息流里附件也堆叠（#215 已合）。

### Step 3 — 走完整分析流水线（最关键，复现画廊/报告 bug）
跟 dogfood 一样回答各反问：模板选 FewZones、分组 XX=对照其余实验、列语义 open→open_arms/closed→closed_arms、要可视化、要报告。跑到 **chart-maker 生成 113 图**。
- **验画廊丢图（三现象修复 §一）**：点"打开产物画廊" → 数画廊里**实际显示几张图**。预期 bug：**只有 1 张 box**（112 张 per_subject 丢失），因 run_chart_plan 的 Command 不上行。**这是核心要复现的。**
- **验报告（三现象修复 §二）**：找 report.md 在哪呈现——预期 bug：画廊只 filter 图、报告藏侧栏 otherFiles、对话流无内嵌报告 → **用户视角看不到报告**。
- **验返回卡顿（三现象修复 §三）**：画廊点"返回对话" → 录 long task + 看滚动位置。预期 bug：`router.push` 重挂载 → **卡顿 + 丢滚动位置（定位不准）**。

### Step 4 — 输入框遮挡（spec 输入框遮挡）
- 当对话流最后是 `ask_clarification` 决策卡时 → 截图确认决策卡选项 + "画图"按钮**是否被悬浮输入框盖住**。预期 bug：被盖、点不到。

### Step 5 — 重进 thread 历史乱序（spec B 历史乱序，🔴红线）
- **跑完后刷新页面 / 重新点进该 thread** → 截图对比消息顺序。预期 bug：**input（28 上传）跑到中间 + 部分之前输出消失**。
- **拿原始数据坐实**：调 `GET /api/threads/{tid}/runs/{rid}/messages` 看后端返回顺序对不对（区分后端乱 vs 前端 mergeMessages 合并乱）。spec B §三要这个数据。

### Step 6 — 切回卡顿（最难复现，需真实条件）
- **开一个新分析任务**（或正在跑的任务）→ 任务**进行中**（SSE 还在到达）时，用 Playwright **切到另一个 tab/page**（`context.newPage()` + `page.bringToFront()` 切换，让原 page 真进后台）→ 停留 30-60s → 切回原 page。
- **录 Performance trace**（CDP `Tracing.start`/`Tracing.end`，category 含 `devtools.timeline`）覆盖"切回"瞬间 → 分析切回后主线程 long task。预期 bug：**切回瞬间一长串 long task（积压的 SSE 一次性 flush + 贴底狂滚）**。
- **若 Playwright 多 page 仍触发不了真实后台节流**（headless 可能不节流），如实记录"自动化无法复现，需真机手测"——别假装测到。本会话 headless 即栽在此。

---

## 三、产出要求

1. **每个 bug 一行结论**：复现✅/未复现/无法自动复现 + 证据（截图路径 / long task 数据 / DOM 断言 / API 响应）。
2. **回填 spec 验证节**：把复现证据写进对应 spec（三现象修复 / 输入框遮挡 / spec B / Chrome 卡顿）的"验证"节，让实施 agent 有真实 repro。
3. **Chrome 卡顿**：若录到火焰图，把"切回后主线程被什么占满"的具体函数（groupMessages? scrollToBottom? setMessages flush?）写出来——这是它还没出 spec 的最后一块（根因方向已定=后台 SSE 积压，缺火焰图坐实）。
4. **新问题**：E2E 路上任何 spec 没预料的（额外报错、布局错位、交互卡）都记下来。
5. **测试数据/脚本别提交**：临时 playwright 依赖、测试脚本、截图放 /tmp 或标注清理。

---

## 四、裸 CDP 范式（若不装 playwright，本会话用过，可复用）

node 22 内置 `WebSocket`，零依赖驱动：
```js
const { spawn } = require("child_process"); const http = require("http");
const CHROME = "~/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome";  // 展开 ~
const proc = spawn(CHROME, ["--remote-debugging-port=9333","--headless=new","--no-sandbox","--disable-gpu","about:blank"]);
// GET http://localhost:9333/json 拿 page.webSocketDebuggerUrl → new WebSocket(url)
// send {id,method:"Page.navigate"/"Runtime.evaluate"/"Tracing.start"...}
```
- 登录：`Runtime.evaluate` 里 `fetch(login, {credentials:"include"})`。
- long task：`Runtime.evaluate` 注入 `new PerformanceObserver(...).observe({entryTypes:["longtask"]})`，事后读 `window.__lt`。
- **教训**：直接改 DOM 模拟 SSE **无效**（不走 React 管线，测不到重渲染）。真实复现必须走真实 SSE（即 Step 6 开真任务），别用 DOM 注入糊弄。
- checkpoints 解码看 state：`backend/.deer-flow/checkpoints.db` + `langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer`（owner user 目录 `cd95effa-...`）。

---

## 五、待派 spec 总览（这次 E2E 服务于它们）

| spec | 文件 | E2E 验哪步 |
|---|---|---|
| 三现象修复 | `2026-06-26-artifact-bubbling-report-display-gallery-return-fix-spec.md` | Step 3 |
| 输入框遮挡 | `2026-06-26-input-box-overlaps-bottom-content-fix-spec.md` | Step 4 |
| 历史乱序（🔴红线）| `2026-06-26-rejoin-thread-history-merge-disorder-fix-spec.md` | Step 5 |
| Chrome 卡顿（未出 spec）| 待 Step 6 火焰图坐实后写 | Step 6 |
| memory UUID | `2026-06-26-memory-context-uuid-load-crash-fix-spec.md` | Step 1（log）|
| 后端质量三小项 | `2026-06-26-backend-quality-three-minor-fixes-spec.md` | 跑完看 gateway.log |
| #5 决策卡 | `2026-06-24-frontend-phase0-5-decision-card-spec.md` | （未实施，E2E 顺带看决策卡现状）|

完整待派清单 + 依赖图见 `2026-06-26-frontend-phase0-finish-and-multi-bug-fix-specs-handoff.md`。

---

## 六、关键陷阱（必读）

- **#213 产物丢失修复已合但方向错**（subagent Command 不上行）——E2E Step 3 会复现"画廊只 1 张"。正确修复=三现象 spec §一（画廊按路径从磁盘取）。别被"#213 已合"误导以为修好了。memory `feedback_subagent_command_artifacts_not_bubble_to_lead_executor_drops_state`。
- **流式红线**：Step 5/6 触及 `mergeMessages`/`setMessages`——E2E **只复现+取证，不改代码**。改它们走对应 spec、需 grill。
- **headless 可能不复现后台节流**：Step 6 若自动化测不到切回卡顿，如实记录"需真机"，别伪造数据（`feedback_code_has_fix_not_equal_bug_eliminated`：现象要真坐实）。
- **测真实数据**：用 owner `.com.cn` 账号；测试 XLSX 问用户要路径。
