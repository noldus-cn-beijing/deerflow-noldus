# 设计 spec：生成式 UX 演进路线图 + A1 后端事件分轨地基（2026-06-30）

> 本文档分两部分：**第一部分**是生成式 UX 演进的总体分层路线图（北极星愿景 + 分步骤排序），用于立项与排序；**第二部分**是路线图第一步 **A1（后端事件分轨地基 + 翻译 middleware）** 的可实施详细设计。A1 是本 spec 的可实施单元；A2/C 系列各自后续单独 brainstorm→spec。
>
> 来源：2026-06-30 dogfood 暴露「输出过载」（14 个 thinking 块 + 12 次工具名 + gate signal 直喷用户）→ 用户提出输出理念重构 → brainstorm 收敛。配套调研报告（ag-ui 评估 + 翻译层最佳实践）见会话记录。

---

# 第一部分：总体路线图（生成式 UX 演进）

## 北极星愿景

从「消息流喷文字」演进为「Agent 驱动的生成式 UI」：界面按需生成/销毁、信息按性质分轨呈现、文件系统成一等公民、用户可对产物追问。**愿景现在启动，分步骤落地，不一次到位。**

## 贯穿全程的架构原则（硬约束）

1. **infra 全用 DeerFlow 已有的，不造轮子**：`astream` 多轨、`ThinkTagMiddleware`、`custom` 轨 + `get_stream_writer()`、中间件链都是上游现成的（已勘察坐实，见第二部分模块 1）。（守 `feedback_sync_full_follow_upstream_infra` + `feedback_harnessx_ideas_on_deerflow_not_harnessx_mechanisms`）
2. **翻译层 = 后端 middleware，由真实状态机驱动**：人话阶段叙事由真实节点/派遣边界事件触发，**不靠 LLM 自报、不靠 prompt 约束**。这是唯一能同时治住「漂移」（工具改了翻译没跟上）和「失真」（说做完其实没做，Air Canada 翻车）两大风险的位置，与已验证的 PR#213（run_chart_plan 确定性登记 artifacts）/ `SealGateMiddleware` 同构。（守「确定性门 > prompt 打地鼠」铁律）
3. **不引入 ag-ui 协议**：我们前后端自研且已打通，ag-ui 最大价值（跨框架互通）对我们≈0，引入=替换现有 SSE 协议的净负担。**只借鉴其「事件分轨」设计语汇**（高层叙事 / 工具轨 / 状态轨分离）。

## 分步骤（按依赖排序）

| 步 | 子项目 | 性质 | 依赖 |
|---|---|---|---|
| **A1** | 后端事件分轨地基 + 翻译 middleware（人话阶段叙事→custom 轨；按需 plan 事件） | 承重墙 | 无 |
| **A2** | 前端分轨渲染（messages 正文 / reasoning 折叠 / custom→stepper+心跳 / values→产物卡） | 呈现层 | A1 |
| **C1** | 画廊纳入 code-executor raw 数据产物 | 文件系统 | 可并行 |
| **C2** | 画廊布局 + UI/UX 优化（ui-ux-pro-max + 日式简洁风格） | 文件系统 | C1 |
| **C3** | 产物框选→喂回输入框追问（Gemini 式数据追问） | 文件系统×对话 | A2+C2 |
| **(愿景层)** | 双向状态同步可冲突调和、能力边界沙盒渲染 | v1.0+ | 全部 |

**本 spec 只完整设计 A1。** A2/C1-3 在此登记排序，各自后续单独 brainstorm→spec→实施。愿景层归 v1.0+，现在不实现（守 `feedback_version_boundary_v01_insight_v10_experiment_harness` 的分界纪律）。

---

# 第二部分：A1 详细设计（可实施单元）

## 目标

后端在 `custom` 轨上，由真实流水线事件确定性地发出「人话阶段叙事 + 按需阶段计划」，使前端（A2）能据此渲染进度 stepper + 心跳，而**工具名 / 子 agent 名 / gate signal 永不进入面向用户的轨道**。

## 模块 1：复用的 DeerFlow 地基（不动，只用）

| 已有设施 | 位置 | 用途 |
|---|---|---|
| `astream(stream_mode=[...])` 多轨 | `runtime/runs/worker.py:280-321`（调用方按需请求，非 default；当前已请求 `messages/custom/updates/values`） | 多轨传输 |
| `custom` 轨 + LangGraph `get_stream_writer()` | LangGraph 原生 | **我们写人话叙事的通道** |
| `ThinkTagMiddleware` | `agents/middlewares/think_tag_middleware.py`（after_model，正则剥 `<think>`→`additional_kwargs.reasoning_content`） | thinking 已分轨到 reasoning，**不动** |
| 中间件链 | lead `agent.py` | 新 middleware 挂这里 |

**A1 不改任何上游 stream 配置、不改 ThinkTagMiddleware、不改 worker。** 仅新增一个 Noldus 中间件 + 复用 custom 轨。

> ⚠️ 实施前必读：custom 轨上 `get_stream_writer()` 的写入是否会被 worker 正确 serialize 并以 `custom` SSE 事件名推送，需在 Step 0 用真实运行坐实（worker.py 的 `_lg_mode_to_sse_event` 对 custom 的处理 + serialize 路径）。这是 A1 的地基假设，必须先验证。

## 模块 2：新增 `StageNarrationMiddleware`（Noldus 定制，承重墙）

挂在 lead 中间件链上的新中间件，职责单一：**在真实流水线节点/派遣边界，确定性地往 custom 轨写「人话阶段事件」**。

- **触发源 = 真实代码事件，不是 LLM**：
  - 意图分类确定时（lead 完成 `E2E_FULL`/`E2E_MIN`/`E2E_FULL_ASKVIZ`/知识问答 判定）→ 发 `stage_plan`（或对非流水线意图**不发**）。
  - lead 派遣某 subagent（`task` 工具调用边界，进/出 code-executor/data-analyst/chart-maker/report-writer）→ 发对应 `stage_update`（active/completed）。
- **落点候选**（Step 0 坐实选哪个）：① 独立 middleware 监听派遣事件；② 复用已有派遣观测点（`subagents/executor.py` 边界 / `task_tool`）。优先不重复造观测——若 executor/task_tool 已有可挂的边界钩子，复用之。
- **不靠 prompt、不靠 LLM 自报**（防失真）。

## 模块 3：两类 custom 事件（payload 设计，借鉴 ag-ui 分轨语汇）

```jsonc
// ① stage_plan —— 仅多阶段流水线意图时发一次（意图分类确定后）
{ "kind": "stage_plan",
  "stages": ["识别范式", "计算指标", "数据解读", "生成图表", "撰写报告"],
  "skipped": ["数据解读"] }   // n=1 等情况标注被跳过的阶段

// ② stage_update —— 每个阶段进/出各发一次
{ "kind": "stage_update",
  "stage": "生成图表",
  "status": "active" | "completed",
  "narration": "正在为 28 个文件生成 113 张图表…" }   // 人话，可空
```

- **`stage_plan` 由「意图分类 → 阶段集」的确定性映射触发**（映射是固定表，不靠 LLM）：
  - `E2E_FULL` → 识别/计算/解读/画图/报告（5）
  - `E2E_FULL_ASKVIZ` → 识别/计算/解读/(询问)，用户答后按 viz_choice 追加 画图/报告
  - `E2E_MIN` → 识别/计算（2）
  - n=1 → 上述基础上 `skipped: ["数据解读"]`
  - **知识问答 / 闲聊 / 单步追问 → 不发 stage_plan**（前端无 stepper，纯流式对话）
- **`narration` 是人话**；工具名、子 agent 名、gate signal **永不写入这两类事件**。
- 阶段名是 SSOT：定义一处（建议与「意图→阶段映射」同处），前端不维护第二份阶段字典（防漂移，守 single-source-of-truth）。

## 模块 4：边界与不做什么

- ❌ 不改 `messages` 轨——LLM 正文照常逐 token 流式（保「还在跑」的心跳感）。A1 只**新增** custom 叙事，不动现有正文流、不动 reasoning 轨。
- ❌ 不引入 ag-ui 协议 / CopilotKit。
- ❌ 不在 A1 动前端（前端分轨渲染是 A2）。
- ❌ 工具名/gate signal 不是「翻译成人话再发」，而是**根本不往面向用户的轨写**——它们留在 reasoning/内部轨，A2 决定默认不渲染。
- ❌ 不把 `stage_plan` 写死成固定 5 步——它是「意图→阶段集」动态映射的产物，非流水线意图不发。

## Step 0（不可跳）：地基假设坐实

实施前先验两条地基假设，错了则整个设计要调：
1. **custom 轨可达性**：在 lead 运行中调 `get_stream_writer()({...})`，确认 worker 正确 serialize 并以 `custom` SSE 事件名推送到客户端（curl SSE 或 dump 验证）。
2. **派遣边界可观测性**：确认 lead 派遣 subagent 的进/出在代码层有确定性可挂的钩子（executor/task_tool/middleware 三者之一），且能拿到 subagent 类型 + 意图模式。

产出 Step 0 小结（落 `docs/superpowers/reports/`），明确两假设成立 + 选定的 middleware 落点。

## 验收（A1 独立可测，不耦合前端）

1. **Step 0 坐实报告**先行。
2. **TDD**（CLAUDE.md 强制）：
   - 跑一次 `E2E_FULL` dogfood，断言 custom 轨事件序列含：1 个 `stage_plan`（5 阶段）+ 每阶段 active/completed 的 `stage_update`，顺序正确。
   - 断言 `E2E_MIN` / 知识问答意图**不发 stage_plan**（前端据此无 stepper）。
   - 断言 n=1 场景 `stage_plan.skipped` 含「数据解读」。
   - **防 vacuous**：断言这些事件的 `narration` 中**不含**工具名（如 `identify_ev19_template`）、子 agent 名（`code-executor`）、gate 关键字（`gate_signals`）——即「内脏不泄漏」是被测断言，不是口头约定。
3. **裸导入两生产入口**（改了中间件链必跑，守闭环铁律）：`python -c "import app.gateway"` + `make_lead_agent` 0 退出。
4. **由真实事件驱动验证**：人为让某 subagent 失败 → 断言对应 stage_update 不报 completed（叙事不撒谎，grounded）。
5. 后端全量 `make test` 绿 + lint。

## 不做什么（A1 范围外）

- ❌ A2 前端渲染（stepper/心跳/折叠）——下一步单独立项。
- ❌ C 系列画廊/文件系统——并行/后续轨道。
- ❌ 愿景层（双向状态调和、沙盒渲染）——v1.0+。
- ❌ 不顺手重构现有 messages/reasoning 轨。

---

## 关联

- 配套调研：ag-ui 协议评估（结论：不引入，借鉴分轨语汇）+ 翻译层最佳实践（结论：后端中间件承重墙）。
- 同源范式：PR#213（run_chart_plan 确定性登记 artifacts）、`SealGateMiddleware`、`DegradationCircuitBreakerMiddleware`——A1 中间件与它们同构（挂中间件链、真实信号触发、确定性产出）。
- 后续子项目：A2（前端分轨渲染）、C1（画廊纳 code-executor 产物）、C2（画廊 UI/UX）、C3（产物框选追问）。
