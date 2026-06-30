# 修复 spec：A1 stage narration 覆盖缺口补齐（识别范式阶段 + knowledge-assistant）（2026-06-30）

> A1（`#250`，已合 dev）上线后，brainstorm A2 时核实 A1 的 narration 覆盖面 vs 即将被删的前端 `stage-broadcast.ts` 覆盖面，发现两个缺口。**本 spec 只写不实施，交别的 agent。是 A2（删 stage-broadcast）的硬前置——A2 实施 gate 在本修复合入之后。**

---

## 一、为什么需要（核实证据）

A2 计划删除前端 `stage-broadcast.ts`（前端查表翻译），改用 A1 后端 `narration`（SSOT）。删前核实两集合的覆盖面，发现 A1 有两处覆盖不到 `stage-broadcast` 原本覆盖的场景 → 删完会出现「状态空白」。

### 核实方法
对比 `stage-broadcast` 的全部 case（`i18n .../stageBroadcast.*`）vs A1 的 `_DISPATCH_STAGE` + `_INTENT_STAGES`（`stage_narration.py`）+ 派发点（`task_tool.py:emit_dispatch_enter`，仅 subagent 边界）。

### 缺口 1：「识别范式」阶段永不点亮 ⚠️（主缺口）

- A1 的 `stage_plan` 会声明 `识别范式` 是 E2E_* 的第一阶段（`_INTENT_STAGES`）。
- 但 `stage_update` **只在派遣 subagent 时发**（`task_tool.py:552`）。
- 「识别范式」阶段由 **lead 自调工具**完成（`identify_ev19_template` / `inspect_uploaded_file` / `prep_metric_plan` / HITL 反问），**不派 subagent** → 永远收不到 `active`/`completed`。
- **后果**：从用户上传 → code-executor 被派遣这段（范式识别 + 列语义 HITL + 生成指标计划，可能数十秒），删掉 `stage-broadcast` 后前端只有灰色「识别范式」待办、**无任何「在动」提示**（旧 `stage-broadcast.parseHeaders`/`resolveCatalog` 恰覆盖此段）。这正是「干等空白」焦虑源。

### 缺口 2：`knowledge-assistant` 漏登记 ⚠️

- `_DISPATCH_STAGE` 只登记 `code-executor`/`data-analyst`/`chart-maker`/`report-writer`，**无 `knowledge-assistant`** → 纯知识问答派遣它时不发 stage_update。删 `stage-broadcast` 后该场景无状态提示。

> 非缺口（澄清，不补）：`runScript`/`genericBash`/`askClarification` 等**子步骤级**文案，A1 故意不发（设计就是「上百子步骤合并进 5 大阶段」，用户已认同该颗粒度）。删它们符合设计，不补。

---

## 二、修法（守 A1 既有架构：真实事件驱动、SSOT、不靠 LLM）

### 修法 1：识别范式阶段的派发观测点（缺口 1）

「识别范式」阶段需要一个**非 subagent 的真实派发点**。候选（Step 0 坐实选哪个）：
- **(a) 工具调用边界**：在 lead 调 `identify_ev19_template`（进入识别）发 `识别范式` active；在 `prep_metric_plan` 成功（识别+计划完成、即将派 code-executor）发 `识别范式` completed。落点：这些工具的执行包装处，或一个监听这些工具名的轻量中间件。
- **(b) 复用 SealGate/中间件链**：加一个 after_model 钩子，识别到 lead 调了识别类工具 → 发对应 stage_update。
- 关键约束：**由真实工具调用触发**（不是 LLM 自报、不是前端推导），与 A1 既有的「task 边界派发」同构；`识别范式` completed 必须由「识别真的完成」（如 `prep_metric_plan` 返回 ok）触发，不撒谎（grounded）。

### 修法 2：登记 knowledge-assistant（缺口 2）

- `_DISPATCH_STAGE` 增加 `knowledge-assistant → <人话阶段名>`（如「查阅领域知识」，作为 STAGE_* 常量加入 SSOT）。
- 注意：知识问答是**非流水线意图**（A1 不发 stage_plan），所以它不是 E2E 阶段集的一员——它的 stage_update 是「独立活动提示」而非「流水线阶段」。设计上确认：knowledge-assistant 的 stage_update 在「无 stage_plan」时前端如何渲染（A2 需对应：无 plan 时把单个 stage_update 渲染成一句活动提示，而非 stepper 节点）。

### SSOT 纪律

新增的阶段名（识别范式已有 `STAGE_IDENTIFY`；知识问答需新增常量）全部进 `stage_narration.py` 一处定义，前端不复刻（守 A1 已确立的 SSOT 原则）。

---

## 三、Step 0（坐实派发点）

1. 确认「识别范式」阶段工具（`identify_ev19_template`/`prep_metric_plan`）的调用边界有确定性可挂的钩子（工具包装 / 中间件 / executor 三者之一），且能拿到 stream writer 发 custom 事件。
2. 确认 `prep_metric_plan` 成功返回是「识别阶段完成、即将派 code-executor」的可靠信号（用于发 `识别范式` completed）。
3. 产出 Step 0 小结（落 `docs/superpowers/reports/`）。

## 四、验收

1. Step 0 报告先行。
2. **TDD**：
   - 跑 E2E_FULL dogfood → 断言 custom 轨在 code-executor 被派遣**之前**就发出了 `识别范式` 的 active→completed（不再空白）。
   - 断言 `识别范式` completed 由真实信号（`prep_metric_plan` ok）触发——人为让识别失败 → 断言不发 completed（不撒谎，grounded）。
   - 知识问答场景 → 断言发 `knowledge-assistant` 对应 stage_update（无 stage_plan 的独立活动提示）。
   - **防 vacuous**：断言 narration 不含工具名（`identify_ev19_template`）/gate 关键字。
3. **裸导入两生产入口**（改中间件链/工具包装必跑）：`import app.gateway` + `make_lead_agent` 0 退出。
4. 后端全量 `make test` 绿 + lint；不回归 A1 既有 4 个测试（`test_stage_narration*`）。

## 五、不做什么

- ❌ 不补子步骤级文案（runScript/genericBash 等，故意合并进阶段，符合设计）。
- ❌ 不让前端推导识别阶段状态（守「只认后端真实事件」）。
- ❌ 不改 A1 的 5 阶段集语义（只补「识别范式」的派发点 + knowledge-assistant 登记）。
- ❌ 不在本 spec 动前端（前端渲染是 A2）。

## 六、关联

- 上游：A1（`#250`，`stage_narration.py` / `stage_narration_middleware.py` / `task_tool.py`）。
- 下游硬依赖：A2（`2026-06-30-a2-frontend-track-rendering-inline-stage-narration-design.md`）——A2 删 `stage-broadcast` 严格 gate 在本修复合入之后。
- 范式：与 A1 / PR#213 / SealGateMiddleware 同构（真实状态机驱动、确定性、不靠 LLM 自报）。
