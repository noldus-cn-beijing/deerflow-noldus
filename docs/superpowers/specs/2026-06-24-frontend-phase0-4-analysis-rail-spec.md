# Spec：分析进度轨 Analysis Rail（Phase 0 · 第 4 项）

> 类型：**一次性实施 spec**（前端，零后端依赖优先）
> 日期：2026-06-24
> 母方案：[docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)（硬伤 B / §3.1）
> 依赖：[spec#1 tokens/motion](2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md)（`--color-stage-*` / 曲线）、[spec#2 run-trace](2026-06-24-frontend-phase0-2-run-trace-live-spec.md)（**数据同源**：复用 `useRunTrace` 派生）
> 适用层：前端 `components/workspace/`（进度轨组件）+ 复用 spec#2 的 `useRunTrace`
> 设计准则来源：`ui-ux-pro-max`（Quick Reference §9 Navigation：`multi-step-progress`、`nav-state-active`、`breadcrumb-web`、`state-preservation`、`adaptive-navigation`）
> 一句话：在 chat 区放一条**常驻的、领域感知的 7 阶段工作流进度轨**（上传→范式→列对齐→指标→质检→解读→报告），把"线性消息流"升级成"有地图的旅程"——研究员一眼知道**走到第几步、卡在哪、为什么停**。状态全部由已有信号**前端推导**,零后端改动。

---

## 〇、为什么需要它（硬伤 B）

EthoInsight 的分析有清晰的**领域工作流**,但前端**没有任何地方呈现这条主轴**：用户看到的是一条线性消息流,subagent 卡片一个个冒出来,但"整体进度/当前位置/为何暂停"全不可见。范式识别、列对齐这些**关键决策点**淹没在消息里,研究员（非程序员）不知道"我刚才那个反问确认了什么、它锁定了哪条范式"。

`ui-ux-pro-max` §9 的 `multi-step-progress`（多步流程要有 step indicator + 允许返回导航）、`nav-state-active`（当前位置必须高亮）在这里**完全缺位**。进度轨正面补这一课。

> 与 spec#2 运行轨迹的分工：**进度轨 = 宏观 7 阶段地图**（永远 7 个节点,给"我在哪"）;**运行轨迹 = 微观每步流水**（几十个事件,给"具体干了啥"）。两者**数据同源**（都从 `useRunTrace`/subtask 派生），不同抽象层。进度轨是常驻轻量条,轨迹是按需抽屉。

---

## 一、7 阶段定义（领域 SSOT）

| # | 阶段 | 完成信号（前端可推导） | 阶段色 token |
|---|---|---|---|
| ① | 上传 | thread 有 `uploaded_files` / human message 带 files | `--color-stage-upload` |
| ② | 范式识别 | `set_experiment_paradigm` tool_call 出现（`message-group.tsx:496` 已渲染该工具） | `--color-stage-paradigm` |
| ③ | 列语义对齐 | `ask_clarification` 且 `clarification_type` 属列对齐类 → 该阶段"等待中"；用户回答后 → done | `--color-stage-align` |
| ④ | 指标计算 | `prep_metric_plan` / `run_metric_plan` tool_call 或 code-executor subagent 派遣；artifacts 出现 .json/.png | `--color-stage-compute` |
| ⑤ | 数据质检 | gate_signals / `extractQualityWarnings` 出现（critical/warning/ok） | `--color-stage-qc` |
| ⑥ | 统计解读 | data-analyst subagent 派遣/完成 | `--color-stage-interpret` |
| ⑦ | 报告 | report-writer subagent 派遣/完成；artifacts 出现 report.md | `--color-stage-report` |

**阶段状态机**：`pending`（灰）→ `active`（品牌绿高亮 + 细描边呼吸）→ `waiting`（琥珀脉冲,等 HITL）→ `done`（✓）→ `warning`（黄,该阶段有非阻断质检警告）→ `failed`（红,subagent failed / gate 阻断）。

> ⚠️ 这 7 阶段定义是 SSOT——进度轨、决策卡（spec#5）、画廊若都要标阶段,**引同一份定义**（放 `core/workflow/stages.ts`），不在各组件重复枚举（CLAUDE.md「同一份知识绝不双存」+ memory `feedback_single_source_of_truth`）。

---

## 二、目标与非目标

### 目标
1. 一条**常驻 7 阶段进度轨**,反映当前 thread 的分析进度。
2. 每阶段：图标 + 名称 + 状态色（pending/active/waiting/done/warning/failed）。
3. **当前阶段高亮**（`nav-state-active`）——品牌绿 + 细描边呼吸（克制,非大色块）。
4. **等待决策态**：HITL 阶段琥珀脉冲 + "等你确认"微标,与 spec#5 决策卡联动。
5. **点阶段 → 锚点滚动**到消息流对应位置（`breadcrumb-web` 思想,`state-preservation`）。
6. 状态全部**前端推导**（复用 spec#2 `useRunTrace`），零后端改动。

### 非目标
- ❌ 不做"阶段可点击跳转执行"（进度轨是**只读地图 + 滚动锚点**,不是控制器;回退/改主意走 spec#5 决策卡）。
- ❌ 不加后端 `workflow_stage` 字段（**先前端推导**;若推导长期不稳,再考虑后端单一字段——记为后续,非本 spec）。
- ❌ 不动流式核心 / `groupMessages`。
- ❌ 不替换 spec#2 运行轨迹（两者并存,数据同源）。

---

## 三、设计

### 3.1 数据层：`useWorkflowStages`（复用 spec#2 派生）

新增 `core/workflow/use-workflow-stages.ts`——**从 spec#2 的 `useRunTrace`（或同源 subtask + messages）派生 7 阶段状态**,纯只读 `useMemo`：

```
输入: TraceEvent[] / subtasks / thread.messages（全部已有）
输出: StageState[7] = [{ id, status, anchorMessageId? }]
推导规则: §一表的"完成信号"映射。例:
  - 见 set_experiment_paradigm tool_call → ②done
  - 见列对齐类 ask_clarification 未答 → ③waiting；已答 → ③done
  - 见 data-analyst dispatch → ⑥active；completed → ⑥done
  - gate critical+blocks_downstream → ⑤failed；warning → ⑤warning
anchorMessageId: 该阶段首个相关 message 的 id（供点击滚动定位）
```

- **不重写解析**：复用 spec#2 已聚合的 `TraceEvent[]`——进度轨是 trace 的"降维投影"（把几十个事件归并到 7 桶）。spec#2 先落,本 spec 复用其 hook。
- **clarification_type 判列对齐**：`ask_clarification` 的 `clarification_type`（`clarification_tool.py` 的 Literal）+ question 内容判断是否属"列语义对齐"。若类型粒度不够,用 question/options 关键词兜底（前端启发,不改后端）。

> 工程纪律：纯派生只读,不写 state、不碰 submit/merge。与 spec#2 同源——**别另起一套解析**（否则两处算法漂移,memory `feedback_handoff_metrics_field_divergence` 类教训）。

### 3.2 视图层：进度轨（日式克制）

**形态**（母方案决策2 已同意做,横/纵此处给推荐）：

- **推荐：顶部横向 stepper**（chat 区顶部,sticky）。理由：占垂直空间小、移动友好（`adaptive-navigation`：窄屏更需省空间）、与现有 workspace header 视觉延续。
- 窄屏（<768px）：横向 stepper 退化为"当前阶段 + N/7"紧凑指示 + 点开看全部（`content-priority`）。

```
顶部横向（宽屏）:
①上传 ──── ②范式 ──── ③列对齐 ──── ④指标 ──── ⑤质检 ──── ⑥解读 ──── ⑦报告
 ✓         ✓          ◖等你确认     ○           ○          ○         ○
                      ↑ 琥珀脉冲，当前焦点

窄屏:
[③ 列对齐 · 等你确认]  3/7  ▾
```

**视觉细节（日式 + spec1 token）**：
- 连接线**极细**（1px),不是粗箭头。
- 节点小圆点,done=✓ + 阶段色实心、active=品牌绿空心 + **细描边呼吸**（`animate-pulse-soft` 已有）、waiting=琥珀脉冲、pending=灰描边、failed=红✗、warning=黄。
- **当前阶段**唯一视觉主角（`primary-action`/`visual-hierarchy`）：其余降为背景层（低对比）。
- 阶段名 hover/点击显 tooltip（该阶段在做什么 + 若 done 显结果摘要）。
- 状态"色 + 图标 + 文字"三件套（`color-not-only`）——色盲下靠图标/文字也能读。
- 阶段切换（pending→active→done）用 spec1 `--ease-brand-out` 过渡,不 snap（`state-transition`）。

### 3.3 交互：锚点滚动（`breadcrumb-web` / `state-preservation`）

- 点某阶段 → 消息流平滑滚动到该阶段 `anchorMessageId` 对应消息（`scroll-behavior: smooth`,reduced-motion 下 auto）。
- 滚动不改变流式状态、不"跳走"——只是定位（`back-stack-integrity`：不重置）。
- 当前 active 阶段在轨上**自动居中/可见**（流式推进时轨自动跟随当前阶段）。

### 3.4 与决策卡（spec#5）联动
- HITL 阶段（③列对齐 / ②范式确认）进入 `waiting` 时,进度轨该节点琥珀脉冲 + spec#5 的决策卡在消息流里同时显眼。两者**同一 waiting 信号**驱动（`useWorkflowStages` 输出 + 决策卡渲染同源）。
- 点 waiting 节点 → 滚动到对应决策卡（让用户快速回到"要回答的地方"）。

---

## 四、实施步骤

### Step 1：阶段定义 SSOT（`core/workflow/stages.ts`）
7 阶段的 id / 名称 key / 图标 / 阶段色 token / 完成信号判定函数。单一来源。

### Step 2：`useWorkflowStages` 派生 hook
从 spec#2 `useRunTrace`（或同源）派生 `StageState[7]`。单测：喂典型消息序列（含范式 + 列对齐反问 + data-analyst + gate warning）→ 断言阶段状态/anchor 正确。

### Step 3：`AnalysisRail` 组件 + `StageNode`
顶部横向 stepper（§3.2）+ 窄屏紧凑态。复用 spec1 token + `animate-pulse-soft`。

### Step 4：挂载 + 锚点滚动
- 挂在 chat 区顶部（`workspace-container` / chat 页 layout），sticky。
- 点阶段 → `scrollIntoView` 到 anchorMessageId（smooth / reduced-motion auto）。
- 确认与 artifacts 侧栏、spec#2 轨迹抽屉、输入框的垂直空间不打架（sticky 顶部 + 内容区 padding 让位）。

### Step 5：i18n + a11y + reduced-motion
- 阶段名/状态/tooltip 进 i18n（不硬编码中文）。
- 键盘：阶段节点可 tab + Enter 触发滚动;`aria-current` 标当前阶段;`nav` 语义 + aria-label。
- reduced-motion：脉动降级为静态描边（spec1 机制）。

---

## 五、验收标准

### 功能
- [ ] 7 阶段进度轨常驻 chat 区顶部,反映当前 thread 进度。
- [ ] 阶段状态正确推导：上传/范式/列对齐/指标/质检/解读/报告,随分析推进实时更新。
- [ ] 当前阶段高亮（品牌绿 + 描边呼吸）;HITL 等待阶段琥珀脉冲。
- [ ] 点阶段 → 平滑滚动到消息流对应位置。
- [ ] gate 警告 → ⑤质检显 warning;阻断/失败 → 显 failed（红）。
- [ ] 窄屏退化为紧凑"当前阶段 + N/7"。
- [ ] 与 spec#2 运行轨迹**数据同源、并存不冲突**。

### a11y / 性能
- [ ] `aria-current` 标当前;节点键盘可 tab + Enter 滚动;`nav` 语义。
- [ ] 状态"色 + 图标 + 文字"三件套（`color-not-only`）。
- [ ] 阶段过渡用 spec1 曲线;reduced-motion 降级（脉动→静态）。
- [ ] sticky 轨不遮挡内容（内容区让位 padding）。

### 工程纪律
- [ ] `useWorkflowStages` 纯派生只读,与 spec#2 同源,**不另起解析**。
- [ ] 阶段定义 SSOT 单一来源（`core/workflow/stages.ts`），决策卡/画廊若标阶段引同一份。
- [ ] 零后端改动（前端推导）。
- [ ] 不动流式核心 / `groupMessages`。
- [ ] `pnpm check` + 单测 + i18n 不硬编码中文。

---

## 六、风险与回退

| 风险 | 缓解 |
|---|---|
| 阶段推导不准（信号歧义，如列对齐 vs 其他反问） | clarification_type + question 关键词双判;不准时该阶段保守显 active 不显 done（宁可"还在进行"不可"假完成"）;长期不稳再上后端 `workflow_stage` 字段（记后续） |
| 与 spec#2 解析漂移 | 强制同源（复用 `useRunTrace`）,不另写;单测锁定 |
| 进度轨占垂直空间挤压聊天 | sticky + 窄屏紧凑态 + 高度克制（一行）|
| 非线性流程（用户中途改范式，阶段回退） | 状态机允许 done→active 回退（重新派生即可,无副作用） |

**回退**：纯新增组件 + 只读 hook;移除挂载即隐藏,`git revert` 零风险。

---

## 七、给实施 agent 的交接

- **依赖 spec#2 先落**（复用 `useRunTrace`）。若 spec#2 未落,可临时直接从 messages/subtasks 派生,但**接口对齐 spec#2**,待 spec#2 落后切到同源。
- 新增：`core/workflow/{stages.ts, use-workflow-stages.ts}`（+单测）、`components/workspace/{analysis-rail, stage-node}.tsx`、i18n、挂载点。
- **不碰**：流式核心、`groupMessages`、消息流渲染。
- 复用 spec1 `--color-stage-*` + 曲线 + `animate-pulse-soft`（不重定义）。
- 与 spec#5 决策卡**同一 waiting 信号**联动——实施 #5 时复用本 hook 的 waiting 输出。
- 横/纵最终形态实施期定（推荐横向 sticky）;窄屏紧凑态必做。

---

*依据：母方案硬伤 B / §3.1 + 决策2（进度轨做）+ `ui-ux-pro-max` §9（multi-step-progress / nav-state-active / breadcrumb-web / state-preservation / adaptive-navigation）。数据同源 spec#2。未写代码。*
