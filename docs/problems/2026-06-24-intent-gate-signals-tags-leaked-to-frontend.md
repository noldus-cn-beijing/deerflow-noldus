# 2026-06-24 问题：lead/data-analyst 的 `[intent]`/`[gate_signals]` 机器控制文法直接泄露到前端对话

> **目的**：把端到端 dogfood（thread `590fbbd3`）暴露的**前端 UX 泄露**问题沉淀为独立诊断材料。
> 本次**只取证 + 定位根因，未改任何代码**（用户未授权修）。
>
> **性质**：🟠 中 · **产品体验 / 专业形象隐患**（非功能性 bug，不影响分析正确性，但直接破坏面向行为学研究员的产品观感）。这是用户主动发现、本次 dogfood 实证确认的真问题。
>
> **关联 thread**：`590fbbd3-9f87-48e5-aca2-8c5d053013e7`（EPM-Xuhui-28）
> **关联 run dir**：`/tmp/noldus-e2e-runs/20260624-161811/`（`final-body.txt` 是前端实际渲染的对话全文，`sse-*.txt` 是逐 turn 流）

---

## 0. 一句话结论

lead agent 和 data-analyst 的 prompt 明确要求它们在 AIMessage **正文**里输出结构化控制信号（`[intent] E2E_FULL` 给 `IntentClassificationGuardrailProvider` 校验、`[gate_signals] ...` 给 lead 做下游决策），**但前端的 tag strip 层（`src/core/messages/utils.ts`）只认尖括号 tag（`<uploaded_files>` 等 4 种），完全不认方括号 tag**。于是 `[intent] E2E_FULL`、`[gate_signals] constitution_acknowledged: true data_quality: critical_count: 0 ...` 这种**机器控制文法被原样渲染到对话气泡里给研究员看**。

**用户的原话观察**：「如果我们直接在前端写出 `[intent] [e2e]` 是不是不太合适？」——**答案是不合适，而且这是真泄露，不是偶发**。本次 dogfood 的 final-body 里 `[intent]` 出现 9+ 次、`[gate_signals]` 在最后两个 turn（chart-maker / report-writer 阶段）分别出现 56、39 次。对行为学研究员而言，这些是毫无意义、且暗示「这工具内部很乱」的噪音。

---

## 1. 上下文（30 秒）

- **EthoInsight**：面向**行为学研究员**（非工程师）的 AI 分析助手。用户是领域专家，对「对话里冒出 `constitution_acknowledged: true`」这类内部状态机字段毫无预期，且会产生「这个工具不专业 / 在对我念代码」的观感。
- **控制信号设计**：lead agent 用 `[intent] <INTENT>` 行做意图分类（被 `IntentClassificationGuardrailProvider` 确定性校验，见 [lead_agent/agent.py:402](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py) W17）；各 subagent（data-analyst / chart-maker / report-writer）用 `[gate_signals] ...` 块给 lead 传递结构化决策信号（见 [data_analyst.py:235](../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py)）。**这些是确定性的控制协议，设计本身是对的**——问题在于它们被写进了「会渲染给用户看的 AIMessage 正文」，而不是独立的 metadata 通道。
- **前端 strip 机制**：`src/core/messages/utils.ts` 有 `INTERNAL_MARKER_TAGS`（尖括号 tag 白名单）+ `stripUploadedFilesTag` + `isHiddenFromUIMessage`（按 `additional_kwargs.hide_from_ui` 过滤）。**这套机制是成熟的**，但只覆盖了尖括号 tag 和 hide_from_ui 消息，**没覆盖方括号控制信号**。

---

## 2. 用户做了什么 / 怎么发现的

用 `/noldus-insight-e2e` skill 对 `Raw data-EPM-Xuhui-28` 跑端到端 dogfood。跑完后用户 review 前端对话，主动提问：

> 「如果我们直接在前端写出 `[intent] [e2e]` 是不是不太合适？」

随后取证确认这是系统性泄露。

---

## 3. 直接观察到的症状（事实）

### 症状 1：`[intent]` 直接出现在对话气泡里，夹在自然语言中间

`final-body.txt`（前端实际渲染的对话全文），第 136 行附近——lead 的回复气泡：

```
请分析这份高架十字迷宫(EPM)实验数据，按 Treatment 分组做组间比较，给出统计结论、效应量和发表级图表。

查看其他 1 个步骤
使用 "identify_ev19_template" 工具

[intent] E2E_FULL            ← ← ← 这一行原样渲染给用户

好的，我看到你上传了28个EPM实验数据文件。让我先识别数据结构和模板
```

**`[intent] E2E_FULL` 独占一行，就夹在「使用工具」步骤条和自然语言回复之间**。研究员看到的是一句没头没尾的 `[intent] E2E_FULL`。

### 症状 2：`[gate_signals]` 块整段泄露，含纯内部字段

`final-body.txt` 第 432 行附近——code-executor 完成后 lead 的转述气泡：

```
run_metric_plan 全部完成，无任何失败。输出最终消息：
OK: handoff written
/mnt/user-data/workspace/handoff_code_executor.json 已由 run_metric_plan 落盘（sealed_by="run_plan"）。

[gate_signals] constitution_acknowledged: true data_quality: critical_count: 0 warning_count: 0 critical_items: [] statistical_validity: ok errors_count: 0
                                  ↑
                                  这一整行内部状态机字段全部渲染给用户
```

`constitution_acknowledged`、`critical_items: []`、`sealed_by="run_plan"` 这些是**纯内部协议字段**，对研究员零信息量，且 `sealed_by` / `handoff` / `.json` 路径会暴露「这是个在跑 JSON handoff 的脚本系统」的工程细节。

### 症状 3：泄露是系统性的、逐 turn 递增，不是偶发

逐 SSE 流统计（`final-body.txt` / `sse-*.txt`）：

| stream | `[intent]` 次数 | `[gate_signals]` 次数 | 对应阶段 |
|---|---|---|---|
| sse-0 | 9 | 0 | 上传 + Gate 1 反问 |
| sse-1 | 7 | 0 | Gate 1（template/column/groups）|
| sse-2 | 7 | 0 | Gate 1 续 |
| sse-3 | 7 | 0 | code-executor 派遣 |
| sse-4 | 13 | 0 | code-executor + data-analyst |
| sse-5 | 10 | 0 | data-analyst |
| sse-6 | 68 | 56 | **chart-maker（两次 max_turns）** |
| sse-7 | 28 | 39 | report-writer |

**结论**：`[intent]` 几乎每个 turn 都泄露；`[gate_signals]` 在后半程（subagent 密集 handoff 阶段）爆炸性增长。chart-maker 那两个 turn（已是本次 dogfood 的重灾区，见 [silent-drop problem doc](2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md)）单 turn 就有 68 个 `[intent]` + 56 个 `[gate_signals]`——**研究员在那个阶段看到的对话几乎一半是机器文法**。

### 症状 4：前端 strip 层完全不认方括号 tag

`src/core/messages/utils.ts:519-529`：
```ts
export const INTERNAL_MARKER_TAGS = [
  "uploaded_files",
  "system-reminder",
  "memory",
  "current_date",
] as const;

const INTERNAL_MARKER_RE = new RegExp(
  `<(${INTERNAL_MARKER_TAGS.join("|")})>[\\s\\S]*?</\\1>`,   // ← 尖括号正则 <(...)>
  "g",
);
```
- 白名单只有 4 种，全是尖括号 tag。
- 正则是 `<(...)>`，**结构上不可能匹配 `[intent]`/`[gate_signals]`**。
- `isHiddenFromUIMessage`（:472）按 `additional_kwargs.hide_from_ui === true` 过滤——但 `[intent]`/`[gate_signals]` 写在**普通 AIMessage 正文**里，**没有** hide_from_ui 标记，所以也逃过了这道关。
- grep 全前端（`src/`）**找不到任何方括号 tag 的 strip / parse 逻辑**（无 `stripTag`/`tagRegex`/`\[intent\]` 匹配）。

---

## 4. 根因分析

### 根因 A（主因）：控制信号协议选错了「传输层」——用「渲染正文」当「信令通道」

`[intent]` 和 `[gate_signals]` 是**确定性的控制协议**（一个被 guardrail 校验、一个给 lead 决策），它们的设计目标是「机器读」，不是「人读」。但 prompt 让 agent 把它们写进 **AIMessage.content 正文**——而 `.content` 正是前端默认渲染给用户看的字段。

**这是协议设计层面的 channel 混用**：本该走独立 metadata 通道（`additional_kwargs`、tool call 参数、或 hide_from_ui 消息）的信号，走了「用户可见正文」通道。一旦走正文，就完全依赖「前端事后 strip」来兜底——而前端 strip 层（根因 B）又没覆盖方括号语法。

**对照**：`<uploaded_files>`、`<system-reminder>`、`<memory>` 这些同样是机器信号，但它们 (1) 用尖括号（被 `INTERNAL_MARKER_RE` 匹配）+ (2) 多数走 hide_from_ui 消息（被 `isHiddenFromUIMessage` 过滤）。**`[intent]`/`[gate_signals]` 两条防线都没沾上**——既不是尖括号，也不在 hide_from_ui。

### 根因 B（直接原因）：前端 strip 白名单 + 正则只覆盖尖括号 tag，未跟上 prompt 的方括号协议

`INTERNAL_MARKER_TAGS` 是个**显式白名单**，需要手工维护。当后端 prompt 新增一种控制信号语法（方括号 `[intent]`/`[gate_signals]`）时，**前端白名单没有同步更新**——典型的「前后端协议漂移」（CLAUDE.md HarnessX「灾难性遗忘」病理的同族：改了一处共享协议，漏同步镜像）。

`utils.ts:531-539` 的注释自己承认了这个风险：「defence-in-depth strip for any message that … slips through without its `hide_from_ui` flag set」——设计者**预见到了**会有信号漏过 hide_from_ui，做了 strip 兜底，**但兜底白名单没把方括号信号加进去**。

### 根因 C（放大器）：subagent 越多、handoff 越密，泄露越严重

`[gate_signals]` 是 subagent 给 lead 的 handoff 信号。本次 dogfood 后半程（chart-maker 两次重派 + report-writer）subagent 密集 handoff，`[gate_signals]` 爆炸性增长（sse-6/7 合计 95 次）。**分析越复杂、pipeline 越长，泄露越严重**——这正好是最需要给研究员「专业感」的复杂场景，却是噪音最密集的场景。

---

## 5. 修复方向（待用户拍板，未动手）

| 候选 | 改动位置 | 治什么 | 风险 |
|---|---|---|---|
| **(F1-frontend)** 前端 strip 层加方括号控制信号：`INTERNAL_MARKER_TAGS` 旁新增 `INTERNAL_BRACKET_SIGNALS = ["intent", "gate_signals", ...]` + 对应正则 `^\s*\[(intent|gate_signals)\][^\n]*$`（行级 strip，单行 `[intent] E2E_FULL` 整行删；`[gate_signals] ...` 整行删）。在 `extractContentFromMessage` / 渲染入口调用 | `src/core/messages/utils.ts` | **治标 + 最快见效**：纯前端，不改后端协议，立即消除用户可见泄露 | 低——需确认没有合法用户输入会以 `[intent]`/`[gate_signals]` 开头（极不可能）；守回归：加 utils.test.ts 用例 |
| **(F2-backend-protocol)** 控制信号改走独立通道：`[intent]`/`[gate_signals]` 从 AIMessage.content 移到 `additional_kwargs`（如 `additional_kwargs.control_signals = {intent: "E2E_FULL", gate_signals: {...}}`），由 lead/中间件在写 message 前提取。前端从 `additional_kwargs` 读（用于 guardrail/决策），不渲染 | lead_agent + 各 subagent prompt + 提取中间件 | **治本根因 A**：从协议层消除「控制信号混入渲染正文」 | 中-高——改 prompt（守 catastrophic forgetting：grep 所有 `[intent]`/`[gate_signals]` 消费者，含 IntentClassificationGuardrailProvider）+ 改 message schema；需全量回归 |
| **(F3-hide_from_ui)** 让携带控制信号的 AIMessage 段落走 hide_from_ui：agent 输出控制信号时，中间件把 `[intent]`/`[gate_signals]` 那几行抽进一条 hide_from_ui 消息，正文只留自然语言 | 新增/复用中间件（类似 ThinkTagMiddleware 抽 `<think>`） | **治本 + 复用现有机制**：`isHiddenFromUIMessage` 已存在，复用其过滤；与 `<think>` 抽取同构 | 中——需保证 guardrail 仍能在 hide_from_ui 前读到信号（提取时机要在 guardrail 之后）|
| **(F4-prompt)** prompt 改用既有的尖括号 tag 语法：`[intent]` → `<intent>E2E_FULL</intent>`、`[gate_signals] ...` → `<gate_signals>...</gate_signals>`，直接进 `INTERNAL_MARKER_TAGS` 白名单 | lead_agent/prompt.py + subagent prompts | **最小协议改动复用现有 strip**：方括号改尖括号即被现有正则匹配 | 低-中——改所有消费者（guardrail 正则、lead 解析逻辑）；守 grep 全量 |

**我的偏好**：**F1（前端立即止血）+ F4（协议对齐到既有尖括号范式）组合优先**。F1 是 5 分钟的前端改动，立即消除用户可见泄露（止血）；F4 是把方括号协议统一到既有尖括号范式（治本 + 复用 `INTERNAL_MARKER_TAGS`，零新机制）。**两者正交**：F1 先上线保体验，F4 再做协议收敛。F2 是最干净的架构解但改动最大，可作为 v0.1 后的长期方向；F3 是 F2 的轻量折中。

**⚠️ HarnessX 三大病理自检**（CLAUDE.md §病理）：
- **Reward hacking**：不涉及（非验收伪造问题）。
- **Catastrophic forgetting**：**F2/F3/F4 改控制信号传输层，必须 grep 所有消费者**（`IntentClassificationGuardrailProvider` 读 `[intent]`、lead prompt 读 `[gate_signals]` 做下游决策、可能的训练数据录制 `TrainingDataMiddleware` 录了这些信号）——漏一个就回归。F1（纯前端 strip）风险最低，**应先做**。
- **Under-exploration**：**不要**用「改 prompt 让 agent 别输出这些信号」修——这些信号是确定性控制协议（guardrail 校验依赖 `[intent]`），不能不输出。问题在「输出后没在前端过滤」，不在「输出本身」。守判据：前端展示层的问题用前端 strip 解（F1），协议层的问题用协议对齐解（F4），不打 prompt 地鼠。

---

## 6. 关联的其他前端/UX 隐患（本次 dogfood 顺带观察，未深挖）

> 这些是同一份 final-body 里观察到的、与 `[intent]` 同族的「内部状态泄露给用户」现象，**未单独取证到代码层**，列出供后续排查：

1. **`sealed_by="run_plan"` / `.json` 路径泄露**：lead 转述 subagent 结果时常带 `/mnt/user-data/workspace/handoff_code_executor.json 已由 run_metric_plan 落盘（sealed_by="run_plan"）`。`handoff` / `sealed_by` / `.json` 路径对研究员是工程噪音，暗示「这是个跑 JSON 文件的脚本」。与 `[gate_signals]` 同源（根因 A：subagent 把内部状态写进给 lead 的转述，lead 又转述给用户）。
2. **「查看其他 N 个步骤」「使用 X 工具」步骤条**：本身是好的（增加透明度），但当工具名是 `identify_ev19_template`、`run_metric_plan` 这种内部命名时，对研究员不友好。**可能是 F1 的同族**：内部工具命名直接暴露。可考虑前端做工具名→人类可读标签的映射表。
3. **chart-maker 重派时无用户感知**：chart-maker 两次撞 max_turns（见 [silent-drop problem doc](2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md)），lead 静默重派，**用户在前端看不到「图表生成失败了两次、正在重试」**——失败被吞，用户以为只是慢。这是**反方向的 UX 问题**：该给用户看到的内部状态（失败/重试）反而藏起来了，不该看到的（控制 tag）反而露出来了。**两个方向都该修**。

> 注：以上 3 条是初步观察，**本文只对 `[intent]`/`[gate_signals]`（症状 1-4）做了代码层取证**。1-3 条若要立项需单独取证确认。

---

## 7. 元信息

- **诊断 agent**：Claude（本会话，`/noldus-insight-e2e` skill 驱动 + 用户主动提问触发深挖）
- **取证时间**：2026-06-24
- **用户授权范围**：写 problem doc，**未授权改代码**
- **关联文档**：
  - [`2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md`](2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md)（同 run 的 chart-maker 问题，`[gate_signals]` 在那个阶段爆炸性泄露）
  - [lead_agent/prompt.py:258](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py)（`[intent]` 协议定义）、[:397](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py)（`[gate_signals]` 消费）
  - [data_analyst.py:235](../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py)（subagent 输出 `[gate_signals]`）
  - [src/core/messages/utils.ts:519](../../packages/agent/frontend/src/core/messages/utils.ts)（前端 strip 白名单，未覆盖方括号）
  - [`../plans/2026-06-24-frontend-generative-ux-upgrade.md`](../plans/2026-06-24-frontend-generative-ux-upgrade.md)（前端生成式 UX 升级计划，本问题应纳入）
  - CLAUDE.md「HarnessX 三大病理自检」
- **forensic 证据**：`/tmp/noldus-e2e-runs/20260624-161811/{final-body.txt,sse-0..7.txt,analyze.txt}`

---

## milestone 建议

「前端 UX / 专业形象」track（若有）：本次 dogfood 实证 lead/data-analyst 的 `[intent]`/`[gate_signals]` 机器控制文法系统性泄露到对话气泡（final-body 里 `[intent]` 出现 9+ 次/turn、`[gate_signals]` 在 subagent 密集阶段单 turn 56 次）。根因是控制信号走「渲染正文」通道 + 前端 strip 白名单只覆盖尖括号 tag。checkpoint：「用户主动发现 → 取证坐实（前端 `INTERNAL_MARKER_TAGS` 不含方括号、无任何方括号 strip 逻辑）→ 修复方向 F1（前端止血）+ F4（协议对齐尖括号）已定」。**这是面向研究员的产品观感硬伤，建议纳入 v0.1 前必清**（9 月硬指标，第一印象）。
