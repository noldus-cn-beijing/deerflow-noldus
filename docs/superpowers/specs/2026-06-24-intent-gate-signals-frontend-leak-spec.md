# Spec：lead/subagent 的 `[intent]`/`[gate_signals]` 机器控制文法泄露到前端对话 —— 前端 render 层 strip

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-24
> 代码基线：dev HEAD `24383de8`
> 性质：🟠 中 · 产品体验/专业形象隐患（非功能 bug，不影响分析正确性）。面向行为学研究员，对话气泡里冒出 `[intent] E2E_FULL`、`[gate_signals] constitution_acknowledged: true ...` 等机器控制文法，破坏专业观感（v0.1 9 月硬指标的第一印象）。
> **诊断文档**：[docs/problems/2026-06-24-intent-gate-signals-tags-leaked-to-frontend.md](../../problems/2026-06-24-intent-gate-signals-tags-leaked-to-frontend.md)
> **⚠️ 本 spec 修法与诊断推荐不同**：诊断推荐 F1（前端 strip）+ F4（改尖括号）。**经核实 F4/F2/F3 有诊断没看透的致命约束**（见 §二）——`[intent]`/`[gate_signals]` 被后端 guardrail/lead **从 message 历史 content 读取**，从 content 抽走/改语法会破坏下游消费方。**唯一外科级正确的是纯前端 render 层 strip（content 对后端保持完整，只 UI 显示时过滤）**。
> 受保护文件：`src/core/messages/utils.ts`（前端，非 deerflow 上游受保护清单，但属共享渲染逻辑，改后跑 `pnpm check`）。

---

## 〇、给实施 agent 的一句话

`[intent] E2E_FULL` / `[gate_signals] ...` 是确定性控制协议（intent 被 W17 guardrail 校验、gate_signals 被 lead 决策），**写在 AIMessage 正文里**；前端 `INTERNAL_MARKER_RE` 只认尖括号 tag，方括号原样渲染给研究员。**修法 = 在前端 render 入口 `extractContentFromMessage`（utils.ts:317）加行级 strip，删掉 `^\s*\[(intent|gate_signals)\].*$` 整行**。**content 在 message state/历史/导出里原样保留**（guardrail/lead 照常从历史读），只 UI 气泡显示时 strip。加 `utils.test.ts` 回归用例。**不动后端协议**（不改方括号→尖括号、不移到 additional_kwargs）——因为那会破坏读取这些信号的 guardrail/lead。

---

## 一、根因（诊断属实，代码核实）

### 泄露事实（final-body.txt 实证）
- `[intent] E2E_FULL`（L136/420 等）、`[gate_signals] constitution_acknowledged: true data_quality: ...`（L432/504 等）**原样出现在前端渲染正文**。
- 后半程（chart-maker/report-writer subagent 密集 handoff）爆炸性增长（sse-6/7 合计 `[gate_signals]` 95 次）——分析越复杂、噪音越密。

### 根因 A：控制信号走「渲染正文」通道
`[intent]`（lead prompt L258 要求输出，guardrail 校验）+ `[gate_signals]`（subagent 输出给 lead，lead 转述时**原样抄进**自己的 AIMessage 正文）都写进 `AIMessage.content` —— 而 `.content` 正是前端默认渲染字段。

### 根因 B：前端 strip 白名单只覆盖尖括号
`utils.ts:526` `INTERNAL_MARKER_RE = new RegExp('<(${...})>...')` —— 用 `<(...)>` 正则，**结构上匹配不了 `[intent]`**。`isHiddenFromUIMessage`（L472）按 `hide_from_ui` 过滤，但这些信号在普通 AIMessage 正文里没这标记。**两道防线都漏。**

---

## 二、为什么不动后端协议（F4/F2/F3）—— 核实出的致命约束

> 诊断推荐 F4（方括号→尖括号）+ F1。但 F4/F2/F3 都要动**后端读取这些信号的消费方**，是协议层改动，违反「展示层问题展示层解」判据，且 catastrophic forgetting 高风险。**核实如下**：

### `[intent]` 被 guardrail 从历史 content 读取
`guardrails/intent_classification_provider.py:37,40-56`：`_INTENT_LINE_RE = re.compile(r"\[intent\]\s+([A-Z0-9_]+)")`，`_extract_declared_intents` **遍历所有历史 AIMessage 的 `msg.content`**（L44-56）扫 `[intent]` 行。guardrail 是 pre-tool-call 检查，下一 turn 评估工具调用时**重新从历史 content 读** lead 上一 turn 声明的 intent。
- → **任何从 content 抽走 `[intent]` 的改动（F2/F3）会让 guardrail 读不到** → W17 意图分类门失效。
- → **改成 `<intent>`（F4）要同步改这个正则 + lead prompt L258/1115 的输出指令** —— 动协议两端。

### `[gate_signals]` 被 lead 从历史读取做决策
lead prompt L397「从 `[gate_signals]` 块或 handoff 摘要里提炼的关键数字/状态」、L399「如果 gate_signals.quality_warnings_critical_count > 0 → 播报」——lead 读前序 subagent 的 gate_signals 决定下一步。从 content 抽走同样破坏。

### 对比 `<think>` 为什么能被 ThinkTagMiddleware 干净抽走
`think_tag_middleware.py` 把 `<think>` 抽到 `additional_kwargs.reasoning_content` 后，**没有任何下游再从 content 读 `<think>`**。`[intent]`/`[gate_signals]` 抽走 = guardrail/lead 瞎了。**这是本质区别——诊断的「F3 ThinkTag 同构」类比不成立。**

### 结论：唯一不破坏下游的是「content 保留 + render 层 strip」
content 必须对后端消费方（guardrail/lead/训练数据录制）完整。**只在前端 render 那一刻 strip 方括号控制行** = F1，且符合判据「展示层问题用展示层解，不动协议层」。

---

## 三、修法（P0 主修：前端 render 层 strip）

### 改动 1 — 新增 `stripControlSignalLines`（`utils.ts`，挨着 `INTERNAL_MARKER_TAGS`）
```ts
/**
 * 后端控制协议信号（方括号语法），写在 AIMessage 正文里供后端消费：
 * - `[intent] <NAME>` — lead 意图分类，被 IntentClassificationGuardrailProvider
 *   从历史 content 读取校验（W17）。
 * - `[gate_signals] ...` — subagent 给 lead 的决策信号，lead 从历史读取做下游决策。
 *
 * 这些是机器控制文法，对研究员零信息量。但【绝不能从 content 抽走】——后端
 * guardrail/lead 依赖从历史 content 读它们。所以只在【前端 render 入口】行级 strip：
 * content 在 state/历史/导出里原样保留，仅 UI 气泡显示时删掉这些行。
 *
 * 与 <think>（ThinkTagMiddleware 后端抽到 additional_kwargs）不同：<think> 抽走后
 * 无下游再读，故可后端处理；[intent]/[gate_signals] 抽走会破坏 guardrail/lead，
 * 故只能前端展示层 strip。
 */
export const CONTROL_SIGNAL_LINE_NAMES = ["intent", "gate_signals"] as const;

// 行级匹配：整行以 [intent] 或 [gate_signals] 开头（容忍前导空白）→ 删整行。
// gate_signals 是多行块（后续缩进行 key: value），也一并删到块尾（下一非缩进行/空行前）。
const CONTROL_SIGNAL_LINE_RE = new RegExp(
  `^[ \\t]*\\[(${CONTROL_SIGNAL_LINE_NAMES.join("|")})\\][^\\n]*(\\n[ \\t]+[^\\n]*)*`,
  "gm",
);

export function stripControlSignalLines(content: string): string {
  return content.replace(CONTROL_SIGNAL_LINE_RE, "").replace(/\n{3,}/g, "\n\n").trim();
}
```
> ⚠️ **`[gate_signals]` 是多行块**（final-body L432 是单行紧凑形，但 chart-maker 的 `[gate_signals]\ncharts_generated: N\nchart_files:\n  - x.png` 是多行）。正则的 `(\n[ \t]+[^\n]*)*` 吃掉块体的缩进续行（`charts_generated:`/`  - x.png`）。**实施时用 final-body + sse-6/7 的真实 `[gate_signals]` 块做正则验证**（两种形态都要 strip 干净）。
> ⚠️ **`\n{3,}→\n\n`**：strip 整行后留下的多余空行收敛，避免气泡里大段空白。

### 改动 2 — 在 render 入口 `extractContentFromMessage` 调用（`utils.ts:317`）
`extractContentFromMessage` 是 UI 气泡渲染入口（`message-list.tsx:118,147` 调）。在它返回前套一层 strip：
```ts
export function extractContentFromMessage(message: Message) {
  let text: string;
  if (typeof message.content === "string") {
    text = splitInlineReasoningFromAIMessage(message)?.content ?? message.content.trim();
  } else if (Array.isArray(message.content)) {
    text = message.content.map(/* ...现有 ... */).join("\n").trim();
  } else {
    return "";
  }
  // P0（spec 2026-06-24-intent-gate-signals-leak）：strip 后端控制协议方括号信号。
  // 只在展示层 strip，content 原样保留供后端 guardrail/lead 从历史读取。
  return stripControlSignalLines(text);
}
```
> ⚠️ **只改 render 路径，不改 state**。`extractContentFromMessage` 是展示派生专用；message 对象的 `.content` 字段不动（后端读的是它）。**已核实它的全部 3 类调用方都是安全 strip 点（无回传后端）**：① `message-list.tsx`/`message-list-item.tsx`（UI render）② `utils.ts:247`（取最近 AI 回复文本的展示派生，只读）③ `export.ts:16`（导出对话文本——该 strip，导出也不该带控制 tag）。**没有任何调用方构造发给后端的 payload**（后端提交走 `core/threads/types.ts` 的 Message 结构，不经此函数）。所以 strip 放进 `extractContentFromMessage` 本身 = render+导出+派生三处一次覆盖，零污染后端。

### 改动 3 — 回归测试（`utils.test.ts`）
`utils.test.ts` 已有 `extractContentFromMessage` 的 `<think>` strip 用例。加：
1. 单行 `[intent] E2E_FULL` → strip 后只剩自然语言。
2. 单行紧凑 `[gate_signals] constitution_acknowledged: true ...` → strip 干净。
3. 多行块 `[gate_signals]\ncharts_generated: 1\nchart_files:\n  - x.png` → 整块 strip。
4. `[intent]` 夹在自然语言中间（final-body L51-53 形态：工具步骤条 + `[intent]` + 回复）→ 只删 `[intent]` 行，保留前后自然语言。
5. **合法用户输入保护**：用户消息正文含 `[intent]`-like 文本（如研究员问「什么是 [intent] 标记」）——确认 user message 不走 `extractContentFromMessage` 的 strip，或 strip 只对 `message.type==="ai"` 生效（**实施时定**：最稳是只对 AI message strip，user message 原样）。
6. **content 完整性**：strip 是 render 专用，message.content 字段本身不变（断言传入的 message 对象未被 mutate）。

---

## 四、P2（可选，谨慎）：`sealed_by`/`.json` 路径泄露

诊断 §6.1：lead 转述 subagent 结果时常带 `/mnt/user-data/workspace/handoff_code_executor.json 已由 run_metric_plan 落盘（sealed_by="run_plan"）`。这是工程噪音（暗示「在跑 JSON 脚本」）。

**本 spec 不含 P2**，理由：
- 与 `[intent]`/`[gate_signals]` 不同，这些是**lead 自然语言转述里嵌的**（不是固定语法前缀），行级正则 strip 容易误删合法内容（如 lead 说「分析结果已保存」是合法的）。
- 真正的修法在**lead prompt**——让 lead 转述时不提 handoff 路径/sealed_by/.json（用户视角的「指标已算完」而非「handoff.json 已落盘」）。但改 prompt 属协议层、需 grep + 回归，单独立项。
- **本 spec 聚焦确定性可 strip 的方括号控制信号**（P0），P2 留给「lead 转述措辞优化」独立 spec/prompt 改动。

> 同理诊断 §6.2（工具名 `identify_ev19_template` 暴露）、§6.3（chart-maker 重派无用户感知）也不在本 spec——前者是「工具名→人类标签映射」（前端独立 feature），后者是「反向 UX：该露的失败状态藏了」（需后端给前端 emit 重试事件，独立 spec）。**本 spec 只治「不该露的控制 tag 露了」这一条，外科级最小。**

---

## 五、验收（确定性 gate）
1. `pnpm check`（lint + typecheck）通过。
2. `utils.test.ts` 新增 6 用例全绿（含 content 完整性、user message 不误 strip）。
3. **用真实 final-body 验证**：取 `/tmp/noldus-e2e-runs/20260624-161811/final-body.txt` 里的 `[intent]`/`[gate_signals]` 实际形态（单行 + 多行块）喂 `stripControlSignalLines`，断言全删干净、自然语言保留。
4. **后端不回归**：改动**只在前端**，`message.content` 字段不变 → guardrail（intent_classification_provider）+ lead（gate_signals 决策）照常从历史 content 读到信号。**已核实 `extractContentFromMessage` 3 类调用方（render/导出/派生文本）无一构造回传后端 payload**（后端提交走 Message 结构不经此函数）→ 后端代码零改动、零回归。
5. 手动：前端跑一次含 `[intent]`/`[gate_signals]` 的对话，气泡里不再出现方括号控制文法。

---

## 六、风险与三大病理自检

1. **Reward hacking**：不涉及（非验收伪造）。
2. **Catastrophic forgetting**：
   - **本 spec 选前端 strip 正是为了避开它**——不动后端协议 = 不碰 guardrail/lead 读取逻辑 = 零下游回归。这是 F1 优于 F2/F3/F4 的核心理由。
   - 改 `extractContentFromMessage` 前 grep 它所有调用方（确认只 render 用、无后端回传），避免 strip 泄漏到非展示路径。
3. **Under-exploration**：
   - **不是改 prompt 让 agent 别输出 `[intent]`**——这些是确定性控制协议（guardrail 依赖），不能不输出。问题在「输出后没在前端过滤」，不在「输出本身」。
   - **也不是上后端结构门**——因为信号必须留在 content 供后端读，后端没有「该删」的诉求；纯展示层问题，展示层解（守判据「展示层问题前端 strip，协议层问题协议对齐」）。

### 前后端协议漂移的根上防护（诊断根因 B 的元教训）
`INTERNAL_MARKER_TAGS` 是手工白名单，后端新增控制信号语法时前端没同步 = 协议漂移。**本 spec 顺带留一道防线**：在 `CONTROL_SIGNAL_LINE_NAMES` 注释里写明「后端新增方括号控制信号（如未来 `[handoff]`）必须同步加这里」，并在 lead/subagent prompt 定义 `[intent]`/`[gate_signals]` 处加反向注释「前端 strip 见 utils.ts:CONTROL_SIGNAL_LINE_NAMES」（双向锚点，降低下次漂移）。**不强求做注册表自动同步**（过度工程，仅 2 个信号）。

---

## 七、关键代码锚点（已核实，行号 = dev 24383de8）
- **前端主修**：`src/core/messages/utils.ts`：`extractContentFromMessage`(317，render 入口)、`INTERNAL_MARKER_TAGS`(519)/`INTERNAL_MARKER_RE`(526，尖括号只覆盖)、`stripUploadedFilesTag`(495)、`isHiddenFromUIMessage`(472)
- **render 调用方**：`src/components/workspace/messages/message-list.tsx`(118/147 调 extractContentFromMessage)
- **测试**：`src/core/messages/utils.test.ts`（已有 extractContentFromMessage `<think>` 用例，加方括号用例）
- **后端消费方（证明不能从 content 抽走）**：`guardrails/intent_classification_provider.py`(37 `_INTENT_LINE_RE`、40-56 `_extract_declared_intents` 遍历历史 content)；lead prompt L258（[intent] 输出）/L397-399（[gate_signals] 消费）/L1115
- **ThinkTag 对比**：`agents/middlewares/think_tag_middleware.py`（抽到 additional_kwargs，无下游再读 = 可后端处理；与 intent/gate_signals 本质不同）
- **诊断文档**：`docs/problems/2026-06-24-intent-gate-signals-tags-leaked-to-frontend.md`
- **forensic**：`/tmp/noldus-e2e-runs/20260624-161811/final-body.txt`（真实泄露形态，正则验证用）

---

## 八、与前端 UX 升级计划的关系
诊断 §6 提到 [docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)。本 spec（P0 控制信号 strip）是**即时止血的最小外科改动**，应先于大计划落地（5 行前端改动 + 测试）。诊断 §6 的其余 UX 隐患（工具名映射、重试可感知、sealed_by 措辞）归入那份 UX 升级计划统筹，不混入本 spec。

---

## milestone 建议
归入「前端 UX / 专业形象」track。checkpoint：「用户主动发现 dogfood 对话气泡冒 `[intent]`/`[gate_signals]` 机器文法 → 取证坐实（前端 strip 只覆盖尖括号，方括号漏）→ **纠正诊断修法**：F4/F2/F3 会破坏后端 guardrail/lead 从历史 content 读信号，唯一外科正确是前端 render 层 strip（content 保留供后端读）→ P0 spec（`stripControlSignalLines` in `extractContentFromMessage`）。面向研究员的产品观感硬伤，v0.1 前必清」。
