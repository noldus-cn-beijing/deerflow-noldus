# Spec：审批/反问决策卡 Decision Card（Phase 0 · 第 5 项）

> 类型：**一次性实施 spec**（前端，零后端依赖）
> 日期：2026-06-24
> 母方案：[docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)（硬伤 C / §5.1）
> 依赖：[spec#1 tokens/motion](2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md)（`--color-status-warning` / 曲线）、[spec#4 analysis-rail](2026-06-24-frontend-phase0-4-analysis-rail-spec.md)（同一 `waiting` 信号联动）
> 适用层：前端 `components/workspace/messages/`（clarification 渲染）+ 输入框态联动
> 设计准则来源：`ui-ux-pro-max`（Quick Reference §8 Forms：`confirmation-dialogs`、`error-recovery`、`progressive-disclosure`；§1/§2：`keyboard-nav`、`escape-routes`、`touch-target-size`；§9：`primary-action`）
> 一句话：把 `ask_clarification` 从"消息流里一段不起眼的 markdown + 按钮"升级成**显眼的决策卡**——左侧状态色 accent bar +"分析已暂停，等待你的确认"强信号 + 决策依据 + 选项（键盘可达）+ 与进度轨/输入框联动。让研究员（非程序员）**绝不会划过去没注意到 agent 在等他**。

---

## 〇、为什么需要它（硬伤 C 前半）

母方案硬伤 C：`ask_clarification` 目前只是消息流里一段 markdown + 几个按钮（`message-list.tsx:86-134` + `clarification-options.tsx`），**没有视觉上"流程在此暂停、等你决策"的强信号**。研究员很容易划过去没注意到 agent 在等他——分析就这么**静默卡住**。用户视角原文："可中断提示""审批链"应该是一等公民。

`ask_clarification` 在 EthoInsight 里承载的是**领域关键决策**：列语义对齐（`中心区`/`边缘区` 反问）、范式确认——这些正是 HITL 铁律所在（CLAUDE.md 第9条「组间比较不猜阈值」+ memory「范式必反问」+ `feedback_identify_zone_info_not_persisted`：lead 要**带列依据**反问）。决策点淹没 = 既伤体验,又让"该用户拍板的"变成"用户没看见就卡住"。

> `ui-ux-pro-max` §8 `confirmation-dialogs`（破坏性/关键操作前确认）、`error-recovery`（必须给清晰恢复路径）、`primary-action`（每屏一个主 CTA）在这里要求：决策点必须**视觉显眼 + 路径清晰 + 一个明确主动作**。

---

## 一、现状（带证据）

### 1.1 已有机制（复用 + 升级）

| 资产 | 位置 | 现状 | 升级 |
|---|---|---|---|
| `ask_clarification` 工具 | 后端 `clarification_tool.py` | `clarification_type: Literal["missing_info","ambiguous_requirement","approach_choice","risk_confirmation","suggestion"]` + `context` + `options` | 前端利用 type 区分卡片样式/图标 |
| ClarificationMiddleware | 后端 `clarification_middleware.py` | 格式化成带 icon + context + 编号选项的 ToolMessage,`Command(goto=END)` 中断 | 不改后端 |
| `message-list.tsx:86-134` `assistant:clarification` 分支 | 前端 | 渲染 markdown + `ClarificationOptions` | **升级为 DecisionCard** |
| `ClarificationOptions` | `clarification-options.tsx` | 竖排 outline 按钮 + 编号 + "或自定义输入" | 复用,加键盘数字键 + 主/次样式 |
| `onSelectClarificationOption` | `message-list.tsx:58` → 页面转发为下条 user message | 点选项 = 发该文本 | 不变（机制正确） |

### 1.2 缺口
- 反问块视觉上与普通 AI 消息**无区别**,无"暂停/等待"强信号。
- `clarification_type` 后端有,前端**没用**（所有反问长一个样,risk_confirmation 和 suggestion 同等对待）。
- 决策**依据**（agent 为什么问、基于哪些列/什么歧义）未突出——研究员看不懂"为什么要我选"。
- 选项**无键盘可达**（不能按 1/2/3 选,`clarification-options.tsx` 只 onClick）。
- 与进度轨无联动（spec#4 的 waiting 态）。
- 输入框无态变化（等待时 placeholder 不提示）。

---

## 二、目标与非目标

### 目标
1. `ask_clarification` 升级为**显眼决策卡**：状态色 accent bar + "分析已暂停，等待你的确认"标题 + clarification_type 图标。
2. **决策依据突出**：context（agent 为什么问 / 基于哪些列）+ question 清晰分层。
3. 选项：主/次视觉层级（`primary-action`）+ **键盘数字键 1-9 可选** + "或自定义输入"。
4. **进度轨联动**（spec#4）：决策卡出现时,对应阶段 `waiting` 琥珀脉冲;点进度轨 waiting 节点滚到卡。
5. **输入框态联动**：等待决策时 placeholder 变"回答上面的问题，或直接输入…"。
6. `clarification_type` 差异化：`risk_confirmation` 更强信号（danger 调）、`suggestion` 较弱（info 调）。

### 非目标
- ❌ 不改后端 clarification 机制（`Command(goto=END)` 中断、ToolMessage 格式不动）。
- ❌ 不改 `onSelectClarificationOption` 的"点选项=发消息"机制（已正确）。
- ❌ 不做"回退/改主意"重型版（母方案 §5.3 回退 → Phase 1;本 spec 只做"显眼 + 易答"）。
- ❌ 不动流式核心 / `groupMessages`。

---

## 三、设计

### 3.1 决策卡视觉（显眼但日式克制）

```
消息流里（替换现 clarification 分支）:
┌▌─────────────────────────────────────────┐   ▌ = 左侧状态色 accent bar
│▌ ⏸  分析已暂停 · 等待你的确认                │   标题（status 色 + 图标 + 文字）
│▌                                           │
│▌ 为什么问：检测到列「中心区」「边缘区」，需   │   ← context（依据），muted 底
│▌ 确认它们对应 OFT 的哪个分析区               │
│▌                                           │
│▌ 这两列分别是中央区还是周边区？               │   ← question，主文案
│▌                                           │
│▌  [1] 中心区=中央区，边缘区=周边区   ← 主样式 │   ← 选项，主/次层级
│▌  [2] 反过来                                │
│▌  [3] 都不是，让我说明                       │
│▌  或在下方直接输入…                          │
└─────────────────────────────────────────────┘
```

**视觉细节（spec1 token + 日式克制）**：
- **左侧 accent bar**（3-4px 状态色竖条）——日式克制的"强信号"（不是整卡变色,不是大红框）。颜色按 `clarification_type`：
  - `risk_confirmation` → `--color-status-danger`（朱）
  - `approach_choice` / `ambiguous_requirement` / `missing_info` → `--color-status-warning`（琥珀,"等待"语义）
  - `suggestion` → `--color-status-info`（沉静蓝,弱）
- **标题**："分析已暂停 · 等待你的确认"（`risk_confirmation` 用"需要你确认风险"）+ ⏸/⚠ 图标 + 状态色。**色 + 图标 + 文字三件套**。
- **决策依据（context）**：muted 底色小块,前缀"为什么问："——直接服务 memory `feedback_identify_zone_info_not_persisted`（带依据反问,让研究员看懂）。context 缺失时不显该块。
- **question**：主文案,字号略大,不埋没。
- **选项**：第一个/推荐项可主样式（brand 描边）,其余次样式（`primary-action`：视觉主次）;每项编号 + 触摸目标 ≥44px（`touch-target-size`,`clarification-options.tsx` 现 `size="sm"` 要确认达标）。
- 卡片入场：spec1 `--animate-fade-in-up` + `ease-brand-out`;**等待中**整卡 accent bar 轻脉动（`animate-pulse-soft` 变体,克制）。
- 已答后：accent bar 转 success 绿 + 收起脉动 + 选中项高亮（`success-feedback`）——给"已确认"的闭环反馈。

### 3.2 键盘可达（`keyboard-nav`）
- 卡片聚焦时,按数字键 `1`-`9` 直接选对应选项（≤9 项）。
- 选项按钮 tab 可达 + Enter 触发。
- "或自定义输入"聚焦输入框（点击/Tab 到输入框）。
- `role="group"` + `aria-label`（现有已有,保留）+ 选项 `aria-keyshortcuts`。

### 3.3 进度轨联动（spec#4 同源）
- 决策卡出现 = spec#4 `useWorkflowStages` 输出对应阶段 `waiting`——**同一信号**。进度轨该节点琥珀脉冲。
- 点进度轨 waiting 节点 → 滚动到决策卡（spec#4 §3.4）。
- 决策卡内不重复算阶段——读 spec#4 的阶段映射（SSOT）。

### 3.4 输入框态联动
- 等待决策时（流非 loading 且最后是未答 clarification）：
  - 输入框 placeholder → "回答上面的问题，或直接输入…"（i18n）。
  - 可选：输入框旁轻提示"⏸ 等待你的确认"。
- 信号来源：母方案已有 `_is_awaiting_clarification` 类判断（memory `feedback_todo_middleware_must_not_force_reengage_while_awaiting_clarification` 提到后端已有此概念）;前端从"最后一条是未答 clarification"派生即可,不改后端。

### 3.5 clarification_type 差异化（利用已有字段）
后端 `clarification_type` 前端当前没用。映射：

| type | accent 色 | 标题 | 强度 |
|---|---|---|---|
| `risk_confirmation` | danger 朱 | 需要你确认风险 | 最强 |
| `approach_choice` | warning 琥珀 | 请选择分析方式 | 强 |
| `ambiguous_requirement` | warning 琥珀 | 需要你澄清 | 强 |
| `missing_info` | warning 琥珀 | 需要补充信息 | 强 |
| `suggestion` | info 蓝 | 一个建议 | 弱（可不脉动） |

> type 从哪取：`message-list.tsx` 已从 `findToolCallArgs(toolCallId)` 拿 `options`（`:90-113`），同处可拿 `clarification_type` / `context`。纯前端,不改后端。

---

## 四、实施步骤

### Step 1：`DecisionCard` 组件（`components/workspace/messages/`）
替换 `message-list.tsx:86-134` `assistant:clarification` 分支的渲染：accent bar + 标题 + context 块 + question + `ClarificationOptions`。读 `clarification_type` / `context` / `options`（均从 `findToolCallArgs` 拿）。

### Step 2：`ClarificationOptions` 升级
- 加键盘数字键 1-9 选（`useEffect` 监听 keydown,卡聚焦时生效）。
- 主/次样式（首项/推荐项 brand 描边）。
- 确认触摸目标 ≥44px。
- 已答态：选中高亮 + 禁用其余。

### Step 3：进度轨联动（依赖 spec#4）
- 决策卡的 waiting 信号与 spec#4 `useWorkflowStages` 同源——不重复算,复用。
- 点进度轨 waiting → 滚到卡（spec#4 提供 anchor）。

### Step 4：输入框态联动
- `input-box.tsx`：检测"等待 clarification"→ placeholder 变文案 + 可选提示。从消息流派生（最后一条未答 clarification）。

### Step 5：i18n + a11y + reduced-motion
- 所有文案（标题/依据前缀/placeholder/类型标题）进 i18n（不硬编码中文）。
- 键盘数字键 + tab + Enter + aria;`escape-routes`（用户可忽略选项直接输入,已有"或自定义"）。
- accent bar 脉动 reduced-motion 降级为静态。

---

## 五、验收标准

### 功能
- [ ] `ask_clarification` 渲染为决策卡：左 accent bar + "分析已暂停·等待确认"标题 + 图标。
- [ ] 决策依据（context）突出显示,前缀"为什么问"；缺失时不显。
- [ ] 选项主/次层级;键盘数字键 1-9 可选 + tab + Enter;"或自定义输入"。
- [ ] `clarification_type` 差异化（risk=danger 最强 / suggestion=info 弱）。
- [ ] 已答后 accent 转绿 + 选中高亮 + 收起脉动（闭环反馈）。
- [ ] 进度轨对应阶段 waiting 琥珀脉冲（spec#4 联动）;点 waiting 滚到卡。
- [ ] 等待时输入框 placeholder 变"回答上面的问题，或直接输入…"。
- [ ] 点选项 = 发该文本（现机制不变）。

### a11y / 性能（红线）
- [ ] 选项触摸目标 ≥44px。
- [ ] 键盘完全可达（数字键 + tab + Enter）;`aria-label` / `aria-keyshortcuts`。
- [ ] 状态"色 + 图标 + 文字"三件套（`color-not-only`）——不靠 accent 色单独表意。
- [ ] accent 色 foreground 对比 ≥4.5:1（danger/warning/info,light）。
- [ ] 脉动 reduced-motion 降级;入场用 spec1 曲线。
- [ ] `escape-routes`：用户可绕过选项直接输入（不强制选）。

### 工程纪律
- [ ] 不改后端 clarification 机制 / ToolMessage 格式 / `Command(goto=END)`。
- [ ] 不改"点选项=发消息"机制。
- [ ] 进度轨 waiting 信号与 spec#4 同源,不重复算阶段（SSOT）。
- [ ] 不动流式核心 / `groupMessages`。
- [ ] `pnpm check` + i18n 不硬编码中文。

---

## 六、风险与回退

| 风险 | 缓解 |
|---|---|
| 决策卡"太显眼"变打扰 | accent bar（细竖条）而非整卡变色/大红框——日式克制的强信号;suggestion 类弱化不脉动 |
| `clarification_type` 粒度不足以判"列对齐" | type + question/options 关键词兜底（同 spec#4）;判不准时统一 warning 调,不影响"显眼"主目标 |
| 数字键与输入框冲突（用户正在输入按到数字） | 仅卡聚焦且输入框未聚焦时数字键生效;输入框聚焦时数字键正常输入 |
| 进度轨未落（spec#4 依赖） | 决策卡的"显眼+键盘+输入框联动"**不依赖** spec#4,可独立先上;进度轨联动作为 spec#4 落后的增量 |

**回退**：纯前端组件升级（替换一个渲染分支）;最坏回退到现 markdown + `ClarificationOptions`。`git revert` 零风险。

---

## 七、给实施 agent 的交接

- 改动文件：`message-list.tsx`（clarification 分支 → DecisionCard）、新建 `messages/decision-card.tsx`、升级 `clarification-options.tsx`（键盘 + 主次样式）、`input-box.tsx`（placeholder 态）、i18n。
- **不碰**：后端 clarification、`onSelectClarificationOption` 机制、流式核心、`groupMessages`。
- 复用 spec1 `--color-status-*` + 曲线 + `animate-pulse-soft`（不重定义）。
- 进度轨联动**依赖 spec#4**——可分两步：先上决策卡本体（独立见效）,spec#4 落后补联动。
- `clarification_type` / `context` 从 `findToolCallArgs(toolCallId)` 取（`message-list.tsx:90` 已有 pattern,扩展取多字段）。
- HITL 是领域铁律所在——决策依据（context）务必突出,这是 memory `feedback_identify_zone_info_not_persisted`「带列依据反问」的前端落点。

---

*依据：母方案硬伤 C / §5.1 + 后端 `clarification_tool.py` 的 type/context/options 字段 + `ui-ux-pro-max` §8（confirmation-dialogs/error-recovery/progressive-disclosure）+ §1/§2（keyboard/escape/touch）。数据/信号同源 spec#4。未写代码。*
