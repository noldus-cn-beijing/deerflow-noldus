# Spec：多文件上传堆叠 + hover/点击扇开（Phase 0 · 第 8 项）

> 类型：**一次性实施 spec**（前端交互组件，零后端）
> 日期：2026-06-25
> 母方案：[docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)（硬伤 A 同源：海量产物/输入也会淹没界面）
> 依赖：[spec#1 tokens/motion](2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md)（`--ease-brand-out` 扇开曲线 / `--dur-*`）、[spec#6 design-language](2026-06-24-frontend-phase0-6-design-language-craft-semiotics-spec.md)（`--shadow-overlap` 堆叠阴影 / scale-feedback / 一屏一主角）
> 适用层：前端 `components/workspace/input-box.tsx`（替换附件渲染用法）+ 新建 `components/workspace/attachments/*`（自建堆叠组件）。**复用** generated `ai-elements/prompt-input.tsx` 的 `usePromptInputAttachments()` store —— **不手改 generated 源**。
> 设计准则来源：`ui-ux-pro-max`（`progressive-disclosure` / `Hover vs Tap`(High) / `scale-feedback` / `touch-target-size` 44px / `overflow` / `modal-escape` / `stagger-sequence` / `excessive-motion`）
> 一句话：研究员一次常传**几十~几百份** EPM trial 文件,现状附件区 generated `PromptInputAttachments`(`prompt-input.tsx:405`)用 `flex-wrap` + **全量 `.map()`** 把每个文件平铺换行 → **一下占满屏幕**(图一/图二)。改成 **Seedance 式堆叠**:超过阈值(≈5)后续文件**堆叠成一叠** + "+N" 计数;**hover(桌面)/点击(触屏)扇开**这叠让用户挑选删除。**自建 Noldus 组件替掉用法,复用现有 attachments store 不重造上传逻辑,不改 generated 源。**

---

## 〇、为什么需要这份 spec

母方案硬伤 A 是"产物侧"几百张图淹没界面;**这份是"输入侧"的同构问题**——研究员上传文件数同样可达几十~几百(一个 EPM 实验 28 trial × 多批 = 轻松上百)。用户原话(2026-06-25,附图一/图二):

> "用户(其实很多情况下)会上传几十份、几百份文件,如果前端输入框上要像现在一样展示每一份文件的上传进度和显示,会造成一下子占满前端……seedance 对素材上传效果显示比较好:超过一定数量(我们场景可能超过 5 个)就让接下来的文件**堆叠**在上传文件上;用户光标浮上去会**单独展开**这些堆叠素材,可选删除(图三/图四)。"

**这是真实高频场景**(EPM/OFT 批量 trial),不是边缘 case。现状把上百个附件 chip 全平铺,直接违反母方案"一屏一主角 + 日式克制"和 `ui-ux-pro-max` `progressive-disclosure`。

---

## 一、现状（带证据）

### 1.1 附件渲染 = generated `ai-elements` 的 flex-wrap 全量平铺（淹没源头）
`input-box.tsx:458-460` 用 generated 组件渲染附件:
```tsx
<PromptInputAttachments>
  {(attachment) => <PromptInputAttachment data={attachment} />}
</PromptInputAttachments>
```
`PromptInputAttachments`(`prompt-input.tsx:392-405`)实现:
```tsx
className={cn("flex w-full flex-wrap items-center gap-2 p-3", className)}  // flex-wrap = 全部换行平铺
{attachments.files.map((file) => (<div className="max-w-60">{children(file)}</div>))}  // 全量 map,无折叠/堆叠/虚拟化
```
→ **几十/几百个 chip 全 wrap 铺开 = 图一/图二**。`p-3` + `max-w-60` 每个还不小,占满更快。

### 1.2 attachments store API（复用，不重造上传逻辑）
`usePromptInputAttachments()`(`prompt-input.tsx:276`)暴露干净 store(本 spec **复用它**):
| API | 类型 | 用途 |
|---|---|---|
| `files` | `(PromptInputFilePart & {id})[]` | 每项 `{id, url, mediaType, filename}`(`:83/177-181`) |
| `add(files)` | `(File[]\|FileList)=>void` | 加文件 |
| `remove(id)` | `(id:string)=>void` | 删单个(扇开后删用它) |
| `openFileDialog()` | `()=>void` | 选文件(`:651`) |

> **关键**:自建堆叠组件只**换渲染**(`.map`→堆叠),**所有 state / 上传 / 删除走现有 `usePromptInputAttachments()`**——不碰上传链路,不重造 state。这是"替用法不改 generated"的支点。

### 1.3 已有可复用
| 资产 | 位置 | 复用 |
|---|---|---|
| `usePromptInputAttachments` store | `prompt-input.tsx:276` | 状态源(files/remove/add) |
| `PromptInputHoverCardContent`(单附件 hover 预览) | `prompt-input.tsx:354` | 扇开后单卡预览可参照 |
| `MAX_UPLOAD_FILES=50`(镜像后端 `uploads.max_files`) | `input-box.tsx:78` | 上限提示已有,堆叠不改上限逻辑 |
| `getFileIcon`/`getFileName` | `core/utils/files` | chip 图标/名 |
| spec#6 `--shadow-overlap` / scale-feedback | spec#6 §1 | 堆叠阴影 + 扇开微交互 |
| spec#1 `--ease-brand-out` / `--dur-*` | spec#1 | 扇开/收起曲线 |

### 1.4 缺口
- 附件区无"超阈值折叠"概念,全量平铺。
- 无堆叠视觉、无 "+N" 计数、无 hover/tap 扇开。
- generated 组件不能手改(sync 冲突),需自建替换。

---

## 二、目标与非目标

### 目标
1. **超阈值堆叠**:附件数 ≤ 阈值(默认 **5**,可配)时正常平铺;超过后,**前 N 个正常显示 + 其余堆叠成一叠**(扇形/层叠视觉)显示 "+M"。
2. **hover/点击扇开**:桌面 hover 那叠 → 扇开/展开成可浏览列表;**触屏点击等价**(`Hover vs Tap` High 红线:hover 不能是唯一入口)。
3. **扇开后可删**:展开态每个文件可单独 remove(复用 `remove(id)`),可滚动浏览全部。
4. **自建替换**:新建 Noldus 组件替掉 `input-box.tsx` 里 `<PromptInputAttachments>` 用法,**复用 store,不改 generated 源**。
5. **日式克制 + 动效**:堆叠用 `--shadow-overlap`,扇开用 `--ease-brand-out` + stagger;不喧宾夺主(输入框仍是主角)。

### 非目标
- ❌ **不改 generated `ai-elements/prompt-input.tsx`**(registry 生成,sync 冲突——CLAUDE.md 铁律)。只**替用法**+复用 store。
- ❌ 不重造上传/附件 state(全走 `usePromptInputAttachments`)。
- ❌ 不改后端 `uploads.max_files` 上限逻辑(堆叠是**显示**层,不动上限校验)。
- ❌ 不做上传进度条精细化(本 spec 聚焦"数量淹没",进度态在堆叠内简化显示即可)。
- ❌ 不动产物侧画廊(那是 spec#3;本 spec 是输入侧,同构但不同组件)。
- ❌ 不动流式核心 / 输入框提交逻辑。

---

## 三、设计

### 3.1 交互模型（Seedance 式堆叠 + 扇开）

```
附件 ≤5（正常平铺，现状保留）:
┌──────────────────────────────────────────┐
│ [📎 Trial1] [📎 Trial2] [📎 Trial3] [📎 Trial4] [📎 Trial5] │
└──────────────────────────────────────────┘

附件 >5（堆叠）:
┌──────────────────────────────────────────┐
│ [📎 Trial1] [📎 Trial2] [📎 Trial3] [📎 Trial4] [▣▣▣ +95] │ ← 前 4 + 一叠"+95"
│                                            └ 堆叠(层叠卡+shadow-overlap)
└──────────────────────────────────────────┘

hover(桌面)/点击(触屏)那叠 → 扇开:
┌──────────────────────────────────────────┐
│  ┌─ 展开面板(可滚动,虚拟化若>50)─────────┐    │
│  │ [📎 Trial5  ✕] [📎 Trial6  ✕] …       │    │ ← 全部剩余,每个可删(✕)
│  │ [📎 Trial7  ✕] [📎 Trial8  ✕] …       │    │
│  └──────────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

- **阈值**:默认 5(用户指定"我们场景可能超过 5 个就有影响")。设常量 `STACK_THRESHOLD = 5`,可调。
- **前 N 显示**:阈值内的正常 chip(如前 4),第 5 位起进堆叠;堆叠卡显 "+M"(M=剩余数)。具体"前几个 + 堆叠"实施期微调(前 4 + 堆叠,或前 5 全显第 6 起堆叠),**原则:平铺数固定不随总数涨**。
- **扇开容器**:`Popover`/`HoverCard`(Radix,已装 `@radix-ui/react-hover-card`/无 popover 则用 dialog 非模态)从堆叠位置展开(`modal-motion` 从触发源)。**桌面 hover 开 + 触屏 tap 开**(双入口)。
- **扇开内**:剩余文件列表,每个 chip 带 ✕ 删除(`remove(id)`);**>50 时列表虚拟化**(复用 spec#7/#3 的 `@tanstack/react-virtual`,与画廊/消息流同引擎)。
- **删除联动**:扇开内删一个 → store `remove(id)` → 计数 -1 → 若降到 ≤ 阈值,自动收回平铺态。

### 3.2 桌面 / 触屏双入口（`Hover vs Tap` High 红线）
- **桌面**:hover 堆叠 → 扇开(`onMouseEnter` 开 + 离开延时关,防误触);hover 单 chip → 高亮 + 露 ✕。
- **触屏**:**点击堆叠** → 扇开(`onClick`,因 hover 在触屏不存在);扇开后点 ✕ 删。触摸目标 ≥44px(`touch-target-size`)。
- **键盘**:堆叠可 focus + Enter/Space 展开;扇开内 Tab 遍历 + Delete/Backspace 删 + ESC 收(`modal-escape` / `keyboard-nav`)。
- `ui-ux-pro-max` `Hover vs Tap`(Severity High)铁律:**hover 只是桌面增强,主交互(展开/删除)必须 tap/click 可达**。

### 3.3 堆叠视觉（日式克制 + spec#6 阴影）
- **层叠卡**:堆叠用 2-3 张轻微偏移 + 旋转的卡叠影(Seedance 那种),最上层清晰、下层渐隐;用 spec#6 `--shadow-overlap` 表达"这是一叠"。
- **"+M" 计数**:堆叠卡上叠一个计数徽章(tabular-nums,spec#1),`text-muted-foreground`,克制。
- **不喧宾夺主**:堆叠 + 计数视觉**安静**(输入框是主角,母方案一屏一主角);堆叠不大、不高饱和、不抢眼。
- **扇开动效**:`--ease-brand-out` + `--dur-base`,展开项 **stagger 30-50ms**(`stagger-sequence`,逐个扇出不齐刷);收起 `--dur-exit`(比展开快)。**reduced-motion 降级**(直接显隐,spec#1 机制)。
- **press 反馈**:chip/堆叠按下 scale 0.97(spec#6 `scale-feedback`),不位移布局。

### 3.4 上传进度态（简化，不淹没）
- 现状每 chip 显"上传中…"(图二),海量时也是淹没源。堆叠后:**正常平铺的前 N 个**可显各自进度;**堆叠那叠**显聚合态(如 "+95 · 上传中 30/95")而非每个都转圈。
- 进度精细化非本 spec 重点(非目标),聚合显示够用。

### 3.5 自建组件结构（替用法，不改 generated）
新建 `components/workspace/attachments/`:
- `stacked-attachments.tsx`(`StackedAttachments`):读 `usePromptInputAttachments().files`,按阈值分"平铺段 + 堆叠段",渲染平铺 chip + 堆叠 + 扇开 Popover。
- `attachment-chip.tsx`:单个 chip(图标/名/✕),平铺与扇开内复用。
- `attachment-stack.tsx`:堆叠视觉(层叠卡 + "+M" + 触发扇开)。
- `input-box.tsx` 改:把 `<PromptInputAttachments>{...}</PromptInputAttachments>`(`:458-460`)**替换**为 `<StackedAttachments />`。**这是唯一改的现有文件**(且只换这一段,不动 input-box 其余)。

> 自建组件**包在 `PromptInput` context 内**(`usePromptInputAttachments` 要 provider),位置不变,所以替换是 in-place 的——store/provider 链不变,只换那段渲染。

---

## 四、实施步骤

> 全在 `frontend/`,非受保护;**不改 generated `ai-elements/`**。

### Step 1：自建堆叠组件骨架（§3.5）
- 新建 `attachments/{stacked-attachments,attachment-chip,attachment-stack}.tsx`。
- `StackedAttachments` 读 `usePromptInputAttachments().files`,按 `STACK_THRESHOLD=5` 分平铺段/堆叠段。
- 先实现**平铺 + 堆叠静态视觉**(不含扇开),`input-box.tsx` 替换用法,确认替换后上传/删除仍正常(走现有 store)。

### Step 2：扇开交互 + 双入口（§3.1 / §3.2）
- 扇开用 Radix HoverCard(桌面 hover)/ Popover(触屏 tap)——**双入口**;扇开内列表 + 每项 ✕(`remove(id)`)。
- 键盘可达(focus/Enter 展开,Tab/Delete/ESC)。
- 删除联动:降到 ≤ 阈值自动收回平铺。

### Step 3：堆叠视觉 + 动效（§3.3）
- 层叠卡 + `--shadow-overlap` + "+M" 计数(tabular-nums)。
- 扇开 `--ease-brand-out` + stagger;收起 `--dur-exit`;reduced-motion 降级;press scale。

### Step 4：扇开列表虚拟化（>50）+ 进度聚合（§3.1 / §3.4）
- 扇开列表 >50 项虚拟化(复用 `@tanstack/react-virtual`,同 #3/#7 引擎)。
- 堆叠那叠显聚合进度("+M · 上传中 X/M"),不每个转圈。

### Step 5：i18n + a11y + reduced-motion
- 文案进 i18n(`t.inputBox.*`,对齐现有 `tooManyFiles`/`addAttachments`,**不硬编码中文**):堆叠 "+N"、扇开标题、删除 aria-label、聚合进度。
- 触摸目标 ≥44px;键盘全可达;扇开 focus trap + ESC;chip ✕ 有 aria-label。
- reduced-motion 全降级。

---

## 五、验收标准

### 功能
- [ ] 附件 ≤5 正常平铺(现状保留);>5 时**前 N 平铺 + 其余堆叠 "+M"**,平铺数不随总数涨。
- [ ] 上传几十/几百份**不再占满屏幕**(图一/图二问题消除)——堆叠后输入区高度恒定。
- [ ] **桌面 hover / 触屏点击**都能扇开那叠(双入口,`Hover vs Tap` 红线)。
- [ ] 扇开后可浏览全部剩余 + 每个单独删除(`remove(id)`);删到 ≤ 阈值自动收平铺。
- [ ] 扇开列表 >50 虚拟化,滚动流畅。
- [ ] 堆叠那叠显聚合进度,不每个转圈。
- [ ] **复用现有 `usePromptInputAttachments` store**——上传/删除/提交链路不变(替换前后行为一致)。

### a11y / 性能
- [ ] hover 非唯一入口(触屏 tap 等价);触摸目标 ≥44px。
- [ ] 键盘:堆叠 focus+Enter 展开、扇开 Tab 遍历、Delete 删、ESC 收。
- [ ] 扇开动效用 spec1 曲线 + stagger;reduced-motion 降级。
- [ ] 堆叠 "+M" tabular-nums;✕ 有 aria-label。

### 工程纪律
- [ ] **不手改 generated `ai-elements/prompt-input.tsx`**——只在 `input-box.tsx` 替用法 + 新建 `attachments/*`。
- [ ] **不重造上传 state**——全走 `usePromptInputAttachments`。
- [ ] 不改后端 `uploads.max_files` 上限逻辑(堆叠是显示层)。
- [ ] 复用 spec#1 曲线 / spec#6 阴影,不重定义。
- [ ] `pnpm check` 通过;i18n 不硬编码中文;零后端改动。

---

## 六、风险与回退

| 风险 | 缓解 |
|---|---|
| 改 generated 引 sync 冲突 | **铁律:不改 generated,自建组件替用法**(§3.5);只动 input-box 一段 + 新文件 |
| 替换破坏现有上传/删除链路 | 复用 `usePromptInputAttachments` store(§1.2),Step1 先静态替换验证行为一致再加交互 |
| hover-only 在触屏失效 | §3.2 双入口(tap 等价),`Hover vs Tap` High 红线纳入验收 |
| 扇开 hover 误触/抖动(进出频繁) | 离开延时关 + 进入阈值;桌面 hover 增强、触屏纯 tap |
| 海量扇开列表卡顿 | >50 虚拟化(同 #3/#7 引擎);堆叠本身不渲染全部(只 "+M") |
| Tailwind v4 自定义 duration 类编译不出(memory 新教训) | 扇开时长用 spec#1 已 `@utility` 定义的 `duration-*`,或直接 inline `style` var;**不假设 `duration-base` 自动生成**(见 memory `feedback_tailwind_v4_dur_namespace_not_real_duration_utility_dead_class`) |

**回退**:纯新增组件 + 替换一段用法;`git revert` 回到 generated `PromptInputAttachments` 平铺。store 不变,上传逻辑零风险。可灰度(阈值调大即近似关闭堆叠)。

---

## 七、给实施 agent 的交接

- **新建**:`components/workspace/attachments/{stacked-attachments,attachment-chip,attachment-stack}.tsx` + i18n 条目。
- **改一处**:`input-box.tsx:458-460` 把 `<PromptInputAttachments>` 段替换为 `<StackedAttachments />`(in-place,context 链不变)。
- **绝不碰**:generated `ai-elements/prompt-input.tsx`(只 import 它的 `usePromptInputAttachments`/类型,不改源);上传链路;后端上限。
- **复用**:`usePromptInputAttachments`(store)、`@tanstack/react-virtual`(同 #3/#7)、spec#1 曲线/时长(注意 memory:自定义 duration 类要 `@utility` 显式,别假设自动生成)、spec#6 `--shadow-overlap`+scale-feedback。
- **顺序**:Step1(静态替换,验行为一致)→ Step2(扇开+双入口)→ Step3(视觉+动效)→ Step4(虚拟化+进度聚合)→ Step5(i18n/a11y)。
- **本会话已定**:① 新建 spec#8(用户拍板)② 自建组件替用法,不改 generated(用户拍板)③ 堆叠+计数,hover(桌面)/点击(触屏)扇开(用户拍板)。阈值默认 5(用户指定)。
- **与其它 spec**:输入侧堆叠(本 spec)与产物侧画廊(#3)是同构"海量淹没"两面,各自组件;虚拟化引擎三处共用(#3 图墙/#7 消息流/#8 扇开列表)。

---

*依据:母方案硬伤 A 同源(输入侧海量淹没)+ 用户 2026-06-25 Seedance 堆叠诉求(图一~图四)+ 现状证据(prompt-input.tsx:405 flex-wrap 全量 map / usePromptInputAttachments store)+ `ui-ux-pro-max`(progressive-disclosure / Hover-vs-Tap High / scale-feedback / touch-target / stagger)。未写代码。*
