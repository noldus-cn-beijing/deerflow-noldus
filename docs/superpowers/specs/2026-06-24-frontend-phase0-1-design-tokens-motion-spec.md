# Spec：前端设计 token + 动效曲线（Phase 0 · 第 1 项）

> **✅ 状态：已实施并合并（PR#201，commit 4d847228）。本 spec 转为「验收/参考存档」，勿重做。** token 已全部落进 `globals.css`（`--ease-brand-*` / `--dur-*` / `@utility duration-*` / `--status-*` / `--stage-*` 均在线）。下文 §一/§三的实施前行号已 stale，仅作设计意图参考；§四已改为「对照已合代码的验收清单」。下游 spec（#2/#3/#5/#6/#7/#8）对本 spec 的依赖是「**复用已合 token**」，不是「等它先落」。
>
> 类型：**一次性实施 spec（已完成）**（前端纯 CSS/token 层）
> 日期：2026-06-24（实施合并 2026-06-25）
> 母方案：[docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)（§2 设计语言 / §7 动效）
> 适用层：`packages/agent/frontend/src/styles/globals.css` + 散落在 `components/` 的 ~38 处 transition/easing 调用点
> 设计准则来源：`ui-ux-pro-max` skill（domain `ux` Animation 类 + Quick Reference §7）
> 一句话：**把"动效曲线"和"语义色/时长"从散落的硬编码升级成一套 token**——核心是用非对称减速曲线替换全站的 `ease-in-out`/`ease-out`/`linear`（直接回应用户"渐出应逐渐变慢、不是 y=2x 匀速"的要求）。零后端依赖、零业务逻辑改动、可灰度。

---

## 〇、为什么这是 Phase 0 第一步

1. **性价比最高**：改的是 `@theme` 里的 token + 少量调用点,**全站立刻有质感提升**,风险接近零（纯样式,无逻辑、无数据、无后端）。
2. **是后面所有组件的地基**：产物画廊、运行轨迹、决策卡（Phase 0 后续 spec）都会用到这套曲线/时长/语义色 token。先立 token,后面组件直接引,不重复定义（SSOT）。
3. **直接交付用户的硬要求**：用户原话——"渐入渐出动画中移出频率应该是一个渐变曲线逐渐变慢，而不是 y=2x 的固定速率"。这条 spec 就是把这个要求**制度化成 token + 验收项**,让全站（含未来组件）都守住。

---

## 一、现状（调研所得，带证据）

### 1.1 现有 token 地基（好的，保留）

`globals.css` 已经相当用心,**不是从零开始**：

| 已有 | 位置 | 保留/调整 |
|---|---|---|
| oklch 色彩空间 + Forest Green `--primary:#1A4840` + Lime `--brand:#10DD8B` + 暖白纸底 | `:root` `globals.css:193-264` | **保留,色相不动**（决策3） |
| radius 阶梯 `--radius-{sm,md,lg,xl,2xl,3xl}` | `@theme inline:144-149` | 保留 |
| `--shadow-float` / `--shadow-modal` / `.glass-card` / `--elevated(-static)` | `:189-263/389-404` | 保留 |
| 三层质感（纸纹 `body::before` + 玻璃卡 + reduced-motion 降级） | `@layer base:339-377` | 保留,**作为日式质感的既有资产** |
| `tabular-nums`（table / token-usage 已用） | `:332-337` | 保留,Phase 0 **推广**（见 §4.4） |
| reduced-motion 全局降级 + per-animation 降级 | `:357-377/422-427` | 保留,**新 token 必须接入此机制** |
| `motion@12` 已装 | package.json | spring 曲线用它 |

### 1.2 真问题：动效曲线全是对称/匀速感（用户批评点）

`@theme` 里所有 `--animate-*` 用的都是浏览器默认的对称三次曲线或匀速：

| 动画 token | 当前曲线 | 证据 |
|---|---|---|
| `--animate-fade-in-up` | `ease-in-out` | `globals.css:83` |
| `--animate-skeleton-entrance` | `ease-out` | `:94` |
| `--animate-suggestion-in` | `ease-out` | `:100` |
| `--animate-wave` | `ease-in-out` | `:106` |
| `--animate-pulse-grid` / `pulse-soft` | `ease-in-out` | `:115/408` |
| 装饰类 `aurora`/`shine` | `linear`/`ease-in-out` | `:122/132` |

散落在组件里的同样是裸 `ease-out`/`ease-in-out`/`duration-300`（grep 出 **~38 处**,如 `chat-box.tsx:122 ease-in-out`、`todo-list.tsx ease-out`、`input-box.tsx:431`、`message-list-item.tsx:80`）。

**为什么这是"廉价感"来源**（`ui-ux-pro-max` domain ux）：
- `Easing Functions`（Result 2）：**"Linear motion feels robotic. Do: ease-out for entering / ease-in for exiting. Don't: linear."**
- `ease-in-out` 是对称曲线,加减速一样,缺"快起→缓停"的减速尾巴——视觉上"平",不高级。
- 用户要的"逐渐变慢的尾巴" = **强 ease-out**（减速段长）。标准 CSS `ease-out`（`cubic-bezier(0,0,0.58,1)`）减速不够明显;要更挺的 quint/expo 曲线。

### 1.3 一个 spec 必须知道的事实：`.dark` 已存在（但是旧值）

`globals.css:269-308` **已有一个 `.dark` 块**（旧 DeerFlow 值 + `@custom-variant dark` 已在 `:43` 注册）。**决策4「dark 先不做」依然成立**——本 spec **不碰 `.dark` 的视觉调校**,但新增 token 必须在 `.dark` 里也给值（即便沿用旧风格），否则将来切 dark 时 token 缺失会塌。原则：**本 spec 让 light 完美;dark 只保证不报错、token 齐全,留待 Phase 2 调校**。

---

## 二、目标与非目标

### 目标
1. 定义一套 **动效曲线 token**（`--ease-*`）+ **时长 token**（`--dur-*`），并映射成 Tailwind v4 可用的 `ease-*` / `duration-*` 工具类。
2. 用非对称减速曲线替换 `@theme` 内所有 `--animate-*` 的 easing,并退役装饰动画（aurora/shine 的引用方）。
3. 定义一套 **状态语义色 token**（`--color-status-{info,success,warning,danger}`）+ **工作流阶段色 token**（给 Phase 0 后续的进度轨/决策卡/质检卡用）。
4. 迁移 `components/` 里 ~38 处裸 easing 调用点到新工具类。
5. 全部接入既有 reduced-motion 降级机制 + 守 a11y 红线。

### 非目标
- ❌ 不改配色基调（决策3：Forest Green + Lime + 暖白纸不动色相）。
- ❌ 不做 dark mode 调校（决策4：Phase 2）。
- ❌ 不动任何组件的**结构/逻辑**——只换 className 里的 easing/duration 工具类 + 加 token。
- ❌ 不引入新动画库（用已装的 `motion@12` 做 spring,不新增依赖）。
- ❌ 不动流式核心 / `groupMessages` / 任何 `core/` 逻辑。

---

## 三、设计：token 体系（核心交付物）

> 所有曲线值给的是**起点**,实施时在真机上微调到"看着对"——但**方向锁定:enter 强减速、exit 快、绝不 linear**。

### 3.1 动效曲线 token（`@theme` 内新增）

```css
@theme {
  /* ── 动效曲线（asymmetric，日式减速尾巴）────────────────────
     命名对齐语义,不对齐数学:brand-out=入场/展开,brand-in=退场/收起 */

  /* 入场/展开：快起 → 长缓停（用户要的"逐渐变慢的尾巴"）。quint ease-out */
  --ease-brand-out: cubic-bezier(0.22, 1, 0.36, 1);

  /* 退场/收起：快、利落（exit 比 enter 短促，避免拖沓）。quint ease-in */
  --ease-brand-in: cubic-bezier(0.64, 0, 0.78, 0);

  /* 双向切换（展开↔收起 用同一条对称强曲线，比默认 ease-in-out 挺） */
  --ease-brand-in-out: cubic-bezier(0.83, 0, 0.17, 1);

  /* 强调/回弹微反馈（press、勾选）——轻微 overshoot，克制 */
  --ease-brand-emphasis: cubic-bezier(0.34, 1.56, 0.64, 1);
}
```

> Tailwind v4 机制：`@theme` 里定义 `--ease-brand-out` 会**自动生成** `ease-brand-out` 工具类（`transition-timing-function: var(--ease-brand-out)`）。组件里写 `className="... ease-brand-out"` 即可。同理 `--animate-*` 自动生成 `animate-*`。这是 v4 原生,无需配置文件。

### 3.2 时长 token（`@theme` 内新增）

依 `ui-ux-pro-max` `Duration Timing`（Result 5：**150-300ms 微交互,≤500ms 复杂,>500ms 禁用**）+ Quick Reference `exit-faster-than-enter`：

```css
@theme {
  --dur-fast:   140ms;   /* 微交互：press、hover、勾选 */
  --dur-base:   220ms;   /* 标准：卡片淡入、tooltip、tab 切换 */
  --dur-slow:   340ms;   /* 复杂：侧栏滑入、面板展开（仍 <500ms 上限）*/
  --dur-exit:   160ms;   /* 退场默认（≈base 的 70%，exit-faster-than-enter）*/
}
```

> ⚠️ **v4 的 `duration-*` 不从 `--dur-*` 自动派生**（与 `--ease-*`→`ease-*`、`--animate-*`→`animate-*` 不同——后两者是真 namespace，`--dur-*`/`--transition-duration-*` 不是）。命名 duration 工具类**必须在 `@theme` 块外用 `@utility` 显式定义**：`@utility duration-fast { transition-duration: var(--dur-fast); }`（已合代码 `globals.css:170-173` 即如此）。**别假设 `duration-base` 自动生成**——否则 className 里 `duration-base` 编译不出、过渡无时长瞬间完成（dead class，见 memory `feedback_tailwind_v4_dur_namespace_not_real_duration_utility_dead_class`）。组件迁移 `duration-300`→`duration-slow`、`duration-200`→`duration-base`（这些 @utility 已就位可直接用）。

### 3.3 状态语义色 token（`:root` + `.dark` 各一份）

为 Phase 0 后续（进度轨/决策卡/质检卡 = §2.2 母方案）预备,**与品牌绿同色系协调,降饱和,非荧光**：

```css
:root {
  /* 原始 token 用裸前缀（不带 --color-），避免 @theme inline 映射时自指循环 */
  --status-info:    oklch(0.62 0.10 230);   /* 沉静蓝 */
  --status-success: oklch(0.60 0.11 155);   /* 苔绿，与 brand 同 family */
  --status-warning: oklch(0.72 0.12  75);   /* 琥珀，降饱和（HITL 等待用）*/
  --status-danger:  oklch(0.58 0.16  25);   /* 朱，非纯红 */
  /* 各配一个极淡底色，给 banner/卡片背景 */
  --status-info-soft:    oklch(0.62 0.10 230 / 0.10);
  --status-success-soft: oklch(0.60 0.11 155 / 0.10);
  --status-warning-soft: oklch(0.72 0.12  75 / 0.12);
  --status-danger-soft:  oklch(0.58 0.16  25 / 0.10);
}
.dark { /* 同 hue，明度抬高一档保证 dark 下 4.5:1（值留 Phase 2 精调，先给不塌的占位）*/
  --status-info:    oklch(0.70 0.10 230);
  --status-success: oklch(0.68 0.11 155);
  --status-warning: oklch(0.78 0.12  75);
  --status-danger:  oklch(0.68 0.16  25);
  /* -soft 同步 */
}
```

并在 `@theme inline` 映射成工具类（**裸 token → `--color-*`，照现有 `--color-*: var(--*)` 模式，不自指**）：`@theme inline { --color-status-info: var(--status-info); --color-status-info-soft: var(--status-info-soft); … }` →生成 `text-status-info` / `bg-status-info-soft` 等。**已合代码 `globals.css:225+` 即此两层结构**（`:root` 裸 `--status-*` + `@theme inline` `--color-status-*: var(--status-*)`）。

> ⚠️ **对比度验收**：四个状态色 + 其 foreground 必须过 4.5:1（正文）/ 3:1（图标/大字）。light 必测;dark 值是占位,做 dark 时再验（决策4）。`color-not-only` 红线:状态**不能只靠颜色**,banner/卡片必须**色 + 图标 + 文字标签**三件套（现有 `QualityWarningBanner` 已是此模式,对齐它）。

### 3.4 工作流阶段色 token（给进度轨,Phase 0 后续 spec 用）

7 阶段各一个低饱和 hue,**全部由品牌绿出发做 hue 偏移**（保持家族感,不引入冲突色）：

```css
:root {
  /* 裸前缀（同 §3.3 理由），@theme inline 再映射成 --color-stage-* */
  --stage-upload:    oklch(0.62 0.06 200);
  --stage-paradigm:  oklch(0.60 0.08 175);
  --stage-align:     oklch(0.60 0.10 155);  /* ≈brand 家族 */
  --stage-compute:   oklch(0.62 0.09 140);
  --stage-qc:        oklch(0.66 0.10 110);
  --stage-interpret: oklch(0.64 0.10  85);
  --stage-report:    oklch(0.62 0.10  60);
}
/* @theme inline { --color-stage-upload: var(--stage-upload); … } 同 §3.3 两层模式 */
```

> 本 spec 只**定义**这组 token,进度轨组件在 Phase 0 spec #4 消费。先定义保证 SSOT（阶段色只存一份,进度轨/决策卡/画廊若都要标阶段,引同一 token）。

---

## 四、实施步骤

> 全部在 `frontend/`,非受保护文件。每步可独立提交、可灰度。

### Step 1：在 `@theme` 注入曲线 + 时长 token（§3.1 + §3.2）

`globals.css` 的 `@theme {}` 块（`:76`）内新增 4 个 `--ease-*` + 4 个 `--dur-*`。

### Step 2：替换 `@theme` 内 `--animate-*` 的 easing

逐条把保留的功能性动画换成新曲线（退役装饰类）：

| token | 改法 |
|---|---|
| `--animate-fade-in-up` | `ease-in-out` → `var(--ease-brand-out)`；时长 `0.15s`→`var(--dur-base)` |
| `--animate-skeleton-entrance` | `ease-out` → `var(--ease-brand-out)` |
| `--animate-suggestion-in` | `ease-out` → `var(--ease-brand-out)`；位移改从下方进（`hierarchy-motion`：进入从下 = 更深一层） |
| `--animate-wave` | 保留（招手是 delight，不动） |
| `--animate-pulse-soft` / `pulse-grid` | 保留曲线（呼吸用对称 ease-in-out 合理）；但见 §4.3 降噪 |
| `--animate-aurora` / `--animate-shine` | **退役**：见 §4.3 |

### Step 3：退役装饰动画（§4.3 `excessive-motion`）

`ui-ux-pro-max` `Excessive Motion`（Result 3，**Severity High**）：**"Animate 1-2 key elements per view max. Don't animate everything."** aurora/shine 是"科技感"装饰,正是母方案 §2.1 批评的"老土 AI 味"。
- grep 引用方：`ui/aurora-text.tsx`、`ui/shine-border.tsx`。
- 做法：**不删 `ui/` 源文件**（generated 组件,改了 sync 会冲突）,而是**移除业务侧对它们的引用**（landing / welcome 等若用了就换成静态）。`@theme` 里的 `--animate-aurora/shine` 可留（无引用即 dead,零成本）,或一并注释。
- 验收：grep 确认 `components/workspace` + `app/` 不再引 aurora-text / shine-border（`ui/` 内部保留）。

### Step 4：迁移组件调用点（~38 处）

grep `components/workspace` + `core` 的 `ease-*` / `duration-[0-9]+` / `transition`,逐处映射：

| 现有 | 迁移到 | 语义 |
|---|---|---|
| `ease-out`（入场/展开类，如 todo-list、input-box） | `ease-brand-out` | 入场强减速 |
| `ease-in-out`（如 chat-box:122/129 侧栏过渡） | 展开 `ease-brand-out` / 收起 `ease-brand-in`（分方向）；纯切换可 `ease-brand-in-out` | 区分进/出 |
| `linear`（若有 UI 过渡，shimmer 类例外） | `ease-brand-out` | 禁 linear（loading shimmer 的 linear 可保留——它是连续循环,linear 合理） |
| `duration-300` | `duration-slow` | 复杂过渡 |
| `duration-200` | `duration-base` | 标准 |
| `duration-150` | `duration-fast` | 微交互 |

**重点处理点**（grep 命中）：
- `chat-box.tsx:122/129`（artifacts 侧栏滑入）：用 `ease-brand-out` + `duration-slow`,且配 §4.2 的"从触发源展开"语义。
- `todo-list.tsx`（4 处 ease-out）：进度类入场,`ease-brand-out`。
- `input-box.tsx:431`（输入框 transition-all duration-300）：→ `duration-base`（输入框反馈要快,300 偏慢）。
- `message-list-item.tsx:80`、`message-group.tsx:143`：hover/toolbar 渐显,`duration-base ease-brand-out`。

> **不要** `transition-all`——只过渡 `transform`/`opacity`（`ui-ux-pro-max` `transform-performance`：不动 width/height/top/left）。迁移时顺手把 `transition-all` 收窄成 `transition-[transform,opacity]` 或 `transition-colors`,但**这属于优化,命中即改,不强求一次清完**（避免 scope 膨胀）。

### Step 5：推广 tabular-nums（§4.4）

母方案 §7.2 `number-tabular`：流式更新的数字（token 计数、p 值、计时、效应量）必须等宽防跳。现有只覆盖 `table` / `[data-slot="token-usage"]`（`:333-335`）。
- 加一个工具类语义点：流式数字组件（`message-token-usage.tsx`、`number-ticker.tsx`、未来统计表）显式加 `tabular-nums` class（已 `@source inline`,可直接用）。
- 本步**只加 class,不改逻辑**。

### Step 6：reduced-motion 接入验证

新增的曲线/动画**自动**被现有全局 `@media (prefers-reduced-motion: reduce)`（`:357-377`）覆盖（它降 `animation-duration`/`transition-duration` 到 0.01ms）。**验证点**：
- spring（motion 库）动画不走 CSS transition,**不被全局媒体查询覆盖**——用 motion 的 `useReducedMotion()` hook 在组件内降级。本 spec 凡引入 spring 处必须配 `useReducedMotion()`。
- 验收：开系统 reduced-motion,全站动效降级/关闭,无残留。

---

## 五、验收清单（对照已合代码 PR#201 核对，非重做）

### 功能/视觉
- [ ] `@theme` 含 `--ease-brand-{out,in,in-out,emphasis}` + `--dur-{fast,base,slow,exit}`；`ease-brand-*`/`animate-*` 自动生成可用，**`duration-*` 经 `@utility` 显式定义可用**（`globals.css:170-173`，非自动派生，见 §3.2）。
- [ ] `@theme` 内所有保留的 `--animate-*` 不再含 `ease-in-out`/裸 `ease-out`（wave/pulse 呼吸类除外,理由见 §4.2 表）。
- [ ] `components/workspace` + `app/` 不再引用 aurora-text / shine-border（`ui/` 内部源保留）。
- [ ] grep `components/workspace` + `core` 无裸 `ease-out`/`ease-in-out`/`linear`（shimmer 连续循环的 linear 例外,需注释说明）。
- [ ] **非对称曲线机读验收**（核心交付物，不靠肉眼）：① 生成的 `ease-brand-out` 工具类 `transition-timing-function` 精确 = `cubic-bezier(0.22,1,0.36,1)`；② 退场时长 `--dur-exit`(160ms) 严格 < 入场 `--dur-base`(220ms)。（"肉眼快起→缓停减速尾巴"仅设计说明，不作判据。）
- [ ] 状态色裸 token `--status-{info,success,warning,danger}(-soft)` + 阶段色 `--stage-*` 定义齐全（`:root` 与 `.dark` 都有），且 `@theme inline` 两层映射成 `--color-status-*`/`--color-stage-*`（非自指，见 §3.3）。
- [ ] **tabular-nums 推广验收**（对应 §四 Step5）：`message-token-usage.tsx` / `number-ticker.tsx`（及流式 token 计数/计时）DOM 带 `tabular-nums`（grep className 确认）；流式数字跳动时字符不横向位移。

### a11y / 性能（`ui-ux-pro-max` 红线）
- [ ] 系统开 reduced-motion → 全站动效降级/关闭,含 motion 库的 spring（`useReducedMotion()`）。
- [ ] 状态色 foreground/background 在 light 下过 4.5:1（danger/warning/success/info）;dark 占位不验（决策4）。
- [ ] 状态呈现"色 + 图标 + 文字"三件套,不靠颜色单独表意（`color-not-only`）。
- [ ] 动画只用 transform/opacity,无 width/height/top/left（命中即改）。
- [ ] 单视图同时入场的动画 ≤1-2 个关键元素（`excessive-motion`）。

### 工程纪律
- [ ] `pnpm check`（lint + tsc）通过。
- [ ] **不改任何 `core/` 逻辑、不改组件结构**——diff 应只含 `globals.css` + 组件 className 字符串 + 移除装饰引用。
- [ ] 视觉回归:`make dev` 起前端,人工过一遍主要页面（聊天流、artifacts 侧栏、welcome、todo、输入框）截图对比,确认无破样式。
- [ ] 不动 `.dark` 视觉值（只补新 token 占位）。

---

## 六、风险与回退

| 风险 | 缓解 |
|---|---|
| 曲线值在真机"看着不对" | 值是起点,§3 明确"真机微调",方向锁定即可;token 集中,调一处全站生效 |
| 迁移漏改某调用点 | grep 清单（§4.4）逐条勾;漏改的退化成旧 easing,不报错（渐进,非破坏） |
| 退役 aurora/shine 影响 landing 视觉 | 只移业务引用、留 `ui/` 源;landing 若强依赖,换静态渐变/留白(本就是去装饰目标) |
| Tailwind v4 没自动生成工具类 | v4 `@theme` 自动生成是文档行为;若某 key 没出,fallback 直接 `style={{transitionTimingFunction:'var(--ease-brand-out)'}}` 或 `@utility` 手写 |
| 改了共享 CSS 影响 `ui/`/`ai-elements/` generated 组件 | 只**新增** token + 改 `--animate-*` 既有值;不删 token,generated 组件引用的 token 仍在 |

**回退**：纯样式,`git revert` 即可;无数据/无后端/无状态迁移。

---

## 七、给实施 agent 的交接

- 改动文件：主要 `src/styles/globals.css`;次要 `components/workspace/**`、`app/**` 的 className 字符串 + 移除 aurora/shine 业务引用。
- **不碰**：`core/`、`components/ui/`、`components/ai-elements/`（generated）、`.dark` 视觉调校、任何流式/逻辑代码。
- 顺序：Step 1-2（token + animate）先提一个 commit（可独立验收）→ Step 3-4（退役装饰 + 迁移调用点）→ Step 5-6（tabular + reduced-motion 验证）。
- 真机微调曲线:`make dev` 后在浏览器 devtools 改 `--ease-brand-out` 实时看,定稿写回。
- 完成后这套 token 即 Phase 0 后续 spec（运行轨迹/画廊/进度轨/决策卡）的**唯一动效与语义色来源**,后续组件**只引不重定义**（SSOT）。

---

*依据：母方案 §2/§7 + `ui-ux-pro-max` domain ux Animation 类（Easing/Duration/ExcessiveMotion/ReducedMotion）+ Quick Reference §7。本 spec 未写代码,留实施 agent 执行 + 真机微调。*
