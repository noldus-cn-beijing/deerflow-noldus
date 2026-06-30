# DESIGN.md — EthoInsight 视觉语言 SSOT

> **这是 EthoInsight 前端的视觉语言唯一事实源（Single Source of Truth）。**
> 所有色板、排版、间距、景深、动效的角色与取值定义在 [`src/styles/globals.css`](src/styles/globals.css) 的 CSS 变量里；本文档是它们的**语义说明 + 使用纪律**，不另立第二份设计标准（守 single-source-of-truth）。
>
> 格式借鉴 [awesome-design-md](https://github.com/voltagent/awesome-design-md) 的 Google Stitch 9 段结构——**借结构与克制理念，不照搬任何品牌身份**。
>
> 适用对象：人设计师 + 所有 coding agent。后续 D2（组件库重构）/ D3（a11y + 响应式）/ C2（画廊 layout）都必须对齐本文档。

---

## 0. 一句话重心

**对话输出为主，hard fact 产物为辅。** 核心是和 agent 对话；图表 / 报告 / 指标表是对话流的有机产出，辅以右侧 / 内联承载。设计语言围绕「对话流的优雅 + 产物的清晰承载」，**不做纯工作台 / BI dashboard**。视觉永远给数据与对话让路。

---

## 1. 视觉氛围（Atmosphere）

**日式简洁**——留白（Ma 間）、降饱和、克制（Shibui 渋味）。安静、不抢戏，让数据与对话是主角。

- **纸感底**：暖白纸色背景（`--background`，oklch 暖白）+ 极轻纸纹（`body::before` 分形噪声，opacity 0.038）。不是冷白/纯白，是「研究员桌面上的纸」。
- **克制用色**：60% 中性灰 / 纸色 + 30% 主数据色（Forest Green 家族）+ 10% 单一强调色。**绝不**用饱和色铺面积；饱和色只点状态与唯一强调。
- **柔而非硬**：边框用半透暖灰（`oklch(0.15 0.01 92 / 0.10)`），不用纯黑实线；分隔靠**留白与升档**，不靠 border 套 border。
- **三层质感**：纸纹底（Layer 0）→ 玻璃卡片（Layer 1，`backdrop-blur`）→ 实白浮层（Layer 2，popover/modal）。z 关系靠**质感 + 阴影差**表达，不靠重描边。
- **安静动效**：入场用减速尾巴（`--ease-brand-out`，快起长缓停），退场利落（`--ease-brand-in`）。绝不用 linear / 对称 ease-in-out 做位移——那是「廉价感」来源。reduced-motion 全局降级。

---

## 2. 色板 + 角色（Color & Roles）

**60 / 30 / 10 分布**：中性为主、数据色为体、强调色为点。明暗双主题（复用 `.dark` token 块）。

### 角色 token（全部定义在 `globals.css` `:root` / `.dark`）

| 角色 | token | light | 用途 | 纪律 |
|---|---|---|---|---|
| **背景底** | `--background` | 暖白纸 | 整页底色 | 不铺饱和色 |
| **前景文字** | `--foreground` | Forest `#1A4840` | 正文 | — |
| **卡片** | `--card` / `--elevated` | 玻璃白 | 内容容器 | 配 backdrop-blur |
| **主色（数据/品牌）** | `--primary` | Forest `#1A4840` | 默认按钮、主数据序列、强调文字 | 占 30% |
| **品牌色（CTA）** | `--brand` | Lime `#10DD8B` | 主 CTA、进度轨活跃 | 点缀，不大面积 |
| **次要** | `--secondary` / `--muted` | 暖奶油 / 弱化底 | 次级容器、占位 | — |
| **次级文字** | `--muted-foreground` | 暖石灰 `#78716C` | 标签、说明、时间戳 | 不用于关键信息 |
| **选中态** | `--accent` | 极淡 Forest 透 | hover/selected 底 | 极淡，不抢戏 |
| **单一强调色** | `--accent-strong` | 蓝 `#0072B2` | **唯一**的高亮/关键数据点强调 | 见下方「强调色铁律」 |
| **边框** | `--border` / `--input` | 半透暖灰 | 分隔、输入框 | 不用纯黑实线 |
| **focus ring** | `--ring` | Forest 半透 | 键盘焦点 | 必须可见（a11y） |

### 状态语义色（仅用于状态，色盲安全）

状态色 **必须「色 + 图标 + 文字」三件套**（color-not-only），绝不只靠颜色表意：

| token | light | 语义 |
|---|---|---|
| `--status-info` | 沉静蓝 | 信息提示 |
| `--status-success` | 苔绿（brand 同 family） | 成功 / 完成 |
| `--status-warning` | 降饱和琥珀 | HITL 等待 / 需注意 |
| `--status-danger` | 朱（非纯红） | 错误 / 危险 |

每个状态色配一个 `*-soft` 极淡底色，给 banner / 卡片背景用。

### 数据可视化分类色（`--chart-1`..`--chart-5`）

Lime 第一序列 + Forest 第二 + 中性补充。**分类对比用色盲安全序列**，绝不靠红/绿区分组（见 §7）。

### 🚨 强调色铁律（colorblind-safe，写进 D2/D3 验收）

- **全站只有一个「强调色」`--accent-strong` = 蓝 `#0072B2`**（Wong 色盲安全板）。用于：唯一需要拉出的关键数据点、当前焦点、重要 callout。
- **绝不用红/绿对编码「正/负」「升/降」「显著/不显著」**——红绿色盲（最常见的 8% 男性）无法区分。统计显著性用**图标 + 文字 + 透明度**，不用红/绿。
- 退路强调色（若蓝与上下文撞）：橙 `#E69F00`（同 Wong 板）。**同一屏只出现一种强调色。**

---

## 3. 排版（Typography）

**层级靠字号 + 留白，不靠粗体轰炸。** 字重克制：标题 600、正文 400、标签 500。

### 字重 token（`globals.css` `@theme inline`）

| token | 值 | 用途 |
|---|---|---|
| `--font-weight-regular` | 400 | 正文（默认） |
| `--font-weight-medium` | 500 | 标签、按钮、次级标题 |
| `--font-weight-semibold` | 600 | 表头、区块标题 |
| `--font-weight-bold` | 700 | 极少——仅强强调 |

> Tailwind v4 从 `--font-weight-*` 生成的工具类名是 **`font-{weight}`**（`font-medium` / `font-semibold`），不是 `font-weight-medium`。直接用 `font-medium`/`font-semibold`。

### 字号 token（type scale，`globals.css` `@theme inline`）

| token | px | 用途 |
|---|---|---|
| `--text-xs` | 12px | 标签、时间戳、caption |
| `--text-sm` | 14px | 次级正文、表单 |
| `--text-base` | 16px | 正文（readable-font-size） |
| `--text-lg` | 18px | 卡片标题 |
| `--text-xl` | 24px | 区块标题 |
| `--text-2xl` | 32px | 页面标题（极少） |

### 数字

- 数值（token 计数、p 值、计时、效应量、统计表）必须 `tabular-nums`（`font-variant-numeric: tabular-nums`）防跳动。已对 `table` / `[data-slot="token-usage"]` 全局生效；流式数字组件显式加 `tabular-nums` class。
- **不全大写**（no ALL CAPS）——日式克制，全大写只用于极少的标签徽章且需谨慎。
- 中文段落自动加大行高（`:lang(zh) p { line-height: 1.7 }`）。

### 字体栈

`--font-sans`: OPPO Sans 4.0（latin + zh 分包，`font-display: swap`）→ system-ui 兜底。**选任何新字体前先验证支持 tabular-nums**（OPPOSans 已验证）。

---

## 4. 组件样式（Component States）

克制——状态变化要**可感知但不刺眼**。

### 按钮（参照 `ui/button.tsx` variants）

| variant | 用途 | 状态规范 |
|---|---|---|
| `default` | 主操作 | `bg-primary` → hover `bg-primary/90`；focus-visible `ring-ring/50` ring 3px |
| `brand` | 主 CTA（分析/生成） | `bg-brand` → hover `bg-brand-hover`；ring `ring-brand/40` |
| `secondary` / `outline` | 次操作 | hover `bg-accent`；不发亮 |
| `ghost` | 工具栏/图标按钮 | hover `bg-accent`，默认透明 |
| `destructive` | 删除/危险 | `bg-destructive`；**必须配确认** |
| `link` | 内联链接 | `text-primary underline-offset-4 hover:underline` |

**统一纪律**：所有交互态过渡用 `transition-all` + `--dur-fast`(140ms) / `--dur-base`(220ms)；focus 必须有可见 ring（a11y，非 `outline:none` 后不补）；disabled 用 `opacity-50 pointer-events-none`，不变灰换色。

### 卡片

- 玻璃卡（`.glass-card`）：`bg-elevated` + `backdrop-blur(16px)` + `border` + 按层阶选阴影（rest / raised / overlap / overlay）。
- 重叠卡片靠**升档阴影**表达 z，不靠 border 套 border。
- reduced-motion：玻璃态降级为 `--elevated-static`（近实白、无 blur）。

### 输入

`--input` 半透暖灰边框；focus 切 `--ring`；aria-invalid 切 `--destructive`。悬浮输入框用 `.shadow-float`。

### 导航

侧栏（`--sidebar` 绿调羊皮纸）+ 顶栏；活跃项用 `--accent` 极淡底 + `--primary` 文字，不用强色块。

---

## 5. 布局原则（Layout）

**对话为主、产物为辅。**

- **对话流是中轴**：消息流左对齐贯穿，产物（图/报告/表）内联或右侧承载，绝不打断对话节奏。
- **留白为分隔**：区块间垂直留白三档（组件内 16 / 组件间 24 / 区块间 32），用距离分组，**不靠全网格线**。
- **亲密性**：关联强的元素间距小（≤8），关联弱的间距大（≥24）——距离即分组。
- **隐形垂直基准线**：同列左边缘对齐、数字右对齐、图标与文字基线对齐——视线顺流。
- **信息层级**：一屏一主角（用户原话「一屏一主角」）；次要信息折叠/降级到 `--muted-foreground`，不堆叠过载。
- **响应式骨架归 D3**，本节只立原则（桌面为主，平板/手机降级承载，不删功能）。

---

## 6. 景深 / Elevation（克制阴影）

**柔、薄、多层，绝不硬黑边/重投影。** 每档 = 1-2 层柔影叠加。

| token | 用途 |
|---|---|
| `--shadow-rest` | 静止卡片（默认） |
| `--shadow-raised` | 悬浮 / hover 抬升的卡片 |
| `--shadow-overlap` | 叠在别的卡片之上 |
| `--shadow-overlay` | 画廊 lightbox / 浮层抽屉 |
| `--shadow-modal` | Modal 最高层 |
| `--shadow-float` | 悬浮输入框（更柔更扩散） |

纪律：升一档 = 换一档阴影，**绝不** `border: 2px solid black` 表达层级。dark 下阴影加深（值已在 `.dark` 给）。

---

## 7. Do / Don't（禁忌清单）

**Do**：
- ✅ 视觉语言**只通过改 token**（`globals.css`）注入——一处定义、全站（含上游组件）跟随，sync 零冲突。
- ✅ 状态色用「色 + 图标 + 文字」三件套。
- ✅ 数值用 `tabular-nums`。
- ✅ 入场减速尾巴、退场利落。
- ✅ 重叠靠升档阴影，不靠 border 套 border。
- ✅ 新结构性 UI 在 `workspace/` 做（Noldus 独占）。

**Don't**（写进 review checklist）：
- ❌ **不暴露内脏**：不把 agent 中间状态/技术栈/stack trace 直接抛给研究员。
- ❌ **不平铺过载**：不把所有产物一次性铺满屏——一屏一主角，其余折叠。
- ❌ **不红/绿对编码语义**（正负/升降/显著与否）——色盲不安全。用图标+文字+透明度。
- ❌ **不全网格线**：分隔靠留白与升档，不用满屏 border。
- ❌ **不双高亮叠加**：同一元素不同时上 brand 色块 + accent-strong + 状态色——只保留一个视觉重点。
- ❌ **不照抄品牌身份**：只借结构与日式克制理念。
- ❌ **不直接改 `ai-elements/` / registry 组件结构**（re-pull 覆盖）——要改在 `workspace/` 包 wrapper，或通过 token 让其跟随。
- ❌ **不破坏 `ui/` copy-in 组件的导出/API**（守 CLAUDE.md copy-in vs registry 规则）——只改 token。

---

## 8. 响应式（Responsive）

> 细节实现归 D3（a11y + 多设备）。本节只立断点原则。

- **桌面为主**（≥1280px）：对话流 + 右侧产物 rail 并存，本产品的主力形态。
- **平板（768-1279px）**：产物 rail 折叠为抽屉/Tab，对话流占满。
- **手机（<768px）**：单列，产物内联在对话流中，底部输入栏固定。
- **降级不删功能**：所有功能在小屏可达，只是承载方式改变。
- **键盘 / 焦点 / 对比度 / 语义** 归 D3（本档立 token 时已预留：`--ring` focus token、状态色三件套、reduced-motion 降级均已就位）。

---

## 9. Agent 提示（给 coding agent 的生成指引）

**当你要生成或修改 EthoInsight 前端 UI 时，按此对齐：**

1. **先读 token，再写样式**：色/间距/字重/阴影/动效**全部从 `globals.css` 的 CSS 变量取**（`bg-primary` / `text-muted-foreground` / `shadow-raised` / `duration-base` / `ease-brand-out`），**绝不硬编码十六进制或裸 `ease-in-out`/`duration-300`**。若缺 token，先在 `globals.css` 补（含 `.dark` 值），再用。
2. **视觉改动只走 token + workspace 结构**：要改上游组件（`ai-elements/` / `ui/`）的**外观**，改 token 让它跟随；要改**结构/交互**，在 `workspace/` 包 wrapper。绝不直接编辑 `ai-elements/` / registry 源码（re-pull 覆盖），绝不破坏 `ui/` copy-in 的导出/API。
3. **守 SSOT**：本 DESIGN.md 是唯一设计标准。D2/D3/C2 的产出都对齐它，不另立第二份色板/字号/阴影表。token 定义在 `globals.css`，本文档只做语义说明。
4. **守色盲安全**：强调只用 `--accent-strong`（蓝）；正/负、显/著与否用图标+文字+透明度，绝不用红/绿对。
5. **守动效纪律**：入场 `--ease-brand-out`、退场 `--ease-brand-in`、duration 用 `duration-fast/base/slow/exit`（`@utility` 显式定义，**不**靠 `--dur-*` 当 Tailwind namespace——它是死类）。reduced-motion 必须降级。
6. **守 Tailwind v4 三坑**（memory `feedback_tailwind_v4_dur_namespace_not_real_duration_utility_dead_class`）：
   - `--ease-*` 是真 namespace（自动生成 `ease-brand-*` 工具类）；
   - `--dur-*` **不是** namespace（`duration-*` 工具类不从它派生），自定义 duration 必须 `@utility` 显式定义；
   - 任何新自定义工具类必须 **PostCSS 真编译坐实生成**（写探针：去掉该类目标行为应失效），否则是死类。
7. **生成后自检**：跑 `pnpm check`（0 error）+ `pnpm test`（不回归）+ 改一个 token 验证全站跟随（坐实 token 注入生效，非硬编码）。

---

## 附：token → 工具类映射速查

| 裸 token（`:root`/`.dark`） | Tailwind 工具类前缀（`@theme inline` 映射） |
|---|---|
| `--background` / `--foreground` / `--card` / `--primary` / `--brand` / `--accent` … | `bg-*` / `text-*` / `border-*` |
| `--status-*` / `--status-*-soft` | `bg-status-*` / `text-status-*` |
| `--accent-strong` | `bg-accent-strong` / `text-accent-strong` |
| `--stage-*` | `bg-stage-*` |
| `--radius-*` | `rounded-*` |
| `--shadow-rest/raised/overlap/overlay` | `.shadow-rest` 等（`@layer base` 工具类） |
| `--ease-brand-*` | `ease-brand-*`（真 namespace，自动生成） |
| `--font-weight-*` | `font-medium` / `font-semibold` 等（真 namespace；类名是 `font-{weight}` 非 `font-weight-{weight}`） |
| `--dur-*` | `duration-fast/base/slow/exit`（`@utility` 显式定义，**非** namespace） |
| `--size-icon-sm/md/lg` | `size-icon-*` |

> 改动 token 时同步更新本表与对应章节，保持 SSOT 不漂移。
