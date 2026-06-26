# Spec：设计语言 · 细节工艺 + 符号学 + 插画美术方向（Phase 0 · 第 6 项）

> 类型：**设计语言规范 spec**（前端样式/资产层为主，定义"好"的可验收标准；少量 token 落 globals.css，其余是规则与纪律）
> 日期：2026-06-25
> 母方案：[docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)（§2 设计语言 / §7 动效与细节）
> 依赖：[2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md](2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md)（曲线/时长/语义色/阶段色/radius token —— **本 spec 不重定义，只在其上补 elevation 阶梯 + 一个降饱和暖中性 accent**）
> 适用层：`packages/agent/frontend/src/styles/globals.css`（elevation token）+ `components/ui/empty.tsx` / 图标层 / 插画资产层 + 各组件的尺寸/层级纪律（验收项，不是新组件）
> 设计准则来源：`ui-ux-pro-max`（`elevation-consistent` / `Stable Interaction States` / `scale-feedback` / `state-transition` / `spring-physics` / `whitespace-balance` / `visual-hierarchy` / `primary-action` / `color-not-only`）；Ant Design 4 原则 + Alibaba Fusion 价值观（普适交互法则）；Peirce icon/index/symbol + Charles Morris 句法/语义/语用（符号系统治理）；Mœbius/ligne claire（插画美术语法）
> 一句话：把母方案 §2/§7 的"日式克制 + 真设计感"从口号**落成四组可验收的硬规则**——① **细节工艺**（重叠 card 的 elevation 阶梯 + 边角 radius 纪律 + hover/press 不抖布局）② **尺寸/比例/层级**（一屏一主角，靠尺寸+留白+对比建层级，不堆色不堆边框）③ **符号系统**（Peirce 三分 + Morris 三轴的 8 条硬规则 + label-off 验收门，锚定 lucide）④ **插画美术方向**（Mœbius **只借语法**：ligne claire 线型 + 平涂无渐变 + 负空间，**硬拒**他的青橙饱和 palette，只取灰化沙漠色温做一个降饱和 accent；只活在空状态 + 图标层，绝不进数据/chrome）。

---

## 〇、为什么需要这份 spec（与 spec#1 的分工）

spec#1 定的是**动效曲线 + 语义色 + radius/时长 token**（"用什么值"）。但用户在本会话补了更细的工艺要求，spec#1 不覆盖：

> 用户原话（2026-06-25）：
> - "重叠 card 的时候需要有阴影"
> - "边边角角应该稍微有点弧度的这种非常高级的设计感，不要沦落俗套"
> - "整体设计风格/图形学风格/各种符号的构建，可以参考莫里斯符号学"
> - "美术风格方面，你觉得莫比斯风格如何？"
> - "各种不同的元素在整体交互画面的比例、尺寸，也是需要考虑的"

这五条分别落在：**深度/阴影工艺**（§1）、**尺寸/比例/层级**（§2）、**符号构建**（§3）、**插画美术风格**（§4）。spec#1 是 token 地基，**spec#6 是地基之上的"品味与构图纪律"**——两者 SSOT 分明：曲线/时长/语义色/radius 在 spec#1，elevation 阶梯 + 暖中性 accent + 符号规则 + 插画公约在 spec#6。

> **守 CLAUDE.md「同一份知识绝不双存」**：本 spec **不重定义** spec#1 已有的任何 token（radius/ease/dur/status/stage）。新增的只有 ①一组 `--shadow-*` elevation 阶梯（现状只有 float/modal 两档，缺中间档）②一个降饱和暖中性 accent（补足纯绿系，Mœbius 灰沙漠色温）。引用 spec#1 token 一律 `var(--ease-brand-out)` 等，不抄值。

---

## 一、细节工艺：重叠 card 的 elevation 阶梯 + 边角 radius 纪律 + 不抖布局

> 直接回应"重叠 card 要有阴影""边角要有高级感弧度，别俗套"。`ui-ux-pro-max` 铁律：`elevation-consistent`（卡片/抽屉/模态用**一致的** elevation 阶梯，**禁止随手写阴影值**）+ `Stable Interaction States`（press/hover 用 color/opacity/elevation 过渡，**绝不改 layout bounds / 抖动周围元素**）。

### 1.1 现状（带证据）
`globals.css` 现有阴影只有**两档**，中间断层：
- `--shadow-float`（`:261`，`0 1px 2px` 极淡）—— 静止卡片。
- `--shadow-modal`（`:258`，`0 16px 48px` 重）—— 模态。
- 中间没有"卡片悬浮/抽屉/重叠抬升"的档 → 现状重叠卡片要么没阴影（扁平、分不清前后），要么直接跳到 modal 级（过重）。

### 1.2 elevation 阶梯 token（补 spec#1 没有的中间档）

在 `globals.css` `:root` 补一套**语义命名、多层柔和**的阴影（`ui-ux-pro-max` Soft UI Evolution：多层 `0 2px 4px` 柔影，比扁平柔、比新拟物清晰）：

```css
:root {
  /* ── elevation 阶梯（日式：柔、薄、多层，绝不硬黑边）──
     语义命名对齐用途，不对齐数字。每档 = 1-2 层柔影叠加。 */
  --shadow-rest:    0 1px 2px rgba(0,0,0,0.04);                                   /* = 现 --shadow-float，静止卡片 */
  --shadow-raised:  0 2px 4px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.04);       /* 悬浮/被 hover 抬升的卡片 */
  --shadow-overlap: 0 6px 16px -4px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04); /* 重叠在别的卡片之上（用户的"重叠 card"）*/
  --shadow-overlay: 0 12px 32px -8px rgba(0,0,0,0.10), 0 2px 6px rgba(0,0,0,0.04);/* 画廊 lightbox / 浮层抽屉 */
  --shadow-modal:   0 16px 48px -12px rgba(0,0,0,0.10);                           /* 保留现值，模态 */
}
.dark { /* 占位，dark 推迟（决策4）；阴影在暗色下加深，给不塌的值，Phase 2 精调 */
  --shadow-rest:    0 1px 2px rgba(0,0,0,0.3);
  --shadow-raised:  0 2px 4px rgba(0,0,0,0.35), 0 1px 2px rgba(0,0,0,0.3);
  --shadow-overlap: 0 6px 16px -4px rgba(0,0,0,0.5), 0 2px 4px rgba(0,0,0,0.3);
  --shadow-overlay: 0 12px 32px -8px rgba(0,0,0,0.55), 0 2px 6px rgba(0,0,0,0.3);
  --shadow-modal:   0 16px 48px -12px rgba(0,0,0,0.6);
}
```

`@layer base` 里补工具类（照现有 `.shadow-float`/`.shadow-modal` 模式，`:397-403`）：`.shadow-rest` / `.shadow-raised` / `.shadow-overlap` / `.shadow-overlay`。

> **`--shadow-float` 不删**（现有组件引用），让 `--shadow-rest` 同值并存；新组件用语义名 `rest/raised/overlap`，旧 `float` 渐进迁移（不强求一次清完）。

### 1.3 重叠 card 的深度规则（用户的核心诉求）

**当两个 card 在视觉上重叠/层叠时，靠上的一张升一档 elevation**——用阴影差表达 z 关系，不用边框堆叠：

| 场景 | 下层 | 上层 | 表达 |
|---|---|---|---|
| 卡片网格里 hover 某张 | `shadow-rest` | hover 张 → `shadow-raised` + `scale(1.02)` | "这张浮起来了" |
| 卡片叠卡片（如决策卡盖在消息流上、画廊缩略图叠在分组头上） | `shadow-rest` | 叠上去的 → `shadow-overlap` | "这是另一层" |
| lightbox / 浮层抽屉盖全屏 | 背景 scrim | 浮层 → `shadow-overlay` | "悬浮于内容之上" |

**铁律**（`ui-ux-pro-max`）：
- **一个层级一种区隔手段**（母方案 §2.1）：要么 elevation（阴影），要么底色，要么一条 1px 细线——**三者不叠加**。重叠优先用**阴影**（最日式、最干净），不靠 `border` 套 `border`。
- **阴影只表达深度，不做装饰**：禁止彩色阴影、禁止 glow（现有 brand glow `:413-425` 仅用于 focus ring，不外溢成装饰）。

### 1.4 边角 radius 纪律（"高级感弧度，别俗套"）

radius 阶梯 spec#1 已有（`--radius-{sm:4,md:8,lg:12,xl:20,2xl:28,3xl:40}`，`globals.css:144-149`），**本 spec 只定使用纪律**：

- **按容器尺寸选档，不随手**：小元件（按钮/输入/chip/小 chip）`md(8)`；卡片 `lg(12)`；大容器/画廊面板/模态 `xl(20)`~`2xl(28)`；整页大区块 `3xl(40)`。**同类元素必用同档**（`ui-ux-pro-max` consistency）。
- **嵌套时外大内小、差≥4px**：外卡 `lg(12)`、内部子卡 `md(8)`——同径嵌套会"贴边硌眼"。但**层级≤2**（母方案 §2.1 反对边框套边框，radius 嵌套同理别超两层）。
- **"高级感"的来源不是大圆角**：是**克制 + 一致 + 与阴影/留白配合**。俗套 = 到处 `rounded-2xl` + 重阴影 + 高饱和（典型"老土 AI 卡片"）。高级 = 中等弧度（卡片 12px）+ 柔薄阴影 + 大留白 + 低饱和。**禁止全站无脑 `rounded-3xl`**。
- **直角的克制使用**：数据表、代码块、紧贴边缘的满宽区可用 `sm(4)` 近直角（理性、信息密度感），不必处处圆。

### 1.5 hover / press 不抖布局（`Stable Interaction States` 红线）

- 可点卡片/按钮 hover：`shadow-rest → shadow-raised` + `scale(1.01~1.02)`；press：`scale(0.97~0.98)`（`scale-feedback`）。曲线 spec#1 `--ease-brand-out`，时长 `--dur-fast`。
- **绝不位移布局**：scale 用 `transform`（不触发 reflow）；阴影变化不改尺寸。**周围元素纹丝不动**（`ui-ux-pro-max`「Layout-shifting transforms 禁止」）。
- 进行中态用 spec#1 `animate-pulse-soft`（brand 呼吸），不是空转 spinner。
- 全部 reduced-motion 降级（spec#1 机制）。

> ⚠️ **本 spec 所有特效（阴影/hover/扇开/脉冲/玻璃）必须过 [spec#7 §3.7「视觉特效的高效实现」硬闸门](2026-06-24-frontend-phase0-7-runtime-performance-spec.md)**：**保视觉、省开销**——动画只 transform/opacity（同视觉不重排）、特效组件 memo+稳定 props（不被无关 render 带着重建）、box-shadow 用本 spec 五档不在 transition 里 animate 模糊半径、will-change 不常驻、同质重复层在肉眼无差时合并。**不是限制做什么视觉,是规范怎么高效实现**（同一个玻璃/阴影/动效,笨重写法卡、高效写法丝滑）。SSOT 在 spec#7 §3.7,本 spec 不重复、只引用——所有视觉 spec（#1/#3/#5/#6/#8）共用那一份。用户原话:"简单有效不是视觉降级,是别写繁琐造额外开销的代码。"

---

## 二、尺寸 / 比例 / 层级：一屏一主角，靠尺寸+留白+对比建层级

> 直接回应"各种元素在整体画面里的比例、尺寸要考虑"。`ui-ux-pro-max`：`visual-hierarchy`（**靠尺寸/间距/对比建层级，不靠颜色单独**）+ `primary-action`（一屏一个主 CTA，次要的视觉降级）+ `whitespace-balance`（用留白分组，不用线堆）+ Ant Design「对比/亲密性/对齐」。

### 2.1 一屏一主角（altitude 纪律）
每个界面状态**只有一个视觉主角**，其余降为背景层（母方案 §2.1「一屏一主角」）：

| 状态 | 主角（放大/高对比/居中焦点） | 配角（缩小/降饱和/退背景） |
|---|---|---|
| 分析进行中 | 当前活跃 subagent / 进度轨当前阶段 | 已完成步骤、侧栏 |
| 等待 HITL | 决策卡（spec#5，accent bar + 居中） | 历史消息、输入框 |
| 看产物 | 选中的图（lightbox 全屏） | 缩略图网格 |
| 画廊浏览 | aggregate 代表图分组 | per-subject 折叠分组 |

**判据**：截一张图，眯眼看——**第一眼落点必须是当前任务的主角**。若有两个东西在抢视线（两个高饱和块、两个大 CTA），就是层级失败。

### 2.2 尺寸阶梯（比例靠系统，不靠手感）

- **字号**：沿用 spec#1/现有 type scale（`12 14 16 18 24 32`），正文 16px（`readable-font-size`），标题靠**字号 + 留白**拉层级，**不靠粗体轰炸**（母方案 §2.1，`weight-hierarchy` 但克制：标题 600，正文 400，标签 500）。
- **间距**：4/8px 节奏（`spacing-scale`），区块间垂直留白分三档（`16 / 24 / 32`，对应组件内/组件间/区块间，Fusion「section spacing hierarchy」）。**亲密性**（Ant）：关联强的元素间距小（≤8），关联弱的间距大（≥24）——距离即分组，省掉分隔线。
- **图标尺寸 token**（`ui-ux-pro-max` Consistent Icon Sizing）：定 `icon-sm(16) / icon-md(20) / icon-lg(24)`，**禁止 18/22/26 随手值**。同一视觉层同一尺寸。
- **元素相对比例**：主 CTA 比次按钮**大一档**（高度/padding），不靠仅颜色区分（`primary-action` + `费茨法则`：主操作越大越易点）。代表图 inline ≤6 张时每张占位适中（不撑满），画廊缩略图统一 `aspect-ratio` 网格（整齐 = 高级）。

### 2.3 对齐与重复（Ant 原则，省心智）
- **对齐**（格式塔连续律）：同列左边缘对齐、数字右对齐（`number-tabular`）、图标与文字基线对齐（`ui-ux-pro-max` Icon Alignment）。**一条隐形垂直基准线**贯穿，视线顺流。
- **重复**：相同元素全站重复（卡片样式、状态徽章、阶段色 spec#1），降学习成本 + 建关联（Ant 重复原则 = `ui-ux-pro-max` consistency）。

---

## 三、符号系统：Peirce 三分 + Morris 三轴的 8 条硬规则

> 回应"参考莫里斯符号学构建符号"。**先澄清歧义**（调研坐实）：

> **"莫里斯符号学" = Charles W. Morris**（句法 syntactics / 语义 semantics / 语用 pragmatics 三轴，1938《符号理论基础》）。UI 符号系统真正最常用的其实是 **Peirce 的 icon/index/symbol 三分**。两者**叠用**才完整：**Peirce 决定"每个符号是哪一类"，Morris 决定"整套符号如何治理一致"**。**与 William Morris（工艺美术纹样）无关**——后者是装饰/纹理另一根杠杆，本 spec 不用。

> **scope 纪律（守 CLAUDE.md 反 under-exploration + v0.1 务实）**：v0.1 **只取下面 8 条可执行硬规则当一页清单，不建学术脚手架、不写 Morris/Peirce 教科书式文档**。符号学买的是"**模糊符号该长什么样、该不该配 legend、该不该 stateful 的决策程序**"——超出"用 lucide 就行"的那部分价值，到此为止。锚定 `lucide-react`（`package.json:67`），不自绘 lucide 已有的。

### 3.1 八条硬规则

**句法轴（syntactics，符号↔符号，让图标像一家人）**
1. **一套图元原语**：统一 stroke width（lucide 默认 2px @24，全站锁定不混粗细）、统一 grid（24 box）、统一 radius 端点风格、optical-size 断点（16/20/24，即 §2.2 的 icon token）。**任何新图标必须能用同一套原语重建**——拒绝一次性异类图标。
2. **图标家族唯一**：只用 lucide 一套（`icon-style-consistent`）；filled vs outline **不在同一层级混用**（`ui-ux-pro-max` Filled vs Outline Discipline）。`@radix-ui/react-icons` 已装但**仅 radix 组件内部用**，业务图标统一 lucide，不混搭两套语言。

**语义轴（semantics，符号↔指称，图标真的指代那个东西吗）**
3. **抽象领域概念优先约定、不强造隐喻**：范式/列对齐/质检/解读这类**没有自然图形**的概念，别硬塞一个研究员要猜的隐喻。
4. **tooltip 依赖 = 语义失败**：一个图标若**离开 tooltip 就看不懂**，要么**永久配文字标签**，要么**降级为带文字的 badge**（不做纯图标）。

**语用轴（pragmatics，符号↔使用者，研究员非程序员怎么读）**
5. **状态/gate/警告信号用研究员的心智词**：「需要你确认」胜过一个扳手图标；gate/审批符号必须清楚区分**阻断 vs 放行**。承载文化负载（红绿色盲不安全 → 永远色+形/文双编码，`color-not-only`）。
6. **状态语言自成一套系统**：`阻断 / 放行 / 待确认 / 运行中 / 已通过` 五态，用研究员词汇设计、色盲安全、**绝不只靠颜色**（接 spec#1 `--color-status-*` + 图标 + 文字三件套）。

**验收与治理**
7. **label-off 验收门（关键）**：任一**独立图标**（无文字）上线前，**关掉文字给真实研究员看能否认出**——认不出就违反规则 4，回退配标签。这是符号系统的硬验收门。
8. **符号注册一处**：约定类符号（范式徽章 EPM/OFT/EZM、质检封印、严重度 tag）作为**约定**记录在一处 SSOT，配 legend，**不指望自明**。（v0.1 这份"注册"= 一段 i18n + 一个常量表即可，不建独立子系统——用户已定"只要 8 条硬规则"。）

### 3.2 Peirce 三分应用到本产品（决定每个信号是哪类）

| 类别 | 含义 | 本产品的符号 | 设计要求 |
|---|---|---|---|
| **iconic（形似）** | 像真实物体 | 上传/文件、图表类型选择、下载、删除、数据表 | 用真实世界指称物，lucide 现成 |
| **indexical（指示/因果）** | 指向"此刻正在发生" | **进度指示、live-trace 脉冲、运行中活动点、sparkline、spinner** | **必须 stateful/动态**，绝不静态装饰（接 spec#2 运行轨迹 + spec#4 进度轨） |
| **symbolic（纯约定）** | 无形似，靠约定 | **品牌标、范式徽章、质检通过封印、严重度 tag** | **必须配 legend**，文档化为约定，不指望自明 |

> 实施落点：spec#2/#4 的运行轨迹/进度轨节点 = indexical（脉冲必须动）；spec#3 画廊的范式/组别徽章 = symbolic（配 legend）；spec#5 决策卡的 clarification_type 图标 = 要过 label-off（规则 7）。**各 spec 标符号时引本 spec 的分类，不各自发明。**

---

## 四、插画美术方向：Mœbius **只借语法不借戏服**

> 回应"莫比斯风格如何"。**调研 verdict（明确，非"看情况"）：可以借，但只借语法，硬拒戏服。** 用户已拍板「只借语法」。

### 4.1 可借的"语法"（grammar，与日式克制天然契合）
- **ligne claire 清晰线条**：等宽、闭合、**无排线/无交叉影线**、清晰优先（源自丁丁 Tintin 一脉，为可读性而生）。
- **平涂色纪律**：闭合细线下平涂填色，**无渐变**（渐变正是"廉价 AI 插画包"的味道来源）。
- **大量负空间**：孤独身影/物体置于空旷之中——**天生适配"无数据"空状态**构图。

### 4.2 硬拒的"戏服"（costume，导致俗套/冲突）
- ❌ 他的**题材**（宇航员、生物机械飞船、外星巨构）——只画**我们自己的领域物体**（叶子、迷宫、传感器、动物剪影、轨迹线）。
- ❌ 他的**青绿+橙原色饱和 palette**——**直接冲突** Forest Green `#1A4840` + 暖白 + 降饱和纪律（决策3）。调研坐实：莫比斯真实作品比"沙漠柔粉"传说**更饱和**，那套柔粉是现代致敬的再调色，不是他本人。
- ❌ 超现实尺度play、漫画分格叙事框。

### 4.3 配色裁决（关键，守决策3）
**采用他的"平涂色方法"，但用我们锁定的 palette 执行，硬拒他的色相策略**：
- 线条 = 由 `#1A4840` 派生的**绿黑**（不是纯黑，与品牌同源）。
- 填色 = 限定在**暖白纸 + 一两个降饱和绿**（spec#1 stage 色家族）。
- **至多一个暖中性 accent**：取莫比斯"灰化沙漠"色温——一个**降饱和 ochre/clay 暖灰**（他的 grayed-brown 一脉是 Mœbius-sanctioned 的，且正好补足现有纯绿系缺的暖色锚点）。

在 `globals.css` 补**一个** accent token（spec#1 没有的暖中性，不动既有色相）：
```css
:root {
  --color-accent-clay: oklch(0.70 0.045 70);        /* 降饱和暖陶土/赭，Mœbius 灰沙漠色温；仅插画/空状态/极少点缀 */
  --color-accent-clay-soft: oklch(0.70 0.045 70 / 0.12);
}
.dark { --color-accent-clay: oklch(0.74 0.045 70); --color-accent-clay-soft: oklch(0.74 0.045 70 / 0.14); }
```
> **纪律**：`accent-clay` **只用于插画层 + 空状态 + 极少装饰点缀**，**绝不**用于数据可视化分类色（会和类别编码打架误导）、不用于状态语义（状态走 spec#1 status 色）、不大面积铺。它是"暖色的一点呼吸"，不是第二主色。

### 4.4 落点排序（哪里用，哪里禁用）
| 排序 | 落点 | 力度 |
|---|---|---|
| ① 最佳 | **图标线型**（等宽 ligne claire，纯语法零戏服，强化 §3 清晰度） | 全量（即 §3 的 lucide 语言，风格一致） |
| ② | **空状态**（`ui/empty.tsx`）——"孤独身影在虚空"=天生 no-data 构图 | 线条+平涂，画领域物体 |
| ③ | onboarding / 区块分隔小插画 | 稀疏 ligne claire 小品 |
| ④ | loading / skeleton 线 motif | 仅安静线条，不成场景 |
| ⑤ 最低（可选/暂不做） | 吉祥物 | **最高戏服风险，v0.1 不做**（用户已定只借语法） |

**绝不进**（调研铁律）：**数据可视化**（平涂+绿移会和类别编码冲突误导）、**密集表格**、**导航 chrome**（ligne claire 当 UI 家具会读成主题乐园）。**插画/图标层 only。**

### 4.5 防"沦落俗套"五铁律（调研 §5）
莫比斯致敬**已是知名 startup 套路**（Robinhood 2020 COLLINS 就是典型）。懒抄 = 和"每个有古怪插画包的 AI startup"一个桶。失败信号 = **借了他的"世界"而非"纪律"**。避免：
1. 借规则不借母题——只画你自己的领域物体。
2. 锁你的 palette，绝不用他的青橙。
3. 只线条+平涂，**禁渐变**（渐变=读成廉价衍生）。
4. **限量用**——只空状态+图标，不到处铺。
5. 不直白引用科幻母题。
> 做到这五条 = 读作"克制绿系里刻意的清晰线条工艺"（tasteful nod），不是戏服。

---

## 五、实施步骤

> 全在 `frontend/`，非受保护文件。多数是**验收纪律**（融进各组件交付），少量是 token（落 globals.css）。

### Step 1：补 elevation 阶梯 + clay accent token（§1.2 + §4.3）
- `globals.css` `:root` + `.dark` 补 `--shadow-{rest,raised,overlap,overlay}`（modal 保留）+ `--color-accent-clay(-soft)`。
- `@layer base` 补 `.shadow-{rest,raised,overlap,overlay}` 工具类（照现有 float/modal 模式）。
- `@theme inline` 映射 `--color-accent-clay` → 生成 `bg-accent-clay-soft` 等工具类。
- 验证：`pnpm check`。**纯 token，零逻辑，可独立提交。**

### Step 2：重叠 card 深度规则落地（§1.3 + §1.5）
- 审已有卡片：决策卡（spec#5）、画廊缩略图分组（spec#3）、消息流卡——重叠处按 §1.3 表升档；hover/press 按 §1.5（scale + 阴影，不抖）。
- **这步与 spec#3/#5 实施合并做**（它们建卡片时就按本规则），不单独大改。

### Step 3：尺寸/层级纪律 → 各 spec 验收项（§二）
- 不是新组件：把 §2.1 一屏一主角 + §2.2 尺寸/图标 token + §2.3 对齐/重复**写进 spec#2/#3/#4/#5 的验收清单**（眯眼测主角、icon token、aspect-ratio 网格）。
- 补 `icon-sm/md/lg` 尺寸 token（若 §2.2 决定 token 化）到 globals.css 或一个 `core/ui/sizes.ts` 常量（SSOT 一处）。

### Step 4：符号 8 条硬规则 → 图标/状态层验收门（§三）
- 建立 label-off 验收门（规则 7）作为图标 PR 的 checklist 项。
- 状态五态（§3.1 规则 6）对齐 spec#1 status 色 + spec#2/#4 的 indexical 脉冲 + spec#3/#5 的 symbolic 徽章。
- 符号注册（规则 8）= 一段 i18n + 常量表（范式徽章/封印/严重度），**不建子系统**。

### Step 5：Mœbius 插画语法 → 空状态 + 图标线型（§四）
- `ui/empty.tsx` 用 ligne claire 线型 + 平涂（领域物体）+ clay accent 重做空状态（第一落点）。
- 图标统一 lucide 2px stroke（已是 ligne claire 亲缘），不引第二套。
- onboarding/分隔/loading 线 motif 顺延（不阻塞）。
- **禁用区核查**：grep 确认数据可视化/表格/nav 不引插画语法。

### Step 6：reduced-motion + a11y + i18n
- 新动效（hover lift、脉冲）reduced-motion 降级（spec#1 机制）。
- 对比度：clay accent 上的文字/图标过 4.5:1（light 必测；dark 占位 Phase 2 验）。
- 插画 alt 有意义；符号 label-off 门已含 a11y。
- 所有新文案进 i18n（不硬编码中文）。

---

## 六、验收标准（每条可勾）

### 细节工艺（§一）
- [ ] `globals.css` 有 `--shadow-{rest,raised,overlap,overlay,modal}` 五档 + 工具类；`:root` 与 `.dark` 都有值。
- [ ] 重叠 card 靠 elevation 升档表达 z（不靠 border 套 border）；一个层级只一种区隔手段。
- [ ] 边角 radius 按容器尺寸选档、同类同档、嵌套外大内小≤2 层；无全站无脑 `rounded-3xl`。
- [ ] hover/press 用 transform scale + 阴影变化，**周围元素零位移**（devtools 查无 reflow/CLS）。
- [ ] 无彩色阴影、无装饰 glow（glow 仅 focus ring）。

### 尺寸/比例/层级（§二）
- [ ] 每个界面状态眯眼测：第一眼落点 = 当前任务主角（无两个元素抢视线）。
- [ ] 字号/间距走系统阶梯；标题靠字号+留白非粗体轰炸。
- [ ] 图标尺寸 token 化（sm/md/lg），无随手 18/22/26。
- [ ] 主 CTA 比次按钮大一档（非仅颜色区分）；画廊缩略图统一 aspect-ratio 网格。
- [ ] 亲密性：关联强元素间距小、弱的大（距离即分组，省分隔线）。

### 符号系统（§三）
- [ ] 图标全 lucide 一套、统一 2px stroke、filled/outline 不同层混用。
- [ ] 抽象领域概念不强造隐喻；tooltip 依赖的图标永久配标签或降级 badge。
- [ ] 状态五态（阻断/放行/待确认/运行中/已通过）色+图标+文字三件套，色盲安全。
- [ ] **label-off 验收门**：独立图标关文字能被真实研究员认出（否则回退配标签）。
- [ ] indexical 信号（进度/脉冲/活动点）stateful 动态；symbolic（徽章/封印）配 legend。

### 插画美术（§四）
- [ ] 空状态/图标用 ligne claire 线型 + 平涂、**无渐变**、画领域物体（非科幻母题）。
- [ ] palette 锁 Forest Green + 暖白 + 至多一个 clay 暖中性 accent；**未引入莫比斯青橙**。
- [ ] `accent-clay` 只在插画/空状态/极少点缀，**不进数据可视化分类色/状态语义/大面积**。
- [ ] 数据可视化/密集表格/导航 chrome **不引插画语法**（grep 核查）。
- [ ] 防俗套五铁律全守（借规则不借母题/锁 palette/禁渐变/限量/不直白科幻）。

### 工程纪律
- [ ] **不重定义 spec#1 token**（radius/ease/dur/status/stage 一律引用，新增只有 shadow 阶梯 + clay accent + icon 尺寸）。
- [ ] `pnpm check` 通过；不改 `core/` 逻辑、不改流式核心。
- [ ] i18n 不硬编码中文；reduced-motion 降级；light 对比度达标（dark 占位 Phase 2）。
- [ ] 不动 `components/ui/`、`ai-elements/`（generated）的源；插画改 `ui/empty.tsx` 等业务层。

---

## 七、风险与回退

| 风险 | 缓解 |
|---|---|
| 阴影/radius/尺寸规则变"全站大改" scope 膨胀 | 多数是**验收纪律**融进 spec#2/#3/#5 交付，不单独大重构；token（Step 1）可独立小提交 |
| Mœbius 借过头变戏服/俗套 | §4.5 五铁律 + 只借语法（用户已定）+ 落点限插画/图标层 + 禁用区 grep 核查 |
| clay accent 被滥用成第二主色 | §4.3 纪律：仅插画/空状态/极少点缀；验收项明确禁用区 |
| 符号学过度工程（学术脚手架） | v0.1 只 8 条硬规则当一页清单（用户已定）；符号注册= i18n+常量表，不建子系统 |
| label-off 门没真实研究员可测 | 退化为团队内部"关文字盲测"；实在无人测则保守**默认配标签**（规则 4 兜底） |
| 改共享 CSS 影响 generated 组件 | 只**新增** token + 工具类，不删不改既有；generated 引用的 token 仍在 |

**回退**：token 是纯样式 `git revert` 即可；纪律类是验收项（不通过就打回，不产生坏代码）；插画限 `ui/empty.tsx` 等少数文件，可灰度。

---

## 八、给实施 agent 的交接

- **改动文件**：`globals.css`（shadow 阶梯 + clay accent + 可选 icon 尺寸 token）；`ui/empty.tsx`（Mœbius 空状态）；图标统一 lucide；其余是**写进 spec#2/#3/#4/#5 验收清单**的纪律，不是独立组件。
- **不碰**：spec#1 已有 token（只引用）、`core/` 逻辑、流式核心、`ui/`+`ai-elements/` generated 源、数据可视化/表格/nav（插画禁入）。
- **与其他 spec 的关系**：
  - spec#1 = token SSOT（曲线/色/radius/时长）；**spec#6 = 其上的工艺+构图+符号+插画纪律**，新增仅 shadow/clay/icon 尺寸。
  - spec#2/#4 的轨迹/进度节点 = §3.2 indexical（脉冲必动）。
  - spec#3 画廊徽章 = §3.2 symbolic（配 legend）；缩略图网格 = §2.2 aspect-ratio + §1.3 重叠深度。
  - spec#5 决策卡 = §1.3 overlap 阴影 + §2.1 主角 + §3.1 规则7 label-off（clarification_type 图标）。
- **顺序**：Step 1（token，零风险，可独立提交）→ Step 5（空状态插画，独立可见）→ Step 2/3/4 随 spec#2/#3/#5 实施时合并落地（纪律融入，不单独大改）。
- **本会话已拍板（不再待定）**：① Mœbius=**只借语法**（线型+平涂+负空间，硬拒青橙 palette，只取灰沙漠暖中性 accent）② 符号学=**只 8 条硬规则**（Peirce 三分+Morris 三轴+label-off 门，不建学术脚手架）③「莫里斯」=Charles Morris+Peirce，**与 William Morris 无关**。

---

*依据：母方案 §2/§7 + 用户 2026-06-25 五点细节要求（重叠阴影/边角弧度/符号构建/莫比斯/元素比例尺寸）+ `ui-ux-pro-max`（elevation-consistent/Stable Interaction States/scale-feedback/visual-hierarchy/primary-action/color-not-only）+ Ant Design 4 原则 + Fusion 价值观 + Peirce/Charles Morris 符号学 + Mœbius/ligne claire 调研 verdict。未写代码。*
