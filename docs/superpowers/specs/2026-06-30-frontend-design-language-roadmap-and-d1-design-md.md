# 设计 spec：前端设计语言轨道路线图 + D1 DESIGN.md（视觉语言 SSOT，sync 友好）（2026-06-30）

> 本文档分两部分：**第一部分**=前端设计语言+设计系统重构的轨道路线图（排序/范围/依赖/贯穿约束）；**第二部分**=路线图第一步 **D1（DESIGN.md 视觉语言 SSOT）** 的可实施详细设计。D1 是本 spec 的可实施单元；D0/D2/D3/C2 各自后续单独 brainstorm→spec。
>
> 缘起：用户提出「重新设计视觉语言 + 人机交互理念，让 layout 更符合使用」，并担心前端是 DeerFlow subtree、sync 会冲突。brainstorm 收敛出「token 注入视觉语言（sync 友好）」的核心解法。

---

# 第一部分：前端设计语言轨道 · 路线图

## 重心定调（已拍板）

**对话输出为主，hard fact 产物为辅。** 核心是和 agent 对话，产物（图表/报告/指标表）是对话的有机产出，辅以右侧/内联呈现。设计语言围绕「对话流的优雅 + 产物的清晰承载」，不做成纯工作台/BI。

## 贯穿约束（已拍板 —— 这是 sync 兼容的核心解法）

前端是 DeerFlow subtree，sync 要拉上游前端修复。为不制造分叉、不和上游打架：

1. **视觉语言只通过「设计 token」注入，不通过「改组件结构」**：色板/排版/间距/景深/动效定义在 `globals.css` 的 CSS 变量（已有 `--ease-brand-*`）。token 一改，上游组件与 Noldus 组件同时变样，而组件代码不动 → **sync 零冲突**（token 是 Noldus 在 globals.css 的定制，上游不碰）。这正是 awesome-design-md 那些 DESIGN.md 的本质（核心是色板+token+原则，不是重写组件）。
2. **结构性重设计只在 `workspace/`（Noldus 独占，75 个 .tsx）做**：对话流/产物面板/画廊/决策卡等 ethoinsight 专属 UI，上游基本没有，自由设计、不碰上游。
3. **`ai-elements/`（27 个，registry 拉取）/registry 组件不直接改**（re-pull 会覆盖）：通过 token 让其跟随视觉语言；要改在 `workspace/` 包 wrapper。`ui/`（41 个 shadcn copy-in）只改 token、保 API（守 CLAUDE.md 的 copy-in vs registry 区分）。

## 子项目排序

| 步 | 子项目 | 性质 | 依赖 |
|---|---|---|---|
| **D0** | UX audit（资深 UX 走查核心流程，问题清单 + 严重度评分 + 修复方向） | 诊断（只看不改） | 无 |
| **D1** | **DESIGN.md（视觉语言 SSOT）+ 落 globals.css token** | 立标准 | D0 输入（本 spec 跳过 D0 直接立 D1，D0 可后补） |
| D2 | 设计系统：分散 UI → 统一组件库（workspace 新 UI 走库，不绕过） | 重构 | D1 |
| D3 | A11y + 多设备响应式（桌面/平板/手机；键盘/焦点/对比度/语义） | 横切 | D1 |
| C2 | 画廊 layout 重设计（硬编码段叠 → 可扩展多产物布局，在 D1/D2 之下） | 应用 | D1/D2 |

**本 spec 只完整设计 D1。** 其余各自后续单独 brainstorm→spec→实施。

> 注：用户提供的「设计系统/a11y/UX audit」三个任务描述含多 agent 并行编排指令——那是**实施层编排**，由实施时决定，不在本设计 spec 范围。

---

# 第二部分：D1 详细设计（可实施单元）

## 目标

确立 noldus-insight 的视觉语言 SSOT，落成 sync 友好的设计 token，让后续 D2/D3/C2 都对齐它。**不重写组件库**（那是 D2）。

## 前端现状（已勘察坐实）

- `globals.css` 已有日式动效曲线 token：`--ease-brand-out`（快起长缓停的"减速尾巴"，cubic-bezier(0.22,1,0.36,1)）/ `--ease-brand-in`（利落退场）/ `--ease-brand-in-out`。Phase0#1 设计 token spec（`2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md`）已立部分 token。
- 前端分层：`workspace/`（75，Noldus 独占）/ `ui/`（41，shadcn copy-in）/ `ai-elements/`（27，registry 拉取）/ `feedback/`。
- Tailwind v4 `@theme`：`--ease-*` 是真 namespace（自动生成工具类），但 `--dur-*` 不是（自定义 duration 需 `@utility` 显式定义——守 memory `feedback_tailwind_v4_dur_namespace_not_real_duration_utility_dead_class`）。

## 产出物（3 件）

### 产出 1：`DESIGN.md`（视觉语言 SSOT 文档）

放前端根（`packages/agent/frontend/DESIGN.md`）。借 awesome-design-md 的 Google Stitch 9 段格式，内容贯彻「日式简洁 + 对话为主 + 色盲安全」：

1. **视觉氛围**：日式简洁（留白/Ma、降饱和、克制 Shibumi）；安静、不抢戏，让数据与对话是主角。
2. **色板 + 角色**：中性灰为主（60%）+ 主数据色（30%）+ **单一强调色**（10%，色盲安全，蓝 #0072B2 或橙 #E69F00，绝不红绿）；语义色仅用于状态。明暗双主题（复用现有 `.dark` token 块）。
3. **排版**：字重层级（表头 semibold 600、正文 400、次要浅灰）；数值 `font-variant-numeric: tabular-nums`（选字体先验证支持）；不全大写。
4. **组件样式**：按钮/卡片/输入/导航的状态规范（hover/focus/active/disabled），克制。
5. **布局原则**：对话为主、产物为辅；留白为分隔（非全网格线）；信息层级清晰。
6. **景深/Elevation**：克制阴影（`--shadow-float` 等），不堆叠重投影。
7. **Do / Don't**：明确禁忌（不暴露内脏、不平铺过载、不红绿、不全网格线、不双高亮叠加…）。
8. **响应式**：桌面/平板/手机断点原则（细节实现归 D3）。
9. **Agent 提示**：给后续 coding agent 的生成指引（怎么用本 DESIGN.md + token 生成对齐 UI）。

### 产出 2：`globals.css` 设计 token 落地

把 DESIGN.md 的色板/间距/排版/景深落成 CSS 变量（complete 现有 token 集）：
- 补齐色板角色变量（中性/主数据/强调/语义）+ 明暗双值。
- 间距/圆角/字重/景深 token。
- 自定义 duration 用 `@utility` 显式定义（不靠 `--dur-*` namespace，守上述 memory）。
- **一处定义，全前端（含上游组件）跟随，sync 零冲突。**

### 产出 3：范本借鉴（研究，不照抄）

从 awesome-design-md 取 **Linear / Notion / Claude / Vercel** 的 `DESIGN.md` + preview.html，研究其 token 结构（色板角色命名、type scale、景深层级）作为我们 DESIGN.md 的结构模板。**借结构与克制理念，不照搬任何品牌身份。**

## 核心约束（写进实现，sync 兼容的命脉）

- ✅ 视觉语言只通过 token（globals.css）+ workspace 结构注入。
- ❌ 不改 `ai-elements/` / registry 组件结构（re-pull 覆盖）。
- ⚠️ `ui/`（shadcn copy-in）只改 token、**保所有 API/导出**（守 CLAUDE.md copy-in 规则）。
- ✅ DESIGN.md 是设计 SSOT，D2/D3/C2 都对齐它，不另立第二份设计标准（守 single-source-of-truth）。

## 验收

1. `DESIGN.md` 9 段齐全，无占位。
2. `globals.css` token 落地：抽样改一个 token（如强调色）→ 现有 workspace 组件 + 上游组件视觉同时跟随（坐实「token 注入」生效）。
3. **sync 兼容验证**：D1 的改动 git diff 证明**未改 `ai-elements/` 结构、未改 `ui/` 的任何导出/API**（只改 token 与 workspace）。
4. PostCSS 真编译坐实自定义工具类生成（守 Tailwind v4 token 三坑，memory `feedback_tailwind_v4_dur_namespace_not_real_duration_utility_dead_class`）。
5. `pnpm check` 0；现有 vitest 不回归。

## 不做什么

- ❌ 不在 D1 重构组件库（D2）、不做 a11y（D3）、不重设计画廊布局（C2）。
- ❌ 不改 ai-elements/registry 组件结构。
- ❌ 不动后端。
- ❌ 不照抄任何品牌 DESIGN.md 身份（只借结构 + 日式理念）。

---

## 关联

- 资源：[awesome-design-md](https://github.com/voltagent/awesome-design-md)（73 个品牌 DESIGN.md，Stitch 9 段格式，token+原则为核心）。
- 现有：`globals.css` 的 `--ease-brand-*`、Phase0#1 设计 token spec、`.dark` 主题块。
- sync 铁律：`feedback_sync_full_follow_upstream_infra`、`feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`、`feedback_tailwind_v4_dur_namespace_not_real_duration_utility_dead_class`。
- 后续：D0（UX audit，可后补做为 D1 修正输入）、D2（组件库重构）、D3（a11y+响应式）、C2（画廊 layout 重设计）。
