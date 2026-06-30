# 设计 spec：D2 设计系统 — workspace/kit 业务组件层（顶层自由 + 底两层 token 对齐）（2026-06-30）

> 本文档是前端设计语言轨道 **D2**（设计系统：分散 UI → 统一组件库）的可实施设计。
> D2 在路线图 spec `2026-06-30-frontend-design-language-roadmap-and-d1-design-md.md` 排为
> D1 之后第一步，性质=重构，依赖 D1。
>
> 缘起：用户继续 brainstorm 设计语言轨道，按依赖排序 D0→D2→C2→D3→C3。
> brainstorm 收敛出「顶层自由建 kit/ 业务组件层 + 底两层靠 D1 token 对齐」的核心解法。

---

## 目标

把散落在 ~10 个 workspace 组件里**重复造**的「状态色竖条 / 卡片壳 / 空态 / 加载态」收敛成
**4 个统一业务原语**，建在 `workspace/kit/`，所有业务卡片改成消费它们。让 workspace 新 UI 走库、不绕过。

---

## 为什么这么做（现状证据）

勘察坐实（grep `border-l-/CheckCircle/XCircle/AlertTriangle/Loader2/状态/accent`）：
状态色 / accent bar / 卡片壳重复散在 ~10 个 workspace 组件——`decision-card`、`subtask-card`、
`message-group`、`artifact-gallery`、`todo-list`、`attachment-chip`、`message-attachments`、
`input-box`、`workspace-nav-menu` 各自实现；空态散在 ~11 个文件各自裸写。
→ 没有统一的「ethoinsight 业务组件层」，每个卡片各搞各的视觉，是 D1 视觉语言无法落地一致的根因。

---

## 分层架构（守 sync 命脉）

```
┌─ ethoinsight 业务组件层  workspace/kit/   ← D2 自由建（Noldus 独占，87 文件域内）
│   StatusCard / StatusBadge+AccentBar / EmptyState / LoadingState
├─ composite 层           ai-elements/  (27, registry 拉取)  ← 禁改结构，只 wrap
└─ primitive 层           ui/           (45, shadcn copy-in)  ← 只改 token，保所有 API/导出
```

- **顶层（kit/）**：D2 真正建的东西，自由设计，消费底两层 + D1 token。
- **底两层不重写**：`ai-elements/` 只能在 kit/ 里 wrap（re-pull 会覆盖结构）；`ui/` 只改 token、
  **保所有导出/API**（守 CLAUDE.md copy-in 规则）。视觉一致靠 **D1 的 globals.css token** 让三层同步变样。
- **kit 不立第二份视觉标准**：kit 原语**只引用 D1 token，不自定义色值/间距硬编码**
  （守 memory `feedback_single_source_of_truth`）。

**与 D1 关系**：D1 立视觉 SSOT（DESIGN.md + token），D2 把这套语言固化成可复用组件。
**D1 实施先于 D2**——token 没落，kit 没得引。

---

## 4 个 kit 原语（核心交付）

### 1. `StatusCard`（卡片壳）

- **职责**：统一「左 accent bar + 标题区 + 内容区」结构，吸附 decision-card / subtask-card /
  quality-warning-banner 的公共壳。
- **API**：`status`（`running`|`success`|`warning`|`error`|`info`）、`title`、`icon?`、
  `accent?`（是否显左竖条，默认 true）、`children`、`className?`。
- **视觉**：左 accent bar = 状态色细竖条（**非整卡变色**，守日式克制）；壳用 D1 `--shadow-float` +
  圆角 token；留白用 D1 间距 token。状态色全走 StatusBadge 色板，不自定义。

### 2. `StatusBadge` + `AccentBar`（状态色原语）

- **职责**：状态的唯一视觉来源（走运/成功/警告/失败/信息）。StatusCard 的 accent bar 复用其色映射。
  **`status → 色/图标/label` 的映射表是这层的 SSOT。**
- **API**：`StatusBadge: { status, label?, size? }`；`AccentBar: { status }`。
- **视觉**：**color-not-only 三件套**（色 + 图标 + 文字，不靠纯色传意，守色盲安全 + a11y）；
  色板**绝不红绿对立**（用 D1 色盲安全强调色，守 memory `feedback_frontend_design_japanese_minimal_motion_craft`）。

### 3. `EmptyState`（空态）

- **职责**：统一所有空态（知识问答无产物 / 未上传 / 无结果 / 无图表），免各组件裸奔。
- **API**：`{ icon?, title, description?, action? }`。
- **视觉**：居中、留白充足、降饱和图标；文案克制；不暴露技术黑话（守输出宪法）。

### 4. `LoadingState`（加载态）

- **职责**：统一加载/流式指示，免各卡片各自用 Loader2 / Shimmer。包装现有 ai-elements 的
  `Shimmer`（wrap 不改）。
- **API**：`{ variant: 'spinner'|'shimmer'|'dots', label? }`。
- **视觉**：入场动效 ease-out 非 linear（守 D1 `--ease-brand-out`）；克制，不抢焦点。

---

## 全量迁移分批（终点是全量，分三批独立可验）

- **批 1（核心对话卡片）**：`decision-card` / `subtask-card` / `quality-warning-banner`
  → 消费 StatusCard + StatusBadge/AccentBar。状态色竖条重复重灾区。
- **批 2（产物 + 列表 + 空态）**：`thread-assets-panel` / `artifacts/gallery/artifact-gallery` /
  `todo-list` / `message-group` 的空态与状态 → 消费 EmptyState + StatusBadge。
- **批 3（边角）**：`attachment-chip` / `message-attachments` / `input-box` / `workspace-nav-menu`
  等剩余 accent/状态用法 → 收敛。

> 「全量」= 三批做完后，grep 不再有绕过 kit 自造 accent bar/状态色/空态的 **对话业务卡片**。
> `settings/`、`agents/` 等非对话面板按需，不强迁（迁移触及时顺手收敛，不专门重构其功能逻辑）。

---

## 回归防护（kit 带测 + 原卡片不回归，守 TDD）

- **kit 4 原语各带 vitest**：
  - StatusCard：accent bar 渲染 + `status → 色` 映射。
  - StatusBadge：color-not-only 三件套（色 + 图标 + 文字都在）。
  - EmptyState：文案/action 渲染。
  - LoadingState：三 variant。
- **迁移后原卡片现有测不回归**：`decision-card.test.tsx` / `subtask-card` / `todo-list` 等
  现有 vitest 全绿。**已知 `utils.test.ts` 2 个 isStreaming 红是 pre-existing baseline，
  非本轮回归**（守 memory `project_frontend_vitest_already_set_up_2_red_streaming_tests`）。
- **`pnpm check` 0**（lint + type）。
- **守 catastrophic forgetting 自检**：改共享卡片前 grep 所有消费点，全量跑 vitest，不只跑新测。

---

## 验收

1. `workspace/kit/` 4 原语齐全，各带 vitest 且绿。
2. 批 1-3 迁移完成，grep 证对话业务卡片不再自造 accent bar/状态色/空态（列出豁免的非对话面板）。
3. **sync 兼容验证**：git diff 证**未改 `ai-elements/` 结构、未改 `ui/` 任何导出/API**
   （只新增 kit/ + 改 workspace 卡片 + 可能改 token）。
4. kit 原语**只引 D1 token，无自定义色值/间距硬编码**（grep 证）。
5. 现有 vitest 不回归（排除 pre-existing 2 红 baseline）。

---

## 不做什么（守边界）

- ❌ 不改 `ai-elements/` 结构（只 wrap）。
- ❌ 不改 `ui/` 导出/API（只改 token）。
- ❌ kit 不自定义视觉值（只引 D1 token，守 SSOT）。
- ❌ 不动后端、不动 `settings/`/`agents/` 等非对话面板的功能逻辑（迁移触及时只顺手收敛视觉）。
- ❌ 不做 a11y 完整审计（那是 D3；kit 的 color-not-only 是顺带守，非 D3 全量）。
- ❌ 不重设计画廊布局（那是 C2；D2 只让画廊空态/状态走 kit）。

---

## 依赖与关联

- **依赖**：**D1 实施先于 D2**（kit 原语引 D1 token，token 没落 kit 没得引）。
- **守 memory**：`feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`（分层约束）、
  `feedback_single_source_of_truth`（kit 不立第二标准）、
  `feedback_frontend_design_japanese_minimal_motion_craft`（L3 克制 + 动效）、
  `project_frontend_vitest_already_set_up_2_red_streaming_tests`（2 红 baseline）。
- **上游路线图**：`2026-06-30-frontend-design-language-roadmap-and-d1-design-md.md`（D2 是其 D1 后第一步）。
- **下游**：C2 画廊布局重设计消费 kit 的 EmptyState/StatusCard；D3 a11y 在 kit color-not-only 基础上做全量。
- **D0 输入**：D0 UX audit 的 findings.md 若指出某卡片视觉问题，归 D2 的可纳入对应批次。
