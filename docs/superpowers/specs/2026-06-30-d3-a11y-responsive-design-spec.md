# 设计 spec：D3 a11y + 多设备响应式（横切治理，立全局标准 + axe 护栏）（2026-06-30）

> 前端设计语言轨道 **D3**（横切，依赖 D1/D2）。路线图 `2026-06-30-frontend-design-language-roadmap-and-d1-design-md.md` 第 D3 步。
> brainstorm 收敛：a11y + 响应式合一份，横切治理立全局标准（写进 D1 DESIGN.md，不另立），axe 自动检测护栏。本 spec 交别的 agent 实施。

---

## 目标

立全局 **a11y（WCAG 2.1 AA）+ 三端响应式**标准，落成断点 token + focus token + axe 自动检测护栏，
D1/D2/D0 都遵守它。**横切治理，不是逐组件零散修。**

## 前端现状（已勘察坐实）

- a11y：87 个 workspace 组件仅 17 个带 aria/focus-visible/role——**覆盖稀疏**。globals.css 已有 `prefers-reduced-motion` 块。
- 响应式：globals.css 已有断点 `@media (width >= 40rem/48rem)` 但只零星用；workspace 零散 `md:/grid-cols`，**无系统三端策略**。
- vitest 已配（memory `project_frontend_vitest_already_set_up_2_red_streaming_tests`，utils.test.ts 2 红是 pre-existing baseline）。

## 三件交付物

### 1. a11y + 响应式标准（写进 D1 DESIGN.md，不另立第二份，守 SSOT）

**a11y（WCAG 2.1 AA）五条可检清单**：
1. **对比度** 4.5:1（正文）/ 3:1（大字/图标）——所有 D1 语义色 token 对达标，脚本校验。
2. **键盘全可达**：交互元素 Tab 可达、Enter/Space 可激活、无键盘陷阱；折叠/lightbox/对比模式纯键盘可操作。
3. **焦点可见**：统一 `focus-visible` ring token，所有可聚焦元素显式焦点环（现有零散 `focus-visible:ring` 统一）。
4. **ARIA 语义**：卡片/反问/产物用正确 role + aria-label/aria-expanded；状态不靠纯色传意（D2 StatusBadge 的 color-not-only 三件套是雏形）。
5. **屏读**：图表带 alt/aria-label（指标名+组）；装饰性元素 `aria-hidden`；动态区（流式/阶段叙事）`aria-live`。

**三端断点策略（全优化）**，写进 DESIGN.md 响应式段 + token：
- **桌面 (≥1024px)**：对话流 + 右侧产物面板并排（现状）。
- **平板 (640–1024px)**：产物面板可收起为抽屉/标签，对话为主；画廊网格降列。
- **手机 (<640px)**：对话流全宽单列；产物面板变底部抽屉/独立 tab；反问卡/subtask 卡纵向堆叠；触摸目标 ≥44px（WCAG AA target size）。
- 三端都守「对话为主、产物为辅」（D1 重心）。

### 2. 断点 token + a11y token（落 D1 已建的 globals.css）

- 统一三端断点变量（扩现有 `@media (width >= 40rem/48rem)`）。
- `focus-visible` ring token。
- 对比度达标的语义色（与 D1 色板联动，不另定色）。
- **一处定义，全前端跟随，sync 零冲突**（同 D1 token 注入解法）。

### 3. axe 自动检测护栏（只 vitest-axe 组件级 + 对比度脚本）

- **新依赖**：`vitest-axe` + `axe-core`（devDependency，前端，sync 友好）。
- `vitest-axe`：D2 kit 4 原语 + 关键 workspace 组件（decision-card/subtask-card/clarification-options/画廊）单测 `toHaveNoViolations()`。
- 对比度脚本：遍历 D1 语义色 token 对，校验达 4.5:1/3:1，不达标 fail。

## 与 D0/D1/D2 关系（不重复劳动）

- **D1**：D3 标准写进 D1 DESIGN.md（SSOT 一处），token 落 D1 globals.css。
- **D2**：D2 的 kit 组件**遵守 D3 标准**；D3 给 kit 补全 ARIA/键盘/焦点 + axe 断言（StatusBadge color-not-only 已是 a11y 雏形）。
- **D0**：D0 audit 发现的 a11y/响应式问题**归 D3 修**。

## 依赖链（实施顺序）

```
D1(token/DESIGN.md) → D2(kit) → D3(a11y+响应式横切)
D0(audit 发现 a11y/响应式问题) → 喂 D3 修
```
D3 依赖 D1 token + D2 kit 就绪（给 kit 补 a11y、对 D1 色板验对比度）。

## 回归防护（axe 护栏 + 三端 + 守 TDD）

- **vitest-axe 断言**：kit 4 原语 + 关键 workspace 组件 `toHaveNoViolations()`；**防 vacuous**：故意给一个组件去掉 aria-label → axe 测应变红（证断言真在跑）。
- **对比度脚本**：D1 语义色 token 对校验达 4.5:1/3:1，不达标 fail。
- **三端响应式**：关键组件在三 viewport 渲染测（桌面/平板/手机）不错位；手机触摸目标 ≥44px 断言。
- **键盘可达**：关键交互（折叠/lightbox/反问）键盘操作测。
- **现有不回归**：vitest 全绿（排除 utils.test.ts 2 红 baseline）；`pnpm check` 0。

## 验收

1. D1 DESIGN.md 含完整 a11y 段（WCAG AA 五条）+ 三端响应式段。
2. globals.css 落 focus-visible ring + 三端断点 + 对比度达标语义色 token。
3. `vitest-axe` 接入，kit + 关键组件 axe 零 violation 且**断言非 vacuous**（去 aria 变红）。
4. 对比度脚本证 D1 语义色全达 AA。
5. 三端 viewport 渲染不错位，手机触摸目标 ≥44px。
6. 关键交互键盘全可达。
7. sync 兼容：只改 token+workspace，未改 ai-elements/ui 结构（ui/ 只保 API 补 aria）。
8. 现有 vitest 不回归 + pnpm check 0。

## 不做什么

- ❌ 不另立第二份设计标准（写进 D1 DESIGN.md，守 SSOT）。
- ❌ 不改 ai-elements/registry 结构（token + wrapper；ui/ 只保 API 补 aria）。
- ❌ 不追 WCAG AAA（AA 够，AAA 对日式降饱和色板过严）。
- ❌ 不动后端。
- ❌ 不接 @axe-core/playwright e2e（只 vitest-axe 组件级）。
- ❌ 不重做 D2 kit（只给 kit 补 a11y 断言/属性）。

## 关联

- **依赖**：D1/D2 先于 D3；D0 audit 喂修复项。
- **守 memory**：`feedback_single_source_of_truth`（写进 D1 不另立）、
  `feedback_frontend_design_japanese_minimal_motion_craft`（对比度/色盲安全与日式联动）、
  `feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`、
  `project_frontend_vitest_already_set_up_2_red_streaming_tests`（2 红 baseline）。
- **新依赖**：`vitest-axe` + `axe-core`（devDep）。
- **上游路线图**：`2026-06-30-frontend-design-language-roadmap-and-d1-design-md.md`（D3 步）。
