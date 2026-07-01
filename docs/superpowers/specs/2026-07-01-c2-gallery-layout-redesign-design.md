# 设计 spec：C2 画廊布局 + UI/UX 重做（概览优先 + 分类可扩展 + 日式简洁）（2026-07-01）

> 生成式 UX 路线图 C 系列。在 C1 已建卡片之上，把画廊从「朴素三段线性堆叠」重做成 SOTA + 日式简洁的**概览优先**布局。**在卡片之上建布局，不推倒重写**；6 条 dogfood 行为不变式逐条继承 + 回归断言。
>
> 依赖 C1（已落地 #255）。**实施前置 = D1（DESIGN.md + token）+ D2（kit 原语）已落地**——当前两者均未实现（坐实：`frontend/DESIGN.md` 与 `workspace/kit/` 均不存在）。故本 spec 现写、**实施排期在 D1/D2 之后**。配套后端注册表见 `2026-07-01-c2b-artifact-registry-backend-design.md`（可先落地）。

---

## 目标

`ThreadAssetsPanel` 现状 = reports→data→charts 三段线性堆叠、全展开、无概览、无分类导航、视觉占位级（C1 自注：「本卡只做最小可用挂载，完整 layout 重设计归 C2」）。C2 重做成**概览优先**（进面板先看到全局有什么）+ **分类可扩展** + **日式简洁**，且**功能行为字节级保留**（视觉升级不推翻正确实现）。

## 重做边界（brainstorm 锁定）

**在 C1 卡片之上建布局**（`metrics-table-card` / `report-card` / `artifact-gallery`），**不推倒重写卡片内部功能逻辑**（懒加载 / 概览优先条件渲染 / 离群标记 / 性能红线 / i18n——这些已做对且被测试锁死，重写只得同样行为却翻倍风险，是 catastrophic forgetting）。

**唯一例外**：某卡内部 DOM 结构挡住 C2 视觉目标时 → **局部外科改那一处结构**，且**必须保 C1 测试绿**。不是「推倒重写全部卡片」。

> 收益论证：C2 想要的收益 100% 在视觉/布局/信息架构层，恰是 C1 没做、留给 C2 的；C1 已做对且测试锁死的是功能层，重写它不产生任何 C2 收益。守 `feedback_perf_is_efficient_impl_not_visual_downgrade` 的镜像：视觉升级不以推翻正确实现为代价。

## 架构：两层（概览层 + 分类详情层）

### ① 概览层（overview-first，4 件，handoff 定稿）

进面板先看到「全局有什么」，不是一头扎进列表：
1. **报告入口卡**：`report.html` 醒目入口（不埋在列表里）。
2. **指标表组综述**：**复用** C1 `metrics-table-card` 的组综述态（不重画）。
3. **代表图缩略**：**复用** aggregate 代表图（`output_mode==="aggregate"`）作概览缩略，不铺全部 113 张。
4. **计数概览**：「N 图 / M 报告 / K 数据表」轻量计数。

### ② 分类详情层（注册驱动可扩展）

- 沿用 C1 已建的 `sections` 描述符数组模式（`ThreadAssetsPanel.tsx:28` `AssetSection`），升级成**概览层在上 + 各分类详情可展开/滚动定位**。
- 加一类产物 = 加一个 section 描述符（继承 C1 可扩展挂载，**不新造 plugin registry**——避免过度工程）。

### ③ 日式简洁视觉（守 `feedback_frontend_design_japanese_minimal_motion_craft`）

- 留白节奏、层级对比（概览层 > 详情层）、降饱和、accent 克制、入场 ease-out 非 linear。
- **依赖 D1 token（globals.css）+ D2 kit 原语**（StatusCard/EmptyState/LoadingState）——这是实施前置 = D1/D2 的根因。

## 6 条 dogfood 行为不变式（逐条继承 + 回归断言）

C2 重做布局时**绝不能破**这 6 条（功能行为原样保留，仅重排视觉）：

| # | 不变式 | 现锚点（`artifact-gallery.tsx`）|
|---|---|---|
| 1 | chart_type 显式映射（不前端正则猜）| `:44` |
| 2 | 防图墙：per_subject 默认折叠 | `:155` |
| 3 | 每 chart_type 独立折叠子区 | `:188` |
| 4 | 失败告警（failed/remaining 计数）| `:73/103` |
| 5 | aggregate 代表图分区 | `:40/120` |
| 6 | lightbox + 对比模式 + ZIP 下载 | `:36/38/135` + facet bar |

## 数据流（不变，仍磁盘为真相）

C2 **不动数据来源**——仍走 C1/C2b 的磁盘端点（`useThreadAssets`：charts/reports/data）。纯呈现层重做：同样 `ArtifactMeta[]` 输入，重排成概览层 + 分类详情层。

```
磁盘端点（C2b 门面）→ useThreadAssets → ThreadAssetsPanel
  → 概览层（报告入口卡 / 指标表组综述 / aggregate 缩略 / 计数）
  → 分类详情层（section 描述符数组，复用 C1 卡片 + artifact-gallery）
```

**关键边界**：C2 不碰后端、不碰数据 hook 契约、不碰 C1 卡片功能逻辑——只重排布局 + 加概览层 + 套 D1/D2 视觉。

## 测试策略（守 TDD 强制 + 防 vacuous）

1. **6 不变式逐条回归**（vitest + testing-library）：每条断言重做后行为不变（chart_type 分组存在 / per_subject 默认折叠 / 独立子区 / 失败告警渲染 / aggregate 分区 / lightbox+对比+ZIP 可达）。
2. **概览层新测**：4 件概览各自渲染断言（报告入口卡点击 / 组综述复用 / aggregate 缩略 / 计数正确）。
3. **性能红线不回归**：C1「per_subject 默认不渲染上百行」性能测试重做后仍绿（概览优先不得引入全量渲染）。
4. **C1 卡片测试保绿**：`metrics-table-card.test.tsx` 等原样通过（局部外科改内部时也必须绿）。
5. **防 vacuous**：概览层「计数」测试——删计数源断言必须红。

## sync 友好性

- 全部 `packages/agent/frontend/src/components/workspace/artifacts/` 下 + 消费 D1 token(globals.css) / D2 kit——**全 deerflow 子树外**。100% sync 友好。
- 不改 ai-elements / registry 结构、`ui/` 只保 API（守 sync 命脉）。

## 依赖时序

- **实施前置 = D1（DESIGN.md + token）+ D2（kit 原语）已落地**。当前两者未实现（`frontend/DESIGN.md` + `workspace/kit/` 均不存在）→ **C2 实施排期在 D1/D2 之后**。
- **C2b 不硬阻塞 C2**（后端门面，`useThreadAssets` 契约字节不变），但先落地更干净。
- **C3 依赖 C2**（C3 选区层挂载到 C2 定稿的画廊里的指标表卡）。
- 链：`C1✅ → C2b（可即派）→ D1 → D2 → C2 → C3`。

## 边界（YAGNI）

- ❌ 不动后端 / 数据 hook 契约（C2b / C1 territory）。
- ❌ 不做产物框选（C3）。
- ❌ 不重写 C1 卡片功能逻辑（之上建布局，局部外科例外见上）。
- ❌ 不引虚拟化（概览优先已避全量渲染；C1 已定「50 行内无需」）。

## 验收标准

1. 概览层 4 件到位（报告入口卡 / 指标表组综述复用 / aggregate 缩略 / 计数）。
2. 分类详情层沿用 section 描述符可扩展模式。
3. 6 dogfood 行为不变式逐条回归绿。
4. C1 卡片测试 + 性能红线测试全绿。
5. 日式简洁视觉落实（留白/层级/降饱和/ease 曲线），消费 D1 token + D2 kit。
6. 前端零改 deerflow 子树 / ai-elements / registry 结构。

## 关联

- 上游：C1（#255 已落地，卡片 + section 数组）、D1（DESIGN.md+token，待实施）、D2（kit，待实施）、C2b（后端门面，可先落地）。
- 下游：C3（`2026-06-30-c3-metrics-selection-followup-design.md`，选区层挂 C2 画廊）。
- 现有代码：`thread-assets-panel.tsx`（section 描述符）、`artifact-gallery.tsx`（6 不变式锚点）、`metrics-table-card.tsx` / `report-card.tsx`（C1 卡片）。
- 守 memory：`feedback_frontend_design_japanese_minimal_motion_craft`、`feedback_perf_is_efficient_impl_not_visual_downgrade`、`feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`（改 ui/ 只 API 保留）、`feedback_chart_requires_columns_gate_distinct_from_zone_alias_overrides`（失败告警不变式）。
