# 设计 spec：C1 指标结果表导出 + 画廊概览优先呈现（2026-06-30）

> 生成式 UX 路线图 C 系列第一步。让用户在画廊看到「**计算出的指标结果**」（非过程/raw 文件）。本 spec 只设计 C1，交别的 agent 实施。C2（画廊 UI/UX）、C3（产物框选追问）各自后续单独 brainstorm。
>
> 呈现范式有调研支撑（数据表最佳实践，见会话记录）：Shneiderman「Overview first → details-on-demand」+ 组摘要懒展开（不平铺/不卡顿）+ 行为学分布缩略（均值会骗人）+ IQR 离群轻量标记（日式克制）。

---

## 一、目标与已拍板决定

让研究员看到 code-executor 算出的指标结果——**干净的结果，不是过程文件、不是 raw 上传**。

已拍板：
1. **(甲) 后端确定性导出**：数据现成（困在内部 `handoff_code_executor.json` 的 `per_subject` + `metrics_summary`），C1 把它确定性导出成干净用户产物（不靠 LLM，同 #245/run_chart_plan 范式），画廊收它。**不让前端耦合内部 handoff 格式。**
2. **(丙) 两产物同源导出**：CSV（研究员下载进 SPSS/Prism）+ JSON（前端渲染表格，剥内脏）。
3. **概览优先呈现**：默认只显示组级综述；用户「像点开画廊某张图片一样」点开 card 下钻 subject 明细。**永不平铺全量**（既避免视觉过载，又从根上避免大表卡顿）。

---

## 二、数据现状（已勘察坐实）

`handoff_code_executor.json`（22KB 内部文件）含：
- **`per_subject`**：`{subject名: {open_arm_entry_count, open_arm_time_ratio, open_arm_time, total_entry_count, ...}}` —— 每 subject × 每指标的原始数值。
- **`metrics_summary`**：`{组名: {指标: {mean, std, n, applicable, ...}}}` —— 组级统计摘要。
- 同文件还混有 `gate_signals`/`statistics`/`assessment`/`confidence` 等**内脏字段**——导出时必须剥除。

现有 `data-table` 端点（`artifacts.py:171`）是 placeholder（只返回第一个 .csv、其实没 csv）。C1 顺手填这个坑。

---

## 三、模块设计

### 模块 1：后端确定性导出指标结果表

- **触发**：code-executor `run_metric_plan` 完成后（指标已算齐），由确定性逻辑导出——**不靠 LLM**。落点：run_metric_plan 工具完成处 / seal 时，择一（Step 0 坐实，优先复用已有确定性 seal/登记点，同 #245）。
- **数据源**：`per_subject` + `metrics_summary`（同源，一次导出两格式）。
- **两产物**（落 `outputs/`，文件名稳定如 `metrics_table.csv` / `metrics_table.json`）：
  - **`metrics_table.csv`**：一行一 subject，列 = `subject, group, <指标1>, <指标2>, ...`。全量全列，给研究员下载。
  - **`metrics_table.json`**：清洗版结构化数据 = `{ groups: [{group, n, metrics:{<指标>:{mean,std, distribution_thumb?}}}], per_subject: [{subject, group, <指标>:值, outlier_flags:{<指标>:bool}}] }`。**剥除 gate_signals/handoff/assessment 等内脏**。outlier_flags 由 IQR（< Q1−1.5·IQR 或 > Q3+1.5·IQR）确定性计算（研究员熟悉的箱线图约定）。
- **SSOT**：CSV 与 JSON 同源一次导出，数值必一致（不双算）。

### 模块 2：后端端点

- `data-table` 端点（placeholder）→ 真正返回 `metrics_table.csv`（下载，`Content-Disposition: attachment`）。
- 新增/扩展端点返回 `metrics_table.json`（前端渲染用）。
- 画廊列表端点把「指标结果表」作为一类 artifact 暴露（与 charts/reports 并列）。

### 模块 3：前端画廊纳入 + 概览优先呈现

**入口**：画廊（`thread-assets-panel` / artifact 列表）新增一张「指标结果表」卡片，与 chart/report 卡并列。

**⚠️ 可扩展挂载（边界，避免硬编码堆叠）**：现状 `thread-assets-panel.tsx` 是**硬编码两段**（`reports` section + `charts` section 上下叠）。C1 加第三类产物若继续硬编码 = 加一类改一次结构。C1 **只做最小可扩展挂载**：把「报告/图表/数据表」改为按产物类型可扩展地渲染（最小骨架，指标表作为新类型挂上去，先能用、不难看），**完整的画廊 layout 重设计（留白/层级/分类导航/日式高级感/可扩展多产物布局）是 C2 的核心任务，不在 C1 做**。C1 的挂载只求「不再硬编码、新类型能挂进去」，不追求最终视觉。

**点开后三层下钻（类比看图片 lightbox）**：
```
画廊「指标结果表」卡 → 点开
  ├─ 综述层（默认只渲染此层，≈组数行，十几行，不卡）
  │    组A  n=24  open_arm_ratio 0.42 ▁▂▃▅(分布缩略)  ⚠2离群
  │    组B  n=21  open_arm_ratio 0.38 ▁▃▅▂
  ├─ 点某组 ▸ 懒加载展开该组 subject 明细（row.getIsExpanded 才渲染）
  │    subject_07  0.41
  │    subject_12  0.79 ●  ← IQR离群:圆点标记+100%不透明
  │    (非离群行 ~70% 不透明度)
  └─ [下载完整 CSV]（全量全列，不受折叠影响）
```

**呈现规范**（调研支撑）：
- **不平铺/不卡顿**：默认只渲染组综述（十几行），subject 明细懒展开。**不上虚拟化**（50 行内无需；默认根本不渲染上百行）；兜底：某组展开后真超百行，仅对该展开面板内部上 TanStack Virtual。
- **均值会骗人**：每个组摘要指标旁配分布缩略（sparkline/raincloud thumb，编码方向不编码精度，邻接真实数字）。
- **离群轻量标记**（日式克制）：IQR 判异 → 离群行圆点/旗标 + 不透明度对比（离群 100% / 其余 70%），**颜色叠加图标冗余、绝不只靠颜色、绝不红绿**；单一强调色（色盲安全，蓝 #0072B2 或橙 #E69F00）。
- **数字排版**：数值列右对齐 + `font-variant-numeric: tabular-nums`（选字体前验证支持）；表头 semibold、不全大写；留白为分隔、不用全网格线。
- 复用现有画廊/lightbox 交互范式，与看图一致。

### 模块 4：边界与不做什么
- ❌ 不把 workspace 的 `handoff_*.json` / `inputs_*.json` / raw 上传塞进画廊（过程/内脏文件）。
- ❌ 不靠 LLM 生成表（确定性导出，数据同源 handoff）。
- ❌ 不在 C1 做画廊整体 layout 重设计 / UI/UX 美化（那是 C2——C1 只做最小可扩展挂载）、不做框选追问（那是 C3）。
- ❌ 不默认平铺全量 subject（性能 + 过载红线）。

---

## 四、验收

1. **后端**（TDD）：code-executor 完成后，outputs/ 确定性出现 `metrics_table.csv` + `.json`；CSV/JSON 数值与 handoff `per_subject` **一致**（断言）；**不含内脏字段**（断言 json 无 gate_signals/handoff/assessment —— 防 vacuous）；outlier_flags 与 IQR 手算一致。
2. **端点**：`data-table` 返回真 CSV（可下载）；JSON 端点返回剥内脏的干净结构；画廊列表含「指标结果表」类。
3. **前端**（TDD）：画廊出现「指标结果表」卡；点开默认只渲染组综述层（断言 DOM 不含上百 subject 行 —— 性能红线被测）；展开某组才渲染该组明细；离群行有标记；CSV 下载全量。
4. **裸导入两生产入口**（改了 code-executor/seal/artifacts 链）：`import app.gateway` + `make_lead_agent` 0 退出。
5. `make test` + `pnpm check` + `vitest` 绿。

---

## 五、关联

- 同范式：#245（source_filename 后端贯穿）、PR#213（run_chart_plan 确定性登记）、seal 确定性产物——C1 与它们同构（后端确定性产干净产物，前端消费）。
- 现有基建：画廊 `thread-assets-panel` / `gallery` 组件（磁盘为真相）、`data-table` placeholder 端点。
- 调研支撑：Shneiderman overview-first、TanStack Table 懒展开（盈亏平衡 50 行）、行为学 raincloud（2024-2025）、IQR 离群、Wong 色盲安全色板。
- 后续：C2（画廊整体 layout 重设计——从硬编码段叠升级为**可扩展多产物布局** + UI/UX，ui-ux-pro-max + 日式；由 C1 加第三类产物驱动，现有真实需求）、C3（产物框选→输入框追问，Gemini 式；依赖 C1 的结果数据可框选）。
