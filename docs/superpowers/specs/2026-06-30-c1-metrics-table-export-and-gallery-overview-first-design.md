# 设计 spec：C1 指标结果表导出（纯后端：确定性导出 + 端点）（2026-06-30）

> 生成式 UX 路线图 C 系列第一步。让用户在画廊看到「**计算出的指标结果**」（非过程/raw 文件）。本 spec 只设计 C1，交别的 agent 实施。C2（画廊 UI/UX）、C3（产物框选追问）各自后续单独 brainstorm。
>
> **⚠️ 2026-06-30 修订（C2 brainstorm 时对齐）**：C1 **收窄为纯后端**——确定性导出 `metrics_table.csv/.json` + `data-table`/JSON 端点 + 画廊列表暴露「指标表」类。**前端画廊呈现（指标表卡挂载 + 概览优先 + 三层下钻）整体移交 C2**。原因：C1/C2 都改 `thread-assets-panel.tsx` 且都未实施，C2 已 brainstorm 出「概览优先 + 分类可扩展（注册驱动）」的完整画廊重设计并吞掉 C1 的最小挂载——让 panel 只被 C2 改一次，消除「C1 最小挂载 → C2 重设计」两次过渡。详见 C2 spec `2026-06-30-c2-gallery-layout-redesign-design.md`。
>
> 呈现范式（Shneiderman overview-first + 组摘要懒展开 + 分布缩略 + IQR 离群轻量标记）现归 **C2**（前端呈现层）；C1 只负责把这些呈现所需的**干净数据**（含 IQR outlier_flags、distribution thumb 数据）确定性导出。

---

## 一、目标与已拍板决定

让研究员看到 code-executor 算出的指标结果——**干净的结果，不是过程文件、不是 raw 上传**。

已拍板（C1 范围 = 纯后端）：
1. **(甲) 后端确定性导出**：数据现成（困在内部 `handoff_code_executor.json` 的 `per_subject` + `metrics_summary`），C1 把它确定性导出成干净用户产物（不靠 LLM，同 #245/run_chart_plan 范式），画廊收它。**不让前端耦合内部 handoff 格式。**
2. **(丙) 两产物同源导出**：CSV（研究员下载进 SPSS/Prism）+ JSON（前端渲染表格，剥内脏；JSON 含 C2 呈现所需的 outlier_flags + distribution thumb 数据）。
3. **(前端呈现归 C2)**：「概览优先 + 三层下钻」的画廊呈现由 C2 做；C1 只保证导出的 JSON 携带 C2 所需的全部干净数据（组综述 + per_subject + outlier_flags + 分布缩略数据）。

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

### 模块 3：前端呈现 —— 整体移交 C2（C1 不做）

**C1 不写任何前端。** 画廊纳入「指标结果表」卡 + 概览优先 + 三层下钻（综述→组→subject 懒展开）+ 离群标记 + 分布缩略呈现，**全部归 C2**（`2026-06-30-c2-gallery-layout-redesign-design.md`，注册驱动的可扩展多产物画廊）。

C1 对 C2 的**唯一契约** = 模块 1 导出的 `metrics_table.json` 结构必须自带 C2 呈现所需的全部干净数据：
- 组综述层数据：`groups[].{group, n, metrics:{<指标>:{mean, std, distribution_thumb}}}`。
- subject 明细层数据：`per_subject[].{subject, group, <指标>:值, outlier_flags:{<指标>:bool}}`。
- 已剥内脏（无 gate_signals/handoff/assessment）。
- IQR outlier_flags 后端确定性算好（前端不重算）。

> 即「呈现所需的判断（离群/分布）在后端定，前端只渲染」——C1 把数据备齐，C2 只管好看地呈现。

### 模块 4：边界与不做什么
- ❌ 不把 workspace 的 `handoff_*.json` / `inputs_*.json` / raw 上传塞进画廊（过程/内脏文件）。
- ❌ 不靠 LLM 生成表（确定性导出，数据同源 handoff）。
- ❌ **不写任何前端**（画廊纳入 + 概览呈现 + 三层下钻 + UI/UX 全归 C2）、不做框选追问（那是 C3）。
- ❌ 不在 JSON 里留内脏字段（防 vacuous，验收断言）。

---

## 四、验收

1. **后端**（TDD）：code-executor 完成后，outputs/ 确定性出现 `metrics_table.csv` + `.json`；CSV/JSON 数值与 handoff `per_subject` **一致**（断言）；**不含内脏字段**（断言 json 无 gate_signals/handoff/assessment —— 防 vacuous）；outlier_flags 与 IQR 手算一致；JSON 自带 C2 呈现所需的 groups 综述 + distribution_thumb 数据（断言结构完整）。
2. **端点**：`data-table` 返回真 CSV（可下载）；JSON 端点返回剥内脏的干净结构；画廊列表含「指标结果表」类。
3. **裸导入两生产入口**（改了 code-executor/seal/artifacts 链）：`import app.gateway` + `make_lead_agent` 0 退出。
4. `make test` + `pnpm check` 绿（C1 不新增前端，前端 vitest 由 C2 负责）。

> 前端验收（画廊出现指标表卡、概览层懒渲染性能红线、离群标记、CSV 下载全量）归 **C2**。

---

## 五、关联

- 同范式：#245（source_filename 后端贯穿）、PR#213（run_chart_plan 确定性登记）、seal 确定性产物——C1 与它们同构（后端确定性产干净产物，前端消费）。
- 现有基建：画廊 `thread-assets-panel` / `gallery` 组件（磁盘为真相）、`data-table` placeholder 端点。
- 调研支撑（呈现层，归 C2 用）：Shneiderman overview-first、TanStack Table 懒展开（盈亏平衡 50 行）、行为学 raincloud（2024-2025）、IQR 离群、Wong 色盲安全色板。
- 后续：**C2（画廊整体重设计 + 吞掉 C1 前端呈现）**——`2026-06-30-c2-gallery-layout-redesign-design.md`，注册驱动的可扩展多产物画廊 + 概览优先 + 指标表三层下钻；C3（产物框选→输入框追问，Gemini 式；依赖 C1 的结果数据可框选）。
