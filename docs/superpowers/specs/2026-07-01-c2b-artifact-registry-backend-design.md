# 设计 spec：C2b 画廊后端产物列表注册表（ext→kind+build_meta SSOT + 薄端点门面）（2026-07-01）

> 生成式 UX 路线图 C 系列。把画廊后端 4 个并列 list 端点各自 ad-hoc 的 kind/ext 判断 + meta 构造**收敛成一个注册表 SSOT**，端点变薄门面。**纯重构、行为字节级不变**——回归测试断言新旧输出逐字节相等。
>
> 依赖 C1（已落地 #255，四端点已在盘）。**纯后端、零 D1/D2 依赖 → 可立即派实施。** 配套前端布局重做见 `2026-07-01-c2-gallery-layout-redesign-design.md`（C2，实施前置 D1/D2）。

---

## 目标

`app/gateway/routers/artifacts.py` 现有 4 个并列 list 端点（`/charts` `/reports` `/data` `/metrics-table`），每个各自判 `ext→kind` + 各自建 `ArtifactMeta`。加一类产物 = 改多处；kind 判断散落无 SSOT。C2b 建一个 `ArtifactRegistry` SSOT + 端点变薄门面，**同样输入同样输出**（不改进输出，只收敛散落逻辑）。

## 范围（brainstorm 锁定）

- **只收敛现有四端点**的散落 kind/ext + meta 构造逻辑。**不新增产物类型。**
- 注册表职责 = `ext→kind` 映射 **+** 每 kind 的 `build_meta` builder 引用（连 meta 构造一起收，不只 ext→kind）。
- `build_meta` 逻辑**字节级从现端点搬出**，不改行为。

## 架构：注册表内核 + 薄端点门面

### 单元 1：`ArtifactRegistry`（注册表 SSOT，纯数据 + 纯函数）

- **ext → kind 映射**（静态 dict，SSOT）：`.png→chart`、`.md/.html→report`、`.csv/.json→data`。
- **每 kind 一个 `build_meta` builder 引用**：
  - `chart` → 现有 `list_chart_artifacts` 的 join `plan_charts.json` 逻辑（保 `chart_type`/`output_mode`/`source_filename` 等元数据）。
  - `report` → 现有 report meta 构造逻辑。
  - `data` → 现有 data meta 构造逻辑（`.csv/.json`）。
- **纯 Python、stdlib**、无 deerflow import（可单测、picklable）。
- **接口**：`kind_for_ext(ext) -> kind | None`；`build_meta_for_kind(kind) -> builder`。

### 单元 2：`list_artifacts_by_kind(kind, thread_id, request)`（统一扫盘 + 分发）

- 扫 `outputs/` → 按 ext 经注册表定 kind → 调对应 `build_meta` builder → 返回 `ArtifactMeta[]`。
- **接口**：输入 kind + thread；输出 `ArtifactMeta[]`（与现端点同结构）。

### 单元 3：端点门面（变薄）

- `/charts` `/reports` `/data` 各变成一行 `return list_artifacts_by_kind("chart"/"report"/"data", ...)`。
- **保持不动**（不进注册表——它们非「列表」逻辑）：
  - `/metrics-table`（返回单个特定 `metrics_table.json` 文件）。
  - `/data-table`（`metrics_table.csv` 下载，`Content-Disposition: attachment`）。
  - catch-all serve 端点 `/{path:path}`。

## 关键约束：行为字节级继承

重构目标是「**同样输入 → 同样输出**」，不是「改进输出」。`build_meta` 逻辑逐字节从现 `list_*` 搬出。回归测试断言新旧输出**逐字节相等**（path/kind/顺序/元数据全同）。

## 数据流（不变）

```
outputs/ 磁盘扫描 (单元2)
  → 按 ext 经 ArtifactRegistry (单元1) 定 kind
  → 调该 kind 的 build_meta builder（chart join plan_charts.json / report / data）
  → ArtifactMeta[]
  → 端点门面 (单元3) 原样返回
  → 前端 useThreadAssets 契约字节级不变
```

## 错误处理 / 韧性（保现状）

| 场景 | 处理（保现有行为）|
|---|---|
| `plan_charts.json` 缺失/坏掉 | chart 端点仍返图（现状「plan 是增强非前提」保留）|
| outputs/ 为空 | 返回 `[]`（现状）|
| 未知 ext | 按现状行为（多半忽略，不进任何 kind 列表）|
| 文件落盘中/半写 | 现有扫盘行为不变（磁盘为真相）|

## 测试策略（守 TDD 强制 + 防 vacuous）

**核心 = 「行为字节级不变」对账测试**（纯重构，验收=输出与重构前逐字节相等）：

1. **注册表映射单测**：`.png→chart`、`.md→report`、`.html→report`、`.csv→data`、`.json→data`、未知 ext→现状行为。
2. **build_meta 对账测试**（防 vacuous 核心）：造磁盘 fixture（几张 .png + plan_charts.json + report.html + metrics_table.csv/json），断言重构后 `list_artifacts_by_kind` 输出与重构前旧端点输出**逐字节相等**。
   - **防 vacuous 探针**：故意把注册表里 chart 的 build_meta 引用指错 builder → 对账测试必须变红（证测试真在比对内容，非恒真）。
3. **chart 的 plan_charts.json join 保真**：断言 chart meta 仍带 `chart_type`/`output_mode`/`source_filename`（6 不变式 + C2 依赖的元数据，重构中不能丢）。
4. **端点门面回归**：`/charts` `/reports` `/data` HTTP 层断言仍返回原结构（保 `useThreadAssets` 契约）。
5. **韧性不变**：plan_charts.json 缺失时 chart 端点仍返图。

**重构安全网**：改前跑现有 `test_artifacts_*` 建 baseline；重构后全绿 + 新对账测试绿。

## sync 友好性

- 全部在 `app/gateway/routers/artifacts.py`（gateway app 层，**deerflow 子树外**）+ 新注册表模块放 `app/gateway/` 下。
- **零改** `packages/harness/deerflow/` 子树 → `sync-deerflow.sh` 不门控。100% sync 友好。

## 边界（YAGNI）

- ❌ 不新增产物类型（只收敛现有四端点）。
- ❌ 不动 `/metrics-table`（单文件 JSON）、`/data-table`（CSV 下载）、catch-all serve 端点。
- ❌ 不做 plugin / entry-point 动态注册（注册表是**静态 dict**，非运行时可插拔框架——避免过度工程）。
- ❌ 不改前端（`useThreadAssets` 契约字节级不变）。

## 依赖时序

- **可立即派实施**：C1 四端点已落地（#255），C2b 零 D1/D2 依赖。
- C2b 先落地让 C2 前端重做时数据源已收敛（更干净），但 **C2b 不硬阻塞 C2**（契约字节不变）。
- 链：`C1✅ → C2b（本 spec，可即派）→ C2 → C3`。

## 验收标准

1. 一个 `ArtifactRegistry` SSOT：ext→kind + 每 kind build_meta builder。
2. `/charts` `/reports` `/data` 变薄门面，各调 `list_artifacts_by_kind`。
3. 对账测试证新旧输出逐字节相等 + 防 vacuous 探针。
4. chart 的 plan_charts.json 元数据保真（chart_type/output_mode/source_filename 不丢）。
5. 现有 artifacts 测试全绿；后端零改 deerflow 子树。

## 关联

- 上游：C1（`2026-06-30-c1-metrics-table-export-and-gallery-overview-first-design.md`，#255 已落地）。
- 下游：C2（`2026-07-01-c2-gallery-layout-redesign-design.md`，消费收敛后的端点）。
- 现有代码：`app/gateway/routers/artifacts.py`（4 list 端点 + catch-all）、`list_chart_artifacts`（chart join 逻辑）。
- 守 memory：`feedback_single_source_of_truth`（kind 判断单一 SSOT）、`feedback_sync_full_follow_upstream_infra`（app 层子树外）、`feedback_pr115_stage1_equivalence_baseline_is_hollow`（对账 baseline 必须打开看内容防假绿）。
