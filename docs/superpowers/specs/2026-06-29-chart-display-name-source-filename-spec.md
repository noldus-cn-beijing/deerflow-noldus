# 实施 spec：图表展示名带来源 raw data 文件名（2026-06-29）

> 实施级文档，供 agent 照此直接写代码。**只加一个 `source_filename` 字段贯穿 plan→json→ArtifactMeta→前端展示，不改物理文件名。**

---

## 〇、现象与根因（已取证）

**现象**：一个 EPM 多文件 thread 生成 113 张图，命名 `plot_heatmap_s0.png`~`plot_heatmap_s27.png`（按 subject index）。28 张 heatmap 在前端全显示成 `sN`/chart_id，用户无法分辨哪张对应哪个 trial。

**关键事实**：subject→源文件映射**磁盘上本来就有** ——
- `user-data/workspace/inputs_heatmap_s0.json` = `["/mnt/user-data/uploads/Raw data-EPM-Xuhui-Trial     1.xlsx"]`
- `inputs_heatmap_s14.json` = `["...Trial    15.xlsx"]`

即 `raw_files[idx]` 在 plan 生成时手上就有，只是没被带进图名/展示。

**根因链**（4 段，都浅）：

| 阶段 | 位置 | 问题 |
|---|---|---|
| Plan 生成 | [resolve.py:1151-1176](../../../packages/ethoinsight/ethoinsight/catalog/resolve.py) `_chart_to_plan()` per_subject 循环 | `for idx, raw_file in enumerate(raw_files)` 里有 `raw_file`，但只用 `idx` 拼 `plot_{id}_s{idx}.png`，没记源文件名 |
| Schema | [schema.py:209-216](../../../packages/ethoinsight/ethoinsight/catalog/schema.py) `PlanChart` dataclass | 无承载源文件名的字段 |
| 序列化 | [resolve.py:1440](../../../packages/ethoinsight/ethoinsight/catalog/resolve.py) `plan_charts_to_dict()` 的 charts 列表推导 | 序列化 `{"id","script","input","output","subject_index"}`，无源文件名 |
| 后端元数据 | [artifacts.py](../../../packages/agent/backend/app/gateway/routers/artifacts.py) `list_chart_artifacts()` | 从 `plan_charts.json` 读 chart 条目填 `ArtifactMeta`，但 plan 里没源文件信息，`filename`/`subject` 空 |
| 前端展示 | [gallery-grid.tsx:41,82-85](../../../packages/agent/frontend/src/components/workspace/artifacts/gallery/gallery-grid.tsx) `ThumbCard` | `alt` 与悬停标题只用 `chart_id+metric+subject`，无源文件名 |

---

## 一、目标与验收

### 目标
每张 per-subject 图在前端展示时带来源 raw data 文件名（如 `Trial 1` / `Trial 15`），使多文件下 N 张同类图可区分。**不改物理 png 文件名**（改名动磁盘约定 + 脚本输出路径 + 所有下游引用，风险大且无必要）。

### 验收标准
1. 跑一个多文件 paradigm（或用现有 thread `dcde1446-...` 的数据复跑），生成的 `plan_charts.json` 每个 per_subject chart 条目含 `source_filename`（值如 `Raw data-EPM-Xuhui-Trial     1.xlsx` 的 basename）。
2. `GET /api/.../artifacts/charts` 返回的 ArtifactMeta 含来源文件名字段（非空，对 per_subject 图）。
3. 前端画廊图卡的 alt + 悬停标题显示来源（去扩展名的 trial 名），28 张 heatmap 不再全同名。
4. **aggregate 图**（output_mode=="aggregate"，多文件合一张）不强加来源文件名（它本就跨文件，`source_filename` 留空/"all"，前端不显示来源段）。
5. 既有测试不红：`cd packages/ethoinsight && pytest tests/`（catalog/resolve 相关）+ `cd packages/agent/frontend && pnpm check`。
6. TDD：先写断言「per_subject plan 的 chart.source_filename == basename(raw_files[idx])」的单测（catalog resolve 层），watched RED → 改后 GREEN。

---

## 二、改动清单（ethoinsight 2 文件 + 后端 1 + 前端 1）

### 改动 1：PlanChart schema 加字段

[schema.py](../../../packages/ethoinsight/ethoinsight/catalog/schema.py) 的 `PlanChart` dataclass（209 行起），在 `subject_index` 附近加：

```python
@dataclass
class PlanChart:
    id: str
    script: str
    # input: str
    output: str
    subject_index: int = 0  # 0-based index into inputs.raw_files
    source_filename: str = ""  # 源 raw data 文件 basename（per_subject 专用；aggregate 留空）
    display_name_zh: str = ""
    # ... 其余字段原样
```

- 字段加在 dataclass 现有字段序列里、**带默认值 `""`**（避免破坏既有按位置构造的调用点；若全是 kwargs 构造则位置不敏感，仍加默认值最安全）。
- **不动** `MetricEntry` / `ChartEntry` 等其它 dataclass（它们是 catalog 定义侧，不承载运行时源文件）。

### 改动 2：`_chart_to_plan()` 填入源文件名

[resolve.py](../../../packages/ethoinsight/ethoinsight/catalog/resolve.py) per_subject 循环（约 1151-1176），在构造 `PlanChart(...)` 时加 `source_filename`：

```python
for idx, raw_file in enumerate(raw_files):
    suffix = f"_s{idx}" if multi else ""
    inputs_name = f"inputs_{ch.id}{suffix}.json"
    inputs_virtual = _materialise_inputs(inputs_name, [raw_file])
    output_path = str(Path(effective_outputs) / f"plot_{ch.id}{suffix}.png")
    # ... 既有 args 处理 ...
    plans.append(
        PlanChart(
            id=ch.id,
            script=ch.script,
            input=str(Path(effective_workspace) / inputs_name),
            output=output_path,
            subject_index=idx,
            source_filename=Path(raw_file).name,  # ← 新增：basename，保留原文件名（含空格）
            display_name_zh=ch.display_name_zh,
            # ... 其余原样
        )
    )
```

- 用 `Path(raw_file).name` 取 basename（如 `Raw data-EPM-Xuhui-Trial     1.xlsx`），保留原始文件名（含多空格——展示层再处理）。
- **aggregate 分支**（同函数内 output_mode=="aggregate" 的另一路径）：`source_filename` 不设（默认 `""`）或显式设 `""`，因为它跨所有文件。

### 改动 3：序列化带出新字段

[resolve.py:1440](../../../packages/ethoinsight/ethoinsight/catalog/resolve.py) `plan_charts_to_dict()` 的 charts 列表推导，加 `source_filename`：

```python
{"id": c.id, "script": c.script, "input": c.input, "output": c.output,
 "subject_index": c.subject_index, "source_filename": c.source_filename}
for c in pc.charts
```

> ⚠️ 同文件 1407/1464 附近还有 metrics 的序列化（`m.id/m.output/m.subject_index`）——**那是 metric 不是 chart，不要动**。只改 1440 这条 chart 推导。

### 改动 4：后端 ArtifactMeta 填来源

[artifacts.py](../../../packages/agent/backend/app/gateway/routers/artifacts.py) `list_chart_artifacts()` 构造 meta dict 处，加一行读 `plan_charts.json` 条目的 `source_filename`：

```python
meta: dict[str, Any] = {
    "path": virtual,
    "kind": "chart",
    "chart_id": chart_id or None,
    "output_mode": entry.get("output_mode") or "per_subject",
    # ... 既有字段 ...
    "source_filename": entry.get("source_filename") or None,  # ← 新增
}
```

- 同步在前端 `ArtifactMeta` 类型（[types.ts](../../../packages/agent/frontend/src/core/artifacts/types.ts)）加可选 `source_filename?: string;`（已有 `filename?` 但语义是「报告产物标题」，**不复用**，避免语义混淆——新加专用字段）。

### 改动 5：前端画廊展示来源

[gallery-grid.tsx](../../../packages/agent/frontend/src/components/workspace/artifacts/gallery/gallery-grid.tsx) `ThumbCard`（约 41/82-85）：

```tsx
// 去扩展名 + 折叠多空格的来源短名
const sourceLabel = meta.source_filename
  ? meta.source_filename.replace(/\.[^.]+$/, "").replace(/\s+/g, " ").trim()
  : null;

const alt = [sourceLabel, meta.chart_id, meta.metric].filter(Boolean).join(" · ") || meta.path;

// 悬停/角标显示：来源优先
{(sourceLabel || meta.metric) && (
  <div className="absolute right-0 bottom-0 left-0 truncate bg-black/40 px-2 py-1 text-xs text-white">
    {sourceLabel ? `${sourceLabel}${meta.metric ? " · " + meta.metric : ""}` : meta.metric}
  </div>
)}
```

- `replace(/\s+/g, " ")` 折叠 `Trial     1` 的多空格为 `Trial 1`，展示干净。
- aggregate 图 `source_filename` 空 → `sourceLabel` null → 退化为原 `chart_id+metric` 显示，不破坏。

---

## 三、不做什么

- ❌ 不改物理 png 文件名（`plot_{id}_sN.png` 保持，磁盘/脚本约定不动）。
- ❌ 不复用 `ArtifactMeta.filename`（那是报告产物标题语义）。
- ❌ 不动 metric 的序列化（1407/1464/1470）。
- ❌ 不给 aggregate 图强加单一来源名。

---

## 四、风险 / 注意

1. **多空格文件名**：源文件名是 `Raw data-EPM-Xuhui-Trial     1.xlsx`（多空格），后端原样存、前端展示时折叠空格。别在后端就改名，保留原始值供其它用途。
2. **PlanChart 按位置构造**：加字段务必带默认值 `""`，否则任何按位置 `PlanChart(a,b,c,...)` 的调用点会错位。改后 grep `PlanChart(` 确认所有构造点。
3. **向后兼容**：旧 `plan_charts.json`（无 source_filename）→ 后端 `entry.get("source_filename")` 返回 None → 前端退化，不报错。
4. 改 ethoinsight 后跑 `pytest tests/`；改后端后**裸导入两入口**（守 CLAUDE.md 闭环铁律）：`PYTHONPATH=. python -c "import app.gateway"`。

---

## 五、实施顺序

1. TDD：catalog resolve 层写单测断言 per_subject 的 `source_filename`（RED）。
2. 改动 1（schema）→ 2（_chart_to_plan）→ 3（序列化）→ pytest GREEN。
3. 改动 4（后端 + types.ts）→ 裸导入验证。
4. 改动 5（前端）→ `pnpm check`。
5. 多文件复跑验证 plan_charts.json + 前端画廊。
6. 精确路径 commit。
