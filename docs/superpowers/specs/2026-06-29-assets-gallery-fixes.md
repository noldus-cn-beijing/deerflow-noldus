# Spec: 右侧产物面板 gallery 三修 —— 图片预览画框 + box filter 漏图 + 单样本图按类型分类

> 状态：待实施（三处根因均坐实）
> 归属：前端 + 1 处后端（2026-06-29 dogfood）
> SSOT 守则：chart 分类元数据**只来自后端**，前端不正则猜（`artifacts.py` 注释明示）。

## Context

右侧 thread 产物面板（`thread-assets-panel` + `ArtifactGallery`）dogfood 暴露三个独立问题，同区一并修。

---

## 问题 1（bug）：图片预览画框不跟随 UI，看不到下载按钮

### 根因（坐实）
`src/components/workspace/artifacts/gallery/gallery-lightbox.tsx`：
- 外框 `div.relative flex max-h-[90vh] max-w-[90vw] flex-col`（line 65）—— **有 max-h 但无 `overflow` 也无 `min-h-0`**。
- 下载/外链/关闭按钮行在**顶部**（line 68-100），图在下方 `<img class="max-h-[80vh] max-w-[80vw] object-contain">`（line 114-119）。
- 按钮行(~40px)+图(可达 80vh)+底部 caption 之和可超 90vh；flex-col 无 overflow 处理 → 内容溢出，而 lightbox 是 `fixed inset-0 items-center justify-center` 居中 → 过高内容**上下等量溢出视口**，顶部按钮行被推到**视口上方**够不到。

### 修法
外框改 `flex flex-col`（保 max-h/max-w），**按钮行 `shrink-0`**，**图区包一层 `flex-1 min-h-0 overflow-auto`**，`<img>` 改 `max-h-full max-w-full object-contain`。这样图区在剩余空间内可滚动，按钮永远可见、随视口/面板尺寸自适应。

---

## 问题 2（bug）：box filter 选中后部分图看不到（如 zone_entry_distribution）

### 根因（坐实 —— 在后端，不是前端）
`packages/agent/backend/app/gateway/routers/artifacts.py` `_derive_chart_type`（line ~207）：
```python
for token in ("trajectory","timeseries","time_series","box","bar","heatmap","violin","scatter","line"):
    if token in text: return ...
return None
```
`zone_entry_distribution` 的 chart_id/script 不含任何 token → 返回 `None` → `chart_type=null`。前端 `use-gallery-filters.ts:34` 用精确匹配 `m.chart_type !== filters.chartType`，`null !== "box"` 为真 → 被过滤掉。证据缩略图 `plot_zone_entry_distribution_s23.thumb.webp`。

> 真实类型：`zone_entry_distribution` 应是 box 类（分区进入次数分布的箱线/分布图）但 token 表漏了它。

### 修法（后端，守 SSOT）
`_derive_chart_type` 的脆弱 substring 启发式 → 补全/改成显式映射表。优先**显式 map**（chart_id → type），把已知图名（zone_entry_distribution 等）正确归类；未知仍 `None`。
- 来源对齐：该函数注释说"与 present_file_tool 同构，SSOT 一处推导"——改时**两处同步**（grep 调用方），别只改一处造漂移。
- 可选前端兜底：`chart_type=null` 的图在「全部/未分类」facet 下仍可见（不被任何具体 type filter 永久吞掉），但**正解是后端把类型补对**。

### TDD
后端 `test_run_chart_plan.py` / artifacts 测试加：`zone_entry_distribution` → `chart_type=="box"`（或正确类）。

---

## 问题 3（feature）：单样本图按类型再细分、各自折叠归类

### 现状
`artifact-gallery.tsx:37-38`：按 `output_mode` 分两组——aggregate/summary（line 87+ 内联展示）+ per_subject 单样本（line 118+ **一个大折叠桶**，展开后单一 virtualized grid，`gallery-grid.tsx:16 VIRTUALIZE_THRESHOLD=24`）。单样本 100+ 张时一坨，难找。

### 可行性
`ArtifactMeta`（`core/artifacts/types.ts`）已有 `chart_type` 字段（后端 plan_charts.json 来）、facet 已提取 chartTypes。**数据齐备**，可按 type 分组。

### 修法
`artifact-gallery.tsx`：per_subject 组按 `chart_type` 二次分组 → 每类型一个**独立可折叠子区**（复用现有折叠按钮 + GalleryGrid 模式），virtualization **按子区各自**触发（box 28 张虚拟化、trajectory 15 张平铺）。
- 依赖问题 2 先修：否则 `chart_type=null` 的图全落「未分类」桶。**实施顺序：问题 2 → 问题 3**。
- 顶层 facet filter 仍跨所有子区生效，不冲突。

### TDD
`artifact-gallery.test.tsx` 加：per_subject 按 type 分出多个折叠子区、空类型不建子区、各子区计数正确。

---

## 改动文件
- 前端：`gallery-lightbox.tsx`（问题1）、`artifact-gallery.tsx`（问题3）、（可能）`use-gallery-filters.ts`（问题2 前端兜底）+ `artifact-gallery.test.tsx`。
- 后端：`app/gateway/routers/artifacts.py` `_derive_chart_type` + present_file_tool 同构处（问题2）+ 测试。

## 验收
- 问题1：lightbox 打开任意图，缩放浏览器/面板，下载/关闭按钮始终可见、图随框自适应。
- 问题2：box filter 选中后 `zone_entry_distribution` 等图可见；后端 `_derive_chart_type` 单测绿。
- 问题3：单样本区按类型分多个折叠子区、各自展开/虚拟化正常。
- `pnpm check` 0 error；`npx vitest run` + 后端 artifact 测试绿；本地 localhost:2026 目测三项。
