# Spec：产物画廊 — 代表图 inline + 独立画廊界面（Phase 0 · 第 3 项）

> 类型：**一次性实施 spec**（前端为主 + 后端产物契约小改）
> 日期：2026-06-24
> 母方案：[docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)（硬伤 A / §4.2）
> 依赖：[2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md](2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md)（曲线/时长/语义色 token）
> 适用层：前端 `components/workspace/artifacts/` + `core/artifacts/` + `core/threads/types.ts`；后端 `present_file` 产物契约（`tools/builtins/present_file_tool.py` + `thread_state.py` 的 `merge_artifacts`）
> 设计准则来源：`ui-ux-pro-max`（domain chart：box plot=AA / large-dataset / data-table；domain ux：Image Optimization、virtualize-lists、progressive-disclosure；Quick Reference §3/§10）
> 一句话：**聊天流里只 inline 少量代表图（aggregate 汇总图）+ 两个并列入口「打开产物画廊」/「下载全部 ZIP」；全部图（上百张 per-subject）收进独立画廊界面**，走图墙最佳实践（分面筛选 + 虚拟化 + 小倍数对比 + 失败/截断显式）。后端把 `plan_charts.json` 已有的 per-chart 元数据接出到前端（决策1=路 A）。
>
> **2026-06-25 回写补强（三处本会话澄清已固化）**：① **渲染压力三成本模型 + 虚拟化是 ②③（DOM/内存）唯一解,`loading=lazy` 治不了**（§3.1.5,实施 agent 必读,别误用 lazy）;② **下载全部 ZIP 提升为第 1 层主路径**（零渲染带走全部图,§3.1.7）;③ **画廊是 per-thread,默认整 thread 累积靠分面组织**（§3.4.1）。另:**PNG 实测拍板后端缩略图=强烈建议做**（trajectory 1–2.7MB 是主成本,§3.1.6）。

---

## 〇、为什么这是 Phase 0 的"重头"

母方案硬伤 A：后端 `run_chart_plan` 能并行产出上百张 per-subject 图,每张经 `present_file` 进 `thread.values.artifacts`——但前端拿到的是**扁平 `string[]`**,`ArtifactFileList` 直接 `grid-cols-2` 全量 `<img>`（`artifact-file-list.tsx:94-135`）。研究员（产品的全部受众）要在 100+ 缩略图里**肉眼大海捞针**,且首屏全量 `<img>` 直接违反 `ui-ux-pro-max` `virtualize-lists`（50+ 必虚拟化）。

**"出图后找不到图"直接抵消整条分析流水线的价值。** 这是研究员体验的最大单点痛。

后端**明明有结构化元数据**（`plan_charts.json`：`chart_id` / `output_mode=per_subject|aggregate` / `script` / paradigm，母方案后端调研 §3-4 坐实）却被前端拍平成路径字符串,丢光了研究员唯一关心的维度（哪张是 aggregate、哪张属于 subject 7 的 open_arm、哪些是 control 组）。

---

## 一、现状（带证据）

### 1.1 前端 `artifacts` 消费面（决策1 改契约必须全改）

`artifacts` 当前是 `string[]`,消费方 grep 出 **4 处 + 1 reducer**：

| 位置 | 用法 | 改动 |
|---|---|---|
| `core/threads/types.ts:8` | `artifacts: string[]` 类型定义 | → `ArtifactMeta[]`（向后兼容,见 §3.1） |
| `components/workspace/artifacts/context.tsx:13-14` | `artifacts: string[]` / `setArtifacts` | → `ArtifactMeta[]` |
| `chats/chat-box.tsx:53/67/69/152/166` | `setArtifacts(thread.values.artifacts)`、空判断、传给 `ArtifactFileList` | 适配新类型 |
| `core/artifacts/utils.ts:22` | `extractArtifactsFromThread` 返回 `artifacts ?? []` | 适配 |
| `core/threads/hooks.ts:385` | `onCreated` 初始化 `artifacts: []` | 不变（空数组兼容） |
| **后端** `thread_state.py` `merge_artifacts` reducer | 合并去重 `list[str]` | → 合并去重 `list[ArtifactMeta]`（按 path 去重） |
| **后端** `present_file_tool.py` | 写 `list[str]` 进 state | → 写 `list[ArtifactMeta]`（附元数据） |

> ⚠️ CLAUDE.md / memory 铁律：**改共享组件先 grep 所有消费者 + 全量回归**。上表即 grep 结果,实施时逐处改 + 加测试。

### 1.2 图片渲染相关基建（复用）

| 资产 | 位置 | 复用 |
|---|---|---|
| `urlOfArtifact` / `resolveArtifactURL` | `core/artifacts/utils.ts:4/55` | 拼缩略图/原图 URL,不变 |
| `ArtifactFileDetail`（单图/代码/md/html 详情 + 下载 + 新窗口） | `artifacts/artifact-file-detail.tsx` | 画廊单图详情复用（lightbox 或侧栏） |
| `ArtifactFileList`（现扁平网格） | `artifacts/artifact-file-list.tsx` | **被替换/重构**为 inline 代表图 + 画廊网格 |
| artifacts overlay 侧栏（ResizablePanel 60/40） | `chats/chat-box.tsx` | 保留（单图详情仍走它）;画廊是**另一个面**（§3.4） |
| 后端 `plan_charts.json`（chart 元数据 SSOT） | `run_chart_plan` 产出 | 元数据唯一来源,接出不重造 |
| `failed_charts` / `remaining_charts` | `run_chart_plan` 返回 | 失败/截断显式呈现的数据源 |

### 1.3 缺口
- `artifacts` 扁平 `string[]`,无 paradigm/output_mode/metric/subject/group 维度。
- `ArtifactFileList` 全量 `grid-cols-2` 渲染,无虚拟化、无筛选、无对比、无 aggregate/per-subject 区分。
- 失败/截断（`failed_charts`/`remaining_charts`）**前端完全不呈现**（静默少图,违反"无声截断"铁律）。

---

## 二、目标与非目标

### 目标
1. **后端（路 A）**：`present_file` 写 artifacts 时附带 per-chart 元数据,`artifacts` 升级成 `ArtifactMeta[]`,向后兼容旧 `string`。
2. **聊天流 inline**：只展示**代表图**（aggregate 全部,通常 ≤6 张）+ **两个并列入口**「打开画廊」（第 2 层探索）/「下载全部 ZIP」（第 1 层带走,零渲染,§3.1.7）+ 失败/截断提示。
3. **独立画廊界面**：分面筛选（范式/图类型/aggregate-vs-per-subject/组/subject + 搜索）+ **虚拟化网格（windowing,治 DOM/内存 ②③——不可用 `loading=lazy` 替代,§3.1.5）** + aggregate 优先 + per-subject 折叠懒加载 + 小倍数对比 + 单图 lightbox + CSV/数据表导出入口。画廊是 **per-thread**（整 thread 累积靠分面组织,§3.4.1）。
4. **失败/截断显式**：画廊明确呈现"N 张未生成（超预算/失败）+ 原因 + 补全入口"。
5. **零渲染逃生口（第 1 层）**：「下载全部 ZIP」一次点击带走全部图,**不经任何缩略图渲染**——后端一个流式 zip 端点,从"附属导出"提升为与"打开画廊"并列的主路径。
6. **（强烈建议,可与前端并行）后端缩略图**：`charts.py` dpi=300 致 PNG 偏大（实测 median 186KB / trajectory 1–2.7MB,§3.1.6）,后端用 Pillow 生成 `thumb_path`,前端缩略图优先用它（治成本 ①）。**不阻塞画廊上线**（无 thumb 时退化原图 + `decoding=async`）。

### 非目标
- ❌ 不改 `run_chart_plan` 的**执行逻辑**（只在 `present_file` 接出已有元数据）。
- ❌ 不重新发明图表分类——元数据唯一来源是 `plan_charts.json`（SSOT,不在前端正则猜）。
- ❌ 不做"图表交互/缩放/重绘"（图是后端渲染的 .png,前端只展示;交互式图表是 v1.0 愿景,不进 Phase 0）。
- ❌ 不动流式核心 / `groupMessages`。
- ❌ 路 B（前端正则解析文件名）**不做**——决策1 已定路 A。

---

## 三、设计

### 3.1 数据契约：`ArtifactMeta`（决策1=路 A，SSOT + 向后兼容）

**前端类型**（`core/threads/types.ts` 或 `core/artifacts/types.ts`）：

```ts
export interface ArtifactMeta {
  path: string;                        // 唯一标识 + URL 基础（必有，向后兼容锚点）
  kind?: "chart" | "report" | "data" | "skill" | "other";
  // 以下 chart 专属，来自 plan_charts.json
  chart_id?: string;                   // e.g. "box_open_arm"
  output_mode?: "aggregate" | "per_subject";
  paradigm?: string;                   // e.g. "epm"
  metric?: string;                     // e.g. "open_arm_time_ratio"
  subject?: string;                    // per_subject 才有
  group?: string;                      // control / treatment（若后端有）
  chart_type?: string;                 // box / bar / trajectory（从 script/chart_id 推或后端给）
  thumb_path?: string;                 // 后端 Pillow 缩略图（§3.1.6，建议做）；缺则前端退化原 path
  run_id?: string;                     // 可选增强（§3.4.1 方案 B：按轮次分面）；Phase 0 默认不填
}

// 向后兼容：老数据是 string → 归一化
export type ArtifactInput = string | ArtifactMeta;
export function normalizeArtifact(a: ArtifactInput): ArtifactMeta {
  return typeof a === "string" ? { path: a } : a;
}
```

**后端契约**（`present_file_tool.py` + `merge_artifacts`）：
- `present_file` 接收 charts 时,从 workspace 的 `plan_charts.json` 按 path 关联出元数据,写 `ArtifactMeta` 进 state（不是裸 path）。
- `merge_artifacts` reducer：**按 `path` 去重合并**（不是按整对象）;新旧并存时新覆盖。
- **向后兼容**：state 里历史的裸 string 仍合法;前端 `normalizeArtifact` 兜底。报告/技能等非 chart 产物 `kind` 标注即可,无 chart 元数据。

> SSOT 守则：`plan_charts.json` 是 chart 元数据**唯一来源**。`present_file` 只"关联接出",不重新计算分类。chart_type 若 `plan_charts.json` 没直接给,由后端从 `script`/`chart_id` 确定性推导（一处推,不在前端各猜一遍）。

> ⚠️ 后端改动属**受保护文件邻域**（`present_file_tool.py` / `thread_state.py`）——按 CLAUDE.md surgical 处理：grep 所有写/读 artifacts 的地方 + `make test` + 裸导入两生产入口（`import app.gateway` / `make_lead_agent`）。`merge_artifacts` 改 reducer **必须**带单测（红→绿：喂混合 string+meta、重复 path → 断言去重正确）。

### 3.1.5 渲染压力的本质：三个独立成本，`loading=lazy` 只治了一个（实施 agent 必读）

> 这一小节是整个画廊的**性能心智模型**。它决定了下面 §3.2–§3.3 为什么这样分层。**实施 agent 最易犯的错是"以为 `loading=lazy` 就够了"——它治不了内存和 DOM。**

研究员看 100+ 张图卡死，是**三个成本叠加**，现状 `artifact-file-list.tsx:89-135`（`imageFiles.map` 全量 `grid-cols-2` `<img loading="lazy">`）只缓解了其中一个：

| 成本 | 现状机制 | `loading="lazy"` 治了吗 | 唯一解 |
|---|---|---|---|
| ① 网络/解码 | 100+ `<img>` 并发拉**原图** PNG（实测最大 2.7MB/张，见 §3.1.6）+ 主线程解码 | **部分**（视口外不发请求） | 缩略图（减小每张体积）+ 虚拟化 |
| ② DOM 节点 | 100+ `<div><img>` 同时挂在 DOM，layout/paint/合成全量 | ❌ **没治**（DOM 节点照样在） | **虚拟滚动（windowing）** |
| ③ 内存（显存） | 进过视口的图，**已解码位图常驻显存**（一张 1766×1765 ≈ **12MB 显存**，与文件大小无关，按像素算）且滚出视口**不释放** | ❌ **没治** | **虚拟滚动（windowing）** |

**核心结论（不可绕过）**：
- **②③ 的唯一解是虚拟滚动（windowing）**——任意时刻 DOM 里只保留"可视区 + 上下缓冲"的十几张 `<img>`，滚出去的**真正卸载**（从 DOM 移除 → 解码位图被 GC → 显存释放）。这样 **DOM 节点数和内存占用恒定，不随图总数增长**（100 张和 1000 张占用一样）。
- `loading="lazy"` 只让"视口外的图不发网络请求"，但**只要 `<img>` 在 DOM 里、进过一次视口，DOM 节点和已解码位图就一直在**。所以 `lazy` 治 ① 的一半，对 ②③ 无效。**实施 agent 千万别用 `loading=lazy` 当虚拟化的替代。**
- 选型：用 `@tanstack/react-virtual`（与现有 `@tanstack/react-query` **同源同作者**，已在 `package.json:48`，轻量、维护活跃）。`react-virtual` 是 headless（只给定位计算，不给样式），契合本仓库 Tailwind + shadcn 风格。

### 3.1.6 PNG 实测大小 → 缩略图决策（已实测拍板）

§六.3 待拍板的"是否做后端缩略图生成"，**已实测一次真实 dogfood 产物（`backend/.deer-flow/threads/*/user-data/outputs/*.png`）+ 核查 `charts.py`**，结论可落：

- **根因**：`charts.py:42-43` 设 `savefig.dpi=300` + `savefig.bbox="tight"` → 发表级高分辨率，文件偏大。
- **实测分布（n=24 真实产出）**：`min=40KB / median=186KB / mean=596KB / max=2740KB`。
- **按图类型**（决定哪类是画廊洪流的主成本）：

  | 图类型 | 典型大小 | 像素 | 评级 |
  |---|---|---|---|
  | **trajectory（轨迹图，per-subject 主力）** | **1.3–2.7 MB** | 1766×1765 | 🔴 主成本 |
  | timeseries（时间序列） | 309–381 KB | 2970×1166 | 🟠 大 |
  | box plot（箱线图） | 40–87 KB | ~1170×1466 | 🟢 中等 |

- **拍板结论**：**强烈建议做后端缩略图生成**。判据（handoff §六.3：≥500KB 强烈建议 / 50–100KB 顺延）落在"建议"侧——**per-subject 洪流的主力恰是 trajectory（1–2.7MB/张）**，100 张轨迹图 = 100–270MB 原图流量 + 巨量解码/显存；缩略图（Pillow `Image.thumbnail((400,400))` → 几十 KB）把 ① 砍 1–2 个数量级。box plot 虽小（40–87KB），但 trajectory 一类就足以让"不做缩略图"在 per-subject 展开时卡死。
  - **后端最小改动**：`present_file` 写 chart 时，若 `kind=="chart"`，额外用 Pillow 生成 `<name>.thumb.webp`（或 `.thumb.png`），在 `ArtifactMeta` 加 `thumb_path?`。前端缩略图网格优先用 `thumb_path`，lightbox/详情用原 `path`。**属本 spec 后端范围的合理增量**，但**不阻塞前端画廊主体**（无 `thumb_path` 时前端退化到原图 + `decoding="async"`，见下过渡方案）。
  - **过渡方案（缩略图后端未就位时）**：前端缩略图 `<img decoding="async" width=.. height=..>`（占位防 CLS）+ 虚拟化。这能扛住 ②③（DOM/显存恒定），但 ① 仍拉原图——**够用但不理想**，trajectory 多时首次进视口仍有解码卡顿。所以缩略图是"该做"，过渡方案是"先能跑"。

> 一句话给实施 agent：**虚拟化是硬性必做（治 ②③，不可用 lazy 替代）；后端缩略图是强烈建议（治 ①，trajectory 是主成本，实测坐实），可与前端并行、不阻塞画廊上线。**

### 3.1.7 分层逃生口：从"零渲染"到"满渲染"的三层（ZIP 是被低估的第 1 层主路径）

研究员"看到所有图"其实是**两个不同需求**，对应不同渲染成本的层。把它们分层，让**大多数人走零渲染或低渲染的路**，只有真要"逐张挑"才触发虚拟化网格：

| 层 | 路径 | 渲染成本 | 满足的需求 |
|---|---|---|---|
| **第 0 层** | 聊天流 inline 只放 aggregate 代表图（≤6 eager）+ 摘要行 | **≈零**（112 张 per-subject 根本不进聊天流 DOM） | "一眼看到这次分析的关键结论图" |
| **第 1 层** | **下载全部 ZIP**（一次点击，零渲染） | **零**（不渲染任何缩略图，浏览器只下文件） | "把所有图**带走**进 PPT/论文"——`看全部图`需求的一半 |
| **第 2 层** | 画廊（分面筛选 → 折叠 → 虚拟化网格 → lightbox 看单张原图） | 零→满（取决于用户展开多少） | "在产品里**自由探索**想看的那几张" |

**关键洞察（本会话澄清，须固化）：第 1 层"下载全部 ZIP"是被低估的逃生口，应从"附属导出按钮"提升为第 1 层主路径**，与"打开画廊"并列：

- 研究员的高频真实诉求是"把这次分析的图**全部拿走**放进论文/汇报"，不是"在网页里一张张滑 112 张"。**ZIP 一次点击、零渲染**就满足这半需求——比让他在画廊里滚动 + 浏览器吃 100+ 解码位图**体验更好且成本为零**。
- 这也是图墙最佳实践的标准做法（Google Photos / Lightroom 都有"导出全部"作一等动作，不是藏在右键菜单）。
- **后端形态**：流式 zip 端点 `GET /api/threads/{thread_id}/artifacts/archive?run_id=...`（边读边压、不全量进内存），或复用 present 清单服务端打包。**列为本 spec 后端范围的合理增量**，规模小（一个流式端点）。
- **前端形态**：第 0 层 inline 摘要行 + 画廊 facet bar **都**放显眼的"⬇ 下载全部 N 张（ZIP）"按钮；点击直接走浏览器下载，**不经任何缩略图渲染**。

> 落地优先级：第 0 层（inline 代表图，§3.2）+ 第 1 层（ZIP 按钮，后端一个流式端点）是**最高性价比组合**——两者都零渲染，就能把"图墙卡死 + 带不走图"两个痛点同时解掉。第 2 层画廊（虚拟化网格）是"自由探索"的完整体验，但**不是每个研究员每次都需要**。

### 3.2 聊天流 inline：代表图 + 入口（克制）

替换 `message-list.tsx` 的 `assistant:present-files` 分支（`:135-155`）渲染：

```
聊天流里（inline）:
┌─────────────────────────────────────────┐
│ 已生成 6 张汇总图 · 112 张单样本图          │  ← 一行摘要（tabular-nums）
│ ┌────┐ ┌────┐ ┌────┐                      │
│ │aggr│ │aggr│ │aggr│  …(仅 aggregate)      │  ← 代表图，≤6 张
│ └────┘ └────┘ └────┘                      │
│ ⤢ 打开产物画廊查看全部 118 张   ⬇ 下载全部(ZIP) │  ← 两个并列主动作（第 1/2 层逃生口）
│ ⚠ 6 张单样本图未生成（超出预算）[详情]       │  ← 失败/截断（若有），status-warning
└─────────────────────────────────────────┘
```

> **两个入口并列、不分主次**（§3.1.7 分层）：「打开画廊」是第 2 层（自由探索，触发渲染）；「下载全部 ZIP」是第 1 层（带走全部，零渲染）。研究员"把图放进论文"走 ZIP 更快更省——**不要把 ZIP 藏进画廊里的次级菜单**。

**代表图选取规则（确定性，不靠 LLM）**：
- `output_mode === "aggregate"` 的图**全部** inline（通常每 metric 1 张,≤6 张）。
- 若**无** aggregate（极少），退化为"每个 (paradigm, metric) 的第一张"。
- 其余全进画廊。
- 纯前端按 `ArtifactMeta` 算,无需 agent 决策。

**入口**：`output_mode` 分组算出"全部 N 张 / aggregate M 张 / per-subject K 张",入口按钮文案动态（`t.gallery.openAll(N)`）。

### 3.3 画廊界面：图墙最佳实践

```
产物画廊（独立界面 / 全屏层）:
┌──────────────────────────────────────────────────────────┐
│ [范式▾][图类型▾][aggregate/per-subject▾][组▾][subject▾]  🔍搜索 │ ← 分面
│ ── Aggregate（6）─────────────────────────────────────── │ ← 默认展开
│ ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐                                 │
│ └──┘└──┘└──┘└──┘└──┘└──┘                                 │
│ ── Per-subject（112）▸ 展开 ──────────────────────────── │ ← 折叠，点开虚拟化
│ ⚠ 6 张未生成：subject 23-28 的 box_open_arm（超预算）[补全] │ ← 失败/截断
│ [✓ 多选模式] → 小倍数并排对比   [下载选中][导出数据表 CSV]   │
└──────────────────────────────────────────────────────────┘
```

视图层（命中 `ui-ux-pro-max` chart + ux 准则）：
1. **分面筛选**（`large-dataset` drill-down）：范式 / 图类型(box/bar/trajectory) / aggregate-vs-per-subject / 组 / subject + 文件名搜索。facet 从 `ArtifactMeta[]` 的 distinct 值动态生成。
2. **aggregate 优先**：默认展开 aggregate 分组（少、最重要）;per-subject 折叠成"展开查看 112 张"。
3. **虚拟化网格**（`virtualize-lists`，50+ 必做）：per-subject 展开后虚拟滚动;缩略图 `loading="lazy"` + `aspect-ratio` 占位防 CLS（`image-dimension`）。`max-w-full h-auto`（`Image Scaling`）。
4. **小倍数对比**（small multiples）：多选 → 并排对齐网格,同范式同指标跨组/跨 subject 对比。
5. **单图详情**：点缩略图 → lightbox（`Dialog`,键盘左右切换 + ESC + 下载 + 新窗口）**或**复用 `ArtifactFileDetail`。
6. **a11y**（chart 准则）：缩略图旁带统计摘要副标（n / 组别,若元数据有）;提供 CSV/数据表导出入口（`data-table` / `export-option`）;组别**色 + 形状/标签**双编码（box plot=AA,但缩略图本身是后端渲染的 .png,前端保证**周边信息**双编码,不靠缩略图配色）。
7. **失败/截断显式**（CLAUDE.md「无声截断」铁律）：从 `failed_charts`/`remaining_charts` 渲染"N 张未生成 + 原因（超预算/失败）+ [补全]入口"。补全入口 = 给用户一个 prompt 模板（"补全剩余单样本图"）或直接发消息触发——具体在实施期定,但**数据必须呈现,不静默**。

### 3.4 画廊"在哪打开"：独立路由 vs 全屏层（布局决策）

母方案定"独立打开界面"。两个落地形态：
- **方案 A（推荐）独立路由** `/workspace/chats/[thread_id]/gallery`：可深链、可分享、可前进后退（`deep-linking`）。点 inline 入口 push 路由;返回回聊天（`back-behavior` / `state-preservation`）。
- **方案 B 全屏 overlay 层**：`Dialog` 全屏,不改路由。更轻,但不可深链。

> **推荐方案 A**（深链 + 研究员可把"这次分析的图集"链接发同事,符合科研协作）。实施期若路由改造成本高可先 B,但 A 是目标态。**与运行轨迹（spec#2 overlay 抽屉）的区别**：轨迹是"瞥一眼过程"（overlay 够）;画廊是"沉浸式探索 + 可分享"（值得独立路由）。

### 3.4.1 画廊是 per-thread 的：边界 = 整 thread 累积（靠分面组织），run 分段作可选增强

> 本会话澄清（用户最后一问"画廊是不是每个 thread 一个"），须固化。

**画廊必然 per-thread（一次分析 = 一个 thread = 一套图集），由数据源和 URL 结构决定，不是可选项**——三处代码证据坐实：
- **数据源 = thread 级 state**：图来自 `thread.values.artifacts`，经 `merge_artifacts` reducer（`thread_state.py:46-53`）在**该 thread** 内累积。不存在跨 thread 的 artifacts 池。
- **URL 带 threadId**：`urlOfArtifact({filepath, threadId})`（`utils.ts:4-16`）拼成 `/api/threads/${threadId}/artifacts${filepath}`——每张图的取图地址绑定 thread。
- **路由 per-thread**：§3.4 方案 A 路由 `/workspace/chats/[thread_id]/gallery` 本就在 thread 命名空间下（现有 `app/workspace/chats/[thread_id]/page.tsx` 的兄弟路由）。
- **科研语义自洽**：一次分析 = 一个 thread = 一套图集，深链可发同事——正是研究员协作心智。

**唯一待拍板的边界：画廊内容放"整 thread 累积"还是"按 run 分段"**：
- `merge_artifacts` 是**累积的**（`list(dict.fromkeys(existing + new))`，按 path 去重、**从不清空**——对比 `merge_viewed_images:67` 有 `len(new)==0 → 清空` 的特例，artifacts **没有**这条，即天然只增不减）。
- 后果：**多轮追问会叠加**——第一轮 60 张 + 追问再画 50 张 = 110 张**全在同一个画廊**里。

| 方案 | 做法 | 成本 | 取舍 |
|---|---|---|---|
| **A 整 thread 累积（推荐，Phase 0 默认）** | 画廊直接吃 `thread.values.artifacts` 全量，**靠 `ArtifactMeta` 分面（metric/group/paradigm/output_mode）让用户自己筛**，不按 run 切 | **零**（就是现有 state，不需后端加字段） | 多轮追问图混在一起，但分面足以区分（"我只看 control 组的 box plot"一筛就出） |
| **B 按 run 分段（可选增强，不阻塞 Phase 0）** | `ArtifactMeta` 加 `run_id`，前端按轮次分面（"第 1 轮 / 追问轮"） | 后端 present 时带 `run_id` + 前端多一个 facet | 想"只看这次追问新出的图"时更直接；但 Phase 0 用 A + 分面已够 |

**结论**：Phase 0 走 **A（整 thread 累积 + 分面组织）**，零成本。B（`run_id` 分段）记为**可选增强**，与 §六低优先级"`ArtifactMeta` 加 `run_id`"对齐，**不阻塞画廊上线**。

### 3.5 视觉/动效（日式 + spec1 token）
- 缩略图网格 stagger 入场（spec1 `--animate-*` + `ease-brand-out`,逐项 30-50ms,不齐刷）。
- 缩略图 hover：轻微 scale（`scale-feedback` 0.97-1.03,不位移布局）+ 阴影抬升（spec1 `--shadow-float`）。
- lightbox 从缩略图位置展开（`modal-motion` 从触发源,`ease-brand-out`+`duration-slow`）。
- 失败/截断条用 `--color-status-warning` + 图标 + 文字（三件套）。
- 全部 reduced-motion 降级。

---

## 四、实施步骤

### Step 1（后端）：`ArtifactMeta` 契约 + present_file 接出元数据 + merge reducer
- `present_file_tool.py`：写 artifacts 时关联 `plan_charts.json` 元数据,写 `ArtifactMeta` dict。
- `thread_state.py` `merge_artifacts`：按 `path` 去重合并 `list[ArtifactMeta]`,兼容裸 string。
- **测试**（红→绿）：merge reducer 喂 [string, meta, 重复path] → 断言去重 + 兼容;present_file 产出含元数据。
- 验证：`make test` + `import app.gateway` + `make_lead_agent` 裸导入 0 退出。

### Step 2（前端）：类型 + 消费面迁移
- `core/artifacts/types.ts`：`ArtifactMeta` + `normalizeArtifact`。
- 迁移 §1.1 表的 4 处消费方到 `ArtifactMeta[]`，全部经 `normalizeArtifact` 兜底旧 string。
- `pnpm check` 通过。

### Step 3（前端）：聊天流 inline 代表图（§3.2）
- 改 `message-list.tsx` 的 `assistant:present-files` 分支：算代表图（确定性规则）+ 摘要行 + 画廊入口 + 失败/截断提示。
- 新组件 `InlineArtifactSummary`。

### Step 4（前端）：画廊界面（§3.3）
- 路由 `/workspace/chats/[thread_id]/gallery`（方案 A）或全屏 Dialog（方案 B）。
- 组件：`ArtifactGallery`（分面 + 分组）、`GalleryGrid`（虚拟化网格）、`GalleryFacetBar`、`GalleryLightbox`（或复用 `ArtifactFileDetail`）、`CompareGrid`（小倍数）。
- **虚拟化是硬性必做**（治 DOM/内存 ②③,§3.1.5）：per-subject 网格用 `@tanstack/react-virtual`（与 `@tanstack/react-query` 同源,已在 `package.json:48`）。**不可用 `loading=lazy` 替代**（lazy 只省网络,DOM/显存照样涨）。
- 缩略图优先用 `ArtifactMeta.thumb_path`（后端 Pillow 生成,§3.1.6）;无则退化原 `path` + `decoding="async"` + `width/height` 占位。

### Step 5（前端 + 后端小端点）：「下载全部 ZIP」第 1 层主路径 + 失败/截断呈现

> §3.1.7：ZIP 是被低估的第 1 层逃生口,**从"附属导出"提升为主路径**（零渲染满足"带走所有图"）。这一步不是 Step 4 画廊的附属,优先级与 inline 入口同级。

- **「下载全部 ZIP」（第 1 层,零渲染）**：
  - 后端流式 zip 端点 `GET /api/threads/{thread_id}/artifacts/archive?run_id=...`（边读边压、不全量进内存),或复用 present 清单服务端打包。**列为本 spec 后端范围,规模小**（一个流式端点 + 不改产物契约）。
  - 前端：第 0 层 inline 摘要行 + 画廊 facet bar **都**放显眼"⬇ 下载全部 N 张（ZIP）"按钮;点击直接走浏览器下载,**不经任何缩略图渲染**。**别藏进次级菜单。**
- **失败/截断呈现**：`failed_charts`/`remaining_charts` 在 chart-maker handoff / `run_chart_plan` 返回里。确认前端能否从消息/state 拿到——若不能,**记为后端小补丁**（present 时一并带出 failed/remaining 摘要）。
- **CSV/数据表导出（顺延）**：数据表来源 = metric 计算结果,可能需 present 数据文件。Phase 0 **先只做上面的"下载全部 ZIP"**（图打包,真主路径）+ 占位"导出数据表";数据表完整实现顺延,不阻塞画廊主体。

### Step 6：i18n + a11y + reduced-motion
- 画廊所有文案进 i18n（不硬编码中文）。
- 键盘：facet 可操作、网格可 tab、lightbox 左右键 + ESC、入口 aria-label。
- 缩略图 alt = `chart_id`/metric/subject 组合（有意义,非文件名）。
- reduced-motion 降级。

---

## 五、验收标准

### 功能
- [ ] 后端 `present_file` 产出 `ArtifactMeta`（含 chart_id/output_mode/paradigm/metric/subject）;旧裸 string 仍合法。
- [ ] `merge_artifacts` 按 path 去重,混合 string+meta 不崩（有单测）。
- [ ] 聊天流只 inline aggregate 代表图（≤6）+ 摘要 + 画廊入口;**不再铺整墙**。
- [ ] 画廊：分面筛选（范式/图类型/mode/组/subject/搜索）可用,facet 从数据动态生成。
- [ ] aggregate 默认展开;per-subject 折叠,展开后虚拟化加载（100+ 张滚动流畅）。
- [ ] 小倍数对比：多选 → 并排对齐。
- [ ] 单图 lightbox/详情：左右切换 + 下载 + 新窗口 + ESC。
- [ ] 失败/截断**显式呈现**（N 张未生成 + 原因 + 补全入口）——**不静默少图**。
- [ ] **「下载全部 ZIP」第 1 层入口**：inline 摘要行 + 画廊都有显眼按钮,点击零渲染走浏览器下载（§3.1.7）。
- [ ] 画廊 **per-thread**：数据源 = `thread.values.artifacts`,默认整 thread 累积 + 分面组织（§3.4.1）。
- [ ] 画廊可深链（方案 A）或全屏层（方案 B）;返回保留聊天状态。

### a11y / 性能（红线）
- [ ] **per-subject 50+ 必虚拟化（windowing,治 DOM/内存 ②③）**——`loading=lazy` **不算数**（它只省网络,DOM 节点/已解码位图照样常驻,§3.1.5）。验收点：滚动 100+ 张时 DOM 里 `<img>` 数量**恒定**（开 devtools Elements 数节点,不随滚动线性增长）。
- [ ] 缩略图 `aspect-ratio`/`width-height` 占位防 CLS;优先 `thumb_path`,无则原图 + `decoding="async"`（§3.1.6）。
- [ ] 缩略图 alt 有意义;facet/网格/lightbox 键盘可达。
- [ ] 组别等信息色 + 形状/文字双编码（`color-not-only`）。
- [ ] 动效用 spec1 曲线;reduced-motion 降级。
- [ ] 首屏不再全量 `<img>`（只 aggregate 代表图 eager,其余虚拟化窗口内才挂载）。

### 工程纪律
- [ ] 改 `artifacts` 契约 = §1.1 全部消费方一起改 + 向后兼容 + 全量回归（`make test` + `pnpm check`）。
- [ ] 后端裸导入两生产入口 0 退出。
- [ ] SSOT：chart 元数据只来自 `plan_charts.json`,前端不正则猜分类。
- [ ] 不动流式核心 / `groupMessages`。
- [ ] i18n 不硬编码中文。

---

## 六、风险与回退

| 风险 | 缓解 |
|---|---|
| 改 `artifacts` 契约漏改某消费方 → 运行时崩 | §1.1 grep 清单逐处改;`normalizeArtifact` 兜底使漏改处退化（旧 string 仍渲染,不崩） |
| `merge_artifacts` reducer 改错 → 产物丢失/重复 | 红→绿单测（混合/重复 path）;CLAUDE.md「改 reducer 必测」 |
| **误用 `loading=lazy` 当虚拟化** → DOM/显存仍随图数涨,100+ trajectory（实测 1–2.7MB/张,12MB 显存/张）仍卡死 | §3.1.5 三成本表已固化:**②③ 唯一解是 windowing**;验收点要求"滚动时 DOM `<img>` 数恒定" |
| 虚拟化依赖 `@tanstack/react-virtual` | **同源同作者**于已装的 `@tanstack/react-query`（`package.json:48`）,headless 轻量,**不算新生态依赖**;若仍不引,fallback IntersectionObserver 懒渲染分页（但 fallback 治标不治本,优先引 react-virtual） |
| `failed_charts`/`remaining_charts` 前端拿不到 | 记后端小补丁（present 时带 failed/remaining 摘要）;Phase 0 内协同 |
| 独立路由改造成本 | 先全屏 Dialog（方案 B）保功能,路由（方案 A）作目标态迭代 |
| 后端缩略图增量是否做 | **已实测拍板（§3.1.6）：强烈建议做**（trajectory 1–2.7MB 是主成本）;但**不阻塞**画廊——无 `thumb_path` 时前端退化原图 + `decoding=async`,可后补 |
| ZIP 端点 / 数据表 CSV 依赖后端 | ZIP 是一个流式端点（小,本 spec 后端范围）;数据表 CSV 顺延占位（不阻塞画廊主体） |

**回退**：前端组件可灰度（先上 inline 代表图 + 入口,画廊后续）;后端契约向后兼容,旧 string 路径不受影响。最坏回退到现 `ArtifactFileList`（保留组件别删,作 fallback）。

---

## 七、给实施 agent 的交接 + 决策点

- 改动文件：后端 `present_file_tool.py` / `thread_state.py`（surgical + 测试 + 裸导入）+ 新增流式 zip 端点 + （建议）Pillow 缩略图写 `thumb_path`;前端 `core/artifacts/types.ts`（新,含 `thumb_path?`）、`core/threads/types.ts`、`artifacts/context.tsx`、`chat-box.tsx`、`message-list.tsx`、新建 `artifacts/gallery/*` + `InlineArtifactSummary` + 路由 `app/workspace/chats/[thread_id]/gallery/`。
- **不碰**：`run_chart_plan` 执行逻辑、`useStream`/`groupMessages`、`ArtifactFileDetail`（复用不改）。
- 复用 spec1 token;与 spec#2 运行轨迹**区分**（轨迹 overlay 看过程,画廊独立路由看产物）。
- **顺序建议**：Step 1-2（契约,后端+前端类型,可独立验收）→ Step 3（inline 代表图 **+ 下载全部 ZIP**,两者都零渲染,立即把"图墙卡死 + 带不走图"两痛点同解,§3.1.7）→ Step 4-5（画廊虚拟化 + 失败呈现）→ Step 6。**Step 3 一上线痛点就大幅缓解**。
- **本会话已拍板（不再是待定）**：
  - ✅ 后端缩略图 = **做**（实测 trajectory 1–2.7MB 是主成本,§3.1.6）;不阻塞画廊（无 thumb 退化原图）。
  - ✅ 虚拟化引擎 = **`@tanstack/react-virtual`**（同源 react-query,非新生态）;**不可用 `loading=lazy` 替代**（§3.1.5）。
  - ✅ 「下载全部 ZIP」= **第 1 层主路径**（不是次级菜单,§3.1.7）。
  - ✅ 画廊 = **per-thread**,Phase 0 默认整 thread 累积 + 分面（§3.4.1）。
- **仍待实施期定（非阻塞）**：①画廊形态 A（独立路由,推荐）vs B（全屏层）②`run_id` 分段（方案 B）是否在 Phase 0 加（建议顺延为可选增强）③数据表 CSV 是否 Phase 0 完整做（建议占位顺延,ZIP 图打包先行）。

---

*依据：母方案硬伤 A / §4.2 + 决策1（路 A）+ 后端 `plan_charts.json` 元数据契约 + `ui-ux-pro-max`（chart large-dataset/box-plot/data-table、ux Image Optimization/virtualize/progressive-disclosure）。未写代码。*
