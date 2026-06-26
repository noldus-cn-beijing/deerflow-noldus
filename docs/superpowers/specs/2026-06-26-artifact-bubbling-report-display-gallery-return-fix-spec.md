# Spec：产物丢失/报告不显示/返回卡顿 三现象修复（作废 #213 错误方向）

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26
> 代码基线：dev HEAD `5616a73f`
> 性质：🔴 高 · 产物链路根因修复 + **纠正 #213 的错误修复方向**
> dogfood 取证：thread `e9837b33-79fc-41c1-985f-807017b4ed8c`（#213 修复**之后** 6/26 09:38 跑的，仍只 1 图进画廊 → 证明 #213 无效）

---

## ⚠️ 〇、#213 修复方向是错的（必须先纠正）

[2026-06-25-run-chart-plan-auto-register-artifacts-spec.md](2026-06-25-run-chart-plan-auto-register-artifacts-spec.md)（已实施=PR#213）让 `run_chart_plan` 在 chart-maker subagent 内返回 `Command(update={"artifacts": 113个meta})`。**dogfood 实测证明无效**：

**取证（thread `e9837b33`，checkpoints.db 解码 + ns 分析）**：
- 磁盘 113 张图全在；`handoff_chart_maker.json` `chart_files` 113 项、`sealed_by=run_plan`。
- **但 `state.artifacts` 只有 2 项**：`box_open_arm`（dict ArtifactMeta）+ `report.md`（str）。
- 所有 `present_files` tool_call 与 artifacts 写入 `checkpoint_ns=''`（**lead 主图**），共 2 次写入。

**根因（铁律级，已记 memory `feedback_subagent_command_artifacts_not_bubble_to_lead_executor_drops_state`）**：
- `run_chart_plan` 是 **chart-maker subagent** 的工具（`chart_maker.py:61`）。subagent executor（`executor.py:1460`）`final_state = chunk` 后**只提取 `chunk.get("messages")`**，artifacts 等其余 state 字段**全部丢弃**。→ **subagent 内返回 `Command(artifacts=113)` 写在 subagent state，不上行到 lead，113 个全丢。**
- 那唯一的 box ArtifactMeta 是 **lead 事后调 `present_files`**（`present_files` 是 lead 工具，`lead_agent/prompt.py:695`）写进 lead 主图的——**不是 #213 起的作用**，lead 只 present 了 box 这一张。
- **#213 + 我和 3 个 Explore agent 的命门假设"subagent Command(artifacts) 会上行"是错的**。错因：查老 thread 时也只 2 张，误把"present_files 上行"当"subagent 通道通"，没分清 present_files 是 **lead** 调的（ns='' 一查就破）。

**结论**：修复**不能靠 subagent 工具返回 Command**。`SealGateMiddleware` 也挂在 subagent（`executor.py:989`），其 `after_agent` 钩子同样跑 subagent 图层——**写 artifacts 一样不上行，排除**。**唯一干净的结构修复点 = executor 这个 subagent→lead 边界。**

---

## 一、现象1 修复：画廊按路径从磁盘/plan 直取图（用户方案 — 磁盘是唯一真相）

> **用户拍板方案**（2026-06-26）：图已成功生成、113 张全在磁盘、`handoff_chart_maker.json`/`plan_charts.json` 已有全部路径+元数据。**不必跟 LangGraph 的 state 冒泡较劲**——绕开 state，让画廊/lead 直接按路径从磁盘+plan 拿图。这符合本项目「磁盘是唯一真相」铁律，改动面比"改 executor 合并 state"小得多，且零依赖 LLM/state 冒泡。
>
> **作废原方向**（executor 合并 subagent state.artifacts）：那是在跟 state 冒泡机制较劲、要动 executor 这个核心。用户方案更优——走已确定性封存的文件。

### 1.1 新增后端只读端点：按 thread 吐全部图的 ArtifactMeta 列表

后端**已有现成的"按 thread 列 outputs 目录"逻辑**（`artifacts.py:111` `archive_artifacts`：`resolve_thread_virtual_path(thread_id, "/mnt/user-data/outputs")` + `rglob("*")` 列全部图、排除 `_THUMB_SUFFIX`）。新增一个端点复用该逻辑 + join `plan_charts.json` 元数据：

```
GET /api/threads/{thread_id}/artifacts/charts   →  ArtifactMeta[]
```
实现：
1. `outputs_dir = resolve_thread_virtual_path(thread_id, "/mnt/user-data/outputs")`，列出所有 `.png`（排除 `.thumb.webp`）—— **磁盘真实文件即真相**（113 张全在）。
2. 读 `plan_charts.json`（`by_output` 用 output 路径做 key，字段齐全：`id`/`output_mode`/`subject_index`/`display_name_zh`/`script`，顶层 `paradigm`）—— 已核实字段够画廊分面。
3. join：每个磁盘图路径 → 命中 `by_output` 升级成完整 ArtifactMeta（`{path, kind:"chart", chart_id, output_mode, paradigm, metric, subject, chart_type, thumb_path?}`）；未命中（极少）退裸 `{path}`。复用前端已有的 `_build_artifact_meta` 等价逻辑（后端版），或直接在端点内构造同形 dict。
4. 缩略图：列目录时若存在 `<stem>.thumb.webp` 就带 `thumb_path`，否则前端退化原图（`_generate_thumbnail` 已在 chart 生成时产出，端点只读不重生成）。

> **为什么这是对的**：画廊要的"113 张图 + 分面元数据"100% 能从 `outputs/ 目录 + plan_charts.json` 重建——这两个都是确定性产物，不需要它们先挤进 LangGraph state。state 冒泡是 LangGraph 的机制噪声，磁盘+plan 是业务真相。

### 1.2 前端画廊数据源：从 `state.artifacts` 换成新端点

- `artifact-gallery.tsx`（现 `artifacts: ArtifactMeta[]` 来自 `thread.values.artifacts`）改为**调 `GET /artifacts/charts` 拉全部图**（`useSWR`/`useQuery` 按 threadId）。
- 这样画廊**不再依赖 artifacts 进没进 state**——磁盘有几张就显示几张，113 张全出，per_subject 分面有图。
- `inline-artifact-summary.tsx`（对话流 inline 代表图）同理可调端点取 aggregate 图，或保留读 state（lead present 的 box 仍在 state，inline 显示它够用）；**画廊全量靠端点**。

### 1.3 lead 代表图（对话流 inline）—— 现状已够，可选增强

lead 收到 chart-maker 结果后已读 handoff、已 present box（aggregate 代表图）进 state.artifacts → 对话流 inline 显示这张够用（研究员在对话流看 1 张代表图 + "打开画廊看全部"）。**inline 代表图不需要改**；要改的只是"打开画廊"后画廊能拿到全部 113 张（§1.1+1.2）。

### 1.4 #213 的去留

#213（run_chart_plan 返 Command 写 subagent state.artifacts）**在本方案下不再是必需**——画廊改走端点，不读 state.artifacts 拿全量。但 #213 让 run_chart_plan 返回结构化结果（status/n_rendered）对 chart-maker 决策树有用，**保留无害**（它写的 subagent state 跑完即弃，不影响）。**本方案不依赖 #213 生效**，两者解耦。

> ⚠️ **必须 dogfood 实测**：`make dev` 跑真 EPM → 打开画廊确认 **113 张全显示**（不是 checkpoints 看 state，而是直接看画廊 UI——因为数据源已是端点不是 state）。

---

## 二、现象2 修复：报告（report.md）在前端可见

> 取证：`report.md` 进了 state.artifacts（str），但**画廊 `artifact-gallery.tsx:28` 只 `filter(isImageArtifact)`——非图产物被过滤**，报告只在侧栏 `artifact-file-list.tsx:92` 的 `otherFiles` 区。用户看画廊看不到报告。

**问题本质**：报告呈现路径不清晰——画廊只放图、侧栏 otherFiles 容易被忽略、对话流无内嵌报告。

**修复选项（实施 agent 择一，建议 A）**：
- **A. 对话流内嵌报告卡**（推荐）：report-writer 完成后，对话流里出现一个「📄 分析报告」卡片，点击展开内嵌 markdown（复用 `markdown-content.tsx`）或打开侧栏详情。研究员最关心报告，应在对话流主路径可见，不藏侧栏。
- **B. 画廊加「非图产物」区**：画廊底部加一个 report/data 文件区（不 filter 掉非图），与图墙并列。
- **C. 仅修侧栏可见性**：确保 report.md 出现在侧栏 otherFiles 且侧栏默认/提示打开。改动最小但仍藏侧栏。

> 注意：report.md 在 state 里是 **str**（lead present 的裸路径），`normalizeArtifact` 兜底成 `{path}`，`kind` 缺失。若要在 UI 按 `kind==="report"` 分类，需后端 present 报告时带 `kind`，或前端按 `.md` 扩展名推 `kind=report`（`types.ts` 已有 `ArtifactKind="report"`）。

---

## 三、现象3 修复：画廊返回对话不卡顿、不丢位置

> 取证：画廊「返回对话」用 `router.push('/workspace/chats/${threadId}')`（`gallery/page.tsx:50`）——**对话页完全重新挂载**：重拉历史 + 重新 groupMessages + 重新虚拟化几百消息（卡顿）+ 丢滚动位置（定位不准）。**`page.tsx` 自己的注释明确警告过**：`Never use next.js router for navigation in this case, otherwise it will cause the thread to re-mount and lose all states`（chats/[thread_id]/page.tsx onStart 注释）——画廊返回恰恰违反。

**修复**：画廊「返回对话」**不用 `router.push`**：
- **首选 `router.back()`**：若画廊是从对话页 push 进来的，`back()` 回到对话页的**既有实例**（不重挂载、保留滚动位置）。但需处理"直接深链进画廊"无 history 的情况（fallback 到 push）。
- **或改成同页面板/抽屉**：画廊不做独立路由页，而是对话页内的全屏 overlay（彻底不离开对话页实例）——改动大，但根治。**Phase 0 先用 `router.back()` + fallback**，overlay 留后续。

```tsx
// gallery/page.tsx:50 改
onClick={() => {
  if (window.history.length > 1) router.back();        // 回既有对话实例，不重挂载
  else router.push(`/workspace/chats/${threadId}`);    // 深链 fallback
}}
```
- 同理检查 `inline-artifact-summary.tsx:101` 进画廊用 `router.push`——进画廊重挂载可接受（画廊轻），但若想返回 `back()` 生效，进入也应是 push（已是）。

---

## 四、改动清单

| 现象 | 文件 | 改动 |
|---|---|---|
| 1 | **新增** `app/gateway/routers/artifacts.py` 端点 `GET /threads/{tid}/artifacts/charts` | 复用 `archive_artifacts` 的列目录逻辑 + join `plan_charts.json` 元数据 → 吐 `ArtifactMeta[]`（磁盘真相，不读 state） |
| 1 | `artifacts/gallery/artifact-gallery.tsx` | 数据源从 `thread.values.artifacts` 换成调新端点（`useSWR`/`useQuery` 按 threadId），画廊不再依赖 state |
| 1 | （保留/解耦）`run_chart_plan_tool.py` #213、`executor.py` | **不改 executor**；#213 保留但本方案不依赖它 |
| 2 | `messages/` 新增报告卡 或 `artifact-gallery.tsx` 加非图区 | 报告在对话流/画廊可见 |
| 3 | `gallery/page.tsx:50` | `router.push` → `router.back()` + fallback |

---

## 五、验证

1. **现象1（最关键）**：`make dev` 跑真 EPM（多 subject）→ **打开画廊确认 113 张全显示**（数据源是端点不是 state，直接看 UI；per_subject 分面有图）。新端点单测：mock outputs 目录 + plan_charts.json → 返回 113 个 ArtifactMeta、aggregate/per_subject 分类正确、缩略图路径正确带出。
2. **新端点是后端只读改动**：跑 `make test` + **裸导入两生产入口**（`PYTHONPATH=. python -c "import app.gateway"` + `make_lead_agent`，CLAUDE.md 铁律）；端点要鉴权（按 thread owner，复用 `archive_artifacts` 的鉴权方式）。
3. 现象2：报告在对话流可见 + 点击展开正常。
4. 现象3：画廊返回对话**不卡顿、滚动位置保留**（人工：滚到对话中部→进画廊→返回→仍在中部）。
5. **回归**：单图/0图 thread 画廊正常（端点返回空列表不报错）；report-writer 的 report.md 仍正常呈现。

---

## 六、关键文件

- `packages/agent/backend/packages/harness/deerflow/subagents/executor.py:1460`（final_state 提取 artifacts）+ `:989`（SealGate 挂载点，说明为何钩子方向不可行）
- `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`（返回 Command 回传 artifacts）
- `packages/agent/frontend/src/components/workspace/artifacts/gallery/artifact-gallery.tsx:28`（isImageArtifact 过滤报告）
- `packages/agent/frontend/src/app/workspace/chats/[thread_id]/gallery/page.tsx:50`（router.push 重挂载）
- `packages/agent/frontend/src/components/workspace/messages/markdown-content.tsx`（报告内嵌渲染复用）
- 参考：`feedback_subagent_command_artifacts_not_bubble_to_lead_executor_drops_state`（根因 memory）

---

*依据：dogfood thread `e9837b33` checkpoints.db 解码（artifacts 数量/类型/ns）+ 读码坐实 executor.py 丢 final_state.artifacts、gallery router.push 重挂载、画廊 isImageArtifact 过滤报告 + memory `feedback_subagent_command_artifacts_not_bubble_to_lead_executor_drops_state`/`feedback_code_has_fix_not_equal_bug_eliminated`。**作废 #213 的"subagent 返 Command"方向**，改 executor 边界合并。*
