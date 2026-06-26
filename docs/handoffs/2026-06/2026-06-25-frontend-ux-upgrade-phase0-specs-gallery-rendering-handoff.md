# Handoff：前端/人机体验大升级 — Phase 0 五份 spec 已出齐 + 画廊渲染压力/per-thread 边界已澄清

> 日期：2026-06-25
> 类型：设计/spec 阶段交接（**未写任何代码**，全部是 plan + spec + 设计决策）
> 上游会话：5f682443-8fbf-4e72-ad24-9167acf42aef（已 compact 一次）
> 母方案：[docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)

---

## 一、当前任务目标

为 EthoInsight 前端（DeerFlow fork，Next 16/React 19/Tailwind 4/shadcn/`@langchain/langgraph-sdk` useStream）做一次**大的人机体验升级**。受众是**行为学研究员（非程序员）**。三个硬伤要治：

- **硬伤 A（最重）**：一次分析可能产出**上百张图**（per_subject），后端 `present_file` 写进扁平 `artifacts: string[]`，前端 `ArtifactFileList` 直接 `grid-cols-2` 全量 `<img>` 渲染成图墙 → 研究员大海捞针 + 浏览器渲染卡死。
- **硬伤 B**：7 阶段领域 workflow（上传→范式→列对齐→指标→质检→解读→报告）前端不可见，无导向。
- **硬伤 C**：审批/反问（`ask_clarification`）淹没在消息流，研究员划过去没注意 agent 在等他 → 分析静默卡住。

**约束（用户硬性）**：不重写不换栈（DeerFlow 上游还会改，要可 sync）；日式简洁美学 + 真设计感（不要千篇一律老土 agent UI）；动效曲线必须**渐变减速（ease-out 尾巴），绝不 linear 匀速**（用户原话举例：移出频率应是渐变曲线变慢，不是 y=2x 固定速率）。

---

## 二、当前进展

### ✅ 已完成（全是文档，0 代码）

1. ✅ **母方案调研稿**：[docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)（含锁定决策表、现状证据、日式设计语言、3 期路线图、工程纪律）。
2. ✅ **Phase 0 五份实施 spec**（均在 `docs/superpowers/specs/`，均用 `ui-ux-pro-max` skill 查实准则）：
   - `2026-06-24-frontend-phase0-1-design-tokens-motion-spec.md` — **SSOT，所有组件动效/语义色源头，必须先落**。`--ease-{brand-out,brand-in,...}`（`brand-out = cubic-bezier(0.22,1,0.36,1)` 减速尾）+ `--dur-{fast:140,base:220,slow:340,exit:160}` + `--color-status-{info,success,warning,danger}` + `--color-stage-*`（7 阶段）；换掉全站 `@theme` 的 `ease-in-out`→非对称曲线；退役 aurora/shine 商业引用。**零后端**。
   - `2026-06-24-frontend-phase0-2-run-trace-live-spec.md` — 运行轨迹 overlay 抽屉，`useRunTrace` 纯派生只读 hook，live 全保真零后端；replay（档 B）留 Phase 1（需 ~30 行 task_running 落 journal + ~50 行 handoff 只读端点）。**overlay 不进 ResizablePanelGroup**（避 3 panel 冲突）。
   - `2026-06-24-frontend-phase0-3-artifact-gallery-spec.md` — **画廊（硬伤 A 主战场）**。聊天流 inline 只放 aggregate 代表图（≤6）+ 入口；独立画廊界面（分面/虚拟化/小倍数/失败显式）。后端 `ArtifactMeta` 契约（路 A），改 `present_file`+`merge_artifacts` reducer。
   - `2026-06-24-frontend-phase0-4-analysis-rail-spec.md` — 7 阶段进度轨，前端推导复用 spec#2 `useRunTrace`，阶段定义 SSOT 放 `core/workflow/stages.ts`。**零后端**。
   - `2026-06-24-frontend-phase0-5-decision-card-spec.md` — `ask_clarification`→显眼决策卡（左 accent bar 按 clarification_type 差异化 + 键盘 1-9 + context 依据突出 + 进度轨/输入框联动）。**零后端**。
3. ✅ **并行推进策略**（已答用户"这几个 spec 能并行吗"）：见下 §四。
4. ✅ **画廊渲染压力 + per-thread 边界澄清**（本会话新增，**尚未回写进 spec#3**）：见下 §三。

### ❌ 未开始
- 任何代码实现（连 spec#1 都没动手）。
- 三个 spec#3 待澄清点回写（§六）。

---

## 三、关键发现（本会话新增，最重要）

### 3.1 画廊渲染压力的本质 = 三个独立成本，`loading=lazy` 只治了一个

研究员看 100+ 张图卡，是三个成本叠加：

| 成本 | 现状（`artifact-file-list.tsx:94-135`） | `loading=lazy` 治了吗 |
|---|---|---|
| ① 网络/解码 | 100+ `<img>` 并发拉原图 PNG + 主线程解码 | 部分（视口外不拉） |
| ② DOM 节点 | 100+ `<div><img>` 同时在 DOM，layout/paint 全量 | ❌ 没治 |
| ③ 内存 | 已解码位图常驻显存（一张 800×600 ≈ 1.9MB 显存，**与文件大小无关**） | ❌ 没治（进过视口就不释放） |

**核心结论**：②③ 的唯一解是**虚拟滚动（windowing）**——任意时刻 DOM 里只有可视区+缓冲的十几张，滚出去真正卸载 → DOM 节点和内存**恒定，不随图总数增长**。`loading=lazy` 治不了 ②③，实施 agent 千万别误以为 lazy 就够了。用 `@tanstack/react-virtual`（和现有 `@tanstack/react-query` 同源，轻）。

### 3.2 分层方案（渲染压力从零到满）

- **第 0 层（聊天流，≈零渲染）**：inline 只放 aggregate 代表图（≤6 eager）+ 摘要行；112 张 per_subject **根本不进聊天流 DOM**。（spec#3 §3.2 已有）
- **第 1 层（下载全部 ZIP，零渲染）**：**这是被低估的逃生口，应从"附属导出"提升为第 1 层主路径**。后端流式 zip（`GET /api/threads/{tid}/artifacts/archive?run_id=...` 或复用 present 清单），研究员"带走所有图进 PPT/论文"一次点击零渲染。满足"看到所有图"里"带走"那一半需求。spec#3 §五 Step5 写得太轻。
- **第 2 层（画廊，windowing）**："自由看想看的"落点：① per_subject 默认**折叠**（没点开零渲染）② 点开后**虚拟滚动** ③ 缩略图≠原图（理想后端 Pillow `thumbnail()` 出几 KB 缩略图；过渡方案不改后端用 `decoding=async`+`width/height` 占位+虚拟化）④ 点缩略图开 lightbox 看**那一张**原图（一次只解码一张）⑤ **分面筛选先把搜索空间砍小**（`ArtifactMeta` 的 group/subject/metric → 筛完可能只剩 8 张，根本不触发大渲染）。

> **缩略图生成值不值得改后端，取决于 PNG 实际大小**：matplotlib 默认 dpi 若每张 ≥500KB → 强烈建议加缩略图；若 50-100KB → 虚拟化+`decoding=async` 够，缩略图顺延。**这是个待拍板的后端范围决策。**

### 3.3 画廊是 per-thread 的（用户最后一问）

- **必然 per-thread**：数据源 `thread.values.artifacts`（thread 级 state，经 `merge_artifacts` 累积）；`urlOfArtifact({filepath, threadId})` 拼 URL 带 threadId；spec#3 §3.4 路由 `/workspace/chats/[thread_id]/gallery` 已 per-thread。科研语义=一次分析=一个 thread=一套图集（深链可发同事）。
- **唯一待拍板边界：画廊放"整 thread 累积"还是"按 run 分段"**。`merge_artifacts` 是累积的（按 path 去重不清空），多轮追问会叠加（第一轮 60 张 + 追问 50 张 = 110 张全在一个画廊）。
  - **建议**：默认 **A 整 thread 累积**（零成本，就是现有 state），靠 `ArtifactMeta` 分面（metric/group）让用户自己筛，**不靠 run 切**。
  - **B 按 run 分段**（`ArtifactMeta` 加 `run_id`，前端按轮次分面）作**可选增强，不阻塞 Phase 0**。

---

## 四、并行推进策略（已答用户）

**不能全并行，依赖图：**

```
#1 token ──────────────┐ (SSOT：#2/#3/#4/#5 的动效+语义色源头)
                        ├──→ #3 gallery (token + 独立改 artifacts 契约)
                        ├──→ #5 card    (status 色 token)
                        └──→ #2 trace ──→ #4 rail (#4 复用 #2 的 useRunTrace)
```

- **第 0 波（串行闸门）**：**#1 token 单独先落**（半天，纯 CSS，零风险）。是所有组件动效/语义色 SSOT，不先落则其余四个引用不存在的 token。
- **第 1 波（#1 合并后，三轨并行）**：
  - 轨道 A：#2 run-trace（建 `useRunTrace`）→ 完成后 #4 rail（复用它）。**A 内部串行**（#4 强依赖 #2，否则两套解析漂移）。
  - 轨道 B：#3 gallery。**其后端 `ArtifactMeta` 契约可在第 0 波就和 #1 并行**（后端改 present_file/reducer 跟前端 token 不沾边）——唯一能提前的后端活。
  - 轨道 C：#5 decision-card 本体（#5 进度轨联动依赖 #4，但卡本体独立）。
- **收尾**：#5 进度轨联动补上（等 #4）。

**仅有的两个文件冲突协调点**（实施前先定布局契约）：
- `message-list.tsx`：#3 改 `assistant:present-files` 分支 + #5 改 `assistant:clarification` 分支（同文件不同分支，轻冲突）。
- `chat-box.tsx` 布局：#2（右侧 trace 抽屉）+#3（画廊入口）+#4（顶部 rail）都要挂东西 → **先花半页定"workspace 三挂载点布局契约"**（顶部 rail / 右侧 trace 抽屉 / 画廊入口），三轨照契约挂就不打架。

---

## 五、关键上下文（代码证据，实施时直接用）

### 5.1 必须改的文件（spec#3 改 artifacts 契约 = 全消费面一起改）
`artifacts` 当前 `string[]`，grep 出 **4 处消费 + 1 reducer**（CLAUDE.md 铁律：改共享契约全改+向后兼容+全量回归）：
- `core/threads/types.ts:8` — `artifacts: string[]` → `ArtifactMeta[]`
- `components/workspace/artifacts/context.tsx:13-14` — `artifacts`/`setArtifacts`
- `chats/chat-box.tsx:53/67/69/152/166` — setArtifacts/空判断/传 ArtifactFileList
- `core/artifacts/utils.ts:22` — `extractArtifactsFromThread`
- 后端 `thread_state.py` `merge_artifacts` reducer — 按 path 去重（**改 reducer 必带红→绿单测**：混合 string+meta + 重复 path）
- 后端 `present_file_tool.py` — 写 ArtifactMeta（从 `plan_charts.json` 关联元数据）

`ArtifactMeta` 字段：`path`(必有,向后兼容锚) / `kind` / `chart_id` / `output_mode`(aggregate|per_subject) / `paradigm` / `metric` / `subject` / `group` / `chart_type`。`normalizeArtifact(string|ArtifactMeta)` 兜底旧裸 string。

### 5.2 绝对不能碰
- 流式核心：`useStream` / `mergeMessages` / `dedupeMessagesByIdentity` / optimistic / summarization（`core/threads/hooks.ts`，1211 行，踩坑沉淀，重写必复发）。
- `groupMessages` 路由（升级一律叠加在它之上）。
- `run_chart_plan` 执行逻辑（spec#3 只在 present_file 接出已有元数据，不改执行）。
- 后端 clarification 机制（`Command(goto=END)` / ToolMessage 格式 / `onSelectClarificationOption` 点选项=发消息机制）。

### 5.3 复用资产
- `urlOfArtifact`/`resolveArtifactURL`（`core/artifacts/utils.ts:4/55`）拼图 URL。
- `ArtifactFileDetail`（`artifacts/artifact-file-detail.tsx`）单图详情，画廊 lightbox 可复用。
- `ClarificationOptions`（`clarification-options.tsx`，63 行）现竖排 outline 按钮+编号+"或自定义"，**只 onClick 无键盘**，spec#5 加数字键 1-9 + 主次样式。
- `plan_charts.json`（chart 元数据 SSOT，`run_chart_plan` 产出，**前端别正则猜分类**）。
- `failed_charts`/`remaining_charts`（`run_chart_plan` 返回，失败显式呈现数据源——**CLAUDE.md「无声截断」铁律，必须呈现**）。

### 5.4 后端契约改动属受保护文件邻域
`present_file_tool.py` / `thread_state.py` 按 CLAUDE.md surgical：grep 所有读写 artifacts 处 + `make test` + **裸导入两生产入口**（`PYTHONPATH=. python -c "import app.gateway"` / `from deerflow.agents import make_lead_agent`，0 退出，绕开 conftest mock 假绿）。

---

## 六、未完成事项（按优先级）

### 高优先级
1. **回写 spec#3 三处**（本会话澄清但还没写进文件）：
   - §3.2/§五：把「下载全部 ZIP」从附属导出**提升为第 1 层主路径**（零渲染满足"带走所有图"）。
   - §3.3：明确写「虚拟化治 DOM/内存 ②③，`loading=lazy` 治不了」原理 + 三成本表，免得实施 agent 误用 lazy。
   - §3.4：补「每个 thread 一个画廊（数据源=thread state，路由 `[thread_id]/gallery`），内容=整 thread 累积靠分面组织，run 分段作可选增强」。
2. **实施 spec#1（token+motion）** — 零风险闸门，所有并行的前置。做完才好铺开三轨。

### 中优先级
3. 待用户拍板的后端范围决策：**是否做后端缩略图生成**（取决于 matplotlib PNG 实际大小——建议先实测一次 dogfood 的 outputs/*.png 大小再定）。
4. 写"workspace 三挂载点布局契约"半页纸（顶部 rail / 右侧 trace 抽屉 / 画廊入口），解锁三轨安全并行。

### 低优先级（Phase 1+，spec 已记）
5. replay 回放（spec#2 档 B，需两个小后端补丁）。
6. `ArtifactMeta` 加 `run_id` 做按轮次分面（画廊 run 分段增强）。

---

## 七、风险与注意事项（容易混淆/不建议方向）

- ❌ **别先做 #2/#3/#4/#5 再做 #1**——它们引用 #1 的 token，#1 不先落=返工或临时硬编码 hex（且违反"不做 dark 期间也用语义 token 不硬编码"纪律）。
- ❌ **别用 `loading=lazy` 当虚拟化的替代**——治不了 DOM/内存 ②③（§3.1）。
- ❌ **`.dark` block 已存在**（`globals.css:269-308`，stale DeerFlow 值）+ `@custom-variant dark` 在 :43——决策"dark 先不做推 Phase 2"仍成立，但 spec#1 新 token 要给 `.dark` 占位值（将来补 dark 零返工）。
- ❌ **别在前端正则猜图表分类**——chart 元数据唯一来源 `plan_charts.json`，走路 A（present_file 接出）。
- ❌ **改 artifacts 契约别漏消费方**——§5.1 grep 清单逐处改，`normalizeArtifact` 兜底使漏改处退化（旧 string 仍渲染不崩）。
- ⚠️ **进度轨/决策卡阶段定义是 SSOT**（`core/workflow/stages.ts`），#3/#4/#5 若都标阶段引同一份，别各组件重复枚举（CLAUDE.md「同一份知识绝不双存」）。
- ⚠️ **`useRunTrace` 一处派生**（#2 建 #4 用），别另起一套解析（漂移=memory `feedback_handoff_metrics_field_divergence` 类教训）。

---

## 八、下一位 Agent 的第一步建议

**若用户要继续推进**，按这个起点：

1. **先读** 母方案 [docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md) + 五份 spec（`docs/superpowers/specs/2026-06-24-frontend-phase0-{1..5}-*.md`）建立全貌。
2. **第一个动作（最稳）**：回写 spec#3 三处（§六.1，纯文档，把本会话澄清固化）——这是把"已想清楚但只在对话里"的结论落盘，避免下次又重推。
3. **第一个代码动作（若用户要写代码）**：实施 **spec#1 design-tokens-motion**（零风险闸门，纯 `globals.css` + ~38 个组件 easing call-site 迁移）。改完 `pnpm check`。这是所有并行的前置，必须最先。
4. **拍板项**（若用户在场，问清再做）：① 画廊内容 = 整 thread 累积（建议）还是 run 分段 ② 是否做后端缩略图生成（先实测 PNG 大小）③ 画廊形态 = 独立路由（推荐，可深链）还是全屏 Dialog。

**关键 memory**（已沉淀，下次会自动 recall）：
- `feedback_frontend_design_japanese_minimal_motion_craft` — 日式简洁+ease-out 减速尾+绝不 linear+spring 优先+ui-ux-pro-max 高饱和 BI 配色往中性留白回拉。
- `project_2026-06-24_frontend_generative_ux_upgrade_plan` — 本方案全进展快照（含五 spec + 锁定决策 + 后端核查）。

---

*依据：母方案 + 五份 Phase 0 spec + 本会话画廊渲染/per-thread 澄清。全程未写代码。`ui-ux-pro-max` skill 查实所有前端准则。*
