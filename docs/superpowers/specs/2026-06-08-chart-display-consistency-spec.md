# Spec E — 图表展示与 handoff 一致性（展示 8 图 vs 实际 2 图）

> 日期：2026-06-08 ｜ 目标分支：从 `dev` 新建 worktree（独立 spec）
> 来源：EPM dogfood（thread `7d4d9b8e`）前端展示异常
> 性质：**调查 + 修复两阶段**。⚠️ 与 Spec A-D 不同：本 spec 的根因**只部分确证**，第一阶段必须先用前端运行时调试钉死根因，再修。**不要跳过调查直接改代码。**
> 这是给执行 agent 的施工单，不是给用户的总结。

---

## 0. 一句话目标

EPM dogfood 结束时，前端展示了 **8 张图**（2 张轨迹 + 5 张柱状 + 1 张重复，文件名如 `epm_trajectory_Trial 1.png`、`epm_bar_open_arm_entry_ratio.png`），但：
- 本 thread outputs 目录**实际只有 2 个 png**（`plot_trajectory.png` + `plot_open_arm_time_ratio_bar.png`）
- chart-maker handoff **只声明 2 张**

展示层与真实产出/handoff **三方不一致**。需先定位"那 8 张哪来的"，再修一致性。

---

## 1. 已确证的事实 vs 待定位的根因（执行 agent 必读）

### 1.1 已确证（现场核实）

1. 本 thread `user-data/outputs/` 真实文件：**`plot_trajectory.png` + `plot_open_arm_time_ratio_bar.png` + report.md**，仅此（`ls` 实测）。
2. `handoff_chart_maker.json` 的 `chart_files`：**正好这 2 个**（`/mnt/user-data/outputs/plot_trajectory.png` + `plot_open_arm_time_ratio_bar.png`），`failed_charts: []`。
3. `merge_artifacts`（`thread_state.py:22`）是**累积 reducer**：`list(dict.fromkeys(existing + new))`——跨轮只去重不清空。`artifacts: Annotated[list[str], merge_artifacts]`（line 66）。
4. **`epm_bar_*.png` / `epm_trajectory_*.png` 这些文件名在整个 `.deer-flow` 里不存在**（`find` 零结果）。
5. 前端 `normalizeArtifactImageSrc`（`frontend/src/core/artifacts/utils.ts`）只做"路径 → /outputs/X.png 规范化"，**不改文件名、不生成 `epm_*` 这种名**。

### 1.2 待定位（现有后端文件系统证据无法钉死）

**那 8 张 `epm_*` 命名的图，display name 从哪来？** 已排除：不在本 thread outputs、不在 .deer-flow 任何地方、前端 utils 不生成。**剩余假设（未确证，执行 agent 须用前端运行时调试验证）**：
- 假设 1：那 8 张是**跨轮/历史累积**进 `artifacts` 状态的条目（merge_artifacts 只增不减）——但文件名对不上（历史也该是 `plot_*`）。
- 假设 2：那 8 张是 report.md 里的 `{{img:...}}` 占位符被某处解析/重命名后渲染的——report.md 引用了图，渲染层可能按某规则改名。
- 假设 3：前端某组件（artifact 列表 / 消息内嵌图）按**指标名**给图生成 display name（`epm_bar_open_arm_entry_ratio` 像是"范式_图型_指标"拼出来的），但底层 src 可能都指向那 2 个真实文件或 404。
- 假设 4：present_files / 某 SSE 事件把同一文件用多个 display 名推给前端。

**关键**：`epm_bar_open_arm_entry_ratio` 这种名像是**程序按"paradigm_charttype_metric"拼的**，而非真实文件——很可能是某层"把指标列表当图列表"渲染。但**这是推测，必须验证**。

---

## 2. 第一阶段：调查定位（必做，先于任何代码改动）

执行 agent **先做完这步、确证根因，再进第二阶段**。用前端运行时调试（项目有 chrome-devtools / playwright MCP）：

1. **复跑 EPM dogfood 到出图那步**（或用现有 thread `7d4d9b8e` 若还能打开），在浏览器里：
   - 对那 8 张图，**抓每张的真实请求 URL**（Network 面板，或 DOM 里 `<img src>`）。看它们请求的是 `/api/threads/<tid>/artifacts/outputs/<什么>.png`。
   - 关键判断：8 张图的真实 src 是指向**同 2 个真实文件**（即重复展示），还是指向 **8 个不同 URL**（其中 6 个会 404，因为文件不存在）。
2. **定位渲染来源**：grep 前端哪个组件渲染这批图：
   ```bash
   grep -rn 'artifacts\|chart_files\|{{img\|normalizeArtifactImageSrc\|outputs.*png' frontend/src/components/workspace/ frontend/src/core/ 2>/dev/null
   ```
   - 是读 `thread.values.artifacts`（累积列表）？还是解析 report.md 的 `{{img}}`？还是读 handoff chart_files？还是 SSE 的 present_files 事件累积？
3. **定位 `epm_*` 命名来源**：grep 谁生成 `epm_`/`<paradigm>_` 前缀的图名：
   ```bash
   grep -rn 'epm_\|f"{paradigm}\|_bar_\|_trajectory_\|displayName\|chart.*name' frontend/src/ backend/packages/harness/deerflow/ packages/ethoinsight/ 2>/dev/null | head -20
   ```
   - 若在 ethoinsight 绘图脚本里 → 真实文件该叫这名（但磁盘没有，矛盾→说明没真产出这些）。
   - 若在前端 → 是 display 层按指标拼名。
4. **产出**：一段"根因确证"说明，明确回答：8 张图 = 重复展示 2 真实文件，还是 6 张 404 幻影？display name 谁生成？accumulate 在哪一层？

> ⚠️ **没完成第一阶段、没确证根因前，不要写修复代码。** 本 spec 的教训正是"别在根因不明时假装知道"（前几轮 dogfood 报告误判根因的复盘）。

---

## 3. 第二阶段：修复（根因确证后，按定位结果选方案）

根据第一阶段定位，**预设几条修复方向**（执行 agent 按实际根因选，不是全做）：

### 若根因 = artifacts 跨轮累积（假设 1/4）
- 展示层应以**当前轮 chart-maker handoff 的 `chart_files`** 为准，而非累积的 `artifacts` 列表。
- 或：展示"本次分析产物"时按 thread 当前 outputs 实际文件过滤，不展示已不存在的历史条目。
- **不要简单清空 merge_artifacts**（它可能有别的正当用途——历史产物下载等）。要在**展示逻辑**层按"本轮 handoff 声明"过滤。

### 若根因 = report.md 的 {{img}} 渲染按指标拼名（假设 2/3）
- 检查 report-writer 写 report.md 时 `{{img:...}}` 用的是否是真实文件名（`plot_trajectory.png`）。若 report-writer 引用了不存在的 `epm_bar_*.png` → 是 report-writer 的 bug（引用了没产出的图）。
- 修：report-writer 只引用 chart-maker handoff 里**真实存在**的 chart_files，不凭指标列表臆造图引用。

### 若根因 = 6 张 404 幻影
- 这是**最该修的**：前端展示了根本不存在的图（坏体验）。展示前应校验文件存在性，或严格按 handoff chart_files（已校验存在）渲染。

> 执行 agent：第二阶段的具体改动**取决于第一阶段定位**。本 spec 不预先写死改哪个文件——因为根因未确证。定位后在本 spec 补"根因确证 + 选定方案"再实施。

---

## 4. 测试

取决于根因。原则：
- 若修前端展示逻辑 → 加前端测试（若有框架；frontend CLAUDE.md 说"No test framework configured"——则靠 playwright 行为验证）。
- 若修 report-writer 图引用 → 加 backend 测试：report-writer handoff 的图引用 ⊆ chart-maker handoff 的 chart_files。
- **dogfood 验证（必做）**：复跑 EPM，确认前端展示的图**数量 = chart-maker 真实产出数**（本例 2 张），无 404 幻影、无重复。

---

## 5. 验收标准

1. **第一阶段**：根因确证文档（§2 产出），明确 8 张图的来源 + 命名来源 + 累积层。
2. **第二阶段**：按确证根因修复，dogfood 复跑展示数 = 真实产出数（2 张），无幻影/重复。
3. 不破坏 artifacts 的正当用途（若有历史产物下载等功能）。
4. 全量回归（若动 backend）：`make test`（已知污染 + config symlink）。

---

## 6. 风险与边界

- **这是 P2、不阻塞分析**（图能看，只是多/重复）。优先级低于 A/B/C/D。
- **根因未完全确证**是本 spec 与其他的最大区别——第一阶段调查是硬门槛，跳过会改错。
- 涉及前端，可能需要前端运行时调试（chrome-devtools/playwright MCP）+ 可能跨前后端。
- 与其他 spec 正交。

---

## 7. 关联

- memory：`project_2026-06-08_epm_dogfood_routing_and_constitution_leak.md`（"其他观察"段，本 spec 来源 + 证据边界）
- 累积 reducer：`thread_state.py:merge_artifacts`（22）、`artifacts` 字段（66）
- 前端图路径规范化：`frontend/src/core/artifacts/utils.ts:normalizeArtifactImageSrc`
- chart-maker handoff chart_files 是真实产出的权威清单
- 同批 spec A/B/C/D
