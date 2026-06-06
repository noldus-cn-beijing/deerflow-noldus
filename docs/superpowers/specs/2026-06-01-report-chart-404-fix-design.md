# 2026-06-01 设计 spec — 修复 report 图表在网页打开 404（图片路径错误）

**类型**：可实施版（已对 dev HEAD `065b4180` 核验；实施前 `git pull` 复核行号）
**对应**：2026-06-01 用户 dogfood 报告「网页打开 report.md 看不到图表，控制台 404」
**估期**：~0.5 天（前端 1 处 + report_writer prompt 1 处）
**前置**：无

> **故障现象**：用户在网页打开 report.md，4 张图表全显示 `Image not available`，控制台报：
> `plot_trajectory_plot_s0.png:1 Failed to load resource: 404 (Not Found)`（4 张图全 404，在 localhost:2027 前端端口）
> `<p> cannot contain a nested <div>`（次要 DOM 警告，修好路径自然消失）

---

## 0. 目标与原则

**目标**：让 report.md 里的图表在网页正常显示。**两者都改**（用户 2026-06-01 决策）：① 前端 rewrite 治本（不依赖 LLM 每次写对）；② report-writer prompt 治源头（输出正确路径）。

---

## 1. 根因（已用证据闭环）

**图片文件全都在**（outputs 目录 4 张 png 完整：1.2MB/989KB/324KB/331KB，生成成功）。问题纯粹是 **report.md 引用了宿主机绝对路径，前端取不到**。

### 证据链

1. **report-writer 写的图片路径是宿主机绝对路径**：
   ```markdown
   ![Figure 1](/home/wangqiuyang/noldus-insight/packages/agent/backend/.deer-flow/users/<uid>/threads/<tid>/user-data/outputs/plot_trajectory_plot_s0.png)
   ```
   浏览器把它当 URL → 解析成 `http://localhost:2027/home/wangqiuyang/...` → **404**

2. **chart handoff 给的本是 sandbox 虚拟路径**（正确内部格式）：
   `handoff_chart_maker.json` 的 `chart_files` = `/mnt/user-data/outputs/plot_*.png`
   → report-writer **自作主张把虚拟路径「还原」成了宿主机绝对路径**（prompt 没教正确转换，`report_writer.py:52/96-98` 只说「直接引用 chart_paths 中的路径」，未规定格式）

3. **前端 Streamdown 不 rewrite img src**：
   `artifact-file-detail.tsx:289` 的 `<Streamdown>` 只对 `a`（链接）配了自定义组件 `ArtifactLink`，**对 `img` 没有任何 src rewrite** → 图片 src 原样塞给浏览器

4. **前端 artifact 的正确 URL 契约**（`core/artifacts/utils.ts:4-19`）：
   `urlOfArtifact = {backendBaseURL}/api/threads/{threadId}/artifacts{filepath}`
   其中 `filepath` 是以 `/` 开头、相对 thread user-data 根的虚拟路径（如 `/outputs/plot_*.png`）

5. **稳定复现的老 bug**：历史 3 个 thread（439b1405 / af5bc3a2 / 3ffb2d20）的 report **全部**用宿主机绝对路径，全部图表打不开。不是这次才有。

**一句话**：report-writer 每次把图片写成宿主机绝对路径 + 前端不 rewrite img → 浏览器永远 404。

---

## 2. 修法（两者都做）

### 修法 A（治本）— 前端 Streamdown 加 img src rewrite

在 `artifact-file-detail.tsx` 的 `ArtifactFilePreview`（`:286-298` markdown 分支）给 `<Streamdown>` 的 `components` 加一个自定义 `img` 组件（**仿照已有的 `a: ArtifactLink` 模式**），对图片 src 做归一化：

- **输入可能的三种 src**，都要正确处理：
  1. 宿主机绝对路径 `/home/.../threads/<tid>/user-data/outputs/X.png`（当前 report-writer 实际产出）
  2. sandbox 虚拟路径 `/mnt/user-data/outputs/X.png`
  3. 相对路径 `outputs/X.png` 或 `plot_X.png`（修法 B 后的理想产出）
- **统一转成** artifact API URL：`urlOfArtifact({ filepath: "/outputs/X.png", threadId })`（复用 `core/artifacts/utils.ts` 现成函数，不要手拼）
- **归一化逻辑**：从任意上述 src 中提取「`/outputs/` 之后的部分」拼成 `filepath="/outputs/<basename>"`；已经是 http(s):// 的外链原样放行

> **实施锚点**：
> - 参考 `:267` `<iframe src={urlOfArtifact({ filepath, threadId, isMock })}>` 看 threadId 怎么拿到（组件已有 `threadId`）
> - 参考 `:292` `components={{ a: ArtifactLink } as Components}` 的写法，加 `img`
> - `urlOfArtifact` / `resolveArtifactURL` 在 `@/core/artifacts/utils`
> - 修复后 `<p> cannot contain a nested <div>` 警告应消失（图能正常 render 不再被异常包裹）；若仍有，是 Streamdown 把 img 包进 p 的渲染问题，次要

### 修法 B（源头）— report_writer prompt 教写正确路径

改 `report_writer.py`（**受保护文件**，subagent prompt）的图表引用段（`:52`、`:95-98`、`:210-215`）：
- 明确：**图片用相对路径或 `/outputs/X.png` 虚拟路径引用，禁止写宿主机绝对路径**
- 从 chart handoff 的 `chart_files`（`/mnt/user-data/outputs/X.png`）取 basename，写成 `![Figure N: 标题](outputs/X.png)` 或保留 `/mnt/user-data/outputs/X.png`（与修法 A 的归一化兼容即可）
- 用 deepseek 正面提示（CLAUDE.md §6）：「图片路径写成 `outputs/文件名.png` 这种相对形式」而非「不要写绝对路径」

> **为什么两者都做**：修法 A 让**已有的坏 report 和未来 report 都能显示**（治本、不靠 LLM）；修法 B 让**输出本身就干净**（源头正确，且未来若前端逻辑变也不脆）。A 是安全网，B 是源头规范。

---

## 3. 实施前核验清单

1. `git pull` 复核行号
2. **前端**：`artifact-file-detail.tsx` 确认 `ArtifactFilePreview` 拿不拿得到 `threadId`（若没有需从父组件传入 — 看 `:286` 的 props）；确认 `Streamdown` 的 `components` prop 支持自定义 `img`（看 Streamdown 版本/类型）
3. **前端验证手段**：用现有坏 report（如 thread `daf5164b` 的 report.md，图片是宿主机绝对路径）在本地网页打开，改后图能显示 = 修法 A 成功
4. **后端**：`report_writer.py` 图表引用段真实行号 + 受保护文件 diff 纪律
5. 真实 chart handoff 的 `chart_files` 格式（确认 basename 怎么提取）

---

## 4. 验收

- [ ] **修法 A**：前端 Streamdown 自定义 `img` 组件，把宿主机绝对路径 / `/mnt/user-data/outputs/X.png` / 相对路径都归一化成 `urlOfArtifact` 的 artifact API URL
- [ ] **🔴 前端回归验证（核心）**：用现有坏 report（thread `daf5164b` 或新跑一个）网页打开 → 4 张图**正常显示**，控制台无 404、无 `<p> cannot contain <div>`
- [ ] **修法 B**：report_writer prompt 教写相对/虚拟路径，禁宿主机绝对路径；受保护文件逐字 diff 确认没动其他报告结构逻辑
- [ ] **新 report 验证**：改后跑一次新分析，report.md 图片引用是相对/虚拟路径（非宿主机绝对路径）且网页能显示
- [ ] 前端 `pnpm check`（lint + typecheck）通过
- [ ] 后端全量 `make test` 不退化（report_writer 是共享 subagent）

---

## 5. 与其他工作的关系

- **独立于** TST spec 和 data-analyst pendulum spec（不同文件、不同层）。可并行实施
- **前端改动** = 唯一一份动 frontend 的 spec；report_writer.py 是受保护后端文件
- report_writer.py 受保护：下次 deerflow sync 按 surgical-merge

---

## 6. 不在范围

- ❌ 改 chart-maker 生成图的逻辑（图生成是好的，问题在引用路径）
- ❌ 改 artifact API 后端路由（契约是对的，前端没用对 + report-writer 没写对）
- ❌ 改 report.md 的其他结构（只修图片引用路径）
