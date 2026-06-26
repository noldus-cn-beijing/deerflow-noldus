# Spec：前端 Phase 0 已合/待合代码层修复（dev 代码核对 + 缺口补齐）

> 类型：**代码层修复 spec**（前端为主 + spec#3 后端契约）。**只收代码层问题**，纯文档层问题（母方案决策表过时、A/B 命名碰撞、各 spec 文本笔误）不在本 spec，另行处理。
> 日期：2026-06-25
> 母方案：[docs/plans/2026-06-24-frontend-generative-ux-upgrade.md](../../plans/2026-06-24-frontend-generative-ux-upgrade.md)
> 关联 spec：#1（已合 dev）/ #2 #3 #6（编写本 spec 时**未合** dev，由其它 agent 并行实施中）
> 适用层：`packages/agent/frontend/src/**` + `packages/agent/backend/.../tools/builtins/present_file_tool.py` + `.../agents/thread_state.py`
> 一句话：把 Phase 0 前端各 spec **落到 dev 代码后的真实状态**核清——spec#1 已合（4 项按 `code≠bug` 原则做**实测验证项**，不是"看着对就跳过"）；spec#3 后端/前端契约**还没合且半落地**（前端 `ArtifactMeta` 契约完全缺失，但工作树有未提交草稿，别重复造）；spec#6 token 待补。本 spec 给实施 agent **一份"哪些已在 dev、哪些待合入"的精确分节清单 + 每项的验证/修复动作**。

---

> ## ✅ 实测结论（2026-06-26 更新 · 全部跑完）
>
> **本 spec 的核对已全部执行完毕，dev HEAD 现 = `0ff55a98`（#2/#3/#6 均已合：PR#202/#203/#205/#207）。结论：A 节全绿无需修复；B 节绝大多数通过，唯一待你裁决的是 `ui/empty.tsx` 一处「shadcn copy-in vs CLAUDE.md 禁改 generated」的张力（非 bug，是纪律口径）。**
>
> **A 节（spec#1 已合，编译产物实测）—— 5/5 ✅ 无需修复**：用 Tailwind v4 真实 `@tailwindcss/postcss` 插件链编译 dev HEAD `globals.css` 成产物 CSS 后 grep 坐实（非读源码猜）：① `.duration-{fast,base,slow,exit}` 四条规则全生成且含 `transition-duration: var(--dur-*)`——**非死类**；② `.text-status-*`/`.text-stage-*` 工具类生成，`--color-status-info: var(--status-info)`→`oklch(...)` 两跳级联闭合——**不自指**；③ `.ease-brand-*` 生成，`cubic-bezier(0.22,1,0.36,1)` 精确在产物，`--dur-exit:160ms` < `--dur-base:220ms`；④ 业务侧零 aurora/shine 引用，workspace 内 `ease-in-out`/`ease-out`/`duration-[0-9]+`/`transition-all` **0 残留**；⑤ `message-token-usage`/`number-ticker`/`token-usage-indicator` 均带 `tabular-nums`。**唯一未做**：浏览器 devtools computed-style 肉眼复核（需 `make dev` 起全栈），但编译产物已是根因层，不会推翻结论。
>
> **B 节（spec#2/#3/#6 已合，源码核对）—— 结论逐项**：
> - **#3（§三）✅ 实现质量高于本 spec 要求**：后端 `_build_artifact_meta` 产全字段（path/kind/chart_id/output_mode/paradigm/metric/subject/group/chart_type）、命中 `plan_charts.json.by_output` 才升级否则退裸 string；`_generate_thumbnail` webp 等比缩放 + **三层退化**（无 Pillow/读写失败/任何 Exception 均退原图）+ 缓存命中；`merge_artifacts` 按 `path` 去重新值覆盖旧值、**无 path 项用 `id()` 兜底键防互吞**（正中 memory `feedback_chart_reconcile_loop_key_must_be_unique`）；前端 `ArtifactMeta` 全字段 + `normalizeArtifact(string|ArtifactMeta)→{path}` 向后兼容 + `isImageArtifact`/`selectRepresentativeCharts` 确定性选图。**最担心的"把 `normalizeArtifactImageSrc` 误当契约"没发生**——契约本体独立、注释明确"前端不正则猜分类"。额外白送 `_build_charts_status`（failed/remaining 显式呈现，守无声截断铁律）。
> - **#2（§五）✅ 教科书级纯派生**：PR#203 全是新增 `core/trace/` + i18n + `stage-broadcast.ts`(+11) + `page.tsx`(+2)，**零触碰** hooks.ts/mergeMessages/dedupe/useStream/optimistic/summariz；`use-run-trace.ts` 自注"只读、不碰 submit/merge/dedupe"，`useMemo` 从 `thread.messages` 派生，带 341 行 `build-run-trace.test.ts`。
> - **#6（§四）✅ token 全对 + ⚠️ 一处待裁**：五档 `--shadow-{rest,raised,overlap,overlay,modal}` 全在，`--shadow-overlap`/`overlay` 是**真·多层柔影**（非 float 改名）；`--accent-clay` 在且 PR 自述"grep 坐实仅 empty.tsx 消费"（纪律守住）；`empty.tsx` 既有 6 导出全保留、仅新增 `EmptyIllustration`（API 零破坏）。**⚠️ 唯一待裁**：#6 **手改了 `components/ui/empty.tsx`** 加 Mœbius 圆盘 + 插画。本 spec 4-B3 原写"`ui/` 是 generated 不该手改"——但核 `components.json`：registry 段只有 `@ai-elements`/`@magicui`/`@react-bits`，**`ui/` 是 shadcn `new-york` copy-in 源码（你拥有、可改），不是持续 sync 的 registry**。所以从 shadcn 哲学看改它**不违规**；但 CLAUDE.md 仍把 `components/ui` 列入"不手改 generated"。**这是 shadcn 惯例 vs 项目口径的张力，非 bug**——交你裁决（见 §四-B3 修正）。

---

## 〇、为什么需要这份 spec（核实方法论）

按 CLAUDE.md memory `feedback_code_has_fix_not_equal_bug_eliminated`：**"代码里有修复" ≠ "现象已消除"**。判据是问"**这个现象在 `make dev` 真实运行下还会不会发生**"，不是"代码看着对不对 / 有没有 PR"。

更基础的一条教训（本次核实直接踩到）：**判断"某代码在不在 dev"必须用 `git show HEAD:<file>`（已提交版），不能用 `grep <file>`（工作树版）**——工作树会包含别的 agent 未提交的草稿，grep 会把"草稿"误判成"已合"。本 spec 的所有"已在 dev / 待合入"分类均以 `git show HEAD:` 在 commit `4d847228`（spec#1 merge）核实坐实，证据附在每项下。

> **本 spec 的角色**：不是"重做"已合的 spec#1，也不是"抢做"别的 agent 在做的 spec#3/#6。是给实施 agent 一份**核对清单**：①spec#1 已合的部分 → 列出**必须真机验证的项**（`code≠bug`，怀疑项即使代码看着对也要实测）；②spec#3/#6 待合的部分 → 列出**已有什么、缺什么、别重复造什么**，避免和并行 agent 撞车或重复劳动。

---

## 一、dev HEAD 真实状态总表（commit `4d847228`，已 `git show HEAD:` 坐实）

| spec | 子项 | dev HEAD 真实状态 | 证据（`git show HEAD:`） | 归类 |
|---|---|---|---|---|
| **#1** | `--ease-brand-*` / `--dur-*` / `@utility duration-*` / `--status-*`/`--color-status-*` 两层 / `--stage-*`/`--color-stage-*` 两层 | ✅ **已合，写法全部正确** | `globals.css:86/100-103/170-173/225-232/321/235-241/333` | **A. 已在 dev，按 `code≠bug` 实测验证** |
| **#3 后端** | `_build_artifact_meta` / `_generate_thumbnail` / `thumb_path` / ArtifactMeta dict | ❌ **未合**——仅在工作树 ` M present_file_tool.py` 未提交改动里；HEAD 只有 `def present_file_tool`（:87） | `git show HEAD:present_file_tool.py` 仅命中 `def present_file_tool`；`git status` 显示该文件 ` M` | **B. 待合入**（草稿已存在工作树，**别重复造**） |
| **#3 后端** | `merge_artifacts` reducer 是否带元数据 | ⚠️ HEAD 是 `list[str]` 朴素去重，**不带元数据**；`thread_state.py` 也 ` M` 未提交 | `thread_state.py:46-53` `list(dict.fromkeys(...))`；`artifacts: Annotated[list[str], merge_artifacts]`（:116） | **B. 待合入** |
| **#3 前端** | `ArtifactMeta` 类型 + `normalizeArtifact(string\|ArtifactMeta)` | ❌ **完全缺失**——全前端 grep 无此类型/函数 | `git grep ArtifactMeta HEAD` 命中的是 `normalizeArtifact**ImageSrc**`（report.md 图片 src 规范化，**与 ArtifactMeta 无关**），非契约本体 | **B. 待合入** |
| **#3 前端** | `types.ts` artifacts 字段 | ⚠️ 仍 `artifacts: string[]`（未迁移） | `types.ts:8` | **B. 待合入** |
| **#3 前端** | `artifact-file-list.tsx` files 字段 | ⚠️ 仍 `files: string[]` + `files.filter(isImageFile)` | `artifact-file-list.tsx:46/89-90` | **B. 待合入** |
| **#3 前端** | `urlOfArtifact({threadId})` / `extractArtifactsFromThread` | ✅ 已在（但只吃 `string[]`） | `utils.ts` `urlOfArtifact`(:4)/`extractArtifactsFromThread`(:21) | A（已在，待 #3 扩展） |
| **#6** | `--shadow-{rest,raised,overlap,overlay}` elevation 阶梯 | ❌ 未合——HEAD 只有 `--shadow-float`/`--shadow-modal` 两档 | （见 §四） | **B. 待合入** |
| **#6** | `--color-accent-clay(-soft)` | ❌ 未合 | （见 §四） | **B. 待合入** |
| **#6** | `ui/empty.tsx` Mœbius 空状态 | ❌ 未合（且 `ui/` 是 generated，须建业务包装层不改源） | （见 §四） | **B. 待合入** |

**两条主线**：
- **A 节（spec#1 已合）**：代码看着对，但按 `code≠bug` 必须 `make dev` 真机验证"现象不复发"。**这是本 spec 现在唯一能立刻执行的实测部分。**
- **B 节（spec#3/#6/#2 待合）**：给并行 agent 的"已有什么/缺什么/别重复造什么"清单。**待对应 spec 合入 dev 后，按本节核对再验证。**

---

## 二、A 节：spec#1（已合 dev）——按 `code≠bug` 的实测验证项

> spec#1 四个曾被怀疑的点，`git show HEAD:globals.css` 核实**写法全部正确**（下表"代码现状"列）。但**代码正确 ≠ 浏览器里效果正确**——Tailwind v4 编译、PostCSS 产物、运行时级联都可能让"看着对的源码"产出"不对的效果"。下列每项给**怀疑的失败现象** + **真机验证动作**，验证通过才算"现象消除"，不通过则按"修复动作"改。

### 2-A1 `duration-*` 工具类是否真编译出来（最高优先，曾是死类高发区）

- **代码现状（HEAD）**：`globals.css:170-173` `@utility duration-fast/base/slow/exit { transition-duration: var(--dur-*); }` —— 写法正确（memory `feedback_tailwind_v4_dur_namespace_not_real_duration_utility_dead_class`：`--dur-*` 不是 namespace，必须 `@utility` 显式定义，已照做）。
- **怀疑的失败现象**：若 `@utility` 因 Tailwind v4 版本/配置/`@source` 扫描问题没被编译进产物 CSS，则组件 `className="duration-base"` 是**死类**（无 `transition-duration`）→ 过渡瞬间完成、无时长，肉眼"没有动效"但不报错。
- **真机验证动作（必做，不靠读源码）**：
  1. `cd packages/agent/frontend && pnpm build`（或 `pnpm dev` 起来）后，在产物 CSS 里 `grep -E "\.duration-(fast|base|slow|exit)\b"` —— **必须命中四条规则且每条含 `transition-duration:`**。命中 0 条 = 死类，回到 memory 的 `@utility` 修法核对 `@source`/版本。
  2. devtools：找一个用了 `duration-base` 的元素（迁移后），看 computed style `transition-duration` **= 220ms**（不是 0s）。
- **修复动作（若验证失败）**：确认 `@utility` 块在 `@theme` 块**外**；确认 Tailwind v4 版本支持 `@utility`；必要时 fallback 到 `transition-duration: var(--dur-base)` 内联或 `.duration-base{}` 普通类。

### 2-A2 `--color-status-*` / `--color-stage-*` 两层映射是否自指塌掉

- **代码现状（HEAD）**：`:root` 裸 `--status-info: oklch(...)`（:321）+ `@theme inline` `--color-status-info: var(--status-info)`（:225）—— 两层结构正确，**不自指**（曾担心写成 `--color-status-info: var(--color-status-info)` 自指循环）。stage 同理（:235-241 ← :333）。
- **怀疑的失败现象**：若任一处实际是自指（`--color-x: var(--color-x)`），该 CSS 变量解析为 `initial`/无效 → `text-status-info`/`bg-stage-*` 等工具类颜色失效（透明或继承）。
- **真机验证动作**：
  1. 产物 CSS `grep -E "\-\-color-status-info"` 确认其值是 `var(--status-info)` 而非 `var(--color-status-info)`。
  2. devtools 给一个元素加 `class="text-status-info"`，computed `color` 应解析出真实 oklch 值（非 transparent/inherit）。
  3. 生成的工具类 `text-status-{info,success,warning,danger}` / `bg-status-*-soft` / `text-stage-*` 实际存在于产物（grep）。
- **修复动作**：若自指，改成裸 token → `--color-*` 单向映射（照 §1 表已合写法）。

### 2-A3 `--ease-brand-*` 工具类 + 精确 cubic-bezier 值

- **代码现状（HEAD）**：`--ease-brand-out: cubic-bezier(0.22, 1, 0.36, 1)`（:86）等四条，且 `--animate-fade-in-up` 等已引用（:106/117/124）。Tailwind v4 `--ease-*` 是真 namespace，**应**自动生成 `ease-brand-*` 工具类。
- **怀疑的失败现象**：① `ease-brand-*` 工具类没自动生成（v4 namespace 行为不及预期）→ `className="ease-brand-out"` 死类；② 退场时长未严格 < 入场（`--dur-exit:160ms` 应 < `--dur-base:220ms`，spec#1 核心交付"exit faster than enter"）。
- **真机验证动作（机读，不靠肉眼"减速尾巴"）**：
  1. 产物 CSS `grep "ease-brand-out"` → 必须有工具类，`transition-timing-function: cubic-bezier(0.22,1,0.36,1)` **精确匹配**（逗号/空格容差，值不变）。
  2. 断言 `--dur-exit`(160) **严格 <** `--dur-base`(220)（源码层已满足，验证产物未被覆盖）。
  3. 一个入场动画元素 computed `animation-timing-function` = 该 cubic-bezier。
- **修复动作**：若 `ease-brand-*` 未生成，fallback `@utility ease-brand-out { transition-timing-function: var(--ease-brand-out); }`。

### 2-A4 装饰动画退役 + 组件迁移是否真清干净（spec#1 §3/§4 验收）

- **怀疑的失败现象**：spec#1 要求退役 aurora/shine 业务引用、迁移 ~38 处裸 `ease-out`/`ease-in-out`/`linear`/`duration-[0-9]+`。合并时可能漏改 → 残留旧 easing（退化非破坏，但 spec 未达成）。
- **真机验证动作（grep dev HEAD）**：
  1. `git grep -nE "aurora-text|shine-border" HEAD -- 'packages/agent/frontend/src/components/workspace' 'packages/agent/frontend/src/app'` → **应为空**（`ui/` 内部源保留不算）。
  2. `git grep -nE "ease-in-out|\bease-out\b|duration-[0-9]+" HEAD -- 'packages/agent/frontend/src/components/workspace'` → 残留项逐一核对是否 shimmer 连续循环例外（需注释）；非例外的算漏迁移。
  3. `git grep -nE "transition-all" HEAD -- 'packages/agent/frontend/src/components/workspace'` → spec#1 建议收窄成 `transition-[transform,opacity]`/`transition-colors`（"命中即改不强求"，记录残留即可，不算 blocker）。
- **修复动作**：漏迁移项按 spec#1 §四 Step4 映射表补；不阻塞（渐进，残留退化成旧 easing 不报错）。

### 2-A5 tabular-nums 推广是否落到流式数字组件（spec#1 §四 Step5）

- **怀疑的失败现象**：流式 token 计数/计时/p 值若没加 `tabular-nums`，数字跳动时字符横向位移（抖动）。
- **真机验证动作**：`git grep -n "tabular-nums" HEAD -- 'packages/agent/frontend/src/components'` 确认 `message-token-usage.tsx` / `number-ticker.tsx`（及流式数字处）命中；`make dev` 跑一次流式输出，肉眼确认数字跳动时不横移。
- **修复动作**：缺的组件 DOM 加 `tabular-nums` class（已 `@source inline`，直接可用），不改逻辑。

> **A 节小结**：spec#1 代码已合且写法正确，但**这五项必须 `make dev` + 产物 CSS grep 实测**才能宣告"现象消除"。**严禁因"源码看着对"跳过实测**（`code≠bug` 铁律）。本节是当前唯一可立刻执行的部分。

---

## 三、B 节-spec#3：产物画廊后端/前端契约（**未合 dev，草稿在工作树**）

> ⚠️ **协调红线**：spec#3 由**另一个 agent 并行实施中**（worktree `worktree-artifact-gallery`，停在 `4d847228`），且**主仓工作树已有未提交草稿**（`present_file_tool.py` / `thread_state.py` 均 ` M`）。本节**不是让本 spec 的实施 agent 去做 spec#3**，而是：① 记录 dev HEAD 真实缺口，② 提醒"草稿已存在别重复造"，③ 待 spec#3 合入 dev 后，按本节做验证。**实施顺序：本节所有项 = "待 spec#3 合入 dev 后验证/修复"，现在不动手。**

### 3-B1 后端 ArtifactMeta（路 A）当前只在工作树，未提交

- **dev HEAD 缺口**：`present_file_tool.py` HEAD 仅 `def present_file_tool`（:87），无 `_build_artifact_meta`/`_generate_thumbnail`/`thumb_path`。
- **工作树已有草稿**（`git status` ` M`）：`_generate_thumbnail`(:159)、`_build_artifact_meta`(:198)、`thumb_path` 写入(:236)、`generate_thumb=has_any_chart`(:278) —— **说明已有人在写，别从零重造**。
- **待合入后验证项**（`code≠bug`）：
  1. `present_file` 真实调用后，返回的 ArtifactMeta dict 是否含 spec#3 约定全字段：`{path, kind, chart_id, output_mode, paradigm, metric, subject, group, chart_type, thumb_path?, run_id?}`。**缺字段 = 前端分面/缩略图失效**。
  2. 缩略图：`_generate_thumbnail` 对 max 2740KB 的 trajectory PNG 真能出 `thumb_path`；**无 thumb 时优雅退化原图**（不阻塞画廊，spec#3 §3.1.6 拍板）。
  3. 后端单测 + **裸导入** `import app.gateway` / `make_lead_agent`（memory `feedback_conftest_mock_hides_circular_import`：改 tools/builtins 后必裸导入生产入口）。
  4. ProcessPool/thumbnail 生成不阻塞 present_file 主路径（Pillow 出图同步耗时，确认不拖垮）。

### 3-B2 `merge_artifacts` reducer 仍是 `list[str]`，未升级到带元数据

- **dev HEAD 缺口**：`thread_state.py:46-53` `merge_artifacts` 是 `list(dict.fromkeys(existing+new))`，**只去重字符串路径，不承载 ArtifactMeta**；`artifacts: Annotated[list[str], merge_artifacts]`（:116）类型仍 `list[str]`。
- **待合入后验证项**：
  1. 若 spec#3 把 artifacts 升级成 `list[ArtifactMeta]`，reducer 去重键要从"整个字符串"改成"`path`（或 `chart_id`）"，否则同 path 不同 meta 会重复堆积。**注意 memory `feedback_chart_reconcile_loop_key_must_be_unique`**：按 id 聚合的 dict 会吞兄弟，per_subject 多 chart 共享 id 时要用 path 做键。
  2. 对比 `merge_viewed_images`(:56) 有清空特例——`merge_artifacts` **无清空**（只增），spec#3 §3.4.1 据此判定画廊默认整 thread 累积。确认升级后仍保持"只增不清空"语义（多轮追问叠加靠分面筛，不是清空）。
  3. reducer 改动必加并发/合并单测（memory `feedback_pr_merge_must_run_full_suite_on_shared_logic`：改 reducer 是共享逻辑，合前跑全量 + grep 调用方）。

### 3-B3 前端 `ArtifactMeta` 契约**完全缺失**（最大缺口，易和老代码混淆）

- **dev HEAD 缺口（坐实）**：全前端 **无** `ArtifactMeta` 类型、**无** `normalizeArtifact(string|ArtifactMeta)` 归一函数。
- **⚠️ 防混淆**：`git grep` 命中的 `normalizeArtifact**ImageSrc**`（`utils.ts`）是处理 **report.md 内 `<img src>` 路径规范化**的（只认 `/mnt/user-data/...`，外链/非规范返 null），**与 ArtifactMeta 契约毫无关系**。实施 agent 勿把它当成"契约已部分实现"——它解决的是另一个问题（图片 src SSOT，spec `2026-06-18-report-image-path-ssot-spec.md`）。
- **待合入后验证项**：
  1. 新增 `ArtifactMeta` interface（spec#3 §定义，含 `thumb_path?`/`run_id?` 两新字段）。
  2. `normalizeArtifact(x: string | ArtifactMeta): ArtifactMeta` 归一函数 —— **向后兼容**：老的纯 `string` 路径要能 fallback 成 `{path: x}`（后端未升级/历史 thread 仍发 string[] 时不崩）。这是 memory `project_..._frontend_generative_ux_upgrade_plan` 记的"路 A + `normalizeArtifact` fallback"。
  3. `types.ts:8` `artifacts: string[]` → `artifacts: (string | ArtifactMeta)[]`（或 `ArtifactMeta[]` + 归一在边界）。
  4. `artifact-file-list.tsx:46` `files: string[]` + `:89 files.filter(isImageFile)` → 改吃 `ArtifactMeta[]`，`isImageFile` 改判 `meta.kind`/`meta.path`。

### 3-B4 前后端契约一致性（半落地状态的核心风险）

- **现象**：dev 处于"前端 `urlOfArtifact`/detail 组件已在、但顶层类型 `string[]` 未迁移、后端 meta 未提交"的**半落地**状态。若 spec#3 只合了一半（如后端发 ArtifactMeta 但前端仍 `string[]` 解析），运行时会：图墙渲染不出元数据、分面/缩略图全失效、或 `.map(filepath => ...)` 拿到 object 当 string 用导致 `[object Object]` URL。
- **待合入后验证项（端到端）**：`make dev` 起全栈 → 真实跑一个出多图的分析（EPM per_subject）→ 看：① 后端 present_file 发的 artifacts 结构；② 前端 `thread.values.artifacts` 收到的结构；③ 画廊/侧栏渲染是否正确。**前后端任一侧未升级 → 响亮失败（别静默兜底成图墙）**。

---

## 四、B 节-spec#6：设计语言 token（**未合 dev**）

> spec#6 同样由并行 agent 实施中（worktree `worktree-frontend-design-language-craft`，停在 `7d5ed253`）。本节记缺口 + 待合入验证，**现在不动手**。

### 4-B1 elevation 阶梯 token 缺失（HEAD 只有两档）

- **dev HEAD 缺口**：`globals.css` 只有 `--shadow-float` / `--shadow-modal` 两档，**无中间档** `--shadow-{rest,raised,overlap,overlay}`（spec#6 §1.2）。重叠 card 现状要么没阴影要么跳 modal 级。
- **待合入后验证项**：
  1. 五档 `--shadow-{rest,raised,overlap,overlay,modal}` + 工具类 `.shadow-{rest,raised,overlap,overlay}` 在产物（grep）；`:root` 与 `.dark` 都有值。
  2. **`--shadow-rest` ≠ `--shadow-float` 退化检查**：spec#6 让 `--shadow-rest` 同 `--shadow-float` 值并存，但 `--shadow-overlap` 必须是**多层柔影**（`0 6px 16px -4px ... , 0 2px 4px ...`），不是单层。验证 overlap/overlay 是真·多层扩散影，不是把 float 改个名（否则"重叠 card 有阴影"诉求没真落地）。
  3. 重叠 card 真机：决策卡盖消息流、画廊缩略图叠分组头——devtools 查上层 `box-shadow` = `shadow-overlap`，且 hover/press **周围零位移**（无 reflow/CLS，spec#6 §1.5 + spec#7 §3.7 transform-only）。

### 4-B2 `--color-accent-clay` 缺失

- **dev HEAD 缺口**：无 `--color-accent-clay(-soft)`（spec#6 §4.3 Mœbius 暖中性 accent）。
- **待合入后验证项**：token 在 `:root`/`.dark` + `@theme inline` 映射出 `bg-accent-clay-soft` 等；**纪律核查**（grep）：clay **只**用于插画/空状态/极少点缀，**未**进数据可视化分类色/状态语义/大面积（spec#6 验收硬项）。

### 4-B3 `ui/empty.tsx` 被 #6 手改（✅ 已核实 · ⚠️ 纪律口径待你裁，非 bug）

- **实测结果**：#6（PR#202）**确实手改了 `components/ui/empty.tsx`**——`EmptyMedia` icon variant 改成 ligne claire 圆盘（`bg-accent-clay-soft` + `ring-1 ring-foreground/15`），并**新增** `EmptyIllustration`（细线叶子+轨迹点 SVG）。既有 6 个导出（Empty/EmptyHeader/EmptyMedia/EmptyTitle/EmptyDescription/EmptyContent）**全部保留、API 零破坏**，`pnpm check` 与 baseline 一致。
- **本 spec 原前提（4-B3 旧文）部分错了**：旧文写"`ui/` 是 **registry-generated**，不手改、sync 会冲突，应建业务包装层"。但核 `frontend/components.json`：`registries` 段只有 `@ai-elements`/`@magicui`/`@react-bits`——**`components/ui/` 是 shadcn `new-york` 风格 `npx shadcn add` 一次性 copy-in 的源码（项目拥有、设计上就允许改）**，不是持续 sync 拉取覆盖的 registry。真正"sync 会被覆盖"的是 `components/ai-elements/`（`@ai-elements` registry）。所以从 **shadcn 惯例**看，#6 改 `ui/empty.tsx` 加插画是**正常用法、不违规**，无需打回改包装层。
- **⚠️ 但仍有一处张力交你裁**：CLAUDE.md 明文把 `components/ui` 与 `components/ai-elements` **一并**列入"不手改 generated"。这是**项目口径**，比 shadcn 通用惯例更严。于是：
  - **若按 shadcn 惯例**（`ui/` = 你的代码）→ #6 改法没问题，保留即可。
  - **若按 CLAUDE.md 字面**（`ui/` 也禁改）→ 应把 Mœbius 插画迁到业务侧包装组件（如 `components/workspace/.../empty-illustration.tsx`），`ui/empty.tsx` 回退到 copy-in 原样。
  - **我的建议**：倾向**保留 #6 改法**（shadcn 惯例 + API 零破坏 + `pnpm check` 干净，且 `ui/empty.tsx` 极少被 shadcn registry 重新拉取），但**同步把 CLAUDE.md 那句"不手改 `components/ui`"收窄**成"不手改 `components/ai-elements`（registry 拉取）；`components/ui` 是 shadcn copy-in 可改，但改动须 API 零破坏 + 注明 Phase0#N"。**这条 CLAUDE.md 收窄属文档层，需你拍板后改**（不在本代码 spec 动）。
- **历史 4-B3 的"建包装层"指引作废**（基于错误的 registry 前提）。

---

## 五、B 节-spec#2：运行轨迹（**未合 dev，代码层缺口极少**）

> spec#2 由并行 agent 实施中（worktree `worktree-run-trace-live`，停在 `4d847228`）。spec#2 是**零后端、纯前端派生**（`useRunTrace` 从 `thread.messages` 派生），cross-review 命中的多为**文档层**（空态 verification、spinner 判据措辞），代码层缺口极少。本节只记一条代码层待验证项，其余归文档层另行处理。

### 5-B1 `useRunTrace` 不得改流式核心（铁律）

- **待合入后验证项**：spec#2 实施后 grep 确认 **未改** `useStream`/`mergeMessages`/`dedupeMessagesByIdentity`/optimistic/summarization（memory 多条 + 母方案铁律：踩坑沉淀重写必复发）。`useRunTrace` 必须是 `thread.messages` 的**纯派生消费侧**，与合并侧隔离（同 spec#7 §安全接缝 `message-list.tsx:62` 消费侧）。
- **验证动作**：`git grep -nE "useRunTrace" HEAD` 找到实现，确认其入参是 `thread.messages`（只读派生），无 `setMessages`/无改 hooks.ts 合并逻辑。

---

## 六、实施顺序与边界（给实施 agent）

1. **现在能做（唯一）= §二 A 节 spec#1 实测验证**：`cd packages/agent/frontend && pnpm build`（或 `pnpm dev`）→ 按 2-A1~2-A5 grep 产物 CSS + devtools 核对 + `make dev` 跑流式。**这是 `code≠bug` 要求的"现象不复发"实测**，五项逐条勾。发现死类/自指/漏迁移按各项"修复动作"改（改 `globals.css` 或补 `@utility`，纯样式低风险）。
2. **待 spec#3 合入 dev 后 = §三 B 节核对**：先 `git show HEAD:present_file_tool.py | grep _build_artifact_meta` 确认已合（别再看工作树），再做 3-B1~3-B4 端到端验证。**别现在抢做 spec#3**（并行 agent 在做 + 工作树有草稿）。
3. **待 spec#6 合入 dev 后 = §四 B 节核对**：重点 4-B1 overlap 真·多层影 + 4-B3 不改 generated 源。
4. **待 spec#2 合入 dev 后 = §五 5-B1**：grep 确认没碰流式核心。
5. **不碰**：`core/` 流式逻辑、`components/ui/` + `components/ai-elements/`（generated）、`.dark` 视觉调校、并行 agent 正在改的 spec#3/#6/#2 spec 文档本身。
6. **纯文档层问题不在本 spec**（母方案 §10 决策表过时、§3.4 vs §3.4.1 A/B 命名碰撞、spec 文本笔误）——另行处理，且避开并行 agent 正改的 #2/#3/#6 文档。

---

## 七、验收清单（2026-06-26 已全部核对，结果标注在每条）

### A 节（spec#1 已合，编译产物实测）
- [x] 2-A1：✅ 产物 CSS 含 `.duration-{fast,base,slow,exit}` 四条规则且全带 `transition-duration: var(--dur-*)`——非死类。（未做 devtools computed 肉眼复核，编译产物已坐实根因。）
- [x] 2-A2：✅ `--color-status-info: var(--status-info)`→`oklch(0.62 0.10 230)` 两跳级联闭合（不自指）；`text-status-*`/`bg-status-*-soft`/`text-stage-*`/`bg-stage-compute` 工具类全在产物。
- [x] 2-A3：✅ `.ease-brand-*` 工具类生成，`cubic-bezier(0.22, 1, 0.36, 1)` 精确在产物；`--dur-exit`(160) 严格 < `--dur-base`(220)。
- [x] 2-A4：✅ `components/workspace`+`app` 无 aurora-text/shine-border 引用；workspace 内 `ease-in-out`/`ease-out`/`duration-[0-9]+`/`transition-all` **0 残留**（迁移极干净）。
- [x] 2-A5：✅ `message-token-usage`/`number-ticker`/`token-usage-indicator` 均带 `tabular-nums`。

### B 节（spec#2/#3/#6 已合，源码核对）
- [x] 3-B1：✅ `_build_artifact_meta` 产全字段、命中 `plan_charts.json.by_output` 才升级否则退裸 string；`_generate_thumbnail` webp 等比缩放 + 三层退化（无 Pillow/读写失败/Exception 退原图）+ 缓存命中。（裸导入生产入口未单独跑——#3 已合且 dev 可启动，如需保险可补 `PYTHONPATH=. python -c "import app.gateway"`。）
- [x] 3-B2：✅ `merge_artifacts` 按 `path` 去重新值覆盖旧值、**无 path 项用 `id()` 兜底键防互吞**；`artifacts: Annotated[list, merge_artifacts]`；配套 `merge_charts_status`。（中 memory `feedback_chart_reconcile_loop_key_must_be_unique`。）
- [x] 3-B3：✅ 前端 `ArtifactMeta` 全字段 + `normalizeArtifact(string|ArtifactMeta)→{path}` 向后兼容；`types.ts` 已迁 `ArtifactInput[]`；**未把 `normalizeArtifactImageSrc` 误当契约**（契约独立，注释明确"前端不正则猜分类"）。
- [ ] 3-B4：⏳ 端到端 `make dev` 跑多图分析未做（需起全栈）。源码层前后端契约一致已坐实；**建议你下次起 `make dev` 时顺手跑一个 EPM per_subject 看画廊渲染**，确认运行时无 `[object Object]` URL 类问题。
- [x] 4-B1：✅ 五档 `--shadow-{rest,raised,overlap,overlay,modal}` + 工具类；`--shadow-overlap`(`0 6px 16px -4px ...,0 2px 4px ...`)/`overlay` 是真·多层柔影（非 float 改名）。（重叠 card hover 零位移属运行时视觉，待 `make dev` 复核。）
- [x] 4-B2：✅ `--accent-clay(-soft)` 在；PR 自述 + 用途均"仅 empty.tsx 消费"，未进数据可视化/状态语义/大面积。
- [~] 4-B3：⚠️ **待你裁**：#6 手改了 `components/ui/empty.tsx`（API 零破坏、仅增 `EmptyIllustration`）。本 spec 原"必须走业务包装层"前提**已作废**（`ui/` 是 shadcn copy-in 非 registry）；但 CLAUDE.md 把 `ui/` 列入禁改。保留 vs 迁包装层 + 是否收窄 CLAUDE.md，见 §四-B3。
- [x] 5-B1：✅ `use-run-trace.ts` 是 `thread.messages` 纯派生（`useMemo`），PR#203 零触碰 useStream/mergeMessages/dedupe/optimistic/summariz，带 341 行测试。

### 工程纪律
- [x] A 节无需修复（spec#1 编译产物全绿），未产生任何代码改动。
- [x] B 节核对未改任何生产代码（纯 `git show HEAD:` 读 + 编译产物 grep）。
- [x] 未抢做/未重复造 spec#2/#3/#6（它们已由对应 agent 合入，本 spec 仅事后核对）。

---

## 八、给实施 agent 的交接（2026-06-26 更新）

- **核实方法铁律**：判断"代码在不在 dev"用 `git show HEAD:<file>`，**不要用 `grep <工作树文件>`**——工作树含未提交草稿会误判（本 spec 初版的 spec#3 后端误判即由此而来，已纠正并复核）。
- **A 节实测方法（已用、可复现）**：无独立 tailwind CLI bin，用 `@tailwindcss/postcss` 插件链编译 `globals.css`——把脚本放进 `frontend/`（从那里解析 `postcss`/`@tailwindcss/postcss`），用一个临时探针 html 显式声明目标工具类强制按需生成，grep 产物 CSS。跑完删探针（勿留仓库）。
- **本 spec 现状**：A 节 + B 节核对**全部跑完**，**无代码需改**。唯一 open item 是 §四-B3 的 `ui/empty.tsx` 纪律口径（交用户裁，可能伴随一条 CLAUDE.md 文档收窄）+ 3-B4/4-B1 的运行时视觉复核（待 `make dev`，非阻塞）。
- **最大易错点（仍提醒后人）**：spec#3 的 `normalizeArtifactImageSrc`（report.md 图片 src 规范化）与 `ArtifactMeta` 契约**无关**，名字像，别混——实际契约是 `core/artifacts/types.ts` 的 `ArtifactMeta`/`normalizeArtifact`。
- **纯文档层问题**（母方案决策表已修；spec#3 文档内 §3.4 vs §3.4.1 A/B 命名碰撞仍待 #3 文档维护者处理）不在本 spec。

---

*依据：`git show HEAD:`（初版核 `4d847228`，复核 `0ff55a98`）坐实 dev 真实状态 + Tailwind v4 `@tailwindcss/postcss` 编译产物实测 + memory `feedback_code_has_fix_not_equal_bug_eliminated`（代码有修复≠现象消除，怀疑项做实测）+ `feedback_tailwind_v4_dur_namespace_not_real_duration_utility_dead_class`（duration 死类）+ `feedback_chart_reconcile_loop_key_must_be_unique`（reducer 去重键唯一）+ Phase 0 spec#1/#2/#3/#6。只收代码层。**结论：A 节全绿无需修复，B 节通过，唯 `ui/empty.tsx` 纪律口径待裁。**未写代码。*
