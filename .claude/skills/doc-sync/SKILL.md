---
name: doc-sync
description: >
  把最近一批 handoff + 新 plan/spec 里已发生的事实和决策"回流"到项目级文档
  (CLAUDE.md / docs/milestone/README.md + 各 milestone / docs/adr/)，同时检测
  并删除/收敛文档里已被推翻或过期的内容。第一性原理是漂移修正(drift
  correction)，不是 append——回流必须包含删除与收敛，CLAUDE.md 行数和 milestone
  活跃表行数是观测指标，只增不减要报警。借鉴 read-handoff 的"git 是真相"原则：
  handoff 里说"X 已合 dev"要先 git log / gh 核实再回流。三类改动分待遇：机械更新
  直接写、语义收敛写但在报告高亮、新建要给判断依据且受配额限制。Use when user
  says "同步文档""文档漂移了""回流 handoff""更新 CLAUDE.md/milestone""项目文档过期
  了"，或批量会话后想把散落的 handoff/plan 沉淀回项目真相。Invoke as
  /doc-sync [N] (N=扫描最近多少天的 handoff，默认 7)。
argument-hint: "[N]  # 扫描最近 N 天的 handoff + 自上次回流起新增的 plan/spec，默认 7"
allowed-tools: [Read, Bash, Write, Edit, Glob, Grep]
version: 0.1.0
---

# doc-sync

把"已发生但未回流"的事实/决策同步进项目级文档，**同时删除/收敛过期内容**。
第一性原理 = **drift correction，不是 append**。项目文档是"当前真相的精炼快照"，
回流 = 把文档拉齐到 handoff 暴露的最新真相，包括删掉过期的。

`<skilldir>` = 本 SKILL.md 所在目录
(`/home/wangqiuyang/noldus-insight/.claude/skills/doc-sync`)。

**目标文档**(只回流到这 4 类，不碰 handoff/plan/spec 源文件)：

| 文档 | 角色 | 基线(首次运行时记录) |
|------|------|---------------------|
| `CLAUDE.md`(仓库根) | 项目级不变事实/约束/当前状态 | 行数观测指标 |
| `docs/milestone/README.md` | roadmap 索引：活跃表 + 已完成表 | 两表行数观测指标 |
| `docs/milestone/<track>.md` | 各 feature track 的 checkpoint | 按需更新 |
| `docs/adr/NNNN-*.md` | 已锁定的架构决策(含反直觉/血泪教训) | 新建受配额限制 |

详细规则按需读 `<skilldir>/references/`：事实提取信号清单→`extraction-checklist.md`，
漂移检测已知模式(预填真实漂移种子)→`drift-signatures.md`，三类改动判定细则+新建判定门→`classification-rules.md`。

## Hard rules（先读，违反即停）

1. **git / gh 是真相，handoff 文字其次。** handoff 里"X 已合 dev / PR #N merged / issue CLOSED"必须先 `git log dev` 或 `gh issue view` 核实再回流。未通过核实的事实**不得**写进 CLAUDE.md / milestone 当既成事实——只能写"handoff 称已合，待核实(per handoff YYYY-MM-DD)"或干脆不回流该条。详见 §git 校准。
2. **回流必须含删除/收敛，禁止只增不改。** 每次回流报告必须列"删了什么 / 收敛了什么"。若某次回流零删除零收敛，必须在报告里显式写原因(如"本期全是新 track 立项，无过期内容可删")。**只找"能加什么"而不找"能删什么"是本 skill 的失败模式。**
3. **观测指标红线。** 写完后报 `wc -l CLAUDE.md` + 活跃表行数 + 已完成表行数，对比上次回流基线。若 CLAUDE.md 行数和活跃表行数**都比上次只增不减**，→ 报警暂停，让用户确认是否真需要增长。自检："连续 5 次运行本 skill，CLAUDE.md 会稳定/变短还是稳定变长？"后者触发报警。
4. **三类改动分待遇。** 机械(mechanical)直接写、语义(semantic)写但报告高亮+给理由、新建(new)写但必须过"新建判定门"且受配额(单次 ADR≤1、milestone≤2)。详见 §三类改动。
5. **新建是例外不是默认。** 默认动作是塞进既有文档(semantic)。起新 milestone / ADR 必须过判定门 + 给"为什么不是旧 track 延续"的判断依据。
6. **改动全留 git，commit 前问用户。** 写完一组改动**先输出完整回流报告**，问用户"是否 commit"，确认后才 commit。commit message 列回流清单。用户可 `git revert` / `git checkout -- .` 干净回退。

## 何时触发 / 输入是什么

**触发**：用户敲 `/doc-sync [N]` 或说"同步文档""文档漂移了""回流 handoff"。

**输入窗口**：
- **handoff 窗口 = 最近 N 天**(默认 7)。按文件名日期前缀 `docs/handoffs/<YYYY-MM>/YYYY-MM-DD-*.md` 筛，**不是 mtime**(handoff 可能回填，文件名日期比 mtime 可靠)。`/doc-sync 14` 扩窗。
- **"新 plan/spec" = 自上次回流起 mtime 变化的** plan/spec/design。扫描目录：`docs/plans/`、`docs/specs/`、`docs/superpowers/plans/`、`docs/superpowers/specs/`、`docs/design/`。基线时间戳读 `<skilldir>/state/last-sync.json`。
- **首次运行**(无 last-sync.json)：默认扫最近 14 天 handoff + 最近 7 天 plan/spec。

## 执行流程（11 步，每步带可检验完成标志）

### Step 1 — collect（收集）
读 `<skilldir>/state/last-sync.json`(没有则用默认窗口)。Glob 出窗口内 handoff + 新 plan/spec。
**完成标志**：输出两个清单(handoff list / plan-spec list)，每条带绝对路径 + 日期。

### Step 2 — extract（提取事实）
逐个读 Step 1 清单，提取**已发生的事实和决策**，丢弃待办/计划中(未实施)/推测/问题描述。聚焦信号：PR 合入、issue OPEN↔CLOSED、track 阶段切换、阻塞解除/新阻塞、方案/架构决策锁定、skill 新增/重命名/启用态变化。判定细则→`references/extraction-checklist.md`。
**完成标志**：输出"事实清单"，每条标来源 handoff 路径 + 日期。格式：
```
[FACT] Phase 0 前端 dogfood bug 修复批 #218-#227 全合 dev
       src: docs/handoffs/2026-06/2026-06-26-frontend-dogfood-bugs-...md (2026-06-26)
```

### Step 3 — snapshot（读当前文档快照）
**读全文**(不是 grep)：`CLAUDE.md`、`docs/milestone/README.md`、`docs/milestone/*.md`(所有)、`docs/adr/*.md`(所有)。记录基线：`wc -l CLAUDE.md docs/milestone/README.md`，活跃表/已完成表行数(`grep -c '^|'` 对应段)。
**完成标志**：基线数字进入报告；所有目标文档已在 context。

### Step 4 — calibrate（git 校准）
对 Step 2 每条"已合/X merged/X CLOSED"类事实用 git/gh 核实，标三态。详见 §git 校准。
**完成标志**：事实清单每条标 `git-verified ✅` / `handoff-only ⚠️` / `unverified ❌`。

### Step 5 — detect drift（检测漂移）
拿 Step 2+4 事实清单对照 Step 3 文档全文，找两类：**过期**(文档写了 X 但 X 已被推翻/完成/改名/废弃)、**缺失**(事实清单有、文档没落地的重大事实/决策)、**矛盾**(文档内自相冲突，如"不手工枚举"却枚举了)。已知模式→`references/drift-signatures.md`。
**完成标志**：输出"漂移清单"，每条 = {文档路径, 现有文字(行号/摘录), 事实, 漂移类型}。

### Step 6 — classify（分类三类改动）
对每条改动标 `mechanical` / `semantic` / `new`。判定细则→`references/classification-rules.md`。
**完成标志**：所有改动已分类，待遇明确。

### Step 7 — apply（执行写入）
按待遇执行。**写入顺序**：CLAUDE.md → milestone/README(含挪表) → 各 milestone → ADR。写完跑 markdown sanity(表格列数对齐、链接路径 `Glob` 验存在)。
**完成标志**：每个被改文件列在报告里，带"改了什么"一行摘要。

### Step 8 — observe（观测指标与报警）
重测 `wc -l CLAUDE.md`、活跃表行数、已完成表行数，对比基线。规则见 §防堆叠硬规则。**报警则暂停后续 commit**。
**完成标志**：指标对比表进入报告；报警(若有)已明确写出"为何只增"。

### Step 9 — report（产出回流报告）
输出 6 节到对话(不写文件)：① 扫描窗口 ② 基线 vs 现状指标表 ③ 三类回流清单(语义/新建高亮) ④ **删除/收敛汇总**(空也写原因) ⑤ git 校准结果(三态分布 + 被挡下没回流的) ⑥ 报警(若有)。
**完成标志**：6 节齐全。

### Step 10 — commit（可选，问用户）
**问用户**"回流报告如上，是否 commit？"，确认后才 commit。message 模板：
```
docs: 回流 <日期区间> handoff/plan 到项目文档

- 机械更新: <N> 处
- 语义收敛: <N> 处
- 新建: <N> 处 (milestone/ADR)
- 删除/收敛: <列表>
- CLAUDE.md <baseline>→<now> 行, 活跃表 <b>→<n> 行

Co-Authored-By: Claude <noreply@anthropic.com>
```
**完成标志**：commit hash 进报告；`git status` clean(或只剩用户其他改动)。

### Step 11 — checkpoint（更新状态）
写 `<skilldir>/state/last-sync.json`：`{last_sync_utc, handoffs_scanned[], claude_md_lines, active_table_rows, completed_table_rows, drift_fixed[]}`。该文件 gitignore(机器本地)，不进 commit。
**完成标志**：该文件更新，下次运行能据此算"新 plan/spec"窗口。

## 三类改动：判定与待遇

| 类别 | 判定 | 待遇 |
|------|------|------|
| **mechanical** 机械 | 仅指针/状态/哈希/emoji 变化，零语义判断 | 直接写，报告列出，不高亮 |
| **semantic** 语义 | 过期段落改写、track 完成挪表、矛盾择新舍旧 | 写，报告高亮+给理由 |
| **new** 新建 | 起新 milestone / 起一个 ADR | 写，但必须过"新建判定门"+受配额 |

- **mechanical 示例**：milestone README"最新 handoff"列换链接；issue OPEN↔CLOSED(git/gh 坐实后)；milestone 的 dev HEAD commit hash 更新；状态 emoji ✅/📋/🔴 切换。
- **semantic 示例**：CLAUDE.md"微调 Qwen3-8B"段落若已被坐实改 30B，改写并在报告说明；track done → 活跃表删行 + 已完成表加行 + 填日期；CLAUDE.md"不手工枚举 skill"却枚举了 → 删枚举、指向 SSOT(`extensions_config.json`)。
- **new 示例**：新 feature track(如"文件路径承重墙")到达 checkpoint → 起新 milestone 文件；架构决策锁定(如"桌面化选 Electron 不选 Tauri")且有反直觉/血泪教训 → 起新 ADR。

判定边界 case 与新建判定门细则→`references/classification-rules.md`。

## 防堆叠硬规则（本 skill 存在的理由，违反即失败）

- **行数观测**：每次回流前后测 CLAUDE.md 行数 + 活跃表行数。CLAUDE.md 只增不减 + 本期无删除/收敛 → 报警暂停。活跃表连续两次只增不减 → 报警(说明 track 只立项不收口)。
- **完成即归档(挪表)**：track 在 handoff/git 里坐实"done" → 必须从活跃表删行 + 已完成表加行 + 填完成日期。**不允许 done 了还挂活跃表**——这是最常见的堆叠源。
- **新建配额**：单次回流新建 ADR ≤ 1、milestone ≤ 2。超额必须分次回流，且每次超额在报告写"为何本期要新建多个"。

## git 校准：先信 git HEAD，后信 handoff 文字

handoff 是"会话末尾的快照"，它说的"已合 dev"可能 PR 还没真 merge，它说的"CLOSED"可能只是 handoff 当时的状态。**回流到 CLAUDE.md/milestone 的事实必须是 git/gh 坐实的。**

**必校准项**：
- "已合 dev / merged into dev" → `git log dev --oneline --since=<handoff日期-1天> | grep -i <关键词>` 或验 commit hash 在 dev
- "PR #N merged" → `git log --all --oneline | grep "#N"` 或看 merge commit
- "commit XXX 在 dev" → `git branch dev --contains <commit>`(exit 0 = 在)
- issue 状态 → **优先 `gh issue view <N> --json state`**(gh v2.45.0 可用)；无证据时保留并注记，不伪造 CLOSED

**校准三态**：
- `git-verified ✅` → 可作为事实回流
- `handoff-only ⚠️` → 回流但加"(per handoff YYYY-MM-DD)"注记
- `unverified ❌` → **不回流**，进报告"待核实"段

**反例(禁止)**：handoff 说"Phase 0 全合 dev"就直接写进 CLAUDE.md；handoff 说"#98 CLOSED"就改 CLAUDE.md issue 状态而无 gh 证据。

## 完成标志（全部满足才算回流完成）

1. ✅ 扫描窗口已列出(handoff list + plan-spec list)
2. ✅ 事实清单已 git/gh 校准(每条标三态之一)
3. ✅ 漂移清单已产出(过期/缺失/矛盾，带文档路径+行号/摘录)
4. ✅ 三类改动已分类且待遇正确(机械直写、语义高亮、新建过判定门)
5. ✅ 所有目标文档已写入(或显式说明本期无需改动)
6. ✅ 删除/收敛汇总已产出(即使为空也写"本期零删除，原因…")
7. ✅ 指标对比表已产出(CLAUDE.md 行数、活跃表/已完成表行数，baseline→now)
8. ✅ 无未解报警(或报警已显式说明原因)
9. ✅ 回流报告已输出给用户(6 节齐全)
10. ✅ last-sync.json 已更新(或用户选择不 commit 时跳过 checkpoint)
