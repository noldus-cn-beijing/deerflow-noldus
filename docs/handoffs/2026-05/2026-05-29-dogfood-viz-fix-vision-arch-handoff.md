# 2026-05-29 Dogfood FST + Viz修复 + 编排哲学 + 愿景架构 — Handoff

**会话主体**：dev `git pull` 卡住排查 → FST dogfood 端到端跑通(E2E_FULL_ASKVIZ) → viz guardrail 竞态 bug 确诊修复(PR #67) → 编排哲学七八轮 grill → 全能行为学 agent 愿景深挖 → 两模式架构假说 → 文档落档 → 脏状态清理
**触发分支**：dev
**dev HEAD**：`ec263b20`（本地 = origin/dev，已同步 push）
**本会话 commit 链**：
- `49ba5983` fix(guardrails): 消除 ask-gate 同批并行竞态误拦(viz dogfood bug)
- `bc42c259` Merge PR #67 (viz race fix)
- `c9bc41f6` Merge PR #66 (Sprint 5.5 handoff 内容非空校验)
- `ec263b20` docs: 沉淀编排哲学收口 + 愿景两模式架构假说

**测试基线**：全量 **3729 passed / 83 skipped / 0 failed**（比上次 3727 多 2 个新回归测试）

---

## 1. 本会话核心产出

### 1.1 FST Dogfood 跑通
- **数据**：大鼠强迫游泳 FST，PorsoltCylinder-NoZones，Drug(5mg/L) vs Saline 对照，每组 n=1
- **流程**：E2E_FULL_ASKVIZ 全程走通（范式识别→指标计算→质量门→解读→图→报告）
- **线程 ID**：`87edb29b-02f2-4970-b5e2-c927f9d6a263`
- **3 个发现**（见 §1.2，§1.3，§1.4）

### 1.2 发现 #1：viz guardrail 竞态 bug（已修 PR #67）

**根因（已用 training-data JSONL 实证，非推断）**：
lead 在**同一个 AIMessage** 里并行发出 `set_viz_choice(choice='yes')` + `task(chart-maker)`。
`IntentPostStepAskGateProvider` 评估 `task()` 时，`set_viz_choice` 的落盘尚未发生
→ `read_context` 读到 gate3 未完成 → 假阳性拦截（`viz_choice_not_acknowledged`）→ retry 才过。

**修复**（治本，`fix/guardrails/intent_post_step_ask_gate_provider.py`）：
- `path_registry.py` 新增 `ASK_GATE_SETTER_TOOL = {"viz": "set_viz_choice"}`
- `IntentPostStepAskGateProvider` 新增 `_current_batch_tool_names()` 读最后一条 AIMessage 的 tool_calls
- deny 前：若同批已含对应 gate-setter → 视为 in-flight 确认，不误拦
- 不过度放行：同批无 setter 时仍 deny（真跳步必须拦）
- 回归：`test_allows_when_set_viz_choice_in_same_batch` + `test_still_blocks_when_no_set_viz_choice_in_batch`

**扩展性**：`report`/`four_choice` 暂无 setter 工具，`ASK_GATE_SETTER_TOOL` 待其工具落地后按格式加。

### 1.3 发现 #2：Sprint 3 参数审计未覆盖 FST mobility threshold（未修，待立案）
- dogfood 中 `parameter_audit_findings_count: 0`，但 data-analyst 正确识别 FST 不动时间 0.56s 异常低（疑 EV XT Mobility detection 未校准）
- **Sprint 3 的参数审计没有触发 FST 的 mobility threshold**，这是真实覆盖缺口
- 关联同事 issue #63（velocity 物种判据），待同事休假回来后按 #63 答复补判据

### 1.4 发现 #3：n=1 处理正确（无需动）
- Gate 2 质量门拦下 → 反问用户 → 记录确认 → 降级描述性比较，全程干净

### 1.5 编排哲学收口（已落档 roadmap v2 + 诊断设计 doc §7）
七八轮 grill 的最终结论，已写入：
- `docs/plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md`（新增「2026-05-29（下午）愿景架构对齐」段）
- `docs/superpowers/specs/2026-05-29-orchestration-path-ssot-diagnosis-design.md`（新增 §7 编排哲学）

**核心论断（防下一 agent 重走认知弯路）**：
1. **声明式编排不等于"没有动态"**：lead 现在就在动态编排（自由判断意图/调哪个 subagent/组织反问），SSOT 是给它装护栏（节点间约束），不是关掉它。**智能在节点内，约束在节点间**。
2. **动态编排（Claude Code 式即兴 JS）不适合我们**：我们的任务空间可枚举（Noldus 自己定义边界），时间敏感不容试错，可复现是科学仪器命脉。这三把判据将来比现在更强，不是更弱。
3. **SSOT 不能写成 JS**：命令式只能执行，不能多视角读（渲染/校验/CI 哨兵）。写成 JS 会把三层真相重新劈裂，复活这一个月所有 bug 的统一根因。
4. **意图路由 ≠ 动态编排**：8→800 个场景只是 SSOT 规模变大，越大越需要声明式（可枚举/可检查/可复现），越不能即兴。

### 1.6 愿景两模式架构假说（⚠️ 待确认，不能据此排 sprint）
EV19 有两个模式（用户在本会话透露）：
- **模板模式**：选已有模板（EPM/OFT…）直接跑
- **自由模式**："无限接近图灵完备"，用户自定义实验/仪器（**但跑不出行为学范畴**）

**假说**：不是"两套编排引擎"，而是「**一套声明式引擎 + 两种粒度构件**」：
- 模板模式 = 粗粒度预组合（如现在的 `PATHS[EPM]`）
- 自由模式 = 行为学**有限基元**（zone 几何/事件谓词/计时器/阈值）的自由组合 + 合法性规则
- 关键：**自由 ≠ 动态**；"接近图灵完备" ≠ 真图灵完备；"跑不出行为学范畴"保证基元有限

**🔑 决定性开关（下一 agent 第一件事）**：
> 向行为学同事确认：**自由模式里用户能定义的"基元"是否真有限？**
> - 是**预定义积木自由组合** → 假说成立，全链路声明式
> - 能**写脚本/公式定义全新检测逻辑** → 真图灵完备，自由层才真需动态，假说推翻

---

## 2. 愿景三深水区（已落档，未展开，待下次聊）

1. **设计层决策智能 ≠ 分析层智能**：设计是前瞻性决策（不可逆），需要 power analysis、权衡协商、**拒绝的勇气**（"用 3 只动物得显著结论"要能说不）。从"听话工具"到"会说不的科研合作者"。
2. **贯穿三层的实验本体 SSOT**：设计→采集→分析的契约链；zone/参数定义必须无损可追溯地流到分析，否则结论全错且数字看起来正常没人发现。
3. **科学仪器级可复现**：整条设计决策链可封存/可审计/三年后可重现（不只是分析层）。

---

## 3. 脏状态清理结果

- ✅ `backend/packages/ethoinsight` symlink 恢复：stray 实体目录（源码与根包字节级一致）已删，`git checkout` 恢复 symlink `-> ../../../ethoinsight`
- ✅ `sprint-5.5-handoff-content-validation` worktree 已删（PR #66 已合，孤儿分支已删）
- ⚠️ `.env.wecom` 留着（永不 commit，不是脏状态）

---

## 4. 当前 worktree 状态

| worktree | 分支 | 状态 |
|---|---|---|
| 主仓 `/home/wangqiuyang/noldus-insight` | `dev` (`ec263b20`) | ✅ 干净 |
| `/home/wangqiuyang/.claude/worktrees/sprint-0-handoff-schema` | `sprint-0-handoff-schema` | ⚠️ **未合并，保留不动** |

3 条 stash（不是本会话的，不要动）：
- `stash@{0}`: WIP on dev: 7046d5df (PR#29 时代)
- `stash@{1}`: WIP on worktree-subagent-role-split-impl
- `stash@{2}`: WIP on worktree-feat+g4-frontend-stage-broadcast

---

## 5. 下一会话执行清单

### A. 第一件事（必须先做）
向行为学同事确认**自由模式基元是否有限**（§1.6 决定性开关）。答案决定愿景架构能否落地。

### B. 同事休假回来后
- issue #63 答复后：Sprint 3 补 FST mobility threshold 判据（换数字不重做结构）
- Sprint 4 调参指南：同事写内容，工程只做通路

### C. 可立即开始的 sprint（不卡同事/不卡愿景开关）
- **Sprint 5 数据质量门**：前置 SSOT-A 已合，可启动（建议等愿景开关确认后）
- **Sprint 4 工程通路**：不等同事，先把 grep 逻辑做好，#63 回来只填内容

### D. 待展开的愿景深挖（下次聊）
- 三深水区选一个深入（建议从**设计层决策智能**或**实验本体 SSOT** 入手）
- 愿景阶段拆分：v0.1（分析）→ v0.2（设计层初步）→ v0.x（全生命周期）的边界在哪

---

## 6. 给下一 agent 的友情提示

1. **dev HEAD = `ec263b20`**，本地=origin/dev，全部 push GitHub
2. **worktree 陷阱**：`EnterWorktree` 默认从 `main` 建（落后 dev N 个 commit）。正解：`git worktree add -b X path origin/dev` 手动建，再 `EnterWorktree path`。
3. **worktree 测试陷阱**：deerflow 是 editable-install 指向主仓。worktree 测试需 `PYTHONPATH=<wt>/packages/harness`；`config.yaml` gitignore 不在 worktree，构造 lead-agent 的测试会 `FileNotFoundError`，需 `cp` 主仓 config.yaml 进 worktree。
4. **编排哲学不要重开**：`docs/superpowers/specs/2026-05-29-orchestration-path-ssot-diagnosis-design.md §7` 已封存完整论证。看到有人提"要不要动态编排/JS 写 SSOT"直接指向那节。
5. **愿景假说待确认**：两模式架构是假说，不是结论。**不要把它当结论排 sprint**，先向同事核实 §1.6 的决定性开关。
6. **Sprint 5.5 已实施**：PR #66 合入 dev（`c9bc41f6`），handoff 旧文档若标"未实施"是 stale memory。
7. **3 条 stash 别动**（不是本会话的）
8. `.env.wecom` 永不 commit
