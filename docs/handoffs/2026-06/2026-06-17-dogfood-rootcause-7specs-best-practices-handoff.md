# Handoff: 2026-06-17 EPM dogfood 根因综合 → 7 问题 spec 化 + 工程最佳实践 + 两修复 review

> 交接对象：下一个接手的 AI Agent。
> 会话主线：用户连续多轮 EPM dogfood，反复遇到同类问题 → 追问"为什么总是重复" → 综合本地 log 判根因 → 提炼 langgraph/deerflow infra 工程最佳实践 + 把 7 个问题各写一份 spec → review 并合入两个修复分支。
> 当前位置：主仓 `/home/wangqiuyang/noldus-insight`（分支 dev，HEAD `a381cb22`）。

---

## 1. 当前状态速览

### 已合 dev 的本会话成果（HEAD a381cb22）
| PR | 内容 | 状态 |
|---|---|---|
| #143 | column_semantics 独立写入通道 + guardrail 模板锁泛化 + Gate1 不 clobber | ✅ 已合 |
| #144 | data-analyst LOO 反事实下沉统计层（compute_outlier_diagnostics） | ✅ 已合（**但引入了 ZeroDivision 回归，见下**） |

### 待用户建 PR 的分支（已 push，review 通过）
| 分支 | 解决 | 基线 | review |
|---|---|---|---|
| `fix/outlier-diagnostics-zerodivision`（`ab3234ef`，P1） | ZeroDivision（当前 dogfood 硬阻塞） | dev a381cb22 ✅ | ✅ 红→绿坐实，真实数据 |
| `fix/loop-detection-tool-semantics-rebased`（`6fba9c6f`，P6） | 报告卡死（write_todos 误伤） | dev a381cb22 ✅ | ✅ 红→绿坐实，无导入环 |
| `fix/standalone-channel-first-call-with-paradigm`（`8c3dfeb9`，PR#145） | 首次 paradigm+列语义同传被 standalone 通道误截 | dev 0b2be473 | ✅ 已开 PR#145，**未合** |

### ⚠️ 必须废弃的错误分支
- `worktree-loop-detection-tool-semantics`（基线 `0696ef79`=PR#140 旧 dev，落后 4 个 PR）。它的 diff 含"回退 seal-gate/列对齐/column-semantics/LOO"灾难性噪音。**建 PR 千万别用它**，已用 `-rebased` 分支替代。建议直接删：`git push origin --delete worktree-loop-detection-tool-semantics`。

---

## 2. 未完成事项（按优先级）

### 🔴 高优先级
1. **建 PR 合 P1 + P6**（解当前 dogfood 两个硬阻塞）：
   - `fix/outlier-diagnostics-zerodivision` → 修 ZeroDivision，dogfood 的 statistics 立刻有结果、data-analyst 不再手算螺旋。
   - `fix/loop-detection-tool-semantics-rebased` → 修报告卡死。
   - **P6 务必用 `-rebased` 分支，不是 `worktree-loop-detection-tool-semantics`**。
2. **10 个 2026-06-17 文档入库**（全部 untracked，**必须 commit 进 dev**）：9 个 spec + 1 个最佳实践。命令见 §6。这是下一个 agent 实施其余 spec 的依据，丢了就断链。
3. **PR#145 合并**（首次同传修复，未合）。

### 🟡 中优先级（5 个待实施 spec，用户已定"交给别的 agent 实施代码"）
按实施顺序（写在最佳实践文档表格末尾）：
- **P3** `charts-column-alignment-self-read-spec` → charts 路径用 runtime 自读 column_aliases（不靠 LLM 拼 CLI）。
- **P5** `chart-budget-by-type-spec` → 聚合图优先（依赖 P3）。
- **P2 + P7**（一起）`statistics-loud-failure-spec` + `data-degradation-circuit-breaker-spec` → 降级信号源 + 降级熔断器（middleware 拦降级→通知→有限自救→HITL 确认→重试）。**P7 是用户产品立场的核心载体**。
- **P4** `shared-state-sourcing-spec` → 元收口：所有路径用 runtime.state 拿 session 态 + parity 回归网（最后织网）。

### 🟢 低优先级
4. **memory 沉淀**（用户问过几次，一直没写）。本会话值得沉淀：① 产品立场=数据分析 agent 可复现性底线 + 降级熔断器机制 ② 4 条工程红线 ③ "statistics={} 是症状非 bug，多成因" ④ worktree 基线漂移会回退已合 PR（review 必查 merge-base）。
5. **清理临时 worktree**：`loo-counterfactual-pushdown`、`loop-detect-rebased`、`standalone-first-call-fix`、`fix-outlier-zerodivision`（用户问过是否清理，未答）。

---

## 3. 关键发现（本会话核心，最重要）

### "为什么总是重复" = 4 条结构性病根（全部写进最佳实践文档）
本轮 dogfood 6+ 个表面不同的故障收敛到 4 条病根：
1. **静默降级**：失败被吞成空结果伪装成功。gateway.log 实证：`statistics 失败 (rc=1) ZeroDivisionError` 下一行 `status=completed n_failed=0`。下游无法区分"崩了"vs"本就没有"。
2. **横切状态靠传参而非共享源**：column_aliases/路径 session 级常量，每条消费路径各自接线，每新增一条多一个"忘了接"的点。metrics 路径对（工具自读 `runtime.state["thread_data"]`），charts 路径错（LLM 拼 `--column-aliases-file`）。
3. **理想化合成数据测试**：真实数据的 0/空串/非标列名是常态，合成 fixture 总是缺席。ZeroDivision 假绿正因 PR#144 用 `[1,...,100]` 而非真实含 0 数据。
4. **通用机制对正常长流程误伤**：loop-detection 把 write_todos（记账工具）当死循环，FORCED STOP 还连坐剥光 task(report-writer)。

### `statistics={}` 是症状不是 bug（用户反复遇到，我曾误说"修好了"）
至少 3 个独立成因汇到同一个静默出口：B（stats gate 误 skip，#6a 已修）、C（env 没包，#5 已修）、**A（ZeroDivision，PR#144 引入，未修=P1）**。每修一个成因不等于修好 statistics 空——**真正治本是 P2/P7：让降级响亮、可区分、走熔断**。

### 用户的产品级立场（必须遵守，写进最佳实践红线一）
- **数据分析 agent，将来是完整实验数据 agent。数据计算的可复现性和准确性是底线，不是"尽量"。** 判读的自然语言表述不可复现无所谓。
- **降级可以，静默不可以**：降级要 (a) 通知用户 (b) 让模型有限自救 (c) 自救有时间/轮次上限 (d) middleware 拦住 → ask_clarification 和用户确认 → 再重试。复刻 SealGateMiddleware（after_model+jump_to）+ ClarificationMiddleware（中断问用户）。这是 P7。
- **测试必须用真实数据**：`/home/wangqiuyang/DemoData`（EPM 在 `real_data/Raw data-EPM-Xuhui-28`，含 Trial 19=0.0 触发 ZeroDivision）。

---

## 4. 关键文件指针

- **工程最佳实践（宪法，给实施 agent 先读）**：[docs/refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md](docs/refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md)。4 红线 + 本仓库真实反/正模式 + CI 判据 + spec 映射表 + 实施顺序。
- **9 个 spec**：`docs/superpowers/specs/2026-06-17-*.md`（每个显式挂对应红线，修法照"正模式"，要求用 DemoData 真实数据测试）。
- **本地 dogfood log**：`packages/agent/scripts/logs/gateway.log`（11:37，含 ZeroDivision line 171）。subagent transcript 不在此（lead-only）。
- **ZeroDivision 崩点**：`packages/ethoinsight/ethoinsight/statistics.py:421-422`（`grp_median / value`，value==0 崩）。
- **statistics 静默吞错点**：`run_metric_plan_tool.py:256-257`（崩了只 warning + statistics_payload=None）。
- **charts 漏列对齐**：chart-maker SKILL.md:22 纯 bash 拼 resolve 没传 `--column-aliases-file`；cli.py:101/247 代码层早支持，只是没喂数据。
- **loop-detection 误伤**：`loop_detection_middleware.py`（per-tool freq 一刀切 + strip 全部）；`loop_detection_config.py`（tool_freq_overrides 机制）。
- **tool 拿 session 态正范例**：`prep_metric_plan_tool.py:124-130`（`runtime.state["thread_data"]["workspace_path"]` + fail-fast）。

---

## 5. 风险与注意事项

1. **worktree 基线漂移会回退已合 PR**（本会话踩到）：P6 原 worktree 基于旧 dev，diff 里混进"删 seal_gate/列对齐"的回退噪音。**review worktree 第一步必查 `git merge-base HEAD origin/dev` 是否=当前 dev HEAD**；不是就 cherry-pick 干净 commit 到当前 dev。
2. **worktree 假绿**（CLAUDE.md 铁律，本会话两次踩）：worktree 多无独立 venv，借主仓 venv 时 editable 指**主仓**。① ethoinsight：`PYTHONPATH` 前置无效（.pth 优先），用 importlib 显式加载 worktree 源验证。② backend deerflow：`sys.path.insert(0, worktree/packages/harness)` 有效（namespace package 无 .pth）。**跑测试前先 print 加载路径确认是 worktree**。
3. **改 harness 核心（agents/middlewares/、subagents/、tools/builtins/）必裸导入两入口**：`python -c "import app.gateway"` + `python -c "from deerflow.agents import make_lead_agent"`，exit 0。（worktree 跑会有 `config.yaml not found` 良性警告，被吞，不影响导入。）
4. **P2/P7 的 prompt 改动是承重消费，不是 prompt 补丁**：主防线在 harness 亮信号（statistics_status / degradation_signals）+ 熔断 middleware；prompt 只消费信号。用户明确"不要改 prompt 就以为解决了"。
5. **9 spec 实施时别打地鼠**：每个 spec 都要求"修结构覆盖全集 + parity/回归网"，不是只补一处实例。先读最佳实践宪法。

---

## 6. 下一位 Agent 的第一步建议

```bash
# 1. 读最佳实践宪法（理解 4 红线，是所有 spec 的纲）
cat docs/refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md

# 2. 把 10 个 2026-06-17 文档 commit 进 dev（高优先级，否则断链）
cd /home/wangqiuyang/noldus-insight
git add docs/superpowers/specs/2026-06-17-*.md docs/refs/2026-06-17-*.md
git commit -m "docs: 2026-06-17 EPM dogfood 根因综合 — 9 spec + agent 工程最佳实践宪法"

# 3. 提醒用户建 PR（P1/P6 解当前 dogfood 阻塞）
git log --oneline origin/dev | grep -iE "outlier|loop-detection"  # 确认是否已合
# P1: fix/outlier-diagnostics-zerodivision
# P6: fix/loop-detection-tool-semantics-rebased（不是 worktree-loop-detection-tool-semantics！）

# 4. 删错误基线分支
git push origin --delete worktree-loop-detection-tool-semantics
```

**之后**：等用户合 P1/P6 → dogfood 复跑确认 statistics 有结果 + 报告能出 → 按实施顺序（P3→P5→P2+P7→P4）逐个实施 spec（用户已定交给别的 agent，每个新 worktree 基于**当前 dev**）。

**若用户给新 dogfood 失败**：先按 §3 的 4 红线框架判断是哪条病根，别再写 prompt 补丁。statistics 空先查是不是成因 A/D（脚本崩 vs 设计内缺席）。

---

## milestone 建议
本会话产出了一份 agent 工程最佳实践宪法 + 9 个根治 spec，是 harness 鲁棒性 track 的重要 checkpoint。建议：
- 更新/创建 milestone「harness 鲁棒性 / dogfood 根因治理」，记录 4 红线 + 7 问题 spec 索引 + 产品立场（数据可复现性底线 + 降级熔断器）。
- 关键摘要：dogfood 反复故障收敛到 4 结构病根；P1/P6 已修待合，P2/P3/P4/P5/P7 待实施；最佳实践宪法是后续所有 harness 改动的依据。
