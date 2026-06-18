# Handoff: 2026-06-18 — 9-spec dogfood 根因治理批次「全部实施 + 合并 dev」完成

> 交接对象：下一个接手的 AI Agent。
> 会话主线：上一会话（2026-06-17）把 EPM dogfood 反复故障收敛成 4 红线宪法 + 9 个根治 spec。**本会话的工作是逐个 review 这些 spec 的代码实施并合入 dev**——用户让不同 agent 实施、本 agent 负责 review（发现问题直接在 worktree 改 + push）。
> 当前位置：主仓 `/home/wangqiuyang/noldus-insight`，分支 `dev`，origin/dev HEAD = `f5237903`。
> **核心结论：9 个 spec 现已 100% 实施完毕且全部合入 dev。该 track 闭环，无 open PR。**

---

## 1. 当前状态速览 — 这批活已经干完了

### 9 个 spec 全部已合 dev（origin/dev `f5237903`）

| Spec 文件（`docs/superpowers/specs/`） | PR | 本 agent 是否 review | 状态 |
|---|---|---|---|
| `2026-06-17-outlier-diagnostics-zerodivision-spec.md`（P1） | #148 | 否（上会话已修） | ✅ 合 |
| `2026-06-17-loop-detection-tool-semantics-spec.md`（P6） | #147 | 否（上会话已修） | ✅ 合 |
| `2026-06-17-column-semantics-standalone-channel-and-guardrail-lock-spec.md` | #143 + #145 | 否 | ✅ 合 |
| `2026-06-17-data-analyst-loo-counterfactual-pushdown-spec.md` | #144 | 否 | ✅ 合 |
| `2026-06-17-charts-column-alignment-self-read-spec.md`（P3） | #149 | ✅ 是 | ✅ 合 |
| `2026-06-17-statistics-loud-failure-spec.md`（P2）+ `2026-06-17-data-degradation-circuit-breaker-spec.md`（P7） | #150 | ✅ 是 | ✅ 合 |
| `2026-06-17-chart-budget-by-type-spec.md`（P5） | #151 | ✅ 是 | ✅ 合 |
| `2026-06-17-shared-state-sourcing-spec.md`（P4） | #153 | ✅ 是 | ✅ 合（最后合入，`f5237903`） |

**open PR：无。** 这批 dogfood 根因治理 track 已彻底闭环。

### 本会话 review 了 4 个 PR（P3 / P2+P7 / P5 / P4），每个都发现并修了真问题

逐个记录（含我修的 bug，便于后续若回归可溯源）：

1. **P3 #149（charts 列对齐自读）**：实现质量高。我发现并修一处 groups 静默降级——`prep_chart_plan` 自读 groups.json 且报 `groups_applied=True`，但 `resolve_charts` 内部 `_build_groups_payload` 用 `arena_key in filename` 子串启发式匹配不了 SSOT `{完整路径:组名}` 形态 → `box_open_arm` 静默丢 `--groups`。修法：完整路径精确匹配优先、子串降级为 fallback（与 `scripts/_cli.py:read_groups_json` 同语义）。红→绿坐实。
2. **P2+P7 #150（降级信号 + 熔断器）**：我修一处红线一静默口——`run_metric_plan_tool.py` Step7 当 statistics 段 `skip_reason=None` 但缺 script/output（残缺段）时，原代码留 `statistics_status` 默认 `absent_by_design`（注释却声称会判 crashed，与代码不符）→ 残缺段伪装成"设计内缺席"绕过熔断。补 `else` 分支判 crashed + 测试。另校正了 handoff_schemas 过度声称"SSOT 唯一定义"的注释。
3. **P5 #151（图预算按类型）**：我修一处 SKILL↔工具契约断裂——chart-maker SKILL 指示给 `seal_chart_maker_handoff` 传 `remaining_charts`、`ChartMakerHandoff` schema 也有该字段，但 **seal 工具签名根本不接受该参数**（`@tool(parse_docstring=True)` 下 docstring 提了签名没有的参数会直接 ValueError；即便不提也静默丢弃降级指纹）。补 seal 工具 `remaining_charts` 形参+payload+docstring + 测试。另强化了一个假信心测试（`test_aggregate_charts_prioritized` 原把 aggregate 放输入数组前面，naive first-N 也能蒙混过 → 改成 per_subject 在前）。
4. **P4 #153（shared-state 元收口）**：最干净的一个，无实质 bug。只修了 1 行 lint（parity-net 测试文件 I001 import 排序）。实现含 4 交付物：B 类 `column_aliases` parity 回归网（`test_column_aliases_parity_net.py`）+ AST 守护测试（`test_no_cli_passes_session_state.py`）+ CLI `--context-file` 兜底（`cli.py`）+ A 类诊断清单（`docs/problems/2026-06-17-a-class-path-resolution-diagnostic.md`，只产出不实施）。

---

## 2. 未完成事项（按优先级）— 这批 track 已无代码待办

### 🟡 中优先级：milestone / 文档沉淀（用户上个 handoff 就提过，仍未做）
1. **为这批 9-spec dogfood 根因治理写/更新 milestone**。建议 track 名「harness 鲁棒性 / dogfood 根因治理」。关键摘要：
   - 4 红线宪法（`docs/refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md`）。
   - 9 spec 全实施全合（见 §1 表）。
   - 产品立场载体：P7 降级熔断器（数据可复现性底线 + 降级→通知→有限自救→HITL）。
   - milestone 索引在 `docs/milestone/README.md`（= 项目 roadmap）。
2. **untracked 文档入库**（`git status` 显示这些 6-16/6-17 文档仍 untracked，需 commit 进 dev，否则断链）：
   - `docs/handoffs/2026-06/2026-06-17-three-specs-seal-gate-statistics-review-handoff.md`
   - `docs/problems/2026-06-17-remote-issues-root-cause-investigation.md`
   - `docs/superpowers/specs/2026-06-16-io-boundary-symmetry-and-aggregator-spec.md`
   - `docs/superpowers/specs/2026-06-16-seal-gate-middleware-engineering-spec.md`
   - `docs/superpowers/specs/2026-06-16-seal-produce-deliver-merge-spec.md`
   - `docs/superpowers/specs/2026-06-16-statistics-path-column-alignment-spec.md`
   - **本 handoff 自身**（`2026-06-18-9spec-dogfood-rootcause-batch-complete-handoff.md`）。
   - ⚠️ 注意：这些是 **docs only**，但 commit 前先 `git status` 看清楚别误带 `packages/agent/backend/uv.lock` 的 ` M`（那个 ` M uv.lock` 从会话开始就在，来历不明，别顺手提交——单独问用户）。

### 🟢 低优先级
3. **A 类物理路径收口分批实施**（P4 只做了 B 类 + 诊断；A 类是 `docs/problems/2026-06-17-a-class-path-resolution-diagnostic.md` 列的 5 批，每批独立 spec + 回归）。**这是新 track，不属于本批 9-spec**——按诊断文档"批 1（run_metric_plan + validate_catalog 的 `_scoped_path_env`，已出过事）"优先，但不急。
4. **memory 沉淀**（用户多次问过，一直没写）。本会话值得沉淀的可复用教训见 §3。
5. **dogfood 复跑验证**：9 spec 全合后，建议用真实数据 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`（含 Trial 19=0.0）端到端复跑 EPM，确认 statistics 有结果 / 图含组间对比 / 报告能出 / 降级走熔断而非静默。

---

## 3. 关键发现（本会话可复用的教训，建议沉淀 memory）

1. **「SKILL 说传 X 但 seal/工具不收 X」是反复复发的契约断裂**（P5 这次又踩，memory 已有 `feedback_code_executor_skill_writefile_contradicts_seal_tool`）。review 任何"SKILL 指示给工具传新字段"的改动，**必看工具签名是否真接受**——尤其 `@tool(parse_docstring=True)` 下 docstring 与签名不一致会直接 ValueError 加载失败。
2. **worktree review 四件套铁律**（本会话每个 PR 都走，建议固化）：① `git merge-base HEAD origin/dev` 查基线漂移，漂了先 rebase；② 跑测试用 worktree 源（`PYTHONPATH=packages/harness` 强制 harness，`PYTHONPATH=...worktree/packages/ethoinsight` 强制 ethoinsight——venv editable 指主仓会假绿/假红）；③ 改 `agents/`/`tools/builtins/`/`subagents/` 后裸导入两入口（`import app.gateway` + `from deerflow.agents import make_lead_agent`）；④ 红→绿坐实（neuter 修复逻辑确认测试真咬）。
3. **「backend 测试在 worktree 里失败」先排除两个环境假象**，再怀疑代码：① worktree **没有 `config.yaml`**（不入 git）→ 任何调 `build_middlewares()`/`get_app_config()` 的测试报 `FileNotFoundError`，用 `DEER_FLOW_CONFIG_PATH=主仓/packages/agent/config.yaml` 解决；② venv ethoinsight editable 指主仓 → 缺 worktree 的 ethoinsight 改动 → 用 `PYTHONPATH` 强制 worktree 源。本会话两者都踩过、都查证为假象（非回归）。
4. **已知 dev 全量测试污染**（与本批无关，别归因自己改坏）：`test_data_analyst_step28_contract`（在干净 dev 上就红）、`test_*_async_delegates_to_sync` 系列（顺序污染，单跑绿）、`test_chart_maker_config_basic_fields`（`cfg.model == "inherit"` 断言与 dev 现状 `deepseek-v4-pro-summary` 不符，pre-existing）。
5. **磁盘曾满**（`/` 100%，非本会话造成）：harness 把命令输出写 `/tmp`（在 `/` 上），满了任何 Bash 命令拿不到输出。诊断时把输出重定向到 `/home` 绕开。用户已清到 71%。若再满：`/tmp/claude-1000/pytest-of-*` + `node-compile-cache` 是可清缓存，但根分区大头需用户定夺，别盲删。

---

## 4. 关键文件指针

- **4 红线宪法（所有 harness 改动的纲）**：`docs/refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md`
- **9 spec**：`docs/superpowers/specs/2026-06-17-*.md`（9 个，全已实施）
- **A 类后续 track 诊断**：`docs/problems/2026-06-17-a-class-path-resolution-diagnostic.md`（P4 产出，5 批收口路线）
- **上一会话 handoff（本会话的起点）**：`docs/handoffs/2026-06/2026-06-17-dogfood-rootcause-7specs-best-practices-handoff.md`
- **本会话各 review 改动的关键文件**：
  - `packages/ethoinsight/ethoinsight/catalog/resolve.py`（P3 `_build_groups_payload` 精确路径匹配；P5 `select_charts_by_priority`）
  - `packages/agent/backend/packages/harness/deerflow/tools/builtins/run_metric_plan_tool.py`（P2 `statistics_status` 三态 + 残缺段 crashed）
  - `packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py`（P5 `seal_chart_maker_handoff` 加 `remaining_charts`）
  - `packages/agent/backend/packages/harness/deerflow/agents/middlewares/degradation_circuit_breaker_middleware.py`（P7 熔断器）
  - `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`（P7 中间件挂载，受保护文件）

---

## 5. 风险与注意事项

1. **这批 track 已闭环，别再"找 spec 实施"**——9 个全合了。下一步是 milestone/文档沉淀（§2.1/§2.2），或开 A 类新 track（§2.3，独立 spec）。
2. **untracked docs 入库时别误提 `uv.lock`**：`git status` 里有个 ` M packages/agent/backend/uv.lock` 从会话起就在，来历不明，单独问用户，别打包进 docs commit。
3. **A 类收口改动面大**（碰几乎所有读写 workspace 的工具/脚本），P4 诊断文档明确"分批、每批独立 spec + 回归、与正在跑的修复隔离"。别一次性大改。
4. **改受保护文件**（`lead_agent/agent.py` / `prompt.py` / `subagents/builtins/__init__.py` / `sandbox/tools.py` 等）后，sync deerflow 时要 surgical 守护——见 CLAUDE.md「同步核心规则」。P7 已往 `agent.py` 中间件链 + CLAUDE.md 受保护清单加了 `DegradationCircuitBreakerMiddleware`。

---

## 6. 下一位 Agent 的第一步建议

```bash
cd /home/wangqiuyang/noldus-insight
git fetch origin dev && git rev-parse --short origin/dev   # 应 = f5237903 或更新

# 1. 确认这批 9-spec 确实全合（无 open PR）
gh pr list --state open   # 预期：仅可能有不相关的新 PR

# 2. 若用户要 milestone：读纲 + 索引，再写
cat docs/refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md
cat docs/milestone/README.md

# 3. 若用户要把 untracked docs 入库（先看清楚，别带 uv.lock）
git status -sb
git add docs/superpowers/specs/2026-06-16-*.md docs/problems/2026-06-17-*.md \
        docs/handoffs/2026-06/2026-06-17-*.md docs/handoffs/2026-06/2026-06-18-*.md
git status   # 确认暂存区无 uv.lock 再 commit
```

**之后**：按用户意图——① 写 milestone（推荐，track 到 checkpoint 了）② dogfood 复跑验证 9 spec 端到端效果（§2.5）③ 开 A 类物理路径收口新 track（§2.3，按诊断文档批 1 起）。**若用户给新 dogfood 失败**：先按 4 红线框架判是哪条病根（`docs/refs/2026-06-17-...best-practices.md`），别写 prompt 补丁。

---

## milestone 建议

本会话让「harness 鲁棒性 / dogfood 根因治理」track 到达 **完成 checkpoint**（9 spec 全实施全合 dev）。建议下一 agent（或用户触发）：
- 创建/更新 milestone「harness 鲁棒性 / dogfood 根因治理」，记录：4 红线宪法 + 9 spec 索引（全 ✅）+ 产品立场（数据可复现性底线 + P7 降级熔断器机制）+ 本会话 4 个 review 修的真 bug（P3 groups 静默降级 / P2 残缺段伪装 absent / P5 seal 契约断裂 / P4 仅 lint）。
- 关键摘要：dogfood 反复故障收敛到 4 结构病根 → 9 spec 逐个根治 → 全合 dev；后续 A 类物理路径收口为独立新 track（诊断已就绪，5 批路线）。
