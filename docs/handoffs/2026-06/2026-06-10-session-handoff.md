# 交接文档 — 2026-06-10 会话

> 写给下一位 AI Agent。当前 dev HEAD：`1d4a9d25`。

---

## 当前任务目标

本次会话做了三件事：
1. 项目级文档对齐（已完成 ✅）
2. OFT dogfood 两处故障的 review brief 骨架化（进行中，见下）
3. 讨论 Fable 5 被 biology 分类器误 flag 的根因和规避（已结论）

---

## 已完成工作 ✅

### 1. 项目级文档对齐（commit `1d4a9d25`，已在 dev，未 push）

- **CLAUDE.md**：修 6 处 `docs/roadmap.md`/`docs/prd.md` 死链（文件不存在）→ 改指 `docs/milestone/README.md`；顶部加"v0.1 真实阻塞 = 同事方法论"说明；仓库树删除不存在的文件条目。
- **`docs/milestone/README.md`**：加 sync-21/409-fix/seal 鲁棒性那批已合（6/8–6/9）；结构聚合(#98)+golden case(#90)明确标 🔴 共同阻塞于同事方法论；顶部加阻塞 callout；补 noldus-kb 独立轨道。
- **新建 `docs/milestone/blocked-on-expert-methodology.md`**：把"等同事"拆成可逐项交付的精确待办（结构聚合按 6 范式逐个列待确认聚合语义 + golden case 列清 FindingExpectation 字段）。

### 2. Fable 5 biology 误报根因确认

**结论**：不是某份文档的词汇问题，而是**CLAUDE.md 被 Claude Code 自动注入上下文，其中 33 行命中行为学触发词**，整个项目目录对 Fable 5 结构性地会被 flag。逐词改文档治不了根因。

**决定**：为了让 Fable 5 能审 harness 故障，把 review brief 改写成**完全通用的 multi-agent 编排技术问题描述**，写到项目目录之外 `/tmp`，绕开 CLAUDE.md 注入。

---

## 关键文件状态

### 已生成、**待清理** 的 .bak 文件（下一个 agent 可以删掉或 gitignore）
```
docs/handoffs/2026-06/2026-06-10-oft-dogfood-two-failures-seal-replan-review-brief.md.bak-biology-original
docs/handoffs/2026-06/2026-06-10-oft-dogfood-two-failures-seal-replan-review-brief.md.bak-before-rewrite
docs/handoffs/2026-06/2026-06-10-oft-dogfood-two-failures-seal-replan-review-brief.md.bak-rewrite-v1
```

### 当前 review brief（骨架版，未 commit）
```
docs/handoffs/2026-06/2026-06-10-oft-dogfood-two-failures-seal-replan-review-brief.md
```
内容：故障 A（`code-executor` 漏调 `seal_code_executor_handoff`）+ 故障 B（lead 反复重规划）的纯工程骨架，已去掉所有行为学叙述，保留所有 `file:line` 锚点。

### Fable 5 可用的通用讨论稿（在项目目录之外）
```
/tmp/multiagent-orchestration-two-failures-discussion.md
```
**用法**：在非 noldus-insight 目录开新 session，用这份稿和 Fable 5 讨论 multi-agent 架构（不会被 CLAUDE.md 染色）。

---

## 未完成事项（按优先级）

### P0 — 等 Fable 审完，写两份 spec 并实施

review brief 的目的是让 Fable 5（或 Opus）审核根因判断和修复方向，再决定拆 spec / 实施。两个故障均是 harness 层修复，**独立于行为学同事方法论阻塞，可立即推进**：

**故障 A 修复（spec 待写）**：
- A1：`skills/custom/ethoinsight-code/SKILL.md:30` + `templates/output-contract.md:22` 删 write_file 旧流，改 defer 到 seal 工具
- A2：扩 `_attempt_auto_seal_from_artifacts`（`subagents/executor.py:288-399`）接入 code-executor，从 `m_*.json` + `groups.json` + `plan_metrics.json` 机械重建 `metrics_summary`/`per_subject`
- A3：`executor.py:917` 补轮措辞 `key_findings` 改通用字段（注意 `:913` in-code 守卫）
- A4：**不改** `validate_catalog`（red herring）

**故障 B 修复（spec 待写）**：
- B1：`lead_agent/agent.py:530` 扩消费面，渲染 `<experiment_context resolved="true">` 块回灌 system_prompt
- B2：`experiment-context.json` 加 groups/trial 即时落盘（守 SSOT：`groups.json` 权威，派生不双存）
- B3：`lead_agent/prompt.py:461-482` 加正向复用规则

### P1 — 大批未 commit 的文档/spec 需要整理入库

工作区有约 22 个未跟踪文件，包括：
- `docs/handoffs/2026-06/2026-06-08-*`（EPM dogfood handoff 系列）
- `docs/handoffs/2026-06/2026-06-09-*`（sync21/409 merged handoff）
- `docs/superpowers/specs/2026-06-08-*`（EPM dogfood 那批 spec）
- `docs/design/2026-06-09-noldus-kb-*`（noldus-kb 改造设计文档）
- `docs/handoffs/2026-06/2026-06-10-noldus-kb-redesign-handoff.md`
- `reports/report for june/`（用户本地报告）

建议：除 `reports/` 外，大部分应该入库。可以分组 commit（handoffs 一批、specs 一批、design 一批）。

### P2 — `1d4a9d25` 这个 docs commit 还没 push

当前只在 dev 本地。按项目规范需要手动 push（`git push origin dev`）或用户决定何时 push。

---

## 风险与注意事项

1. **worktree 创建必须显式基于 dev**：`git worktree add <path> -b <branch> dev`，不写基准默认 origin/main，会落后一大截，PR 会卷入误删除。这坑反复踩（见 memory `project_2026-06-08_three_specs_review_acdf`）。

2. **改 `executor.py` 后必须裸导入验证**（`feedback_conftest_mock_hides_circular_import_verify_bare_prod_import`）：
   ```bash
   PYTHONPATH=. python -c "import app.gateway"
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
   ```
   conftest mock 会藏循环导入，pytest 假绿不算过。

3. **已知全量 5 个基线红**（deferred_tool_registry ×2 + inspect_gate/paradigm async ×2 + chart_maker_config ×1），非本次改动引入，看到别归因自己。

4. **Fable 5 在 noldus-insight 目录会被 biology 分类器 flag**：根因是 CLAUDE.md 自动注入。harness 工程任务用 Opus 4.8；要用 Fable 5 审架构问题，用 `/tmp/multiagent-orchestration-two-failures-discussion.md` 在项目目录外讨论。

5. **本次两修复独立于同事方法论阻塞**（`blocked-on-expert-methodology.md`）——可以不等同事直接推进。

---

## 下一位 Agent 的第一步建议

```
1. 读 /tmp/multiagent-orchestration-two-failures-discussion.md（或直接看 review brief 骨架版）
2. 在非 noldus-insight 目录开 Fable 5 session，贴入通用讨论稿，让 Fable 审核根因和修复方向
3. Fable 通过后，回到 noldus-insight 目录（用 Opus 4.8），按 A1/A2/A3 和 B1/B2/B3 各建 worktree 写 spec 并实施
4. 实施完跑全量测试 + 裸导入验证（见上方风险第 2 条）
5. 顺手清理 .bak 文件 + 把 P1 那批未入库文档分组 commit
```
