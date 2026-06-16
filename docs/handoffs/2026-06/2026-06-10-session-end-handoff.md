# 交接文档 — 2026-06-10 会话终止

> 写给下一位 AI Agent。当前 dev HEAD：`751be167`（spec-a PR #116 已合入）。

---

## 当前任务目标

本次会话完成了两项 OFT dogfood 故障的 harness 修复：
1. **Spec A**（code-executor 封存边界对账兜底）— ✅ 已实施、review 通过、PR #116 合入 dev
2. **Spec B**（澄清答案 write-through + 每轮回灌）— ✅ 已实施、bug 修复已 push，**等待你发 PR**

---

## 已完成工作 ✅

### Spec A（PR #116，已合入 dev @ 751be167）
- A1：删 `skills/custom/ethoinsight-code/SKILL.md:30` write_file 旧流，改 defer 到 seal 工具；`output-contract.md:22` [gate_signals] 降级为中间产物；`code_executor.py` step 6 seal 重构为正面唯一终结动作
- A2：`executor.py` `_AUTO_SEALABLE` 加 code-executor；`_attempt_auto_seal_from_artifacts` 加分支（plan vs 磁盘 m_*.json 完整性对账，missing=∅→completed/missing≠∅→partial，缺项列 errors）；`sealed_by=framework_rebuild`；结构化日志（触发率可观测）
- A3：seal-resume 补轮去 `key_findings` 专有名词
- 全量测试：branch vs baseline 同 config.yaml 时失败集 byte-identical（spec-a 引入 0 新失败）

### Spec B（已 push spec-b-resolved-facts @ bb0e25f1，等待 PR）
- B1：`prompt.py` 新增 `_get_resolved_facts_context(thread_id)` + `<resolved_task_facts>` 独立块注入 `apply_prompt_template`，实现 Fable 三隔离条件（C1 thread-scope/C2 独立块/C3 last-writer-wins）
- B2：`experiment_context.py` `set_experiment_paradigm_tool` 接受 `resolved_facts` 参数，同时写 context 文件权威 `resolved` 键 + memory facts 投影（非阻塞）
- B3：`prompt.py` 澄清段加正向复用规则
- **Bug 修复（bb0e25f1）**：`_thread_id_from_runtime` 读嵌套 `configurable.thread_id`→改扁平 `ctx.get("thread_id")`（与 8 处现有中间件对齐）；修正了一个把 buggy 行为写死的旧测试；加回归红锚点

### git stash pop 冲突解决
- `packages/ethoinsight/ethoinsight/parse/_core.py` 的 stash pop 冲突（`标题行数`/`number of header lines` 大小写匹配）已取 dev 版（`.lower()` 不敏感），staged，阻塞 `git pull` 的问题已解除

---

## 关键文件状态

### spec-b worktree
```
/home/wangqiuyang/noldus-insight-spec-b/  ← 本地 worktree
分支: spec-b-resolved-facts @ bb0e25f1（已 push origin）
```

### P1 未入库文档（25 个 untracked，主 checkout）
主要集中在：
- `docs/handoffs/2026-06/2026-06-08-*` （EPM dogfood handoff 系列）
- `docs/handoffs/2026-06/2026-06-09-*` （sync21/409 系列）
- `docs/handoffs/2026-06/2026-06-10-*` （本次会话 + noldus-kb）
- `docs/superpowers/specs/2026-06-08-*` （EPM dogfood 那批 spec）
- `docs/design/2026-06-09-noldus-kb-*` （noldus-kb 改造设计）

### 两份已 commit 的 spec（在 dev）
- `docs/superpowers/specs/2026-06-10-spec-a-code-executor-autoseal.md`
- `docs/superpowers/specs/2026-06-10-spec-b-resolved-facts-readback.md`

---

## 未完成事项（按优先级）

### P0 — 发 spec-b PR（你来做）
```
spec-b-resolved-facts @ bb0e25f1 已 push
目标 base: dev
建议 PR 标题：feat(harness): Spec B — 澄清答案 write-through + 每轮回灌 (resolved facts readback) [bug fix: thread_id flat path]
```

**PR 后验收**：dogfood OFT 时确认——连问三轮澄清不再每轮重读输入文件、已答的不重问、`<resolved_task_facts>` 确实回灌到 lead system_prompt（`sealed_by` + 触发率遥测）。

### P1 — 25 个未入库文档分组 commit（可选，不阻塞功能）
建议分 4 组 commit：
1. `docs/handoffs/2026-06/2026-06-08-*`（EPM dogfood handoffs）
2. `docs/handoffs/2026-06/2026-06-09-*` + `2026-06-10-*`（sync/409/本次）
3. `docs/superpowers/specs/2026-06-08-*`（EPM dogfood specs）
4. `docs/design/2026-06-09-noldus-kb-*`（noldus-kb design docs）

### P2 — spec-b worktree 清理
```bash
# spec-b 合入后可删 worktree
cd /home/wangqiuyang/noldus-insight
git worktree remove /home/wangqiuyang/noldus-insight-spec-b
```

### P3 — 其他已有 worktrees（不急，用户已知）
```
/home/wangqiuyang/noldus-insight/.claude/worktrees/ 下有多个旧 worktree
见 git worktree list 全览
```

---

## 关键决策记录（本次会话做出，下一 agent 直接复用）

### Fable 三轮审判的最终结论
1. **Spec A**：data-analyst 不接 auto-seal（维持 prompt-only）；触发判据：A1 dogfood 通过后指令一致条件下复发一次即启动转录兜底 follow-up spec
2. **Spec B**：`_thread_id_from_runtime` 必须用扁平 `runtime.context.get("thread_id")`，不能用嵌套 `configurable` 形
3. **同构本质**：A 和 B 共享"write 侧边界对账"原语，但 read 侧只 B 需要（消费者是 LLM 需回灌）；不造 BoundaryReconciler 基类（n=2 早熟）

### 全量测试关键发现
- 50 个失败是 **config.yaml 内容漂移**（summarization `summary_prompt: null` + models 字段）导致的，与 spec-a/b 代码无关
- 两个分支与 baseline 同 config.yaml 对比：失败集 byte-identical，branch-only failures = 0
- `test_subagent_executor.py` / `test_task_tool_core_logic.py` 大批失败 = 这 50 个的子集，同因

---

## 风险与注意事项

1. **spec-b PR 基于 dev（751be167）**，不要基于 main，worktree 基准见 spec-b 的 `git log --oneline`
2. **动 `experiment_context.py`/`prompt.py`/`agent.py` 后必须裸导入**：
   ```bash
   cd /home/wangqiuyang/noldus-insight-spec-b/packages/agent/backend
   PYTHONPATH=. uv run python -c "import app.gateway; print('OK')"
   PYTHONPATH=. uv run python -c "from deerflow.agents import make_lead_agent; print('OK')"
   ```
3. **已知 5 个基线红**（deferred_tool_registry ×2 + inspect_gate/paradigm async ×2 + chart_maker_config ×1）+ 50 个 config 漂移红，非本次改动引入，看到别归因自己

4. **Fable 5 不能在 noldus-insight 目录用**（CLAUDE.md biology 分类器触发）。要问架构问题用 `/tmp/` 外的通用稿。

5. **data-analyst 转录兜底留 follow-up**（Spec A §6.3 触发判据已定死），不在本批实施。

---

## 下一位 Agent 的第一步建议

```
1. 确认 spec-b PR 是否已发（如未发，从 spec-b-resolved-facts @ bb0e25f1 建 PR 到 dev）
2. 若已发，等合入后在真实 OFT 数据上 dogfood（验 <resolved_task_facts> 回灌 + 不重问）
3. P1 文档入库：按上面 4 组 commit，git add 各组文件 + git commit -m "docs: ..."
4. spec-b worktree 合入后清理
```
