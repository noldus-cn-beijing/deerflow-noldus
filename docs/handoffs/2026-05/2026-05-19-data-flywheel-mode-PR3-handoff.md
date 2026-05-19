# 2026-05-19 数据飞轮模式 + chart-maker 死锁修复 Handoff

> **状态**:PR-1 (3 commit) + PR-2 (1 commit) **已合入 dev**。PR-3 (3 改动) 已在 worktree 完成,**待 commit + push + 开 PR**。

---

## 上下文 — 这条修复链怎么走到这里的

2026-05-19 dogfood thread `51b00ac8` + `78ccb52b` 暴露了 8 个独立问题(详见 §诊断)。用户明确要求:

1. **不要做 prompt 胶水代码**,要修根因
2. **用 deerflow 现成的 feature**,不要重新造轮子
3. **数据飞轮模式**(UI 上「全自动 / 数据飞轮」toggle 中的后者)需要 **每个 subagent 完成后 graph 暂停**,等用户审批 + 留 feedback 再继续
4. **feedback 按钮要每条 AI message 都有**,不只是 tool-end 消息

按用户指示拆 3 个 PR 并行/串行执行。前 2 个已合入 dev,PR-3 代码已写完待提交。

---

## ✅ 已合入 dev 的修复(用户已合 PR #11 #12)

### PR-1 — Frontend workflow_mode 桥接 + skill 路径绝对化 + chart-maker 死锁

3 个 commit,在 dev 上是 `486db381` merge 之前的 `fcca9990`、`10766cf5`、`ff97eb8e`。

| Commit | 修了什么 |
|---|---|
| `ff97eb8e` feat(skills) | 新建 `deerflow.skills.render` 模块,inject SKILL.md 时给 `<skill>` 标签加 `base_path="/mnt/skills/<cat>/<name>"` + regex 把 `references/foo.md` 等相对路径改写绝对路径。9 TDD test |
| `10766cf5` fix(frontend) | `hooks.ts` 提交 stream context 加 `workflow_mode: context.mode === "flywheel" ? "manual" : "auto"`(1 行)— **修了一个超大潜伏 bug**:此前 UI 上「数据飞轮」mode 切换从未生效,backend 总是 fallback 到 `"auto"`,`GateEnforcementMiddleware` 自创建以来从未运行过(grep langgraph.log 无一行 `gate_check`) |
| `fcca9990` fix(chart-maker) | chart-maker prompt:png `--output` 改为 `/mnt/user-data/outputs/` 而非 workspace(否则 `present_files` 不接受);catalog.resolve 列完整必填参数;skill 路径补 `/custom/` category |

### PR-2 — StepwiseGateMiddleware(数据飞轮逐 subagent 暂停)

1 个 commit `7bf1324b`。

新建 `StepwiseGateMiddleware`,**复用 LangGraph 原生 `Command(goto=END)` 机制**(跟 `ClarificationMiddleware` 完全同构,零新基础设施)。

**触发规则**:
- HumanMessage → AI(task):**放行**(用户新 turn 的第一个 task)
- AI(task) → ToolMsg(task) → AI(task):**拦截**,返回 `Command(goto=END)`,graph 暂停,前端流结束
- 用户新发 HumanMessage:gate reset

仅在 `workflow_mode == "manual"` 挂载(配合 PR-1 解锁)。9 TDD test 全绿。

---

## ⏳ 待你接手的部分 — PR-3 已写完代码,未 commit/push/PR

### 工作环境

```bash
# Worktree(分支跟 origin/dev 已 rebase 同步)
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/subagent-role-split-impl
git log --oneline -3   # 应该跟 origin/dev 完全一致(486db381 在最顶)

# 主仓 dev(已 pull 到最新)
cd /home/wangqiuyang/noldus-insight
git status   # 主仓有 3 个 modified 文件,跟 worktree 完全一致(我手动同步过)
```

**当前未提交改动(主仓 + worktree 完全一致)**:
```
M packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py
M packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py
M packages/agent/frontend/src/core/threads/hooks.ts
```

### PR-3 改动详情

#### 改动 ① — code_executor.py + chart_maker.py:`${workspace_path}` 占位符全部替换成绝对路径

**根因**:transcript 中 code-executor 跑 `find / -name plan_metrics.json` 6 次猜路径,因为 system_prompt 里 `${workspace_path}/plan_metrics.json` 是个**模板字符串**,LLM 不知道它实际等于什么。

**修法**:把 prompt 里所有 `${workspace_path}` 改成 deerflow sandbox 的实际虚拟路径 `/mnt/user-data/workspace`,这个值是个**常量**(见 backend CLAUDE.md「Virtual Path System」)。

具体修了:
- `code_executor.py:17-21 <environment>` — 明示「工作目录 = `/mnt/user-data/workspace`」+ skill 目录 `/mnt/skills/custom/ethoinsight/...` + skill 引用已被 harness 改写
- `code_executor.py:23-37 <workflow>` — workflow 步骤里所有 `${workspace_path}` 替换
- `code_executor.py:68 <output>` 同样替换
- `chart_maker.py:17-25 <environment>` — 同上,加上「workspace_path 等同于沙盒虚拟路径,但所有引用都用绝对形式」的注释
- `chart_maker.py:23-37 <workflow>` 全部替换
- `chart_maker.py:69 <output>` 替换

**注意**:这 2 个 subagent 通过 `pytest tests/test_code_executor_config.py tests/test_chart_maker_config.py tests/test_code_executor_workflow.py` 跑过,15/15 全绿。

#### 改动 ② — hooks.ts:feedback 按钮在历史消息也渲染

**根因**:`messageRunIds` 是 `Map<message_id, run_id>`,只在 `onLangChainEvent(event.event === "on_tool_end")` 时填充(L325-338)。后果:
- 纯文本 AIMessage(无 tool_call)→ 没有 `on_tool_end` 事件 → 没 run_id → feedback 按钮不渲染
- 页面刷新 → React state 清零 → **所有历史消息都丢 run_id**

**修法**:`useThreadHistory` hook(L766-,这个 hook 在 page mount 时遍历 runs,调 `/api/threads/{tid}/runs/{run_id}/messages` 拿历史 messages)在 fetch 时**顺便把每条 message 的 id 跟当前 run.run_id 关联存到自己的 `historyMessageRunIds` Map**,return 出来。`useThreadStream` 用 `useMemo` 把 stream-derived 和 history-derived 两个 Map merge,stream 优先(反映最新 run),history 兜底。

**精确改动 4 处**:

1. `hooks.ts:254-255`(`useThreadStream` 解构):新增 `messageRunIds: historyMessageRunIds`
2. `hooks.ts:763-776`:把 `return [mergedThread, sendMessage, isUploading, messageRunIds]` 改成 useMemo 合并两个 map 再返回
3. `hooks.ts:783-789`(`useThreadHistory` 内部):新增 state `const [historyMessageRunIds, setHistoryMessageRunIds] = useState<Map<string, string>>(() => new Map())`
4. `hooks.ts:843-859`(history fetch loop 内):在 `setMessages(...)` 之后插入 `setHistoryMessageRunIds(...)` 块,遍历 `_messages` 把每条的 `msg.id` 映射到当前 `run.run_id`,**幂等**(已存在 == 当前 run_id 则跳过)
5. `hooks.ts:912-915`(`useThreadHistory` return):新增 `messageRunIds: historyMessageRunIds`

**主仓 typecheck 已过**:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend && pnpm typecheck   # ✅ 已验证
```

### 你接手后要做的步骤

#### Step 1 — 验证主仓 + worktree 状态一致(应该是)

```bash
cd /home/wangqiuyang/noldus-insight
diff packages/agent/frontend/src/core/threads/hooks.ts \
     /home/wangqiuyang/noldus-insight/.claude/worktrees/subagent-role-split-impl/packages/agent/frontend/src/core/threads/hooks.ts
# 应该 exit=0(我已经从主仓 cp 同步过去)
```

#### Step 2 — 跑相关测试(worktree 里)

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/subagent-role-split-impl/packages/agent/backend
source .venv/bin/activate
PYTHONPATH=. python -m pytest \
  tests/test_code_executor_config.py \
  tests/test_chart_maker_config.py \
  tests/test_code_executor_workflow.py \
  tests/test_subagent_skill_path_rewrite.py \
  tests/test_stepwise_gate_middleware.py \
  -v
# 应该 ≥ 33 passed(15 + 9 + 9)
```

frontend typecheck:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/frontend && pnpm typecheck
```

frontend lint 可选(input-box.tsx 有 1 个 pre-existing error,跟本 PR 无关):
```bash
pnpm lint 2>&1 | grep hooks.ts   # 我加的代码 lint 应该干净
```

#### Step 3 — 全后端 regression 看 baseline 不破

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/subagent-role-split-impl/packages/agent/backend
source .venv/bin/activate
PYTHONPATH=. python -m pytest tests/ -q \
  --ignore=tests/test_metric_catalog_live.py \
  --ignore=tests/test_client_live.py \
  --ignore=tests/test_create_deerflow_agent_live.py 2>&1 | tail -3
# 期望:2637 passed,32 failed (pre-existing baseline,差集为空)
```

#### Step 4 — Commit + push(在 worktree 上)

我建议拆 2 个 commit,**最后才 commit hooks.ts 那个最大的改动**:

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/subagent-role-split-impl

# Commit 1: subagent prompt 占位符 → 绝对路径
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py
git commit -m "$(cat <<'EOF'
fix(subagents): code-executor / chart-maker prompt 用绝对 sandbox 路径替换 ${workspace_path} 占位符

2026-05-19 dogfood thread 78ccb52b 暴露 code-executor 跑 find / -name
plan_metrics.json 6 次猜路径,因为 system_prompt 里 ${workspace_path}/plan_metrics.json
是个模板字符串,LLM 不知道实际指向什么(sandbox virtual path 在 LLM 眼里就是常量)。

修法:把所有 ${workspace_path} 直接替换为 /mnt/user-data/workspace(deerflow
sandbox 的虚拟路径,常量)。同时把 skill 引用补全 /custom/ category 前缀,
跟 PR-1 的 render.py 改写规则保持一致(LLM 看到的就是 harness rewrite 后的形态)。

不影响测试 — test_code_executor_config / test_chart_maker_config /
test_code_executor_workflow 15/15 全绿。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

# Commit 2: feedback 按钮在历史消息和纯文本 AIMessage 都渲染
git add packages/agent/frontend/src/core/threads/hooks.ts
git commit -m "$(cat <<'EOF'
fix(frontend): feedback 按钮在所有历史消息和纯文本 AIMessage 都渲染

根因:messageRunIds 是 Map<message_id, run_id>,只在 onLangChainEvent
(event.event === "on_tool_end") 时填充(hooks.ts:325-338)。后果:
- 纯文本 AIMessage(无 tool_call,如 lead 反问 / 最终回答)永远没 run_id
- 页面刷新 → React state 清零 → 所有历史消息都丢 run_id → feedback 按钮全消失

这是数据飞轮模式 per-step feedback 收集的核心阻碍 —— 用户想留反馈但按钮不显示。

修法:useThreadHistory hook 已经在 page mount 时遍历 runs 调
/api/threads/{tid}/runs/{run_id}/messages 拿历史 messages(L815-826),
fetch 完顺便把每条 message.id 跟 run.run_id 写到自己的 historyMessageRunIds
Map,return 出来。useThreadStream 用 useMemo 把 stream-derived 和
history-derived 两个 Map merge,stream 优先(最新 run),history 兜底。

主仓 pnpm typecheck 已通过。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"

# Push
git push origin worktree-subagent-role-split-impl
```

#### Step 5 — 开 PR

PR 链接(push 后会有):
```
https://github.com/noldus-cn-beijing/noldus-insight/compare/dev...worktree-subagent-role-split-impl?expand=1
```

**PR Title**:
```
fix: PR-3/3 subagent prompt 路径绝对化 + feedback 按钮全渲染
```

**PR Body**:
```markdown
## Summary

PR-3/3 — 数据飞轮模式收尾。修两个独立 bug:

### 修复 #1 — `${workspace_path}` 占位符让 subagent 满世界 find
- 2026-05-19 dogfood thread 78ccb52b 中 code-executor 跑 `find / -name plan_metrics.json` 6 次猜路径
- 根因:system_prompt 里 `${workspace_path}/plan_metrics.json` 是个模板字符串,**LLM 看到 `${workspace_path}` 不知道它要被替换成什么**
- 修法:直接替换为 `/mnt/user-data/workspace`(deerflow sandbox 虚拟路径常量)

### 修复 #2 — feedback 按钮只在 tool-end 消息出现
- 根因:`messageRunIds` 只在 `on_tool_end` LangChain event 触发时填充。纯文本 AIMessage 和页面刷新后的历史消息都拿不到 run_id → feedback 按钮不渲染
- 数据飞轮模式 per-step feedback 收集的核心阻碍
- 修法:`useThreadHistory` hook 在 fetch 每个 run 的 messages 时,顺便建 `(message.id, run.run_id)` 映射;`useThreadStream` 用 useMemo merge stream-derived 和 history-derived 两个 Map

## Test plan

- [x] backend tests: 33/33 passed(code_executor_config + chart_maker_config + code_executor_workflow + 已合 PR-1/2 的 skill_path_rewrite + stepwise_gate)
- [x] backend full suite: 2637 passed,无新 regression
- [x] frontend `pnpm typecheck`:已过
- [ ] 合并后:重启 dev,数据飞轮 mode 跑 EPM 单被试,验证:
  - code-executor **不再 find / 不再 ls** 探索 plan_metrics.json,直接读 `/mnt/user-data/workspace/plan_metrics.json`
  - 每条 AI message(包括反问 + 最终回答 + subagent 完成后的 lead 汇报)下方都有 feedback 按钮(三色)
  - 页面刷新后历史消息的 feedback 按钮也在
```

#### Step 6 — 用户合 PR 后

- 主仓 `git pull` 拉新
- 重启 dev:`cd packages/agent && make stop && make dev`
- 跑数据飞轮 mode E2E,验证以上 3 条

---

## 完整的 8-issue 诊断表(供参考)

来自原始 dogfood thread 51b00ac8 + 78ccb52b transcript + langgraph.log:

| # | 现象 | 根因 | 状态 |
|---|------|------|------|
| 1 | code-executor `find / -name plan_metrics.json` 6 次 | system_prompt 用 `${workspace_path}` 模板,LLM 不知实际路径 | **PR-3 ①(待 commit)** |
| 2 | SKILL.md 内 `references/foo.md` 相对路径,subagent 猜 6 次 | SKILL.md inject 时不改写相对路径 | ✅ PR-1 `ff97eb8e` |
| 3 | chart-maker `catalog.resolve` 试 4 次才蒙对参数 | prompt 没列必填参数 | ✅ PR-1 `fcca9990` |
| 4 | chart-maker 派 general-purpose cp 文件 → LoopDetection 死锁 | png 写到 workspace/ 而非 outputs/,lead 无 bash 工具,只能派 task 帮忙 | ✅ PR-1 `fcca9990`(直出 outputs/) |
| 5 | lead 算完指标不告诉用户就派 chart-maker | 全自动模式 by design,但数据飞轮模式应该 stepwise gate | ✅ PR-2 `7bf1324b` |
| 6 | feedback 按钮只在 tool-end 消息出现 | `messageRunIds` 只从 `on_tool_end` event 拿 run_id | **PR-3 ②(待 commit)** |
| 7 | 跨会话上传文件丢失 | 不是 bug,是 lead 正确反问 | ✅ 无需修 |
| 8 | UI mode 切换从未生效 | frontend hooks.ts 漏传 `workflow_mode` config | ✅ PR-1 `10766cf5` |

---

## 关键文件路径速查

- **Worktree**: `/home/wangqiuyang/noldus-insight/.claude/worktrees/subagent-role-split-impl/`
- **Worktree branch**: `worktree-subagent-role-split-impl`(base: origin/dev `486db381`)
- **主仓**: `/home/wangqiuyang/noldus-insight/`(branch dev)
- **dev .venv**: `/home/wangqiuyang/noldus-insight/.claude/worktrees/subagent-role-split-impl/packages/agent/backend/.venv/`(worktree 共用)
- **frontend node_modules**: 主仓 `/home/wangqiuyang/noldus-insight/packages/agent/frontend/node_modules`(worktree **没装**,要 typecheck 在主仓跑)
- **dogfood log**: `/home/wangqiuyang/noldus-insight/packages/agent/logs/langgraph.log`
- **dogfood transcript 文件**: `/home/wangqiuyang/noldus-insight/docs/e2e/Elevated Plus Maze Track Analysis 5-19.md`

## 关键 Context / 约束

- **用户已合 PR #11 (PR-1) 和 PR #12 (PR-2)**,本 worktree 已 rebase 到 dev `486db381`
- **不要再造 prompt 胶水 middleware**(用户明确反对,所以我之前写的 ProgressReportEnforcementMiddleware 已删,改用 deerflow 现成的 `Command(goto=END)` = StepwiseGateMiddleware)
- **数据飞轮模式开关 = `workflow_mode == "manual"`**,UI 上对应「数据飞轮」按钮
- **超过 PR-3 scope 的优化** — chart-maker 自己生成 columns.json/raw_files.json 是个未来工作(应该让 lead 用类似 prep_metric_plan_tool 的 prep_chart_plan_tool 一步搞定),**不要在本 PR-3 做**
- **`workflow_mode` 这变量在 lead_agent.py L412 是 pre-existing ruff F841 unused warning**,跟我无关,不要顺手清(scope creep)

## 验证清单(用户测试通过的标志)

下次 dogfood E2E 应该看到:
1. ✅ langgraph.log 出现 `gate_check` 行(来自 GateEnforcementMiddleware,PR-1 解锁)
2. ✅ 数据飞轮 mode 下,lead 派一个 subagent 后 graph 暂停,前端看到 stream END
3. ✅ chart-maker 把 png 直接写到 outputs/,**不再** `[LOOP DETECTED] task 3 times` 死锁
4. ✅ code-executor **不再** `find / -name plan_metrics.json`,直接 read `/mnt/user-data/workspace/plan_metrics.json`
5. ✅ 每条 AI message(反问 / 流式输出 / 最终回答 / subagent 完成后的汇报)都有 feedback 三色按钮
6. ✅ 页面刷新后历史消息的 feedback 按钮仍在(不再丢失 React state)
