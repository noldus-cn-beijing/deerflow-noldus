# 2026-04-21 Subtask 可见性 + 语言一致性 + 洞察深度 交接

> 给下一位 Agent：上下文延续点。读这份文档 + `docs/plans/2026-04-21-subtask-visibility-and-language.md`（本轮执行的计划）即可无缝接手。

---

## 1. 当前任务目标

解决 fix4 E2E 暴露的三类问题：

1. **过程可见性**：用户点开 SubtaskCard 只看到"读取 skill 文件"一行，看不到 subagent 内部"专家工作过程"
2. **语言一致性**：Lead 在中文会话里仍会在 AIMessage 正文主动写 `## Extracted Context: Zebrafish Shoaling Analysis` 这种英文结构化 dump
3. **洞察深度回归**：data-analyst 相较 fix3 退化——fix3 能点出"Subject 3 的 NND=70.02，排除后均值降至 37.23 mm"，fix4 只会说"存在至少一个离群个体"

**架构约束**（重要，不要动）：
- 不改 DeerFlow 基建（middleware / StreamBridge / message routing）
- 只改 fork 受保护文件（lead prompt / subagents/builtins/*）+ EthoInsight 包 + 前端渲染层
- Prompt 改动必须用**正面指令**（GLM-5.1 对"禁止 X"会反向激活，user memory 里有这条）

---

## 2. 当前进展

**已完成（本地 dev 分支 8 commit，未 push）**：

| Commit | 范围 | 说明 |
|---|---|---|
| `4927baea` C1 | 前端 | `Subtask.messages: AIMessage[]` 累积保留，替代 latestMessage 单条覆盖 |
| `918bb8a2` C2 | 前端 | SubtaskCard 展开态渲染统一 CoT 时间线（reasoning / tool call / text）|
| `c686729f` C3 | 前端 | 拆分 `LEAD_HIDDEN_TOOL_CALL_NAMES` vs `SUBTASK_HIDDEN_TOOL_CALL_NAMES`|
| `33e273f4` C4 | 后端 | Lead prompt 加 `<用户语言锁定>` + `<回答风格>` |
| `26e585e5` C5 | 后端 | 三个 subagent prompt 各加 `<语言>` 小节 |
| `04b76199` C6 | 后端 + ethoinsight | DataAnalystHandoff.outlier_findings / OutlierFinding schema；per_subject 写入 handoff JSON；data-analyst prompt 新增 Step 6 按受试者检查 + 反事实 |
| `6bd12282` fix | 前端 | ToolCall bash 分支缺 description 时返回 JSX（E2E 首次崩溃原因）|
| `3c51a235` fix | 前端 | thread 切换时把 setSubtasks 挪到 useEffect（setState-in-render warning）|

**测试全绿**：
- backend: 1660 passed, 14 skipped（基线 1651 + 新增 9）
- ethoinsight: 131 passed, 3 skipped（基线 130 + 新增 1）
- 前端: `pnpm check` + `SKIP_ENV_VALIDATION=1 pnpm build` 全绿

**E2E 部分验证**：
- 用户跑了一次 shoaling 5 文件分析 → 后端正常
- `handoff_code_executor.json` 确认包含 `per_subject` 字段，Subject 3 `mean_nnd = 70.02223` 正确
- 前端在打开 SubtaskCard 时崩溃（两个 fix 已修）
- 用户尚未在修复后重新跑 E2E 确认修复生效

---

## 3. 关键上下文

### 3.1 项目与分支

- 仓库根：`/home/qiuyangwang/noldus-insight`（WSL2）
- 分支：`dev`，从 `ec62f918 wrap up for 420` 起 8 个 commit 待 push
- 执行的 plan：`docs/plans/2026-04-21-subtask-visibility-and-language.md`（未追踪，**不要 git add**）
- E2E 产物目录：`docs/e2e_tests/`（未追踪，**不要 git add**）
- `docs/ethoinsight-architecture.html` 有未提交的无关改动（上一轮遗留，不要动）

### 3.2 关键文件路径

**前端**（C1-C3 + 两个 fix）：
- [packages/agent/frontend/src/core/tasks/types.ts](packages/agent/frontend/src/core/tasks/types.ts) — Subtask 类型（新加 `messages: AIMessage[]`）
- [packages/agent/frontend/src/core/tasks/context.tsx](packages/agent/frontend/src/core/tasks/context.tsx) — useUpdateSubtask 按 message.id 去重追加
- [packages/agent/frontend/src/core/threads/hooks.ts:310-315](packages/agent/frontend/src/core/threads/hooks.ts#L310-L315) — thread 切换 useEffect 清空 subtasks
- [packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx](packages/agent/frontend/src/components/workspace/messages/subtask-card.tsx) — SubtaskCoTTimeline + CoTStepRenderer
- [packages/agent/frontend/src/components/workspace/messages/message-group.tsx:455-489](packages/agent/frontend/src/components/workspace/messages/message-group.tsx#L455-L489) — LEAD / SUBTASK 两组白名单、`convertToSteps` 第二/第三参数

**后端**（C4-C6）：
- [packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:739-762](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L739-L762) — `<用户语言锁定>` + `<回答风格>`，插在 `<critical_reminders>` 之前
- [packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py](packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py) / [data_analyst.py](packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py) / [report_writer.py](packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py) — 各有 `<语言>` 小节
- [packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py:47-76](packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py#L47-L76) — workflow Step 6 "按受试者逐一检查 + 反事实"
- [packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py) — CodeExecutorHandoff.per_subject + OutlierFinding + DataAnalystHandoff.outlier_findings

**EthoInsight**（C6）：
- [packages/ethoinsight/ethoinsight/templates/tool.py:1022-1043](packages/ethoinsight/ethoinsight/templates/tool.py#L1022-L1043) — `assess_and_handoff_tool` 把 `m_result["per_subject"]` 写入 handoff JSON

**新增测试**：
- `packages/agent/backend/tests/test_lead_prompt_language_and_style.py`
- `packages/agent/backend/tests/test_subagent_language_rule.py`
- `packages/agent/backend/tests/test_data_analyst_insight_contract.py`
- `packages/ethoinsight/tests/test_code_summary_per_subject.py`
- `packages/ethoinsight/tests/test_granular_tools.py`（扩展 per_subject 断言，demo 数据缺失时跳过）

### 3.3 常用命令

```bash
# 启动 / 停止服务
cd /home/qiuyangwang/noldus-insight/packages/agent
make dev          # 启动全栈，localhost:2026
make stop

# 后端测试
cd packages/agent/backend && source .venv/bin/activate
make test
PYTHONPATH=. pytest tests/test_data_analyst_insight_contract.py -v

# ethoinsight 测试
cd packages/ethoinsight
source /home/qiuyangwang/noldus-insight/packages/agent/backend/.venv/bin/activate
python -m pytest tests/ -q

# 前端
cd packages/agent/frontend
pnpm check
SKIP_ENV_VALIDATION=1 pnpm build
```

### 3.4 E2E 产物位置（验证时用）

最新一次 shoaling E2E 的产物：
```
packages/agent/backend/.deer-flow/threads/6f046cc7-775a-4eb9-9027-2022e50781ca/user-data/workspace/
  ├── handoff_code_executor.json  ← 含 per_subject（验证 C6）
  ├── code_summary.json
  ├── metrics.pkl
  ├── statistics.json
  └── ...
```

---

## 4. 关键发现

### 4.1 已验证有效

- **C6 per_subject 链路打通**：latest E2E 的 `handoff_code_executor.json` 包含完整 per_subject，Subject 3 mean_nnd = 70.02223 与 fix3 一致。这说明"让 data-analyst 拿得到数据"这一步 OK，剩下的是让它"真的去看 + 用"（prompt 生效靠运行时）。
- **ToolCall bash 裸字符串 bug 是预先存在的**：不是我们引入的，只是被 C3 之前的"bash 全局隐藏"掩盖了。C3 把 bash 从 SUBTASK 白名单去掉后暴露出来。修法是 `label={description ?? fallback}`，永远返回 JSX。
- **setTasks-in-render 规则**：Context 来的 setState 不能在 render 函数里直接调；ref 赋值可以（ref 不是 state）。thread 切换的"清旧状态"逻辑必须放在 `useEffect([threadId])` 里。

### 4.2 数据位置

- `per_subject` 原始数据住在 `metrics.pkl`（ethoinsight compute_metrics 写入）和 `metrics.csv`（output 目录）里，一直都在，只是 `assess_and_handoff_tool` 没有把它暴露到 handoff JSON。
- `m_result["per_subject"]` 形状：`{subject_name: {metric: value, velocity_stats: {mean, std, max, min, median}}}`

### 4.3 测试覆盖边界

- Prompt 契约测试**只验证关键 phrase 存在**，不验证模型真的听。C4/C5/C6 的效果必须靠 E2E 确认。
- `test_granular_tools.py::test_full_pipeline_writes_valid_handoff` 需要 `demo-data/*.txt` 但当前该目录结构已变（demo-data/DemoData/...），测试跳过。不是回归，是 pre-existing skip。新加的 `test_code_summary_per_subject.py` 直接构造 metrics.pkl，不依赖 demo 数据。

---

## 5. 未完成事项（按优先级）

### 5.1 【高】用户的 E2E 验证（进行中）

用户在两次 fix 之后需要**重跑 shoaling 5 文件分析**，对照清单：

| # | 验收点 | 证据位置 |
|---|---|---|
| 1 | SubtaskCard 展开显示完整 CoT（不只是"读取 skill 文件"）| 前端界面 |
| 2 | parse_trajectories / compute_metrics 等专家工具可见，read_file/write_file/ls 不可见 | 前端界面 |
| 3 | Lead 主时间线仍然简洁 | 前端界面 |
| 4 | 全链路语言一致，无 `## Extracted Context` 英文 dump | 前端 + 导出 |
| 5 | data-analyst 点名 Subject 3 + 给出反事实（排除后降至 37.23 mm）| 前端界面 + `handoff_data_analyst.json` 的 outlier_findings（如写）|
| 6 | 无崩溃 / 无 setState warning | 浏览器 DevTools Console |
| 7 | 后端日志干净 | `packages/agent/logs/langgraph.log` |

**如果用户报新错**：让他贴浏览器 DevTools Console 的**完整错误栈**，不要只贴 error message。

### 5.2 【中】导出 + 归档 fix5 对话

用户手动从前端导出新的 E2E 对话到：
```
docs/e2e_tests/斑马鱼鱼群行为轨迹数据分析-fix5.md
```
这步由**用户做**，不要 git add（`docs/e2e_tests/` 是未追踪的）。

### 5.3 【中】DeerFlow 上游同步（用户已问过，**现在不做**）

用户问过"要不要拉上游"，我回答"**现在不拉**"，理由：
1. 改的几乎全是受保护文件，拉取必触发冲突
2. 本地有 8 commit 未 push，先 push 再同步更安全
3. 用户在做 E2E，引入未知变化会污染结果

正确时机：用户 E2E 验证通过 → push dev → 再跑 `./scripts/sync-deerflow.sh --dry-run` 看上游 diff。

### 5.4 【低】Push 决策

不要自己 push。问用户。当前 `dev` 领先 `origin/dev` 共 **8 commit**：
- C1-C6（6 个）
- 2 个 fix commit

用户之前提过还有 "Phase A/B 15 commit + pipeline 1-6a 的 7 commit" 等待 push——check `git log origin/dev..dev` 的实际数字再说，不要硬记 plan 里写的"29 commit"数字。

### 5.5 【低】Prompt 有效性调优（E2E 失败时的备选）

如果 C4/C5/C6 某条规则在 E2E 中失效：

- **语言混用仍发生**：看 `langgraph.log` 启动时的 system prompt dump，确认 phrase 真的渲染到了系统消息里
- **data-analyst 没点名 Subject**：看 `handoff_code_executor.json` 确认 `per_subject` 已写出；若已写出但模型没用，强化 workflow Step 6 的 imperative 措辞（现在是"关键！"标注，可以加"必须"）
- **Lead 仍写英文 dump**：GLM-5.1 对正面指令效果可能有限（user memory 里的原则），考虑加"好回答 vs 不好回答"的对照示例

---

## 6. 建议接手路径

**第一步：看用户当前状态**
```bash
cd /home/qiuyangwang/noldus-insight
git log --oneline dev -5       # 确认 HEAD 是 3c51a235 或更新
git status -s                  # 预期只有 docs/ethoinsight-architecture.html M + 两个未追踪
ls packages/agent/backend/.deer-flow/threads/ | tail -3  # 看用户跑了几次 E2E
```

**第二步：读核心文件（按重要性排序）**
1. `docs/plans/2026-04-21-subtask-visibility-and-language.md` — 完整执行计划（Task 1-8）
2. `/home/qiuyangwang/noldus-insight/CLAUDE.md` — 项目全景
3. `docs/handoffs/2026-04-20-phase-ab-handoff.md` — 上一轮交接，了解 `B1/B2/B3` 的 InternalNotesMiddleware 废弃背景
4. 本文件

**第三步：根据用户反馈分支**

- 若用户说"E2E 通过了" → 进入 Task 7 Step 6（写 handoff-done 文档）+ 问 push
- 若用户贴出新的前端崩溃 → 看 console 栈 → 定位新 bug → 小 commit 修
- 若用户说"data-analyst 还是没点名 Subject" → 进入 §5.5 备选路径
- 若用户说"要拉上游" → 先 `./scripts/sync-deerflow.sh --dry-run` 给用户看

---

## 7. 风险与注意事项

### 7.1 不要做的事

- **不要 git add `docs/e2e_tests/` 或 `docs/plans/2026-04-21-...md`**——这两个是未追踪、刻意不提交的
- **不要动 `docs/ethoinsight-architecture.html`** 的修改（上一轮遗留，不是本轮范围）
- **不要自己 push**——必须问用户
- **不要改 DeerFlow middleware 框架**（StreamBridge / 消息路由 / agent graph 结构）
- **不要回退到 InternalNotesMiddleware heading 白名单**——B3'' 已废弃，C4 用正面 prompt 是正解
- **不要把 bash 加回 SUBTASK_HIDDEN_TOOL_CALL_NAMES**——真正的修复是 ToolCall 组件在缺 description 时也返回 JSX（已修），把 bash 隐藏只是绕过 bug
- **不要 skip hooks、--amend、--no-verify**——除非用户明说

### 7.2 容易误判

- **code-executor 没有把 bash 设为 disallowed**，所以它可能调 bash 做 sandbox 诊断。真实 E2E 里我们没看到它调 bash，但理论上会。若新崩溃点和 bash 相关，优先怀疑 args shape。
- **前端 AIMessage.content 对 Sonnet 是数组**（`[{type:"thinking",thinking:"..."}]`），不是 string。`convertToSteps` 的 `includeText` 路径用 `typeof message.content === "string"` 守卫，Sonnet 不产生 text step——这是对的，别去"修"它。
- **Plan 里写的测试数字**（1654 / 1657 / 1660）是 backend 测试的累计预期，若未来加/改测试这些数字会漂移。用 `make test` 实测，不要硬比。
- **GLM-5.1 vs Claude Sonnet**：当前 lead 跑的是 claude-sonnet-4-6（见 langgraph.log: `model_name: claude-sonnet-4-6`）。user memory 说的"GLM-5.1 对'禁止 X'反向激活"未必在 Sonnet 上成立，但**正面指令对 Sonnet 也 OK**，保持就行。

### 7.3 已知的 pre-existing issue（不要顺手"修"）

- Pydantic `PydanticSerializationUnexpectedValue` warning（langgraph.log 里刷屏）— 测试和 E2E 都不受影响，上游问题
- `test_granular_tools.py::test_full_pipeline_writes_valid_handoff` skip — demo-data/ 结构变了，已用 `test_code_summary_per_subject.py` 覆盖同样断言
- `docs/ethoinsight-architecture.html` 的 M 标记 — 来自上一轮会话

---

## 8. 下一位 Agent 的第一步建议

**第一个动作**：读用户的新 message，判断他在哪一步。

**如果用户说"E2E 通过了"/"看到 Subject 3 了"/"语言一致"**：
1. 恭喜并问是否导出 E2E 对话到 `docs/e2e_tests/斑马鱼鱼群行为轨迹数据分析-fix5.md`（用户手动做）
2. 跑 `git log origin/dev..dev --oneline` 看真实待 push 数量
3. 问用户是否 `git push origin dev`

**如果用户贴出新的前端错误**：
1. 让用户贴 **DevTools Console 完整栈**（不只是 message）
2. 在本文档 §3.2 的前端文件里按栈顺序定位
3. 小 commit 修（一次一个 bug，别批量）

**如果用户说"data-analyst 还是只说'存在异常个体'"**：
1. 打开最新 `handoff_code_executor.json` 确认 per_subject 真的在里面
2. 打开 `handoff_data_analyst.json`（如果存在）看 outlier_findings 是不是空
3. 若 per_subject 在但 outlier_findings 空 → 加强 data-analyst prompt Step 6 措辞
4. 若 per_subject 不在 → C6 的 ethoinsight 改动没生效，查 `assess_and_handoff_tool` 是否在新 thread 里跑过

**如果用户问"拉上游吗"**：回答"不拉，先 push，再 dry-run 看"，跑：
```bash
cd /home/qiuyangwang/noldus-insight
./scripts/sync-deerflow.sh --dry-run
```

---

**签名**：Claude Opus 4.7 (1M context)，2026-04-21 会话 handoff
