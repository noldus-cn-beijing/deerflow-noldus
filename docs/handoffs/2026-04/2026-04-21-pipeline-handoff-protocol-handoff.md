# 2026-04-21 前端卡死修复 + handoff 协议重构 交接

> 给下一位 Agent：本文件 + 上一份 [2026-04-21-subtask-visibility-handoff.md](2026-04-21-subtask-visibility-handoff.md) 是完整上下文。这一份是"卡死修复 + pipeline 协议切换"之后的最新状态。

---

## 1. 当前任务目标

承接 fix4 → fix5 两轮 E2E 暴露的问题，这一轮做了两件事：

1. **前端卡死修复**（用户报告"Page Unresponsive"）—— render-time setState 循环
2. **pipeline 协议重构**（用户提出"handoff JSON 是文档交接第一标准"的架构原则）—— subagent 之间从混杂的 markdown + Lead 中转，统一为纯 JSON 链

E2E 目标仍然是 shoaling 5 文件完整分析跑到底，不被 compaction 打断。

---

## 2. 当前进展

**分支**：`dev`，从 `ec62f918 wrap up for 420` 起**累计 14 commit 待 push**（`origin/dev..dev`）。

**本轮新增的 6 个 commit**（在原 8 个 C1-C6 + 两 fix 之上）：

| Commit | 范围 | 说明 |
|---|---|---|
| `3c51a235` fix | 前端 | thread 切换 setSubtasks 挪进 useEffect（render-time setState warning） |
| `d735fef3` fix | 前端 | `useUpdateSubtask` 加 isSameSubtask 短路（挡住幂等调用的 setState 风暴） |
| `81f67bd7` fix | 前端 | `useUpdateSubtask` 恢复双路径：render-time mutate（不 setTasks）/ SSE-time setState——这是真正根除卡死的版本 |
| `6a6fa91e` fix | 前端 | 导出 markdown/JSON 过滤 `hide_from_ui` 消息（compaction pointer 不该出现在导出文件） |
| `f45705cc` SKILL | compaction-recovery | SKILL.md 加"读完摘要看到 pending items 直接 `task` 推进"小节 + 章节重排 |
| `6fa6962a` pipeline | 后端 + skill + 测试 | **协议重构**：data-analyst ↔ report-writer 中间交付物从 `analysis_summary.md` 切到 `handoff_data_analyst.json`；`max_turns` 6→12；整个 pipeline 变成纯 JSON 链 |
| `1b605d35` subagent | 后端 + 测试 | data-analyst / report-writer prompt 加 `<json_writing>` 节（JSON 字符串转义约束） |

**测试**：
- backend: **1664 passed, 14 skipped**（baseline 1660 → 本轮 +4 新断言）
- ethoinsight: 不变（本轮未动）
- 前端：`pnpm check` + `pnpm build` 全绿

**E2E 验证（三轮）**：
- **fix5 (14:10 UTC 06:10)**：前端卡死、Lead 被 compaction 打断停下问"有什么新需求"
- **fix1 (14:11 UTC 06:33)**：卡死修了，但 data-analyst `max_turns=6` 被强杀，Lead 凭印象 fallback 补分析
- **fix2 (14:33 UTC 07:30)**：架构层完全跑通——Lead 自主派 code-executor → data-analyst，两个 handoff JSON 都落盘，Lead 调 `ask_clarification` 三选一等用户。**用户看到 question 和 options**。唯一瑕疵：`handoff_data_analyst.json` 有未转义双引号导致不是合法 JSON（`1b605d35` 已加 prompt 约束防止下次复发）

**下一步验证**：用户在 fix2 的 thread 里点击三选一（要 APA 报告），看 report-writer 分支能否跑通（会触发 `1b605d35` 的 prompt 约束生效 + handoff_report_writer.json 生成）。

---

## 3. 关键上下文

### 3.1 分支与仓库

- 仓库根：`/home/qiuyangwang/noldus-insight`（WSL2）
- 分支：`dev`
- **14 commit 领先 `origin/dev`**——用户尚未决定 push（按 handoff §7.1 不要自己 push）
- 工作区未追踪（按项目惯例不要 git add）：
  - `docs/e2e_tests/`（E2E 导出产物）
  - `docs/plans/2026-04-21-subtask-visibility-and-language.md`
  - `docs/handoffs/2026-04-21-subtask-visibility-handoff.md`（上一份 handoff）
- 工作区一直存在的 M：`docs/ethoinsight-architecture.html`（上一轮遗留，不是本轮范围）

### 3.2 核心文件路径

**本轮协议重构的主战场**（`6fa6962a` + `1b605d35`）：

- [packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py](packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py) — 新 contract / workflow / max_turns=12 / `<json_writing>` 节
- [packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py](packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py) — 输入改为两个 handoff JSON，输出加 `handoff_report_writer.json`
- [packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py) — `DataAnalystHandoff` 删除 `analysis_summary_path` 字段
- [packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) — 删除"Lead write_file analysis_summary.md"中转步骤（多处）；subagent 交付表更新
- `packages/agent/skills/custom/ethoinsight-planning/references/{failure-recovery,intent-classification,quality-gates}.md` — 3 个 reference 同步

**本轮前端修复**：

- [packages/agent/frontend/src/core/tasks/context.tsx](packages/agent/frontend/src/core/tasks/context.tsx) — `useUpdateSubtask` **双路径**：有 `latestMessage` 才 setTasks，render-time 调用只 mutate
- [packages/agent/frontend/src/core/threads/hooks.ts:310-315](packages/agent/frontend/src/core/threads/hooks.ts#L310-L315) — thread 切换 useEffect 清空 subtasks
- [packages/agent/frontend/src/core/threads/export.ts](packages/agent/frontend/src/core/threads/export.ts) — 导出时过滤 `hide_from_ui`

**SKILL**：

- [packages/agent/skills/custom/compaction-recovery/SKILL.md](packages/agent/skills/custom/compaction-recovery/SKILL.md) — "读完摘要之后：无缝续跑未完成的工作"章节

**测试**：

- `packages/agent/backend/tests/test_subagent_prompt_clarity.py` — 加了 `test_warns_about_json_string_escaping` × 2、`test_declares_handoff_output` × 2、负面断言防 `analysis_summary.md` 回归
- `packages/agent/backend/tests/test_subagent_contracts.py:TestDataAnalystHandoffSchema` — 重写（不再要求 `analysis_summary_path`）
- `packages/agent/backend/tests/test_data_analyst_insight_contract.py` — 去掉 `analysis_summary_path=...` 参数

### 3.3 Pipeline 的新文件流

```
handoff_code_executor.json → handoff_data_analyst.json → handoff_report_writer.json
       (code-executor)          (data-analyst)              (report-writer)
```

每个 subagent 完工 = 一个 handoff JSON 落盘。Lead 不在中间 write_file markdown；报告（`report.md`）只有 report-writer 在最终用户请求时才产。

这是架构原则，用户原话："**handoff json 是需要的！每一个 subagent 都应该在完成自己的任务 scope 之后，写一个 handoff，因为文档永远是交接的第一标准。**"

### 3.4 用户交互的若干关键决策

会话中用户纠正了几次我走偏的方向，记下来：

1. **Lead 的职责边界**：Lead 只负责"拆需求 → 派遣 → 或自己答简单问题"。**不该承担 pipeline 编排**（我一度想给 Lead prompt 加"ls workspace → 对照表"状态机，用户否决——那是让 Lead 降级成执行器）
2. **subagent 交付物必须是 handoff JSON**：我一度想让 data-analyst 只返回 AIMessage 不写文件，用户否决——文档是交接第一标准，不能丢
3. **data-analyst 不写正规报告**：只写 handoff JSON；markdown 报告是 report-writer 的职责，且只在用户明确要求时才生成
4. **compaction 应该是后台机制**：用户看不见 pointer；Lead 看到 pending items 就默默推进，不对用户汇报"对话已压缩"

### 3.5 常用命令

```bash
# 启动 / 停止
cd /home/qiuyangwang/noldus-insight/packages/agent
make dev      # localhost:2026
make stop

# 后端测试
cd packages/agent/backend && source .venv/bin/activate
make test     # 预期 1664 passed, 14 skipped

# 前端
cd packages/agent/frontend
pnpm check
SKIP_ENV_VALIDATION=1 pnpm build

# 单独跑本轮新测试
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_subagent_prompt_clarity.py -v
```

### 3.6 fix2 E2E 产物位置（用户正在交互的 thread）

```
packages/agent/backend/.deer-flow/threads/b50eba41-364b-4fd7-87f0-37cce6755492/
├── user-data/workspace/
│   ├── handoff_code_executor.json   ← code-executor 产出 ✓
│   ├── handoff_data_analyst.json    ← data-analyst 产出 ✓（JSON 语法有未转义引号 bug，此 thread 已定格）
│   ├── conversation_summary.md      ← compaction 摘要
│   ├── code_summary.json            ← code-executor 的精简快照
│   └── ...
└── archived_messages/
    └── 20260421T073351634895.json   ← 压缩归档的 7 条消息
```

---

## 4. 关键发现

### 4.1 前端卡死真正的根因链

**并不是我最初以为的"render 里 setState 抛 warning"**，而是 C1 (`4927baea`) 把 `useUpdateSubtask` 从"有 latestMessage 才 setTasks"改成**无条件 setTasks**——render 里同步调的多个 `updateSubtask(...)` 立即触发 setState → Context 变 → 整棵重渲染 → 再 render 再调——**紧密循环直到浏览器冻结**。

修法经历了三代：
- `d735fef3` 加浅比较短路（挡住幂等调用，但首次创建 task 时仍 setTasks，加上 useMemo+useEffect 依赖 `[messages]` 流式每 chunk 重算，叠加仍然重）
- `81f67bd7` **恢复昨天 baseline 的双路径模式**——有 latestMessage（SSE 事件路径）才 setTasks，无 latestMessage（render-time 路径）只 mutate tasks[id]——**根治**

教训：**C1 commit message 说"修复了 status-only 更新不触发 setTasks 的 bug"——那不是 bug，是刻意保护**。render 里调 `updateSubtask` 是架构遗留，改的时候要尊重它"只 mutate 不 setState"的隐式契约。

### 4.2 max_turns 的影响

fix1 E2E 时 data-analyst `max_turns=6` 不够——它 workflow 有 9 步，5 分钟跑完 6 个 AI turn 就被强杀。`6fa6962a` 把 workflow 压到 4 步 + `max_turns=12`；fix2 实测 data-analyst 1:15 内跑完，没被杀。

其他 subagent 目前：code-executor=12, report-writer=15, knowledge-assistant=6。data-analyst=12 对齐 code-executor 合理。

### 4.3 handoff JSON 的语法风险

fix2 的 `handoff_data_analyst.json` 因模型在字符串里塞了未转义的半角双引号（`"统计功效不足"`）而不是合法 JSON。`1b605d35` 加 `<json_writing>` prompt 约束顶住了下一轮。

**但这个问题的根治方案**是：给 subagent 一个专用的 `write_json_file(path, data_dict)` 工具——工具内部 `json.dumps`，模型只填字段不拼字符串。列入"下一轮工作"（§5.3）。

### 4.4 compaction 触发条件

[config.yaml](packages/agent/config.yaml) 里 `summarization.trigger: messages=15, keep=8`。不是按 token，是按消息条数。shoaling 分析 pipeline 大约 15-20 条消息就一定会触发。

曾经讨论过调高 trigger（到 30 / 40）降低触发概率，但**Qwen3-8B 原生 context 仅 32K**——当前触发点已经接近边界，不能调高（调高逼迫开 YaRN，会损失短文本性能）。正确方向是让 compaction 更无痛：SKILL 教 Lead 读完摘要继续、pipeline 状态靠 handoff 文件而非对话记忆——都已做。

### 4.5 Sonnet vs Qwen3-8B

当前跑 Sonnet 4.6（见 [config.yaml:models[0]]）。架构改动的深层动机是为未来 Qwen3-8B：
- 结构化 JSON handoff 比自然语言 markdown 对小模型友好
- 文件作为交接优先，不依赖小模型的对话记忆判断
- prompt 的正面指令（"用 Y 替代 X"）比"禁止 X"对小模型有效（见 user memory `feedback_positive_prompting.md`）

---

## 5. 未完成事项（按优先级）

### 5.1 【高】用户继续 fix2 E2E 验证 report-writer 分支

用户已在 fix2 thread 看到三选一 `ask_clarification`（thread_id = `b50eba41-364b-4fd7-87f0-37cce6755492`）。**还没点**。下一步：

- 让用户点"需要 APA 格式报告"
- 观察 report-writer 是否：
  1. 成功 read `handoff_data_analyst.json`（尽管 JSON 语法有 bug——但因为那是在这次 thread 被冻结的状态，**这个 thread 里重新派 report-writer 会读坏 JSON**——建议用户**新建 thread 重跑一次完整的**，让 data-analyst 在 `1b605d35` 约束下写出合法 JSON，再走到 report-writer）
  2. 产出 `report.md`
  3. 产出 `handoff_report_writer.json`（新字段验证）
- 最终 Lead `present_files` 展示报告 + 图表

**如果用户说"直接在 fix2 thread 点三选一"**：提醒他 data-analyst 的 JSON 坏了，report-writer 可能崩——更好的做法是**新开 thread 重跑**。

### 5.2 【中】导出并归档 fix2 之后的新 E2E

完整跑完后用户导出到：
```
docs/e2e_tests/Fish Shoaling Behavior Trajectory Analysis -fix3.md
```
由用户手动导，**不要 git add**（`docs/e2e_tests/` 未追踪）。

### 5.3 【中】给 subagent 加 `write_json_file` 专用工具

**动机**：prompt 约束（`1b605d35`）是顶住，不是根治。模型写 JSON 字符串拼 `,` / `"` / `{}` 本来就不可靠，小模型更甚。

**设计**：
- 新工具 `write_json_file(path, data)`，参数是字典（不是字符串），工具内 `json.dumps(data, ensure_ascii=False, indent=2)` 落盘
- data-analyst / report-writer prompt 改为"用 `write_json_file` 填字段，不要自己拼 JSON 字符串"
- 保留 `write_file` 给 markdown / py / txt

**位置**：[packages/agent/backend/packages/harness/deerflow/sandbox/tools.py](packages/agent/backend/packages/harness/deerflow/sandbox/tools.py) 附近。测试 + subagent prompt 联动更新。

**时机**：等 fix3 E2E 验证完，下一轮迭代做。

### 5.4 【中】Push 决策

用户曾问过"要不要拉上游"，当时答"不拉，先 push"。现在 14 commit 待 push，等用户发话：

```bash
cd /home/qiuyangwang/noldus-insight
git log origin/dev..dev --oneline    # 确认 14
git push origin dev                   # 用户指令后
```

### 5.5 【低】Prompt 有效性继续打磨

如果未来 E2E 再出现 Lead 在 compaction 后不推进 / subagent 语言混用等问题：
- 看 `langgraph.log` 启动时 system prompt dump 确认 phrase 真的渲染到
- 考虑把 compaction-recovery SKILL 的"post-compaction 续跑"提升到 lead_agent/prompt.py 的 `<critical_reminders>` 级别

### 5.6 【低】DeerFlow 上游同步

**不要现在做**。用户 push 后、下一轮 E2E 完成后再跑：
```bash
./scripts/sync-deerflow.sh --dry-run
```

---

## 6. 建议接手路径

**第一步：读用户当前状态**
```bash
cd /home/qiuyangwang/noldus-insight
git log --oneline dev -5              # 确认 HEAD = 1b605d35 或更新
git status -s                         # 预期 M docs/ethoinsight-architecture.html + 若干未追踪
ls packages/agent/backend/.deer-flow/threads/ | tail -5  # 看 thread 数
```

**第二步：核心文件快速过一遍**
1. 本文档
2. 上一份 handoff: `docs/handoffs/2026-04-21-subtask-visibility-handoff.md`
3. `/home/qiuyangwang/noldus-insight/CLAUDE.md`（项目全景）
4. `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`（最新 contract）
5. `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`（最新 contract）
6. `packages/agent/skills/custom/compaction-recovery/SKILL.md`

**第三步：根据用户反馈分支**

- **"fix2 thread 跑通了 / 不通了"** → §5.1 / §5.2
- **"我要 push 了"** → §5.4，告知确切数字
- **"能不能让 subagent 不再搞坏 JSON"** → §5.3 设计 `write_json_file` 工具
- **新问题 / 新崩溃** → 先问 DevTools Console 完整栈 / 后端 traceback，不要盲改

---

## 7. 风险与注意事项

### 7.1 不要做的事

- **不要 git add `docs/e2e_tests/`、`docs/plans/2026-04-21-*.md`、`docs/handoffs/2026-04-21-*.md`**——这些未追踪是故意的
- **不要动 `docs/ethoinsight-architecture.html` 的 M 标记**（上一轮遗留）
- **不要自己 push**——必须问用户（§5.4）
- **不要改 DeerFlow middleware 框架**（StreamBridge / 消息路由 / agent graph 结构）
- **不要在 Lead prompt 里加"pipeline 状态机对照表"**——这违反用户的"Lead 只做决策不做执行"原则
- **不要删 handoff JSON**——用户的架构原则："**文档永远是交接的第一标准**"
- **不要把 data-analyst 输出退回 analysis_summary.md**——`1b605d35` 加的负面断言 (`test_subagent_prompt_clarity`) 会抓住回归
- **不要在 render 里直接调 setTasks**——`81f67bd7` commit message 详细解释了为什么必须保持双路径

### 7.2 容易误判

- **前端 `Cannot update a component while rendering` warning 本身不是 bug**——是 C1 后暴露出来的。`81f67bd7` 已经根治这条 warning（因为 render-time 不再调 setTasks）。如果再看到类似 warning，先检查有没有回归到 C1 的"无条件 setTasks"
- **handoff_data_analyst.json 在 fix2 thread 里是坏的**（未转义引号）——但 `1b605d35` 的 prompt 约束只对未来运行有效，那个文件本身不会被回溯修复。如果用户坚持在该 thread 继续跑 report-writer，需要提醒他可能会崩
- **Lead 在 compaction 后呈现"## 提取的关键上下文"类 markdown heading 不是 bug**——那是 `ask_clarification` 前的 context 呈现，给用户看当前状态作为三选一的依据。和 C4/C5 想压制的"英文结构化 dump"是不同情境
- **max_turns 的计数**：data-analyst 的 `max_turns=12` 指的是**subagent 自己产生的 AI message 数**，不是 tool call 数也不是总 LLM 调用数。一次 "read_file → 思考 → write_file → 思考 → 最终消息" 大约 3-4 个 AI message
- **测试数字**（1660 / 1662 / 1664）会漂移，用 `make test` 实测
- **compaction trigger=15 messages 的"messages"含 system prompt 吗？** 不确定，以 `archived_messages/*.json` 的 `message_count=7` + `keep=8` = 15 为事实经验

### 7.3 已知 pre-existing issue（不要顺手修）

- Pydantic `PydanticSerializationUnexpectedValue` warning —— 上游问题
- `test_granular_tools.py::test_full_pipeline_writes_valid_handoff` skip —— demo-data 结构变了
- `docs/ethoinsight-architecture.html` 的 M 标记 —— 上一轮遗留

---

## 8. 下一位 Agent 的第一步建议

**读用户的第一条消息，判断在哪一步。**

**如果用户说"fix3 E2E 跑通了 / 走到 report-writer 了"**：
1. 恭喜 + 问是否导出到 `docs/e2e_tests/`
2. `git log origin/dev..dev --oneline | wc -l` 看实际待 push 数
3. 问是否 `git push origin dev`
4. push 后可问是否 `./scripts/sync-deerflow.sh --dry-run`

**如果用户说"点三选一后崩了 / report-writer 读 JSON 报错"**：
1. 说明 fix2 thread 的 handoff_data_analyst.json 本身就坏了（§4.3 / §7.2）
2. 建议**新开 thread 重跑**——让 `1b605d35` 的 prompt 约束生效
3. 如果新 thread 还是坏 JSON → 进入 §5.3 设计 `write_json_file` 工具

**如果用户说"Lead 在 compaction 后还是没继续 / 又停下来问了"**：
1. 看 `langgraph.log` 确认 compaction 时点和 Lead 的 LLM 调用
2. 确认 Lead 有没有 `read_file conversation_summary.md`（应该有）
3. 可能需要把"post-compaction 续跑"规则从 SKILL 提升到 `<critical_reminders>`

**如果用户说"前端又卡了"**：
1. 让用户贴 DevTools → Performance profile 或 Console 完整栈（不是只贴第一行）
2. 先检查有没有回归 `81f67bd7` 的双路径模式
3. 如果是新的 render loop，按 `81f67bd7` commit message 的诊断方法定位

**如果用户问"要 push 上游吗"**：答"**不拉，先 push，再 dry-run**"，按 §5.4 → §5.6 顺序

---

**签名**：Claude Opus 4.7 (1M context)，2026-04-21 会话 handoff（dev/1b605d35）
