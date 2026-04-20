# 2026-04-20 — EthoInsight 流水线重设计 (commit 1-6a+6) 交接

## 1. 当前任务目标

落地 `docs/plans/2026-04-20-ethoinsight-pipeline-redesign.md` 定义的 6 个 commit + 额外 1 个 commit (6a)，让 EthoInsight 从"流水线工厂"改造为"交互式助手"：
- 分析先行，APA 报告按需（ask_clarification 三选一触发）
- subagent 失败统一走 ask_clarification，不静默 bypass
- 修复 shoaling 群体指标"零方差"伪数据 bug（commit 2）
- write_file 对大 content 的鲁棒性（commit 3）
- 前端隐藏 lead 内部"记笔记"消息、过滤 I/O 工具调用、为 ask_clarification 选项渲染可点按钮

**完成标准**：全部 7 个 commit 都在 `dev` 分支落地，backend `make test` + frontend `pnpm check` + `pnpm build` 全绿。E2E 需要用户亲自跑一次确认 UX 符合预期。

**当前状态**：代码已全部 commit，**用户正在做最后一轮 E2E 验证 commit 6 和 6a 的实际效果**。

## 2. 当前进展

全部 7 个 commit 已在 `dev` 上落地（按时间正序）：

| Commit | Hash | 内容 | 测试增量 |
|---|---|---|---|
| 1 | `37104e26` | subagent handoff Pydantic schemas + 契约测试（`handoff_schemas.py` + `test_subagent_contracts.py`） | backend +14 |
| 2 | `dbe8f3b0` | ethoinsight 数据边界防御：移除 per_subject 的 IID/polarity 伪广播；新增 `group_level_metrics` + `data_quality_warnings` | ethoinsight +9 |
| 3 | `94b4f6ec` | write_file 显式 Pydantic `args_schema=_WriteFileArgs` + 8000 字符阈值 fail-fast 指引 append 分段 | backend +12 |
| 4 | `59e13c19` | 三个 subagent prompt 补 `<failure>` 节 + report-writer 加 `<write_file_chunking>`；`ethoinsight-analysis` skill 参考文档同步新字段 | backend +10 |
| 5 | `8f54c053` | **核心改动** lead prompt：流水线默认只到 data-analyst → ask_clarification 三选一（APA报告/不需要/解释XX）→ 选后才派 report-writer；失败统一澄清；新增"过程透明原则"+"分析结果呈现模板" | backend +14 |
| 6a | `ab89e5cf` | 新中间件 `InternalNotesMiddleware` 给 lead 以 `## 提取的关键上下文` 开头的 AI 消息打 `additional_kwargs.hide_from_ui=True`；`ArchivingSummarizationMiddleware._build_new_messages` override 给 "Here is a summary of the conversation to date:" HumanMessage 同样打标 | backend +16 |
| 6 | `76b7323d` | 前端 `ClarificationOptions` 选项按钮组件；`MessageList` 新 prop `onSelectClarificationOption` 接 `sendMessage`；`convertToSteps` 的 `HIDDEN_TOOL_CALL_NAMES` 过滤 I/O + ethoinsight 细粒度工具；i18n 新增 `clarification.chooseOption` / `orTypeCustom` | frontend lint/tsc/build 全绿 |

**测试累计**：
- backend: **1617 passed, 14 skipped**（基线 1551 + 本次 66）
- ethoinsight: **130 passed, 3 skipped**（基线 121 + 本次 9）
- frontend: `pnpm check`（eslint + tsc）全绿，`SKIP_ENV_VALIDATION=1 pnpm build` 成功

## 3. 关键上下文

**分支 & 工作目录**
- 工作分支是 **`dev`**（不是 `feature/etho-skills`，用户明确指示）
- 有两个**始终未跟踪**的文件不属于本次任务，不要动：`docs/e2e_tests/`、`docs/plans/2026-04-20-ethoinsight-pipeline-redesign.md`
- 仓库结构见 `/home/qiuyangwang/noldus-insight/CLAUDE.md`

**用户偏好 / 已做决策**
- 用户是 Noldus 软件开发工程师，偏好实用工程化方案
- 中文 commit message
- TDD 强制，每个新功能带单测
- **不要用 GLM，模型是 Sonnet**（E2E 跑的就是 Sonnet，我曾误判过是 GLM 的"记笔记"习惯）
- GLM-5.1 对"禁止X"反向激活——用正面指令（这次没直接触发，但记忆中有）
- commit 前后跑 `make test`（`packages/agent/backend/`），lint 只看自己新增/改动的文件
- 不要擅自 push
- 修内部行为优先用 **DeerFlow 已有基建**（例如 `additional_kwargs.hide_from_ui` 这个前端早已支持的机制），不要去改 LLM/prompt 层

**测试基线校准**
- commit 1 原本给 backend `test_subagent_contracts.py` 里写了 2 个跑真实 ethoinsight 工具的"集成测试"（使用 `tmp_path` 作为 workspace 路径）
- 用户指出：那不是真的集成测试，沙箱虚拟路径没走到，工具的 `thread_data=None` 让 `_resolve_virtual_path` 退化成 no-op
- 已 `git commit --amend` 把这 2 个集成测试删掉，只保留 14 个纯 schema 测试
- **因此 commit 1 的基线 baseline 从 1567 → 1565，后续数字都基于此**

## 4. 关键发现

### 4.1 commit 2 的"零方差"bug 真正根因

原先以为是"单鱼文件产出假 IID"，**错**。真正 bug 是：
- `compute_paradigm_metrics` L393-398（改动前）把一个群体级时序均值 `iid["mean_iid"].mean()` **广播**到每条鱼的 `per_subject[name]["mean_iid"]`
- 5 条鱼拿到相同标量 → 组间 std=0 → 下游 `compute_metrics_tool` 的 `quality_warnings` 报"zero variance"
- 修复：IID/polarity 彻底从 `per_subject` 移除，放到新字段 `group_level_metrics`；`mean_nnd` 保留 per-subject（NND 本质就是每条鱼独立的最近邻距离）
- 单鱼输入时 `group_level_metrics["mean_iid"]` 结构是 `{"applicable": false, "reason": "..."}`，这个形态由 `CodeExecutorHandoff.MetricStat`（commit 1 schema 就预留了）承载

### 4.2 commit 6a 的 `hide_from_ui` 基建

- 前端 `src/core/messages/utils.ts:327` 的 `isHiddenFromUIMessage` 检查 `message.additional_kwargs?.hide_from_ui === true`
- `groupMessages` L55 在渲染前 `continue` 跳过
- 这是 DeerFlow 早有的机制，`/workspace/agents/new/page.tsx:201` 已在使用
- 我们只是把"lead 内部状态笔记"+"summarization 注入的 HumanMessage"也接入这个机制
- **后端改动最小面**：新 middleware 只读 content 开头做精确白名单匹配，不改 LLM 行为，不删消息

### 4.3 E2E 流式"闪一下"限制

- `InternalNotesMiddleware` 在 `after_model` 末尾打标，前端已经流完内容才 hide
- 结果：用户会短暂看到状态块内容，然后它从渲染中消失
- 这是 commit 6a 已知限制，写在其 commit message 里
- 根治需要介入流式层（拦截首 chunk 看 heading 是否匹配），本次不做

### 4.4 Summarization 行为

- `config.yaml` 里 `summarization.enabled: true`、`summary_prompt: null` → 使用 LangChain 默认 `DEFAULT_SUMMARY_PROMPT`（"Context Extraction Assistant"）
- 该 prompt 会让模型输出 `## Extracted Context / ## 提取的关键上下文` 风格的 dump
- 压缩完注入的 `HumanMessage("Here is a summary of the conversation to date:\n\n...")` 被前端默认渲染为用户气泡
- commit 6a 在 `ArchivingSummarizationMiddleware._build_new_messages` override 处打标解决

### 4.5 `ask_clarification` options 的数据路径

- 模型调 `ask_clarification(question=..., options=[...])`
- `ClarificationMiddleware` 在 `wrap_tool_call` 里拦住，把 options 拼进 content 的 ToolMessage（格式化成 `1. opt / 2. opt / 3. opt`），跳 END
- 前端分组把 ToolMessage 拎到 `assistant:clarification` 分组里
- **选项列表只存在于触发 ToolMessage 的 AIMessage 的 `tool_calls[i].args.options`**，不在 ToolMessage 本身
- 故 commit 6 加了 `findToolCallArgs(toolCallId, messages)` helper，兼容 LLM 把 options 序列化成 JSON 字符串的情况

## 5. 未完成事项

按优先级排序：

### P0（用户正在做）

- **E2E 验证 commit 6 + 6a**：重跑斑马鱼 shoaling 5 条文件，观察
  1. lead 内部状态块（`## 提取的关键上下文`）不再作为正常消息出现（可能流式闪一下）
  2. summarization 消息不再以用户气泡形式出现
  3. Chain-of-Thought 里不再有 `read_file` / `write_file` / `bash` / `parse_trajectories` 等原始 tool 字样
  4. data-analyst 完成后 lead 呈现"分析结果 / 关键指标 / 关键洞察 / 数据质量提示"四段模板
  5. 下方出现 3 个可点按钮（APA报告 / 不需要 / 解释XX），点击等价于手打文本

### P1（E2E 发现问题时）

- 如果流式期的"闪一下"体验差：加一个 commit 介入 `StreamBridge` 或前端流式渲染层，检测到首 chunk 以 `## 提取的关键上下文` 开头就暂不 flush，等到 `after_model` 标签到达后统一决定是否展示
- 如果某个 subagent 失败路径没按预期走 ask_clarification：检查 `lead_agent/prompt.py` 对应节（失败处理规则的 data-analyst/report-writer 小节）的措辞

### P2（路线图外的待办，写在 plan doc 里）

- v0.1 之后再做：arena-grouping 的彻底重构（按 `(trial, arena)` 分组后再算 IID/polarity），commit 2 只做了最小修正
- `summary_prompt` 是否要自定义一个更简洁的（当前是 LangChain 默认，内容冗长）——commit 6a 只是把它从前端隐藏，模型自己还要消化那一大坨
- **不要做**的事情（YAGNI，plan doc §10 明确写了）：
  - `write_file_streaming` 新工具
  - 前端事件过滤白名单配置文件
  - SSE 通道流 thinking tokens
  - lead agent 自动重试机制

## 6. 建议接手路径

### 首先读这几个文件理解设计

1. `docs/plans/2026-04-20-ethoinsight-pipeline-redesign.md` — 整套设计 10 个章节
2. `/home/qiuyangwang/noldus-insight/CLAUDE.md` — 仓库全貌 + 开发规范
3. `packages/agent/backend/CLAUDE.md` — DeerFlow 后端架构（harness / app split、中间件链顺序）
4. `git log --oneline -10` 看 7 个 commit 的 message（每个 commit message 都写得很长、包含"为什么"和"怎么测"）

### 想了解某个具体改动

- **handoff schemas**: `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py` + `tests/test_subagent_contracts.py`
- **ethoinsight 边界防御**: `packages/ethoinsight/ethoinsight/metrics.py::compute_paradigm_metrics` + `packages/ethoinsight/tests/test_metrics.py` 的 `TestShoaling*` 三个类
- **write_file schema**: `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py` 搜 `_WriteFileArgs` / `WRITE_FILE_MAX_CONTENT_CHARS`
- **subagent 契约文档**: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/{code_executor,data_analyst,report_writer}.py` 看 `<failure>` 节
- **lead prompt 改造**: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` 搜 `过程透明原则` / `分析结果呈现模板` / `Step 3: 自然语言呈现` / `Step 4a: 派遣 report-writer` / `data-analyst 失败` / `report-writer 失败`
- **中间件 hide_from_ui**: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/internal_notes_middleware.py` + `archiving_summarization.py::_build_new_messages` override
- **前端按钮组**: `packages/agent/frontend/src/components/workspace/messages/clarification-options.tsx` + `message-list.tsx` 的 `assistant:clarification` 分支
- **前端工具过滤**: `packages/agent/frontend/src/components/workspace/messages/message-group.tsx` 搜 `HIDDEN_TOOL_CALL_NAMES`

### 跑测试

```bash
# backend
cd /home/qiuyangwang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make test           # 1617 passed, 14 skipped
make lint           # 只看自己改的文件；prompt.py 有 2 个 pre-existing 错误（F841 / E501）不是我们引入的

# ethoinsight
cd /home/qiuyangwang/noldus-insight/packages/ethoinsight
python -m pytest tests/ -q   # 130 passed, 3 skipped

# frontend
cd /home/qiuyangwang/noldus-insight/packages/agent/frontend
pnpm check
SKIP_ENV_VALIDATION=1 pnpm build
```

### 起服务做 E2E

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent
make dev          # 所有服务 → localhost:2026
```

然后上传 `demo-data/DemoData/斑马鱼鱼群行为/*.txt` 中的 5 条 Subject 1-5 文件，对照"分组：1 和 2 是对照组，3 4 5 是实验组"走一遍。

## 7. 风险与注意事项

### 容易跑偏的点

- **不要碰 lead prompt 去"禁止"模型写状态块**。用户明确说"那个状态可以有，但只给 AI 看"——commit 6a 就是这个路径，别回头改 prompt。
- **不要直接改 LangChain 上游的 `SummarizationMiddleware.DEFAULT_SUMMARY_PROMPT`**。我们只在 `_build_new_messages` 覆盖，内容照旧，只加 `hide_from_ui`。
- **commit 2 的改动是正向修正不是回归**。如果下游 `assess.py` 或 `statistics.py` 发现"少了 mean_iid"，那是期望行为——去 `group_level_metrics` 取；`per_subject` 里的 mean_iid/mean_polarity 被故意移除，存在即 bug。
- **不要同一轮重新派遣同一 subagent**。prompt 里明确写了"绝对禁止"，失败必须走 `ask_clarification`。

### 容易误判的点

- `_build_subagent_section(max_concurrent: int)` 是那个 ~500 行的大字符串生成函数，commit 5 改的很多内容都在它里面。`SYSTEM_PROMPT_TEMPLATE` 是另一个顶层字符串。`orchestration_guide` 是 `apply_prompt_template` 里动态拼的第三块。测试要 assert 在正确的块上——之前我有过一个失败就是把 Step 3 内容放错块导致测试误报。
- 中间件链顺序在 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py::_build_middlewares`，`InternalNotesMiddleware` 放在 `LoopDetectionMiddleware` 之后、`ClarificationMiddleware` 之前，因为它只是打标不影响路由，位置不敏感。
- 前端的 `groupMessages` 在 `src/core/messages/utils.ts`；`assistant:clarification` 分组特殊——它是 tool message 也同时 push 到 lastOpenGroup，之后又独立开一个分组（L67-77）。修改时别破坏这个双入口。

### 已验证不再继续的方向

- 让 lead 输出结构化 state dump 写到某个内部 tool（比如假想的 `update_planning_state`）——改动面太大，用户明确选择"中间件接住笔记"路径
- 根治流式期"闪一下"——本次不做，等用户 E2E 反馈
- 把 summarization 关掉——会丢长对话上下文，不是解决方向

## 8. 下一位 Agent 的第一步建议

**先别写代码**。优先做这几件事：

1. `git log --oneline -10` 看 7 个 commit message，它们是本次设计的精确记录
2. 问用户 E2E 跑得怎么样、哪几项清单项通过 / 哪项失败
3. 如果用户说"都好"→ 没有活要干，清理 `docs/e2e_tests/` 的临时文件问一下是否要归档，或等新任务
4. 如果用户说"ask_clarification 按钮没出现"→ 检查后端 `ask_clarification` 调用是否带了 `options=[...]` 数组参数（在 lead prompt 里检索 `options=[`）；再检查前端 `findToolCallArgs` 是否从 AIMessage 正确拿到
5. 如果用户说"流式闪一下太难受"→ 讨论要不要做 P1 的流式层介入（方案：在 `StreamBridge` 或前端 `useThreadStream` 检测首 chunk heading，延迟渲染到 `messages` state 更新）
6. 如果用户说"状态块还是漏了"→ 去看新 E2E 文件，看是什么新 heading 没被 `_INTERNAL_NOTE_HEADINGS` 白名单覆盖，补一条即可（精确匹配，保持窄白名单）

**今天累计 7 个 commit 已全部 commit 到 `dev`，还没 push。如果用户让 push，用 `git push origin dev` 而不是 force-push。**

绝对不要 rebase 或改写这 7 个 commit 的历史，它们代表了设计文档里每个决策的可追溯单元。
