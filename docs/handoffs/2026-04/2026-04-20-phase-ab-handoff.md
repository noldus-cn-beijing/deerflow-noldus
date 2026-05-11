# 2026-04-20 — Pipeline 重设计 E2E 修复（Phase A + B）交接

## 1. 当前任务目标

延续 [2026-04-20-pipeline-redesign-handoff.md](2026-04-20-pipeline-redesign-handoff.md)。
commit 1-6a 用户验收通过后，E2E 跑斑马鱼 shoaling 发现三个问题：

1. lead 在 AIMessage 正文里写 `## Extracted Context` 结构化 dump（commit 6a 的 heading 白名单只认中文，英文漏判）
2. summarization 注入的 HumanMessage 以用户气泡出现（commit 6a 的 `_build_new_messages` override 未完全生效）
3. ask_clarification 的 3 个选项被渲染两次（文字列表 + 按钮组件）

本轮**不用白名单**的方式治本：机制化分离"给自己看的思考 vs 给用户看的输出"。方案是 **Phase A（从上游 cherry-pick 4 条）+ Phase B（4 个治本 commit）**。

**完成标准**：8 个 commit 全部在 `dev` 上；backend `make test` + frontend `pnpm check` + `pnpm build` 全绿；用户跑 E2E 验证 UX。

**当前状态**：全部 15 个 commit（commit 1-6a + A1-A4 + B1-B4）已落地 `dev`，**E2E 验收已通过（2026-04-20 fix3，5/5 项清单全部通过）**。等用户确认 push。

## 2. 当前进展

全部 8 个 Phase A/B commit 已在 `dev` 上落地（按时间正序）：

| Commit | Hash | 内容 | 测试 |
|---|---|---|---|
| A1 | `dad3281c` | 补齐 title middleware think 标签剥离测试（代码上游 c91785dd 已 sync） | backend +4 |
| A2 | `89853e5b` | 补齐 ClarificationMiddleware options 类型强化测试（代码 ad6d934a 已 sync） | backend +9 |
| A3 | `b801ef71` | frontend 用户消息禁用 incomplete-markdown 解析（52718b0f） | frontend check 绿 |
| A4 | `6780e811` | 补齐 loop detection 稳定 key hash 测试（代码 c3170f22 已 sync） | backend +7 |
| B1 | `66dedf93` | **核心** 新增 `ThinkTagMiddleware` + 公用 `strip_think_tags` helper；think 标签内容搬到 `reasoning_content` | backend +23 |
| B2' | `5dc0af86` | **核心** summarization 文件化：摘要写 `{thread_dir}/user-data/workspace/conversation_summary.md`；注入轻量指针 HumanMessage（`hide_from_ui=True`）；失败 fallback 到旧行为 | backend +7, delete 1 过时测试 |
| B3'' | `e5db0562` | 新增 `compaction-recovery` skill（`skills/custom/`，custom 默认 enabled）；**废弃 `InternalNotesMiddleware`**（代码+测试一并删除） | backend -15 |
| B4 | `7abe6d11` | 前端剥离 clarification 的 1/2/3 文本（IM channels 不受影响——后端 content 不变） | frontend check/build 绿 |

**测试累计**：
- backend: **1651 passed, 14 skipped**（基线 1617 + 本次净增 34）
- frontend: `pnpm check` 绿，`SKIP_ENV_VALIDATION=1 pnpm build` 绿

## 3. 关键上下文

**分支 & 工作目录**
- `dev` 分支（未 push，15 个 commit 累计等待）
- 不跟踪的文件不要动：`docs/e2e_tests/`、`docs/handoffs/2026-04-20-pipeline-redesign-handoff.md`、`docs/plans/2026-04-20-ethoinsight-pipeline-redesign.md`

**用户偏好 / 已做决策**
- Noldus 软件工程师，偏好实用工程化方案；中文 commit；TDD 强制；模型是 Sonnet（不是 GLM）
- commit 前后跑 `make test`；不要擅自 push
- **"Text > Brain" 哲学**（用户引用）：状态在文件里，不在对话里。文件是因果连续性的介质
- **反对硬编码白名单**（"治标不治本"）——本轮把 InternalNotesMiddleware 彻底废弃
- 修内部行为**优先复用 DeerFlow 已有基建**（`additional_kwargs.reasoning_content`、`hide_from_ui`、`ExtensionsConfig.is_skill_enabled` custom default=True）
- Skill vs Middleware 职责拆分：Skill 负责**静态行为规则**，Middleware 负责**运行时事件通知**

**上游 sync 策略决策**
- 用户做了一次完整 sync（`c91785dd` / `ad6d934a` / `c3170f22` 等已经在代码里，只缺测试）
- 全量 cherry-pick 1897 条 commit 不做——本轮只挑 4 条跟当前问题直接相关的
- 未来追齐上游应独立开任务，不与 UX 修复混合

## 4. 关键发现

### 4.1 前端 reasoning 三条渠道（B1 的机制基础）
`extractReasoningContentFromMessage` (`core/messages/utils.ts:198`) 认三种来源：
1. `additional_kwargs.reasoning_content`（字段级，B1 用的是这条）
2. `content` 数组里带 `thinking` 字段（Anthropic 原生）
3. 正文里 think 内联标签（`THINK_TAG_RE`）

ThinkTagMiddleware 把源于渠道 3 的内容搬到渠道 1，跟下游 Reasoning 折叠块（`components/ai-elements/reasoning.tsx`）对接，零前端改动。

### 4.2 Summarization 污染源（B2' 的根因）
- LangChain 默认 summary prompt 是 `Context Extraction Assistant`，强制产出 `## Extracted Context\n**Task**:...` 格式
- 这个格式塞进 HumanMessage 回上下文 → lead 看到 → **模仿输出同样格式的 AI message**（这次 E2E 泄漏的根因）
- B2' 把摘要写文件、只注入"指针"HumanMessage（内容只一句话），lead 看不到 dump 内容就没样可模仿

### 4.3 IM channels 消费 ToolMessage.content（B4 的约束）
`app/channels/manager.py:149` 把 `tool_message.content` 当作要发给用户的文本（Feishu/Slack/Telegram）。IM 用户没按钮 UI，**必须靠 content 里的 1/2/3 列表**。所以 B4 不能改后端 format，只在**前端 Web 渲染**时剥离。

### 4.4 摘要文件的两层归档
- 原有：`{thread_dir}/archived_messages/{timestamp}.json` — **给前端刷新恢复历史用**（上游设计，保留）
- 新增 B2'：`{thread_dir}/user-data/workspace/conversation_summary.md` — **给 lead 读用**（虚拟路径 `/mnt/user-data/workspace/conversation_summary.md`），多次压缩追加到同一文件

### 4.5 已知残留风险（B3'' 废弃 InternalNotesMiddleware 的代价）
B1 + B2' 切断了**两个已知污染源**（think 标签、summarization 注入消息），但**不能保证 lead 不会主动在 AIMessage 正文里写 dump 格式**——commit 6a 的 heading 白名单是唯一能接住这种情况的防线，废弃后这条路暴露。

如果 E2E 发现 lead 仍然在正文写 `## Extracted Context` 这类 dump，**不要回退到白名单**（用户明确反对），应该：
- 检查 lead prompt 是否有某段诱导了这种输出
- 检查是否某个 skill（如 ethoinsight-planning）的示例里有这种格式
- 考虑 prompt 里加 positive 引导（"用自然语言回答，不要用 markdown heading + 粗体键值对的结构化列表"）

## 5. E2E 验收结果（2026-04-20 fix3）

### 测试场景
斑马鱼 shoaling 5 条文件，分组 Control=1,2 / Treatment=3,4,5。
- 后端日志：`packages/agent/logs/langgraph.log`
- 前端导出：`docs/e2e_tests/斑马鱼群行为轨迹数据分析-fix3.md`

### 5 项清单全部通过

| # | 检查项 | 结果 | 证据 |
|---|--------|------|------|
| 1 | 内部状态块（`## Extracted Context` / `## 提取的关键上下文`）不作为 AI 气泡出现 | 通过 | 前端导出无此类 AI 气泡；中间 `[系统]` 上下文里出现仅是历史记录 |
| 2 | summarization 注入消息不以用户气泡出现 | 通过 | 压缩消息显示为 `[系统] 前序对话已压缩并追加到 conversation_summary.md`，非用户气泡 |
| 3 | 原始工具名（read_file/write_file/bash/parse_trajectories）不在 UI 显示 | 通过 | 用户确认：前端实际界面上看不到这些工具名（导出中 `Tool: xxx` 是 Notion 格式标注） |
| 4 | data-analyst 完成后呈现四段模板 | 通过 | 分析结果 / 关键指标表格 / 关键洞察 3 条 bullet / 数据质量提示 3 条，完整呈现 |
| 5 | ask_clarification 按钮只出现一次 | 通过 | 后端日志 L238 `Intercepted clarification request`；前端按钮正常渲染，无文字列表重复 |

### 后端健康指标
- 无 error、无 failure、无 429（有 1 次 retry 但随即成功）
- 流水线完整：code-executor（7 轮 AI msg，~75s）→ data-analyst（3 轮 AI msg，~77s）→ ask_clarification 拦截跳 END
- 总耗时 ~4.5min（08:21:17 → 08:26:16）
- 两次 summarization archive 正常（文件写入 + 指针注入，无 fallback 路径触发）

## 6. 未完成事项

### P0（无——E2E 通过）

没有阻塞性事项。等用户决定：
- 是否 `git push origin dev`（15 个 commit 一起 push）
- 是否开始 P1/P2 待办
- 是否归档/清理 `docs/e2e_tests/` 下的临时测试文件

### P1（路线图外的待办）

- IM channels 里的 clarification 文字列表用户体验没改过（保留原行为）——未来如果 IM 也要按钮化，需要专门做
- 摘要文件的管理（清理、下载、前端展示入口）——可以做但非必需
- Phase C（独立任务）：真想追齐 1897 条上游 commit 时单独开任务
- compaction-recovery skill 的引导效果本次 E2E 没专门验证（lead 触发压缩后是否主动 `read_file` summary）——下次长对话 E2E 留意

### P2（已知残留 / 观察点）

- §4.5 的残留风险仍在：lead 如果在未来主动写 dump 格式，没有防线接住；靠 prompt/skill 层面的预防
- 流式期"闪一下"的根治未做（不影响本次 E2E，Sonnet 不倾向写这类 dump）

## 7. 建议接手路径

### 首先读这些文件理解现状

1. `docs/handoffs/2026-04-20-pipeline-redesign-handoff.md` — 前一轮（commit 1-6a）的交接
2. **本文件** — 本轮（A1-A4、B1-B4）的交接 + E2E 验收结果
3. `CLAUDE.md` / `packages/agent/backend/CLAUDE.md` / `packages/agent/frontend/CLAUDE.md` — 仓库结构 + 开发规范
4. `git log --oneline -20` 看 15 个 commit（每个 commit message 有完整的"为什么"和"怎么测"）

### 想了解某个具体改动

- **ThinkTagMiddleware 机制**: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/think_tag_middleware.py` + `tests/test_think_tag_middleware.py`
- **文件化 summarization**: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/archiving_summarization.py` + `tests/test_archiving_summarization.py`（搜 `_build_file_backed_messages`、`_write_summary_file`）
- **Compaction Recovery 行为规则**: `packages/agent/skills/custom/compaction-recovery/SKILL.md`
- **Clarification 去重**: `packages/agent/frontend/src/core/messages/utils.ts` 搜 `stripClarificationOptionsFromContent`；`message-list.tsx` L106-119
- **Middleware 链顺序**: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py::_build_middlewares`（注意 ThinkTagMiddleware 插在 LoopDetectionMiddleware 之后；InternalNotesMiddleware 已删除）

### 跑测试

```bash
# backend
cd /home/qiuyangwang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make test           # 1651 passed, 14 skipped

# frontend
cd /home/qiuyangwang/noldus-insight/packages/agent/frontend
pnpm check
SKIP_ENV_VALIDATION=1 pnpm build
```

### 起服务做 E2E

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent
make dev          # → localhost:2026
```

上传 `demo-data/DemoData/斑马鱼鱼群行为/*.txt` 中的 5 条 Subject 1-5 文件，分组"1 和 2 对照，3 4 5 实验"。

## 8. 风险与注意事项

### 容易跑偏的点

- **不要回退到白名单方案**。用户明确反对硬编码 heading 白名单。如果 lead 还在主动写 dump，走 prompt 引导或 skill 示例层面的修正
- **不要改 ToolMessage content 的后端格式**（B4 的设计原则）——IM channels 依赖它
- **不要把 compaction-recovery skill 搬到 public/**。`skills/public/` 是上游 DeerFlow 的 skill 目录，放那里会增加未来 sync 成本；放 `custom/` 对 custom 默认 enabled
- **`extensions_config.json` 是 gitignored**——不要尝试提交它的改动；compaction-recovery 默认 enabled 靠的是 `ExtensionsConfig.is_skill_enabled` 对 custom 的默认行为（L196 `return skill_category in ("public", "custom")`）

### 容易误判的点

- `_format_clarification_message` 上游 `ad6d934a` 的 defensive 代码**已经在我们仓库**（sync 带入），A2 只补了测试
- `title_middleware._strip_think_tags` 现在是 `strip_think_tags` 的 thin wrapper（B1 提公用函数后保留 instance method 是为了兼容既有测试）
- `_build_file_backed_messages` 有两条分支：正常路径写文件 + 只返回指针；fallback 路径保留**原 upstream 行为**（inline HumanMessage with "Here is a summary..."），**但仍打 hide_from_ui**。前端对这两种都隐藏，但内容上后者会让 lead 看到完整摘要（回到污染风险）——只在 `thread_id` 缺失或写文件失败时触发
- B2' 之前 commit 6a 的 `_build_new_messages` override 已经**完全删除**；如果有代码还调用它会报错（目前没有）
- 前端导出格式（Notion md）会自动在工具调用处加 `Tool: xxx` 标注，**这不代表前端 UI 显示了工具名**。判断工具过滤是否生效要看实际浏览器界面，不是看导出文件

### 已验证不再继续的方向

- Heading 白名单（InternalNotesMiddleware）—— 已废弃
- Summarization inline HumanMessage 打 `hide_from_ui` 作为长期方案 —— 只作为 fallback
- Prompt 层面"禁止输出 X"的约束 —— 不鲁棒（用户反对）

## 9. 下一位 Agent 的第一步建议

**先别写代码。**

1. `git log --oneline -20` 看 15 个 commit message
2. 读 §5 E2E 验收结果，确认上一轮结论
3. 问用户这轮想做什么：
   - push 到远程？→ `git push origin dev`（15 个 commit 一起 push）
   - 开 P1/P2 待办？→ 按优先级挑一个（推荐先做 compaction-recovery skill 的效果观察）
   - 归档 e2e_tests/？→ 问用户是否要 commit 到仓库或删除
   - 新任务？→ 按用户描述展开
4. **不要主动 push**——用户明确要求 push 前要确认

**15 个 commit（1-6a + A1-A4 + B1-B4）还没 push**。绝对不要 rebase 或改写历史，每个 commit 是一个可追溯的决策单元。
