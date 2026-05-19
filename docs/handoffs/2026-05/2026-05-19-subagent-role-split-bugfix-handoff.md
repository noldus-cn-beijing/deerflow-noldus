# 2026-05-19 Subagent Role Split — Bug Fix Handoff

> **状态**：W1-W21 完成合入 dev。W22 dogfood S1 中暴露 3 个 bug，正在诊断修复中。

## 当前任务目标

修复 W22 dogfood 中暴露的 3 个 bug，然后重新跑 dogfood S1/S2/S8 全绿。

## 已完成 (✅)

| 项目 | 状态 |
|------|------|
| W1-W21 代码全部落地，PR #8 merge 到 dev | ✅ |
| Backend 2604 tests pass (33 pre-existing 无关) | ✅ |
| Ethoinsight 267 tests pass (1 pre-existing 无关) | ✅ |
| Thread 5288a885 E2E 诊断 — 根因全部定位 | ✅ |
| Bug #1 根因(`_find_usage_recorder` TypeError)定位 | ✅ |
| Bug #2 根因(ArchivingSummarizationMiddleware 归档过敏 + summary 不记 handoff)定位 | ✅ |
| Bug #3(前端白框)待排查 | ⬜ |

## 三大 Bug 详情

### Bug #1: `_find_usage_recorder` — `TypeError: 'AsyncCallbackManager' object is not iterable`

**文件**: [task_tool.py:193-206](packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py#L193-L206)

**根因**: LangGraph runtime 的 `config["callbacks"]` 不保证是 `list`。在 async path 上是 `AsyncCallbackManager` 实例(无 `__iter__`)，而当前代码直接 `for cb in callbacks:`。

**来源**: 上游 commit `58abb0b0 sync(deerflow): A-9 subagent token usage 流式到 header` 引入，不是 W1-W21 改的。

**已探明的 CallbackManager API**:
```
AsyncCallbackManager 有:
  .handlers: list           ← 可 iter
  .inheritable_handlers: list  ← 可 iter
  无 __iter__
  
LangChain runtime config["callbacks"] 可能是:
  A) list[BaseCallbackHandler]
  B) AsyncCallbackManager / CallbackManager
```

**修复方向**(1 行变更):
```python
callbacks = config.get("callbacks", [])
# 如果 callbacks 是 BaseCallbackManager，取 .handlers
if hasattr(callbacks, 'handlers'):
    callbacks = callbacks.handlers
if not callbacks:
    return None
for cb in callbacks:
    ...
```

**连锁影响**: 该 TypeError 发生在 task tool 返回值的 `_report_subagent_usage` 调用链中，导致 subagent 虽然跑完了(产物都落盘了)但返回值是 error → lead 收到 error → 加上 archiving 同时把 ToolMessage 归档 → lead 失忆 → 重派。

### Bug #2: ArchivingSummarizationMiddleware 阀值过敏

**文件**: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/archiving_summarization.py`

**现象**: thread 5288a885 一次会话内触发了 **2 次归档**:
1. `03:46:09` 归档 7 条消息(summary 仅 58 字符 — "用户要分析 EPM 数据")
2. `03:49:06` 归档 6 条消息 — 就在 code-executor #1 完成瞬间，把 ToolMessage 也归档了

**关键证据 — 03:49:06 归档的 payload**:
```json
{"type":"human","data":{"content":"[系统] 前序对话已压缩并追加到 conversation_summary.md..."}}
```
然后 lead 下一条 message 说"上传目录目前是空的"——它完全不记得已派过 code-executor。

**summary 缺失**: 两次 summary 都没记下"code-executor 已 dispatched + completed + handoff 已写盘"。lead 在 compaction 后只能靠 summary 重构上下文，但 summary 里没这些关键信息。

**修复方向**:
- A) 调高 trigger threshold(token 数 / message 数)
- B) keep policy 保证最近 N 条 tool_call + ToolMessage 不归档
- C) summary LLM prompt 加指令"必须记下已 dispatched 的 subagent 名 + 完成状态"

建议 A + C 组合(最小改动 + 最高 leverage)。

### Bug #3: 前端白框压缩文字

**现象**: 聊天界面出现白框，压缩输出文字，用户看不到部分内容。同时向上翻聊天历史发现之前的聊天记录也消失了。

**Bug #2 的解释**: "向上翻看不到上文"是 ArchivingSummarizationMiddleware 的设计 — 被归档的消息从 LangGraph thread state 中移除写入 `archived_messages/` JSON 文件，前端 hist API 返回的 messages 不含已归档部分。这是 **后端 behavior** 非前端 bug。

**但是** "白框压缩文字" 是独立前端 UI bug — 可能是消息气泡容器的 `max-height` / `overflow: hidden` 或 streamdown render 块样式问题。**需要看前端代码**。

## 关键文件路径速查

| 文件 | 用途 |
|------|------|
| `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py:193-206` | Bug #1 所在 |
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/archiving_summarization.py` | Bug #2 所在 |
| `packages/agent/frontend/` (Next.js) | Bug #3 前端 |
| `packages/agent/backend/.deer-flow/users/*/threads/5288a885-97cd-4ead-b2b9-23ab4c31a8eb/` | thread 本地记录 — workspace / archived_messages / conversation_summary.md |
| `packages/agent/logs/langgraph.log` | 后端完整日志 |
| `docs/superpowers/specs/2026-05-18-subagent-role-split-capability-exposure-spec.md` | 当前 PR 的 spec |
| `docs/superpowers/plans/2026-05-18-subagent-role-split-implementation-plan.md` | 实施 plan(W22 部分) |

## Worktree 状态

- **Worktree 路径**: `/home/wangqiuyang/noldus-insight/.claude/worktrees/subagent-role-split-impl`
- **Branch**: `worktree-subagent-role-split-impl`
- **Base**: dev HEAD `31275138` (仍在此 commit，因为 W1-W21 是 PR merge 进 dev 的，本 worktree 未更新到 dev 最新 `cd5cceca`)
- **⚠ 需要 `git pull` 或 `git rebase dev`** 把 worktree 更新到包含 W1-W21 的最新 dev

## 下一步(建议顺序)

1. **先更新 worktree**: `cd /home/wangqiuyang/noldus-insight/.claude/worktrees/subagent-role-split-impl && git rebase dev`
2. **修复 Bug #1**: 改 task_tool.py `_find_usage_recorder` 防御 `AsyncCallbackManager` — 1 行变更 + TDD
3. **修复 Bug #2**: 调 ArchivingSummarizationMiddleware trigger/keep + summary prompt — 可能 2-3 行
4. **排查 Bug #3**: 看前端 chat bubble / markdown render 组件，定位白框样式
5. **重新 dogfood S1**: 单被试 EPM + "再画几个图"，确认 3 个 bug 不再复现
6. **继续 S2 + S8**: 按 plan W22 验收标准跑
7. **全绿后 push + PR 合 dev**

## 关键 Context

- **模型已切 `deepseek-v4-pro[1m]`** — 用户在分析过程中切换了模型
- **dev 服务目前运行中** (`make dev`)，可在 `localhost:2026` 访问
- **Thread 5288a885 本地记录完整**，用于复现
- Bug #1 修复后必须确认 `_report_subagent_usage` 在两种 callbacks 形态下都正常(是 list 时走 for，是 CallbackManager 时走 `.handlers`)
- Bug #2 的 archiving_middleware 设计意图正确(省 token)，只是触发太激进 + summary 不够 informative。**不要完全禁用它** — 只调参数

## 风险

- **不要动 `Ev19TemplateGuardrailProvider` / `HandoffIsolationProvider`** — 它们是正交组件
- **不要改 `ArchivingSummarizationMiddleware` 的核心逻辑** — 只改 trigger / keep 参数或 summary prompt
- Bug #1 修复后要跑完整 test suite 确保不影响 token usage 统计
- 前端白框可能是 streamdown 2.5.0 或 markdown render 的已知问题，检查 frontend README / recent commits
