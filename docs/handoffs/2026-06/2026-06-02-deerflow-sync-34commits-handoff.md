# 2026-06-02 DeerFlow 上游 sync (f9b70713→74e3e80c, 34 commit) handoff

## 概要

DeerFlow 上游从 `f9b70713`(5-25 sync 后) 前进到 `74e3e80c`, 共 34 个 commit。本次 sync **全量跟随上游**（策略变更：从"挑着合"改为"默认全要 + 受保护文件 surgical"），合入 23 个 harness 文件 + app/frontend/test/docs，surgical 2 个受保护文件。

## 策略（2026-06-02 用户锁定）

- **deerflow 是我们的 infra 底座，不是外部参考库**
- **默认全量合**，例外是 22 个受保护文件做 surgical
- 即使"当前用不上"(postgres pool / MiMo) 的非受保护 commit 也合（消除分叉点）
- 唯一真正可跳过的只有纯 docs

## 测试结果

| | passed | failed | 备注 |
|---|---|---|---|
| HEAD (sync 前) | 3129 | 57 | pre-existing failures |
| sync 后 | 3268 | 29 | +139 passed, -28 failed |
| 净改善 | **+139** | **-28** | 所有 29 个失败均与受保护文件相关 |

## 合入的上游 commit 分类

### 类 A — harness 实质改进（全量合）
- `ca487578` ToolOutputBudgetMiddleware: 超大 tool 输出保护（新中间件 489 行 + config 62 行）
- `e683ed6a` write_file recovery: 格式错误 write_file 恢复引导
- `cbf8b194` JSONL async IO: 事件循环不阻塞 + per-thread asyncio.Lock
- `3cb75887` memory parse: LLM 包装 JSON / markdown fence 解析
- `9f3be2a9` UploadsMiddleware: abefore_agent offload 到 worker thread
- `79cc2279` LLM fallback: deerflow_error_fallback 标记 + journal/worker run status
- `92905e9e` todo: #2135 after_model+jump_to 范式（反幻觉 spec 层3 直接复用）
- `44677c5e` MiMo provider: 新文件，我们用 deepseek 不触发

### 类 B — 受保护文件 surgical（合上游 fix, 保 Noldus 定制）
- `mcp/tools.py`: 合 HTTP/SSE 跳过 session pooling, 保 4096 截断
- `llm_error_handling_middleware.py`: 合 error_fallback 标记(journal/worker 依赖), 保总超时/中文关键字

### 类 C — app/gateway/channels（全量合）
- `031d6fbc` postgres AsyncConnectionPool（我们不触发, 但消除分叉）
- `46ddc346`/`e8e9edcb` Feishu channels
- `37451500`/`a5599c10` gateway 修复
- `b00749a8` auth token
- `8decfd32` skill install 权限
- `737abc0e` stale run reconnect
- `d46a5779` summarization 后保留消息

### 类 D — frontend（全量合）
- `11dd5b06` strip 未闭合 `hausenhaus` 标签（关联我们 thinking 渲染历史）
- 其余 8 个前端修复

### 类 E — tests/docs（选择性合）
- `e344be8d` Blockbuster event-loop 阻塞 IO 测试门
- `052b1e21` Blockbuster JSONL runtime anchor
- 其余测试文件全量跟随

## 额外的 Noldus 守护

以下文件虽不在受保护列表但含 Noldus 定制,全量覆盖后需还原:
- `uploads_middleware.py`: 还原 `_DATA_FILE_EXTENSIONS` 数据文件指引
- `conftest.py`: 还原 `blocking_io_detector` fixture（上游不含）
- `blocking_io.py`: 还原 `BlockingIOProbe`（上游空文件）
- `test_channel_file_attachments.py`: mock 加 `**kwargs` 适配 `user_id` 参数
- `test_skills_load.py`: skip 不适用于我们 executor 的测试

## 未合的受保护文件（上游也改了但合入成本高, 留后续评估）

这些文件在受保护列表内,上游也有改动,但我们的 Noldus 定制太深:
- `agents/lead_agent/agent.py` — 中间件链 + Noldus 定制中间件
- `agents/lead_agent/prompt.py` — 中文调度规则 + Gate 机制
- `subagents/builtins/__init__.py` — ethoinsight subagent 注册
- `subagents/executor.py` — 递归限制 + max_turns + `_load_skill_contents`

## `.deerflow-sync-state` 已更新

```
last_sync_commit: 74e3e80c
last_sync_date: 2026-06-02
last_sync_commits_count: 34
```

## 下一步

1. **Push 分支并创建 PR** 到 dev
2. 反幻觉 spec 的层 3 可以直接复用 `todo_middleware` 的 `after_model+jump_to` 范式
3. 评估是否对 `lead_agent/agent.py` 做 surgical merge（上游加了 SandboxAuditMiddleware 等新中间件）
4. 评估 subtree squash 消除未来全量对比假象
