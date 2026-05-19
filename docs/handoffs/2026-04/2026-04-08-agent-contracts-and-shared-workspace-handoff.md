# 交接文档：Agent 契约体系 + 共享 Workspace + Summarization 归档 + 模型升级

> 日期: 2026-04-08
> 前序交接: `docs/handoffs/2026-04-08-knowledge-assistant-and-mcp-guard-handoff.md`
> 前序交接: `docs/handoffs/2026-04-08-model-upgrade-and-summarization-handoff.md`

---

## 1. 本次会话完成的任务

1. GLM-5.1 主模型升级 + Summarization 调优
2. SummarizationMiddleware 消息归档（解决前端消息消失问题）
3. Agent 契约体系 + 共享 Workspace `/mnt/shared/` + 占位符系统 `{{shared://...}}`
4. Data-analyst 升级为"分析+洞察"双角色
5. 模板 handoff JSON 增加 `metrics_summary` + `statistics` 字段

---

## 2. 当前进展

### Part A：模型升级 ✅

| # | 改动 | 文件 |
|---|------|------|
| 1 | 主模型 glm-5 → glm-5.1（编码 +28%，thinking 默认开启） | `config.yaml` |
| 2 | glm-5 保留为第二模型（摘要专用，thinking=false） | `config.yaml` |
| 3 | Summarization 触发条件：messages:15（去掉了 fraction 因 LangChain 不支持无 profile 模型） | `config.yaml` |
| 4 | Summarization 保留消息数 6→8，model_name 指定 glm-5 | `config.yaml` |

### Part B：Summarization 消息归档 ✅

| # | 改动 | 文件 |
|---|------|------|
| 5 | 新建 ArchivingSummarizationMiddleware（子类化 LangChain） | `deerflow/agents/middlewares/archiving_summarization.py` |
| 6 | 替换 SummarizationMiddleware → ArchivingSummarizationMiddleware | `deerflow/agents/lead_agent/agent.py` |
| 7 | Gateway API: GET /api/threads/{id}/archived-messages | `app/gateway/routers/threads.py` |
| 8 | 前端消息缓存 + 归档加载（useEffect fetch + messageCacheRef） | `frontend/src/core/threads/hooks.ts` |

归档路径：`.deer-flow/threads/{thread_id}/archived_messages/{timestamp}.json`
格式：`{archived_at, message_count, messages: [messages_to_dict 序列化]}`

### Part C：Agent 契约 + 共享 Workspace + 占位符 ✅

| # | 改动 | 文件 |
|---|------|------|
| 9 | 新增 SHARED_PATH_PREFIX="/mnt/shared" + Paths.shared_dir() | `config/paths.py` |
| 10 | ThreadDataState 加 shared_path 字段 | `agents/thread_state.py` |
| 11 | ThreadDataMiddleware 返回 shared_path | `agents/middlewares/thread_data_middleware.py` |
| 12 | replace_virtual_path 添加 /mnt/shared 映射 | `sandbox/tools.py` |
| 13 | task_tool 添加 _resolve_placeholders() | `tools/builtins/task_tool.py` |
| 14 | orchestration_guide 重写 Step 1.5/2/2.5/3（lead agent 写共享文件 + 占位符派遣） | `agents/lead_agent/prompt.py` |
| 15 | data-analyst: 添加 `<contract>` + 升级为"分析+洞察"双角色 | `subagents/builtins/data_analyst.py` |
| 16 | report-writer: 添加 `<contract>`，输入来自共享文件 | `subagents/builtins/report_writer.py` |
| 17 | code-executor: 添加 `<output_contract>` | `subagents/builtins/code_executor.py` |
| 18 | knowledge-assistant: 添加 `<contract>` | `subagents/builtins/knowledge_assistant.py` |

### Part D：模板 handoff 增强 ✅

| # | 改动 | 文件 |
|---|------|------|
| 19 | _write_handoff 新增 metrics_summary（去掉 values，只保留 mean/std/n）+ statistics | `ethoinsight/templates/shoaling.py` |

所有后端文件路径前缀：`packages/agent/backend/packages/harness/deerflow/`

---

## 3. 关键架构知识

### 新的数据流

```
code-executor → handoff.json（含 metrics_summary + statistics）
    ↓
lead agent read_file handoff.json
    ↓
lead agent write_file /mnt/shared/code_summary.json（精简摘要）
    ↓
task(data-analyst, prompt="分析 {{shared://code_summary.json}} ...")
    ↓ task_tool._resolve_placeholders()
data-analyst 收到 prompt："分析 /mnt/shared/code_summary.json ..."
    ↓ read_file /mnt/shared/code_summary.json
data-analyst 写分析 + 返回摘要文本
    ↓
lead agent write_file /mnt/shared/analysis_summary.md
    ↓
task(report-writer, prompt="基于 {{shared://code_summary.json}} + {{shared://analysis_summary.md}} ...")
    ↓
report-writer read_file 2 个共享文件 → 写报告
```

### 占位符机制

- 格式：`{{shared://filename}}`
- 解析位置：`task_tool.py:_resolve_placeholders()` — subagent 启动前
- 替换结果：`/mnt/shared/filename`
- subagent 用已有 `read_file` 按需读取，**prompt 不膨胀**

### 共享 workspace

- 虚拟路径：`/mnt/shared/`
- 物理路径：`.deer-flow/threads/{thread_id}/shared/`
- ThreadDataMiddleware 自动创建
- 隔离于 uploads/outputs/workspace

### Agent 契约结构

每个 subagent 的 system_prompt 中包含 `<contract>` 标签：
```
<contract>
输入: {{shared://...}} 引用
输出: 文件路径 + 最终消息格式
禁止: 不允许的操作
</contract>
```

### 模型配置

| 用途 | 模型 | thinking | Coding Plan 倍率 |
|------|------|----------|-----------------|
| 主模型（lead agent + subagents） | glm-5.1 | ✅（lead only，subagent 硬编码 False） | 3x |
| 摘要（SummarizationMiddleware） | glm-5 | ❌ | 3x |

### Summarization 归档链路

```
SummarizationMiddleware 触发（15条消息）
  → ArchivingSummarizationMiddleware._archive_messages()
  → 写入 .deer-flow/threads/{id}/archived_messages/{timestamp}.json
  → RemoveMessage 正常执行
  → 前端 streaming 期间：messageCacheRef 保留旧消息
  → 刷新页面：fetch /api/threads/{id}/archived-messages → 恢复历史
```

---

## 4. 关键发现

1. **DeerFlow subagent 是 tool 调用**：`task` tool → SubagentExecutor → 后台线程 → 轮询 → "Task Succeeded. Result: ..." 返回给 lead agent
2. **Subagent 之间零直接通信**：通过文件系统间接传递，编排由 lead agent 决定
3. **fraction 类型 summarization 不兼容智谱模型**：LangChain 需要 model profile 的 max_input_tokens，智谱无内置 profile
4. **SummarizationMiddleware 是 LangChain 库代码**：用 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 删除旧消息，不能改库，只能子类化
5. **前端 `useStream` 的 values 模式**：返回完整 state 快照，summarization 修改 state 后前端丢失旧消息

---

## 5. 未完成事项

### 高优先级

- [ ] **端到端验证**：重启服务，上传数据 → 分析 → 确认：
  - lead agent 是否写了 /mnt/shared/code_summary.json
  - data-analyst 是否从共享文件读取数据
  - report-writer 是否只读 2 个共享文件
  - 归档文件是否生成在 `.deer-flow/threads/{id}/archived_messages/`
  - 刷新页面后消息是否恢复
- [ ] **knowledge-assistant 追问场景验证**：同 thread 追问 → 确认路由到 knowledge-assistant + 读取 {{shared://code_summary.json}}
- [ ] **Summarization 触发验证**：连续追问 15+ 轮 → 确认摘要触发 + 归档生成

### 中优先级（继承）

- [ ] code-executor 性能优化（3 个方向未实施）
- [ ] 数据质量关卡验证
- [ ] 新图表类型端到端验证（raincloud_plot）
- [ ] noldus-kb MCP search_knowledge 500 错误排查
- [ ] title 和 memory 的 model_name 改为轻量模型（当前用主模型）

### 低优先级

- [ ] 更新 README.md / CLAUDE.md
- [ ] 评估更多图表类型（山脊图、桑基图）
- [ ] 考虑其他范式模板也加 metrics_summary + statistics 到 handoff

---

## 6. 建议接手路径

### 启动

```bash
cd /home/qiuyangwang/noldus-insight
make stop
rm -f packages/agent/backend/.deer-flow/checkpoints.db*
rm -rf packages/agent/backend/.deer-flow/threads/*
make dev
```

### 验证顺序

1. `tail -f packages/agent/logs/langgraph.log`
2. Ultra 模式 → 上传数据 → "请分析" → 观察：
   - 日志中 `glm-5.1` 出现
   - lead agent 调用 `write_file("/mnt/shared/code_summary.json", ...)`
   - data-analyst 调用 `read_file("/mnt/shared/code_summary.json")`（而不是 metrics.csv）
   - `.deer-flow/threads/{id}/shared/` 目录下有 code_summary.json
3. 同 thread → "这个 p 值为什么不显著？" → 确认 knowledge-assistant 路由
4. 连续追问 15 轮 → 确认 summarization 触发 + `archived_messages/` 下有 JSON
5. 刷新页面 → 确认早期消息恢复

### 关键文件

| 文件 | 用途 |
|------|------|
| `packages/agent/config.yaml` | 模型 + summarization + memory 配置 |
| `deerflow/agents/middlewares/archiving_summarization.py` | Summarization 归档 middleware |
| `deerflow/tools/builtins/task_tool.py` | 占位符解析 _resolve_placeholders() |
| `deerflow/config/paths.py` | SHARED_PATH_PREFIX + shared_dir() |
| `deerflow/agents/lead_agent/prompt.py` | orchestration_guide（数据流核心） |
| `deerflow/subagents/builtins/*.py` | 4 个 subagent 契约 |
| `ethoinsight/templates/shoaling.py` | 模板 handoff 含 metrics_summary |
| `frontend/src/core/threads/hooks.ts` | 消息缓存 + 归档加载 |
| `app/gateway/routers/threads.py` | archived-messages API |

---

## 7. 风险与注意事项

1. **不要改 `executor.py:136`** 的 `object.__setattr__` — Pydantic 兼容性修复
2. **不要删 glm-5 模型配置项** — summarization 依赖 `model_name: glm-5`
3. **has_noldus_agents 有两处** — prompt.py line ~205 和 ~755，修改必须同步
4. **占位符只支持 shared://** — 不要用 `{{workspace://...}}`，那不存在
5. **lead agent 必须先写共享文件再派遣** — 如果 lead agent 跳过 write_file 直接 task()，subagent 会 read_file 404
6. **前端归档恢复仅在刷新后生效** — streaming 期间靠 messageCacheRef 保持消息
7. **Coding Plan 配额** — GLM-5.1 和 GLM-5 都是 3x 倍率（高峰），注意监控
8. **seaborn 需手动安装** — `uv pip install --python .venv/bin/python seaborn`
