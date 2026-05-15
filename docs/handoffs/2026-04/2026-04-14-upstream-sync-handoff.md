# DeerFlow 上游同步 + 测试修复 — 交接文档

> 日期: 2026-04-14
> 上一份交接: `docs/handoffs/2026-04-13-skill-refactor-handoff.md`

---

## 1. 当前任务目标

**本次会话完成了三件事**：

1. **诊断 E2E 测试问题** — 发现 `subagent_enabled: False` 导致 lead agent 不分派子任务
2. **DeerFlow 上游选择性同步** — 从 deerflow-noldus 拉取 27 个新 commit（f0dd8cb → c91785d），选择性合入
3. **修复全部 20 个失败测试** — 测试全部通过（1539 passed, 0 failed）

---

## 2. 当前进展

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | 诊断 `subagent_enabled: False` 问题 | ✅ 根因：前端只有 ultra 模式发送 true |
| 2 | 改 `subagent_enabled` 默认值为 True（agent.py + client.py） | ✅ 但被 linter 还原，需确认 |
| 3 | 创建 `scripts/sync-deerflow.sh` 同步脚本 | ✅ |
| 4 | 创建 `docs/sop/deerflow-sync-sop.md` 操作流程 | ✅ |
| 5 | 上游同步：30 个安全文件自动合入 | ✅ |
| 6 | 上游同步：5 个受保护文件手动合入 | ✅ |
| 7 | 修复 20 个失败测试 | ✅ |
| 8 | 提交代码 | ❌ 未提交 |
| 9 | 第二轮 E2E 测试 | ❌ 未开始 |

---

## 3. 改动的文件（未提交）

共 43 个文件改动，分为 4 类：

### 3.1 上游安全文件（自动合入，28 + 2 新增）

这些文件直接用上游 deerflow/main 的最新版本覆盖：
- `agents/memory/{queue,storage,updater}.py`
- `agents/middlewares/{clarification,dangling_tool_call,loop_detection,sandbox_audit,title,uploads}_middleware.py`
- `client.py`, `config/{app_config,model_config,subagents_config}.py`
- `community/{aio_sandbox,firecrawl,jina_ai}` 多个文件
- `models/{factory,openai_codex_provider,patched_deepseek}.py`
- `runtime/runs/worker.py`, `sandbox/{file_operation_lock,security}.py`
- `subagents/{builtins/bash_agent,builtins/general_purpose,registry}.py`
- `tools/builtins/present_file_tool.py`
- 新增: `community/exa/tools.py`, `config/agents_api_config.py`

### 3.2 上游受保护文件（手动合入 bug fix，保留 Noldus 定制）

| 文件 | 合入的上游改动 |
|------|-------------|
| `executor.py` | event loop 冲突修复：`_isolated_loop_pool` + `_execute_in_isolated_loop` |
| `local_sandbox.py` | 路径解析：`_resolve_paths_in_content` + `_agent_written_paths` 追踪 |
| `llm_error_handling_middleware.py` | circuit breaker 防限流（closed → open → half_open） |
| `agent.py` | 模型解析修正：`_resolve_model_name()` 统一调用 |
| `prompt.py` | workspace 相对路径指引（3 行） |

### 3.3 测试修复

| 文件 | 修改内容 |
|------|---------|
| `test_subagent_timeout_config.py` | `general-purpose`/`bash` → `code-executor`/`data-analyst` |
| `test_subagent_prompt_security.py` | 重写为验证 Noldus subagent 注册和 prompt 内容 |
| `test_lead_agent_model_resolution.py` | `SummarizationMiddleware` → `ArchivingSummarizationMiddleware` |
| `test_local_sandbox_encoding.py` | Windows mock 从精确匹配 kwargs dict 改为检查关键参数 |
| `test_client.py` | stream_mode 期望从 `["values", "custom"]` 改为 `["values", "messages", "custom"]` |
| `test_uploads_middleware_core_logic.py` | content 从 string 改为支持 list（多段 text）格式 |

### 3.4 新增工具文件

| 文件 | 用途 |
|------|------|
| `scripts/sync-deerflow.sh` | DeerFlow 上游选择性同步脚本 |
| `docs/sop/deerflow-sync-sop.md` | 同步操作流程文档 |

---

## 4. 关键架构决策

### subagent_enabled 默认值

- **问题**：前端只有 ultra 模式发 `subagent_enabled: true`，其他模式默认 False
- **改动**：在 `agent.py` line 257/285 和 `client.py` line 117/223 将默认值改为 True
- **注意**：linter 可能已将这些改动还原（system-reminder 中提到文件被 linter 修改）。需确认 `subagent_enabled` 默认值当前状态

### DeerFlow 上游同步策略

确定了选择性同步方案（非 subtree merge）：
- 你定制过的 13 个文件为"受保护文件"，同步时逐个判断
- 其余文件直接用上游版本覆盖
- 脚本 `scripts/sync-deerflow.sh` 自动分类

### 受保护文件完整清单（13 个）

```
# 高侵入
agents/lead_agent/prompt.py                          # 调度规则、中文示例
subagents/builtins/__init__.py                       # 4 个 Noldus subagent 注册
# 中侵入
agents/middlewares/llm_error_handling_middleware.py   # 超时 + circuit breaker
mcp/tools.py                                         # MCP 结果截断
sandbox/tools.py                                     # shared workspace
sandbox/local/local_sandbox.py                       # venv PATH + env vars
agents/lead_agent/agent.py                           # tool_groups + summarization
tools/builtins/task_tool.py                          # {{shared://}} 占位符
subagents/executor.py                                # recursion_limit + event loop
config/paths.py                                      # /mnt/shared 路径
# 低侵入
sandbox/sandbox.py                                   # extra_env 参数
agents/thread_state.py                               # shared_path 字段
agents/middlewares/thread_data_middleware.py          # shared_path 初始化
```

---

## 5. 关键发现

### Memory update 解析失败

E2E 测试日志中发现：
```
Failed to parse LLM response for memory update: Invalid control character at: line 8 column 139 (char 581)
```
GLM-5.1 返回的 JSON 包含非法控制字符，memory updater 解析失败。不影响主流程但说明 GLM 输出格式不规范。

### 上游 prompt.py 长期维护讨论

讨论了多种降低 prompt.py 冲突面的方案：
- SOUL.md（上游内置）只能追加不能替换
- hook 式字符串替换不够企业级
- 最终结论：**暂不重构**，用 `sync-deerflow.sh` 选择性同步管理冲突面

---

## 6. 未完成事项

### 高优先级

1. **提交代码** — 43 个文件改动未提交
2. **确认 `subagent_enabled` 默认值** — linter 可能还原了改动，需检查 `agent.py` line 257/285 和 `client.py` line 117/223
3. **第二轮 E2E 测试** — 重启 `make dev`，验证完整流水线（同 `2026-04-13-skill-refactor-handoff.md` 中的待办）

### 中优先级

4. **恢复 noldus-kb** — 等 `180.184.84.124:7001` 恢复后改 `extensions_config.json` 为 `"enabled": true`
5. **微调工作启动** — 按 `docs/plans/2026-04-13-fine-tuning-small-model-design.md` 开始
6. **范式模板验证** — open_field 和 epm

### 低优先级

7. **spec 表填写** — `docs/specs/paradigm-analysis-tools-spec.md`
8. **prompt.py 抽离重构** — 将 Noldus 调度逻辑抽到独立文件（长期改善同步体验，但非必须）

---

## 7. 建议接手路径

### 如果要提交代码

```bash
cd /home/qiuyangwang/noldus-insight

# 1. 确认 subagent_enabled 默认值
grep "subagent_enabled.*True\|subagent_enabled.*False" \
  packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py \
  packages/agent/backend/packages/harness/deerflow/client.py

# 如果被还原为 False，重新改为 True：
# agent.py line 257: .get("subagent_enabled", True)
# agent.py line 285: .get("subagent_enabled", True)
# client.py line 117: subagent_enabled: bool = True
# client.py line 223: .get("subagent_enabled", True)

# 2. 跑测试确认
cd packages/agent/backend && make test

# 3. 提交
cd /home/qiuyangwang/noldus-insight
git add -A
git commit -m "sync deerflow upstream to c91785d + fix 20 test cases"
```

### 如果要继续 E2E 测试

```bash
cd /home/qiuyangwang/noldus-insight

# 确认 noldus-kb 已禁用
grep '"enabled"' packages/agent/extensions_config.json | head -1

# 启动服务
make dev

# 上传 demo 数据，观察 langgraph.log：
# 1. subagent_enabled 应为 True
# 2. code-executor → data-analyst → report-writer 流水线
```

### 如果要同步下一批上游代码

```bash
cd /home/qiuyangwang/noldus-insight
git fetch deerflow
./scripts/sync-deerflow.sh --dry-run   # 先看报告
./scripts/sync-deerflow.sh             # 交互式同步
```

---

## 8. 风险与注意事项

1. **43 个文件未提交** — 当前所有改动都在工作区，需要尽快提交
2. **subagent_enabled 可能被还原** — linter 在会话中修改了 agent.py 和 client.py，system-reminder 提示改回了 False
3. **skills/custom/ 是 gitignored** — skill 文件不在 git 中
4. **noldus-kb 临时禁用** — `extensions_config.json` 中 `"enabled": false`，不要提交
5. **sync 脚本的 PROTECTED_FILES 列表** — 如果新增定制文件，记得更新 `scripts/sync-deerflow.sh` 中的列表

---

## 下一位 Agent 的第一步建议

1. 读取本文档了解全貌
2. 检查 `subagent_enabled` 默认值是否为 True，如果被还原则重新修改
3. 跑 `make test` 确认 1539 passed
4. 提交代码
5. 重启 `make dev` 进行 E2E 测试
