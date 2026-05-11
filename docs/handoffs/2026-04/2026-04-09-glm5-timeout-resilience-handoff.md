# 交接文档：GLM-5 超时韧性修复 + 前端孤立消息处理

> 写给下一位接手的 AI Agent。假设你无法访问之前的对话上下文。

## 1. 当前任务目标

**核心问题**：data-analyst subagent 在"行为数据学解读"环节卡住很久（10+ 分钟），前端显示"temporarily unavailable"错误。

**根因链条**：
```
GLM-5 API 在大 context 下响应时间 >60s（甚至 >90s）
  → openai SDK request_timeout 触发
  → httpx.ReadTimeout（流式读取超时）未被 middleware 识别为 retriable
  → middleware 只尝试 1 次就放弃
  → 同时 openai SDK retry × middleware retry 叠加，最坏情况 18 分钟
  → 前端收到孤立 tool message → console.error
```

**预期产出**：
- GLM-5 超时时合理重试（而非卡死或直接放弃）
- 前端不再报 "Unexpected tool message outside a processing group" 错误

## 2. 当前进展

### ✅ 已提交（commit cd2a2f9 + 6600728 + c24e575）

#### 1. `config.yaml` — GLM-5 超时参数调优

**文件**: `packages/agent/config.yaml` (lines 21-22, 35-36)

- `request_timeout`: 120s → 90s（折中：GLM-5 正常 5-30s，大 context 可达 60-80s）
- `max_retries`: 3 → 2（SDK 层面减少无效等待）

#### 2. `llm_error_handling_middleware.py` — 三项修复

**文件**: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py`

**(a) 总重试时间上限** (line 72)
```python
retry_total_timeout_s: float = 180.0
```
`wrap_model_call` 和 `awrap_model_call` 开头记录 `start_time = time.monotonic()`，每次 retry 前检查是否超过 180s，超过直接放弃。日志输出 `"LLM call aborted after Xs (total timeout Ys)"`。

**(b) `ReadTimeout` 加入 retriable 列表** (lines 86-93)
```python
exc_name in {"APITimeoutError", "APIConnectionError", "InternalServerError",
             "ReadTimeout", "ConnectTimeout", "RemoteProtocolError"}
```
之前 `httpx.ReadTimeout` 被归为 `"generic"` 不重试，现在正确归类为 `"transient"` 会重试。

**(c) `_BUSY_PATTERNS` 新增超时匹配** (lines 34-36)
```python
"readtimeout", "read timed out", "connect timed out"
```

#### 3. `data_analyst.py` — 减少无效 LLM 调用

**文件**: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`

- `max_turns`: 10 → 6（read → analyze → write → return，4-5 轮足够）
- system prompt 新增：
  - "code_summary.json 是唯一且完整的数据源，无需逐一验证各字段"
  - "一次 read_file 即可获取全部数据，不要多次反复读取同一文件"
  - 禁止列表新增"反复多次读取同一文件确认数据"

#### 4. `prompt.py` — Lead agent 输出规则

**文件**: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

- 新增"输出规则"区块：禁止在用户消息中复述 bash 命令、JSON 数据、代码片段
- Step 1.5 强化：写共享文件必须用 `write_file` 工具，禁止用 bash `cat/echo`
- 背景：GLM-5 之前用 `bash cat > ... << ENDJSON` 写 JSON，整段数据流式展示给用户

#### 5. `utils.ts` — 前端孤立 tool message 处理

**文件**: `packages/agent/frontend/src/core/messages/utils.ts` (line 83)

- `console.error("Unexpected tool message...")` → 静默忽略 + 注释说明
- 触发场景：LLM 超时返回纯文本 AIMessage（无 tool_calls）后，残留 tool result 消息到达前端

## 3. 未完成事项

### 应该做（P1）

- **端到端验证**：修复后还未在干净环境下完整跑通一次分析流程
  ```bash
  cd /home/qiuyangwang/noldus-insight
  make stop
  rm -f packages/agent/backend/.deer-flow/checkpoints.db*
  rm -rf packages/agent/backend/.deer-flow/threads/*
  make dev
  ```
  新建 thread → 上传斑马鱼数据 → 检查 `logs/langgraph.log`：
  - `ReadTimeout` 现在应该触发 retry 而非直接 fail
  - data-analyst 应在 6 轮以内完成
  - 不应出现 >5 分钟的空白等待期
  - 前端不应出现 "Unexpected tool message" console error

- **GLM-5 API 稳定性监控**：GLM-5 在大 context 下间歇性超时（60-90s），如果 90s timeout 仍不够，可能需要：
  - 进一步降低 data-analyst 的 context 大小（精简 code_summary.json）
  - 或把 request_timeout 提到 120s 但配合总超时 180s 兜底

### 可选（P2）

- **Lead agent 并行派遣问题**（从上一轮遗留）：lead agent 有时同时派遣 code-executor 和 data-analyst
- **noldus-kb MCP 验证**：MCP server 已连接（6 tools），但尚未在知识问答场景验证调用
- 评估是否将 ethoinsight skill 拆分为多个独立 skill

## 4. 关键上下文

### 超时参数体系（修复后）

| 层级 | 参数 | 值 | 说明 |
|------|------|-----|------|
| openai SDK | `request_timeout` | 90s | 单次 HTTP 请求超时 |
| openai SDK | `max_retries` | 2 | SDK 内部重试次数 |
| middleware | `retry_max_attempts` | 3 | middleware 层重试次数 |
| middleware | `retry_total_timeout_s` | 180s | middleware 总重试时间上限 |
| subagent | `timeout_seconds` | 600s (data-analyst) | subagent 总执行时间 |

**最坏情况**：90s(timeout) × 2(SDK retry) = 180s，刚好触发 middleware 总超时 → 放弃。不会再出现 18 分钟卡住的情况。

### GLM-5 API 实测数据

| 请求类型 | 响应时间 | 备注 |
|----------|---------|------|
| 简单 hello | ~8s | 正常 |
| 大 context (~10K tokens) | ~47s | 正常但慢 |
| data-analyst 第 4 轮 | >60s | 触发超时 |

### 关键文件路径

| 文件 | 本 session 改动 |
|------|----------------|
| `packages/agent/config.yaml` | request_timeout 90s, max_retries 2 |
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py` | 总超时 180s + ReadTimeout retriable + busy patterns |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` | max_turns 6 + prompt 精简 |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | 输出规则 + Step 1.5 write_file 强制 |
| `packages/agent/frontend/src/core/messages/utils.ts` | 孤立 tool message 静默忽略 |

### 之前的交接文档

- `docs/handoffs/2026-04-08-phase3-final-handoff.md` — 上一轮完整交接（含 path_mappings 根因修复、agent 角色分工、skill 限制等）
- `docs/milestone/2026-04-07-deerflow-merge-restoration.md` — 三个阶段的完整 DeerFlow merge 恢复记录

## 5. 下一位 Agent 的第一步建议

1. 读取本文档
2. 清理旧 checkpoint 并重启：
   ```bash
   cd /home/qiuyangwang/noldus-insight
   make stop
   rm -f packages/agent/backend/.deer-flow/checkpoints.db*
   rm -rf packages/agent/backend/.deer-flow/threads/*
   make dev
   ```
3. 新建 thread，上传斑马鱼数据，发起分析
4. 检查 `logs/langgraph.log` 确认：
   - `ReadTimeout` 触发 retry（而非 "failed after 1 attempt"）
   - data-analyst 在 6 轮以内完成
   - 前端无 console error
5. 如果 90s timeout 仍不够，先检查 GLM-5 官方控制台 [open.bigmodel.cn](https://open.bigmodel.cn) 的调用日志确认延迟
6. 完整三阶段记录见 `docs/milestone/2026-04-07-deerflow-merge-restoration.md`
