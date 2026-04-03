# EthoInsight 交接文档

> 日期：2026-04-03（第五次）
> 会话：P2-P5 实现 + 端到端验证调试
> 状态：P0-P5 代码完成，端到端验证发现 4 个阻塞问题待修

---

## 1. 当前任务目标

端到端验证 EthoInsight：上传斑马鱼 demo 数据 → Agent 分析 → 输出报告。验证中发现 4 个阻塞问题需要修复。

**核心实施方案**：`/home/qiuyangwang/noldus-insight/docs/plans/2026-04-02-implementation-spec.md`
**前次交接**：`/home/qiuyangwang/noldus-insight/docs/handoffs/2026-04-03-p5-handoff.md`

---

## 2. 端到端验证发现的 4 个阻塞问题

### 问题 1: subagent_enabled: False（最关键）

**日志证据**：
```
Create Agent(default) -> subagent_enabled: False
```

**config.yaml 已改**（`subagents.enabled: true`），但日志显示仍为 False。

**根因排查方向**：DeerFlow 的 `subagent_enabled` 不是从 `config.yaml` 的 `subagents.enabled` 读取的——它是前端通过 runtime configurable 参数传递的。需要查看：
- 前端发送 `runs/stream` 请求时 `configurable.subagent_enabled` 是否为 true
- `make_lead_agent()` 中 `cfg.get("configurable", {}).get("subagent_enabled", False)` 的默认值
- 前端 `useThreadStream` 或 `useSubmitThread` 中是否传递了 subagent_enabled

**相关文件**：
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py:278-290`
- `packages/agent/frontend/src/core/threads/hooks.ts` — 搜索 `subagent_enabled` 或 `configurable`
- `packages/agent/frontend/src/core/settings/` — 用户设置中是否有 subagent toggle

### 问题 2: MCP streamable-http 不支持

**日志证据**：
```
Failed to configure MCP server 'noldus-kb': MCP server 'noldus-kb' has unsupported transport type: streamable-http
```

**根因**：DeerFlow 的 `deerflow/mcp/client.py` 只支持 `stdio` 和 `sse` 两种 transport type，不支持 `streamable-http`。

**修复方向**：
- 方案 A：将 `extensions_config.json` 中的 type 改为 `sse`（如果 noldus-kb 的 MCP endpoint 兼容 SSE transport）
- 方案 B：在 `deerflow/mcp/client.py` 中添加 `streamable-http` 支持（需要 `langchain-mcp-adapters` 版本是否支持）
- 方案 C：暂时禁用 noldus-kb MCP，后续再接入

**相关文件**：
- `packages/agent/backend/packages/harness/deerflow/mcp/client.py` — 搜索 `transport type` 或 `unsupported`
- `packages/agent/extensions_config.json` — noldus-kb 配置

### 问题 3: sandbox 中 python 找不到依赖

**日志证据**：Agent 执行 `python -c "import pandas..."` 失败，然后尝试 `pip install`。

**根因**：`local_sandbox.py` 的 `execute_command` 用 `subprocess.run(shell=True)` 执行命令，继承父进程环境。但系统 `python3`（`/usr/bin/python3`）没有装 pandas 等依赖——它们在 uv venv（`backend/.venv/bin/python`）里。

**已尝试的修复**（在 `local_sandbox.py` 中）：把 `sys.executable` 所在的 bin 目录加到子进程 PATH 前面。代码已写入但**还没测试**（因为 log 是修改前的运行）。

**当前代码状态**：
```python
# local_sandbox.py execute_command() 中：
import sys
env = os.environ.copy()
venv_bin = os.path.dirname(sys.executable)
current_path = env.get("PATH", "")
if venv_bin not in current_path:
    env["PATH"] = venv_bin + os.pathsep + current_path
# 然后 subprocess.run(..., env=env)
```

**需要验证**：重启后 `python` 命令是否解析到 venv 中的 python。

### 问题 4: NewAPI 中转超时断连

**日志证据**：
```
LLM call failed: peer closed connection without sending complete message body (incomplete chunked read)
```
发生在 input=114K tokens 的请求上。

**根因**：NewAPI 中转（`https://newapi.noldusapi.com/v1`）对长请求超时断连。config.yaml 中 `request_timeout: 600` 但中转可能有自己的超时限制。

**修复方向**：
- 增大 `max_retries`（当前 2）
- 检查 summarization 配置是否合理（当前 `trigger: tokens: 15564`，但 input 已到 114K）
- 减少 system prompt 大小（orchestration_guide + skill 内容可能太大）

---

## 3. 本次会话已完成的修复

以下修复已写入代码但大多**还未被端到端测试验证**：

| 修复 | 文件 | 状态 |
|------|------|------|
| ExperimentContextMiddleware async→sync | `middlewares/experiment_context.py` | ✅ 已修复 |
| read_file UTF-16 LE fallback | `sandbox/local/local_sandbox.py:240-248` | ✅ 已写入 |
| sandbox python PATH 注入 | `sandbox/local/local_sandbox.py:195-205` | ✅ 已写入 |
| 前端 orphan tool message | `frontend/src/core/messages/utils.ts:74-83` | ✅ 已修复 |
| 后端 orphan ToolMessage 清理 | `middlewares/dangling_tool_call_middleware.py` | ✅ 已写入 |
| subagents.enabled: true | `config.yaml:88-90` | ✅ 已写入，但 runtime 可能不读这里 |
| allow_host_bash: true | `config.yaml:81` | ✅ 已写入 |
| noldus-kb MCP type | `extensions_config.json` | ❌ 需改 type |

---

## 4. 关键上下文

### 4.1 测试状态
```bash
cd /home/qiuyangwang/noldus-insight/packages/agent/backend
uv run pytest packages/ethoinsight/tests/ -v  # 80 passed
uv run pytest tests/test_harness_boundary.py -v  # 1 passed
```

### 4.2 项目文件结构
（见 P5 交接文档 `docs/handoffs/2026-04-03-p5-handoff.md`）

### 4.3 DeerFlow 核心未修改
Memory 系统、Skills 渐进加载、Sandbox 隔离、Context Engineering、Gateway API 全部未改。

---

## 5. 下一位 Agent 的第一步建议

**优先级排序**：

### P0: 修复 subagent_enabled（最关键）
1. 搜索前端代码确认 `subagent_enabled` 怎么传递：
   ```bash
   grep -r "subagent_enabled\|subagent\|ultra" packages/agent/frontend/src/ --include="*.ts" --include="*.tsx"
   ```
2. 查看前端设置界面是否有 "Ultra" 或 "Sub-agent" 模式开关
3. 如果是前端控制的，确保默认启用或在 UI 上选择正确模式

### P1: 修复 MCP transport type
1. 检查 DeerFlow 支持的 MCP transport types：
   ```bash
   grep -r "transport\|stdio\|sse\|streamable" packages/agent/backend/packages/harness/deerflow/mcp/client.py
   ```
2. 尝试将 `extensions_config.json` 中 type 改为 `sse`
3. 如果 noldus-kb 不支持 SSE，暂时 disabled

### P2: 验证 sandbox python PATH
1. 重启 `make dev`
2. 新建 thread，发送：`请执行 python -c "import ethoinsight; print(ethoinsight.__version__)"`
3. 确认输出 `0.1.0`

### P3: 处理 NewAPI 超时
1. 增大 `config.yaml` 中 `request_timeout` 到 900
2. 增大 `max_retries` 到 3
3. 检查 summarization trigger 是否太高（15564 tokens 但实际 input 到了 114K）

---

## 6. 风险与注意事项

- **subagent_enabled 是运行时参数**：不是从 config.yaml 读的，是前端通过 LangGraph SDK 传递的 configurable 参数。这是 DeerFlow 的设计——用户在 UI 上选择执行模式（standard/pro/ultra）来控制是否启用 subagent。
- **noldus-kb MCP server 在 180.184.84.124:7001**：需要确认该服务是否在运行、是否支持 SSE transport
- **local_sandbox.py 改动较多**：read_file UTF-16 fallback + execute_command PATH 注入，需要仔细测试
- **前端 orphan tool message 修复**：是防御性修复，根因在 summarization 压缩消息配对断裂
