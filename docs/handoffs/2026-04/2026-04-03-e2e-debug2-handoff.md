# EthoInsight 交接文档

> 日期：2026-04-03（第六次）
> 会话：端到端调试
> 状态：P0-P5 代码完成，端到端调试进行中，多个 bug 已修但需重启验证

---

## 1. 当前任务

端到端验证：上传斑马鱼 demo 数据 → Agent 用 Ultra 模式派遣 subagent 分析 → 输出报告。

---

## 2. 本次会话已修复的 bug

| # | Bug | 修复 | 文件 | 状态 |
|---|-----|------|------|------|
| 1 | ExperimentContextMiddleware 是 async def，框架期望 sync | `async def` → `def` | `middlewares/experiment_context.py:37` | ✅ 已修 |
| 2 | read_file 无法读 UTF-16 LE 文件 | 加 UnicodeDecodeError fallback 到 utf-16-le | `sandbox/local/local_sandbox.py:240-248` | ✅ 已修 |
| 3 | sandbox bash 中 python 找不到依赖 | 把 `sys.executable` 的 bin 目录加到子进程 PATH | `sandbox/local/local_sandbox.py:195-205` | ✅ 已修 |
| 4 | 前端 orphan tool message console error | 创建 processing group 而非报错 | `frontend/src/core/messages/utils.ts:74-83` | ✅ 已修 |
| 5 | 后端 orphan ToolMessage 传给 LLM | DanglingToolCallMiddleware 清理无匹配 AI tool_call 的 ToolMessage | `middlewares/dangling_tool_call_middleware.py` | ✅ 已修 |
| 6 | subagent_enabled 始终 False | 发现是前端 mode 控制：只有 Ultra 模式才启用。GLM-5 的 `supports_thinking: true` 使 Ultra 可选 | `config.yaml:24` | ✅ 已修 |
| 7 | orchestration_guide 中 `{paradigm}` 被 `.format()` 当占位符 → KeyError | 改为具体示例文本，不用花括号变量 | `prompt.py` orchestration_guide 段 | ✅ 已修 |
| 8 | Lead Agent 直接 read_file 读大数据文件导致 token 爆炸 | 在 orchestration_guide 加醒目警告禁止读数据文件 | `prompt.py` orchestration_guide 段 | ✅ 已修 |
| 9 | code-executor 不知道 ethoinsight 库 | system_prompt 加了 `<ethoinsight_library>` 段落 | `subagents/builtins/code_executor.py` | ✅ 已修 |
| 10 | NewAPI 超时断连 | request_timeout 600→900, max_retries 2→3, summarization trigger 15564→8000, keep messages 10→6 | `config.yaml` | ✅ 已修 |
| 11 | MCP noldus-kb streamable-http 不支持 | **未修**——DeerFlow MCP client 不支持此 transport type | `extensions_config.json` | ❌ 待修 |

---

## 3. 当前 config.yaml 状态

```yaml
models:
  - name: glm-5
    display_name: GLM-5
    model: glm-5
    base_url: https://newapi.noldusapi.com/v1
    api_key: sk-iSPNEf...
    request_timeout: 900.0
    max_retries: 3
    supports_thinking: true   # 使前端能选 Ultra 模式
    supports_vision: false

sandbox:
  allow_host_bash: true       # code-executor 需要执行 python

subagents:
  enabled: true
  timeout_seconds: 900

summarization:
  trigger:
    - type: tokens
      value: 8000             # 从 15564 降低
  keep:
    type: messages
    value: 6                  # 从 10 降低
```

---

## 4. 关键发现

### 前端模式与 subagent 的关系

```
Flash    → thinking: false, plan: false, subagent: false
Thinking → thinking: true,  plan: false, subagent: false
Pro      → thinking: true,  plan: true,  subagent: false
Ultra    → thinking: true,  plan: true,  subagent: true  ← 必须选这个
```

前端代码位置：`frontend/src/core/threads/hooks.ts:387-394`

如果模型 `supports_thinking: false`，前端会强制 fallback 到 Flash 模式（`input-box.tsx:92-93`），永远无法选 Ultra。所以 GLM-5 必须设为 `supports_thinking: true`。

### prompt.py 中不能用花括号变量

`SYSTEM_PROMPT_TEMPLATE` 通过 Python `.format()` 渲染，所有花括号 `{xxx}` 都会被当作变量。orchestration_guide 中的示例代码如果包含 `{paradigm}` 等，必须用具体值替代或用双花括号 `{{paradigm}}` 转义。

### local_sandbox execute_command PATH 注入

```python
import sys
env = os.environ.copy()
venv_bin = os.path.dirname(sys.executable)
if venv_bin not in env.get("PATH", ""):
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
subprocess.run(..., env=env)
```

这让 sandbox 中 `python` 命令解析到 uv venv 的 python，包含所有已安装依赖。

---

## 5. 未完成事项

### P0: 重启验证（最高优先级）
- [ ] 重启 `make dev`
- [ ] 新建 thread，选 **Ultra** 模式
- [ ] 上传 5 个基础斑马鱼文件（`轨迹-*Subject ?.txt`）
- [ ] 发送分析指令
- [ ] 确认 Agent 派遣 code-executor subagent 而非自己读文件
- [ ] 确认 code-executor 使用 ethoinsight 库

### P1: MCP noldus-kb transport type
- [ ] 检查 `deerflow/mcp/client.py` 支持哪些 transport types
- [ ] 将 `extensions_config.json` 中 type 从 `streamable-http` 改为 `sse`（如果 noldus-kb 支持）
- [ ] 或暂时 disabled noldus-kb

### P2: 其他范式模板
- [ ] templates/epm.py, open_field.py, novel_object.py 等

### P3: 前端品牌
- [ ] DeerFlow → EthoInsight（header, landing page）

---

## 6. 下一位 Agent 的第一步

1. **重启 `make dev`**（所有修复需要重启才生效）
2. **新建 thread**（旧 thread 可能有脏 checkpoint），选 **Ultra** 模式
3. **只上传 5 个基础文件**（不要上传全部 35 个）：
   ```
   轨迹-Shoaling behavior with JavaScript XT180-Trial 1-Arena 1-Subject 1.txt
   轨迹-Shoaling behavior with JavaScript XT180-Trial 1-Arena 1-Subject 2.txt
   轨迹-Shoaling behavior with JavaScript XT180-Trial 1-Arena 1-Subject 3.txt
   轨迹-Shoaling behavior with JavaScript XT180-Trial 1-Arena 1-Subject 4.txt
   轨迹-Shoaling behavior with JavaScript XT180-Trial 1-Arena 1-Subject 5.txt
   ```
4. **发送**：`请分析这批斑马鱼数据。Subject 1-2 对照组，Subject 3-5 实验组。`
5. **观察日志**确认：
   - `subagent_enabled: True`（不是 False）
   - Agent 使用 `task()` 工具派遣 code-executor
   - code-executor 使用 `from ethoinsight import parse, metrics`
   - 没有 read_file 读取 .txt 数据文件

---

## 7. 风险与注意事项

- **旧 thread 有脏 checkpoint**：之前的失败运行可能导致 checkpoint 中有 orphan tool messages。务必新建 thread 测试。
- **NewAPI 中转仍可能超时**：如果 GLM-5 的响应时间 > 900s 或中转有自己的限制，仍会断连。考虑直接连智谱 API 而非中转。
- **noldus-kb MCP 暂不可用**：streamable-http 不被支持，Agent 无法查询知识库。不影响核心分析流程。
- **config.yaml API key 明文**：生产环境应改为 `$GLM_API_KEY`。
