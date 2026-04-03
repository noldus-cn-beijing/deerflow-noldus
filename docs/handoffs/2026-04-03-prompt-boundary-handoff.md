# EthoInsight 交接文档

> 日期：2026-04-03（第七次）
> 会话：Agent 提示词边界修复 + Bug 修复
> 状态：提示词改造完成，需重启验证

---

## 1. 当前任务

修复端到端调试中发现的 3 个问题 + 全面梳理 agent 提示词的责任边界。

---

## 2. 本次会话已完成的修改

| # | 修改 | 文件 | 状态 |
|---|------|------|------|
| 1 | Thread 404 修复：删除 thread 前先取消 active runs | `frontend/src/core/threads/hooks.ts:498-510` | ✅ |
| 2 | GraphRecursionError 修复：code-executor max_turns 25→50，强化 prompt 防止探索循环 | `subagents/builtins/code_executor.py` | ✅ |
| 3 | Lead Agent subagent_section：从 DeerFlow 通用模板改为 EthoInsight 专用版 | `agents/lead_agent/prompt.py:_build_subagent_section()` | ✅ |
| 4 | Lead Agent subagent_reminder/thinking：改为中文 EthoInsight 专用指令 | `agents/lead_agent/prompt.py:apply_prompt_template()` | ✅ |
| 5 | data-analyst：移除不可用的 noldus-kb MCP 引用，移除 bash/web 工具，加边界 | `subagents/builtins/data_analyst.py` | ✅ |
| 6 | report-writer：移除 bash 工具，加"不要跑代码"边界 | `subagents/builtins/report_writer.py` | ✅ |
| 7 | Pydantic 序列化警告：确认是 LangGraph 框架内部问题，无法修复 | N/A | ⚠️ 已知问题 |

---

## 3. 责任边界矩阵（修改后）

| 能力 | Lead Agent | code-executor | data-analyst | report-writer |
|------|:---:|:---:|:---:|:---:|
| 理解用户意图、确认需求 | **YES** | - | - | - |
| 派遣 subagent (task()) | **YES** | - | - | - |
| 写 Python 分析脚本 | - | **YES** | - | - |
| 执行 bash 命令 | - | **YES** | - | - |
| 读原始数据文件 (.txt) | **禁止** | **禁止** | **禁止** | **禁止** |
| 读输出文件 (CSV/JSON) | - | - | **YES** | **YES** |
| 解读统计结果 | - | - | **YES** | - |
| 写分析报告 (analysis_report.md) | - | - | **YES** | - |
| 写科学论文 (report.md) | - | - | - | **YES** |
| 综合呈现结果给用户 | **YES** | - | - | - |

### 各 Agent 工具权限

| Agent | 允许工具 | 禁止工具 |
|-------|---------|---------|
| code-executor | bash, read_file, write_file, ls, str_replace | task, ask_clarification, present_files, web_search, web_fetch, image_search |
| data-analyst | read_file, write_file, ls | task, ask_clarification, present_files, bash, str_replace |
| report-writer | read_file, write_file, ls | task, ask_clarification, present_files, web_search, web_fetch, bash, str_replace |

---

## 4. 关键发现

### subagent_section 与实际注册表不匹配（已修）

原 `_build_subagent_section()` 硬编码引用 `general-purpose` 和 `bash` agent，但 `BUILTIN_SUBAGENTS` 只有 `code-executor`、`data-analyst`、`report-writer`。GLM-5 看到 prompt 说"用 general-purpose"但实际没有这个 agent，导致混乱。

修改后：动态从 `get_available_subagent_names()` 读取实际注册的 agent 列表。

### GraphRecursionError 根因

LangGraph 的 `recursion_limit` 计算的是**图节点步数**（model + tools = 2 步/轮），不是对话轮数。`max_turns=25` 实际只允许约 12 轮对话。code-executor 在 12 轮中反复 `ls`、`python3 -c "import os; ..."` 探索文件，耗尽了限制。

修复：max_turns 提高到 50（约 25 轮对话），prompt 加强禁令和 few-shot 示例。

### noldus-kb MCP

- `extensions_config.json` 中已 `enabled: false`
- 代码中所有引用已移除
- 等 noldus-kb 服务准备好后再启用

### Pydantic 序列化警告

```
PydanticSerializationUnexpectedValue(Expected `none` - field_name='context')
```

LangGraph 框架内部的 `ToolRuntime.context` 泛型默认为 `None`，但运行时传入 dict。这是框架 bug，无法从应用层修复，不影响功能。

---

## 5. 未完成事项

### P0: 重启验证（最高优先级）
- [ ] 重启 `make dev`
- [ ] 新建 thread，选 **Ultra** 模式
- [ ] 上传 5 个基础斑马鱼文件
- [ ] 发送：`请分析这批斑马鱼数据。Subject 1-2 对照组，Subject 3-5 实验组。`
- [ ] 确认日志中：
  - `subagent_enabled: True`
  - Lead Agent 派遣 `code-executor`（不是 `general-purpose`）
  - code-executor **第一步是 write_file 写脚本**（不是 ls 或 find）
  - code-executor 使用 `from ethoinsight import parse, metrics`
  - 没有 GraphRecursionError
  - 没有 Thread 404 错误

### P1: MCP noldus-kb transport type
- [ ] 等 noldus-kb 服务准备好后，在 `extensions_config.json` 中设置 `enabled: true`
- [ ] 检查 DeerFlow MCP client 是否支持 `http` type（或需改回 `sse`）

### P2: 预存在的测试失败（14 个）
- [ ] `test_subagent_timeout_config.py` — 测试引用 general-purpose/bash 但 BUILTIN_SUBAGENTS 已改
- [ ] `test_subagent_prompt_security.py` — 同上
- [ ] `test_local_sandbox_encoding.py` — Windows-only 测试，WSL 环境下失败
- 需要更新测试以匹配 EthoInsight 的实际 agent 注册表

### P3: 其他范式模板
- [ ] ethoinsight 库的 templates/epm.py, open_field.py, novel_object.py 等

### P4: 前端品牌
- [ ] DeerFlow → EthoInsight（header, landing page）

---

## 6. 当前 config.yaml 状态

```yaml
models:
  - name: glm-5
    display_name: GLM-5
    model: glm-5
    base_url: https://newapi.noldusapi.com/v1
    api_key: sk-iSPNEf...
    request_timeout: 900.0
    max_retries: 3
    supports_thinking: true
    supports_vision: false

sandbox:
  allow_host_bash: true

subagents:
  enabled: true
  timeout_seconds: 900

summarization:
  trigger:
    - type: tokens
      value: 8000
  keep:
    type: messages
    value: 6
```

---

## 7. 下一位 Agent 的第一步

1. **重启 `make dev`**（本次所有修改需要重启才生效）
2. **新建 thread**（旧 thread 有脏 checkpoint），选 **Ultra** 模式
3. **只上传 5 个基础文件**：
   ```
   轨迹-Shoaling behavior with JavaScript XT180-Trial 1-Arena 1-Subject 1.txt
   轨迹-Shoaling behavior with JavaScript XT180-Trial 1-Arena 1-Subject 2.txt
   轨迹-Shoaling behavior with JavaScript XT180-Trial 1-Arena 1-Subject 3.txt
   轨迹-Shoaling behavior with JavaScript XT180-Trial 1-Arena 1-Subject 4.txt
   轨迹-Shoaling behavior with JavaScript XT180-Trial 1-Arena 1-Subject 5.txt
   ```
4. **发送**：`请分析这批斑马鱼数据。Subject 1-2 对照组，Subject 3-5 实验组。`
5. **观察日志** `packages/agent/logs/langgraph.log` 确认：
   - code-executor 第一步是 `write_file`（不是 `ls` 或 `bash("find ...")`）
   - 没有 GraphRecursionError
   - data-analyst 只读取 CSV/JSON，不跑 bash
   - report-writer 只读取文件和写报告

---

## 8. 风险与注意事项

- **旧 thread 有脏 checkpoint**：之前的失败运行可能导致 checkpoint 中有 orphan messages。务必新建 thread。
- **GLM-5 可能仍不完全遵循 prompt**：即使加了强禁令，中国模型对英文指令的遵从度可能不如 GPT-4/Claude。如果 code-executor 仍然探索文件，考虑进一步简化 prompt 或切换到更强的模型。
- **14 个预存在测试失败**：与本次修改无关，但需要后续更新测试文件以匹配 EthoInsight 的 agent 注册表。
- **config.yaml API key 明文**：生产环境应改为 `$GLM_API_KEY`。
