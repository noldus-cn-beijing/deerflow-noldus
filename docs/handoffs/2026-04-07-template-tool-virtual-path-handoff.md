# 交接文档：get_analysis_template 工具 + 虚拟路径解析

> 写给下一位接手的 AI Agent。假设你无法访问之前的对话上下文。

## 1. 当前任务目标

**核心问题**：code-executor subagent（GLM-5 驱动）在分析行为数据时不使用预制模板，而是从头写 400+ 行代码然后陷入调试循环。

**解决方案**：把模板包装成 LangChain `@tool`（`get_analysis_template`），code-executor 必须先调用工具获取参数化脚本，再按需微调执行。

**预期产出**：
- code-executor 第一个动作是调用 `get_analysis_template`
- 返回的脚本能在 sandbox 中直接执行成功
- 无需 agent 手动修改路径

## 2. 当前进展

### ✅ 已完成
- `ethoinsight/templates/tool.py` — `get_analysis_template` LangChain 工具，自动发现模板、填参数、返回可执行脚本
- `ethoinsight/templates/shoaling.py` — 加入 `# CUSTOMIZABLE` 标记系统 + `_resolve_path()` 环境变量路径解析
- `ethoinsight/templates/__init__.py` — 导出公共 API
- `config.yaml` — 注册 `get_analysis_template` 工具到 ethoinsight group
- `code_executor.py` — 重写中文 system_prompt，tools 加入 `get_analysis_template`
- `lead_agent/prompt.py` — 更新 orchestration_guide，加强文件路径传递示例
- `executor.py:232` — `recursion_limit` 从 `max_turns` 改为 `max_turns * 3`（修复 GraphRecursionError）
- `sandbox/local/local_sandbox.py` — `execute_command` 新增 `extra_env` 参数，注入 `DEERFLOW_PATH_*` 环境变量
- `sandbox/sandbox.py` — 基类签名同步更新
- `sandbox/tools.py` — `bash_tool` 传入 `_build_path_env(thread_data)`，新增 `_build_path_env()` 辅助函数
- `frontend/src/app/workspace/layout.tsx` — `CommandPalette` 改用 `dynamic(..., { ssr: false })` 修复 hydration mismatch
- 单元测试：`test_template_tool.py`、`test_analysis_template_integration.py`

### 🔄 刚完成但未测试
- **虚拟路径环境变量注入**：sandbox bash 执行时注入 `DEERFLOW_PATH_MNT_USER_DATA_UPLOADS` 等环境变量，模板脚本通过 `_resolve_path()` 读取 → 解决 Python 进程内 `glob.glob("/mnt/user-data/...")` 找不到文件的问题

## 3. 关键上下文

### 项目架构
- **DeerFlow**：LangGraph-based agent system，Lead Agent + subagents（code-executor, data-analyst, report-writer）
- **GLM-5**：中文 LLM，中文 prompt 比英文效果显著更好
- **sandbox 虚拟路径系统**：agent 看到 `/mnt/user-data/{workspace,uploads,outputs}`，物理路径是 `backend/.deer-flow/threads/{thread_id}/user-data/...`

### 路径翻译的两层机制
1. **bash 命令层**：`replace_virtual_paths_in_command()` 在 `tools.py` 中替换命令字符串里的虚拟路径 → 只对 bash 命令有效
2. **Python 进程内**（新增）：`DEERFLOW_PATH_*` 环境变量 + 模板脚本的 `_resolve_path()` → 解决 `glob.glob()`, `open()` 等 Python API 的路径问题

### 环境变量命名规则
```
/mnt/user-data          → DEERFLOW_PATH_MNT_USER_DATA
/mnt/user-data/uploads  → DEERFLOW_PATH_MNT_USER_DATA_UPLOADS
/mnt/user-data/outputs  → DEERFLOW_PATH_MNT_USER_DATA_OUTPUTS
/mnt/user-data/workspace → DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE
/mnt/skills             → DEERFLOW_PATH_MNT_SKILLS
```

### 关键决策
- Tool > Skill：Skill 是给 Lead Agent 的 Markdown 文档，subagent 看不到；Tool 是确定性函数调用
- `# CUSTOMIZABLE` 标记系统：agent 只能修改带标记的行，不能重写整个脚本
- 中文 prompt：GLM-5 对中文指令遵从度远高于英文
- `recursion_limit = max_turns * 3`：LangGraph 每轮消耗 2 个 node step（model + tools）

## 4. 关键发现

1. **路径翻译是根本障碍**：模板脚本使用虚拟路径 `/mnt/user-data/uploads/轨迹*.txt`，但 Python 进程内的 `glob.glob()` 无法解析。之前测试中 agent 花了 25+ 轮反复执行 `ls`, `python -c`, `file` 探索文件系统，最终自己用 `str_replace` 把路径改成物理路径。环境变量方案应该彻底解决这个问题。

2. **GLM-5 仍然会违反 prompt 禁令**：即使中文 prompt 明确禁止 `python -c` 探索和 `ls` 文件系统，agent 在遇到错误时还是会这样做。脚本执行成功是让 agent 快速完成的关键。

3. **Lead Agent 文件模式传递不可靠**：GLM-5 有时会把 `轨迹*.txt` 缩短为 `.txt`。已在 prompt 中加了 CRITICAL 标注 + 正确/错误示例对比。

4. **checkpoint 恢复问题**：重启 LangGraph 后会从 SQLite checkpoint 恢复旧的未完成 run。测试前必须清理 `backend/.deer-flow/checkpoints.db*` 和 `threads/`。

## 5. 未完成事项

### P0 — 必须测试
- [ ] **端到端测试虚拟路径环境变量方案**：重启 `make dev`，清理 checkpoints，新建 thread，上传 5 个斑马鱼文件，发送分析请求，确认脚本一次执行成功
- [ ] **确认 `_resolve_path()` 正确匹配环境变量**：`/mnt/user-data/uploads/轨迹*.txt` 应该先匹配 `DEERFLOW_PATH_MNT_USER_DATA_UPLOADS`，然后拼接 `轨迹*.txt`

### P1 — 功能完善
- [ ] **AioSandbox（Docker）兼容**：当前 `extra_env` 只在 `LocalSandbox.execute_command` 中实现。如果 Docker sandbox 也需要，需要在 `AioSandboxProvider` 中做类似处理
- [ ] **其他范式模板**：目前只有 `shoaling.py`。新增的模板需要同样包含 `_resolve_path()` 函数和 `global` 声明
- [ ] **单元测试未跑通**：`uv run pytest` 因网络超时（下载 fonttools/matplotlib 失败）而无法运行。网络恢复后需要跑一遍

### P2 — 已知问题
- [ ] 14 个预存测试因新 agent registry 需要更新
- [ ] Pydantic serializer warning（`PydanticSerializationUnexpectedValue` for `context` field）— 不影响功能，但日志刷屏

## 6. 建议接手路径

### 优先查看的文件
| 文件 | 作用 |
|------|------|
| `packages/ethoinsight/ethoinsight/templates/shoaling.py` | 模板脚本，含 `_resolve_path()` |
| `packages/ethoinsight/ethoinsight/templates/tool.py` | `get_analysis_template` 工具 |
| `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py` | `bash_tool` + `_build_path_env()` |
| `packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py` | `execute_command(extra_env=...)` |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` | code-executor prompt |
| `packages/agent/backend/packages/harness/deerflow/subagents/executor.py:232` | `recursion_limit = max_turns * 3` |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | orchestration_guide |

### 应先验证什么
1. 清理 checkpoints：`rm -f backend/.deer-flow/checkpoints.db* && rm -rf backend/.deer-flow/threads/*`
2. `make dev` 重启
3. 前端新建 thread → Ultra 模式 → 上传 5 个斑马鱼文件
4. 发送："请分析这批斑马鱼数据。Subject 1-2 对照组，Subject 3-5 实验组。"
5. 观察 `logs/langgraph.log`：
   - code-executor 是否第一步调用 `get_analysis_template`？
   - `analysis.py` 是否一次执行成功？
   - 是否有 `ls`, `python -c` 等探索行为？

### 推荐的下一步
如果环境变量方案测试通过 → 功能基本完成，转入 P1 完善。
如果测试失败 → 检查 `_resolve_path()` 的环境变量匹配逻辑，用 `bash("env | grep DEERFLOW")` 确认变量是否注入。

## 7. 风险与注意事项

1. **每次测试必须用新 thread**：旧 thread 有 dirty checkpoint，会导致恢复旧状态
2. **清理 checkpoints**：`rm -f backend/.deer-flow/checkpoints.db*` 是必需的，否则重启后旧 run 会自动恢复
3. **不要用英文 prompt**：GLM-5 对英文 prompt 遵从度极低，之前的英文版本被完全忽略
4. **`_resolve_path()` 需要在每个新模板中重复**：这是模板脚本的运行时代码，不是框架代码。未来可考虑抽到 ethoinsight 库中作为 utility
5. **`extra_env` 只在 LocalSandbox 实现**：Docker sandbox（AioSandbox）目前不传环境变量，如果切换到 Docker 部署需要额外处理

## 下一位 Agent 的第一步建议

1. 读取本文档
2. 读取 `packages/agent/backend/CLAUDE.md` 了解项目全貌
3. 清理 checkpoints：`rm -f /home/qiuyangwang/noldus-insight/packages/agent/backend/.deer-flow/checkpoints.db* && rm -rf /home/qiuyangwang/noldus-insight/packages/agent/backend/.deer-flow/threads/*`
4. `make dev` 重启所有服务
5. 前端新建 thread，上传斑马鱼数据文件，发送分析请求
6. 观察 `logs/langgraph.log` 确认 code-executor 行为：
   - 第一步是否调用 `get_analysis_template`？
   - `DEERFLOW_PATH_*` 环境变量是否被注入？（在日志中搜索或让 agent 执行 `env | grep DEERFLOW`）
   - `analysis.py` 是否一次成功？
7. 如果成功 → 进入 P1 任务（AioSandbox 兼容、新范式模板、跑单元测试）
8. 如果失败 → 检查 `_resolve_path()` 逻辑和环境变量命名是否匹配
