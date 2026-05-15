# 交接文档：DeerFlow Merge 后 Bug 修复与 Agent 边界定义 — 第三阶段

> 写给下一位接手的 AI Agent。假设你无法访问之前的对话上下文。

## 1. 当前任务目标

**核心问题**：第二阶段恢复了被 deerflow subtree pull 覆盖的 Noldus 专有代码（commit f61f75b），但端到端测试仍然失败。本阶段定位并修复了导致失败的 bug，并定义了 agent 职责边界。

**预期产出**：
- `make dev` → 新建 thread → 上传斑马鱼数据 → code-executor 成功执行分析脚本 → data-analyst 解读 → report-writer 报告 → 前端展示结果

## 2. 当前进展

### ✅ 已完成（本 session）

#### Bug 修复

1. **`local_sandbox.py` — `path_mappings.items()` TypeError（根因）**
   - **文件**: `packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py:248`
   - **问题**: 上游 DeerFlow 新版把 `path_mappings` 从 `dict[str, str]` 改为 `list[PathMapping]`，但 Noldus 恢复代码在 `execute_command()` 中仍用 `.items()` 遍历，导致 `AttributeError: 'list' object has no attribute 'items'`
   - **症状**: 每次 code-executor 调用 `bash("python analysis.py")` 都失败，触发反复重试直到 recursion limit
   - **修复**: 改为 `for mapping in self.path_mappings:` + `mapping.container_path` / `mapping.local_path`
   - **验证**: 手动运行 `analysis.py` 脚本成功生成全部输出（7张图 + metrics.csv + statistics.json）

2. **`executor.py` — `recursion_limit` 确认已正确**
   - **文件**: `packages/agent/backend/packages/harness/deerflow/subagents/executor.py:236`
   - **状态**: 已是 `self.config.max_turns * 3`（旧版逻辑，在第二阶段已恢复）
   - code-executor `max_turns=15` → `recursion_limit=45`

3. **`builtins/__init__.py` — 确认只注册 Noldus subagents**
   - **文件**: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/__init__.py`
   - **状态**: `BUILTIN_SUBAGENTS` 只包含 code-executor、data-analyst、report-writer（已在第二阶段恢复）

#### Agent 职责边界

4. **Lead Agent prompt — noldus_rules 重写为中文角色定义**
   - **文件**: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
   - 旧版：一堆英文 MUST NOT 规则 + 代码示例
   - 新版：简洁的中文角色分工表（调度员 / code-executor / data-analyst / report-writer）
   - `subagent_reminder` 简化为 "调度员模式：检测到数据分析需求时，按 orchestration_guide 派遣专员"
   - `subagent_thinking` 简化为 "派遣优先：涉及数据分析、图表、报告时，直接派遣对应专员"

5. **data-analyst — 加严 disallowed_tools**
   - **文件**: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`
   - 新增禁用：`web_search`、`web_fetch`、`image_search`

6. **config.yaml — 移除 web tools**
   - **文件**: `packages/agent/config.yaml`
   - 移除了 `web` tool group 及 `web_search`、`web_fetch` 工具
   - 保留：file:read、file:write、bash、ethoinsight

### ✅ 已确认完好（无需修改）
- sandbox.py 基类 `extra_env` 签名（已正确）
- tools.py `_build_path_env()` 函数（已正确）
- code-executor / data-analyst / report-writer 的 system_prompt（中文，已正确）
- EthoInsight skill（SKILL.md，已正确）
- extensions_config.json（ethoinsight enabled，其他 skill 保持原状）
- config.yaml GLM-5 配置（已正确）
- 前端品牌恢复（已在第二阶段完成）

### ⚠️ 已提交但未验证
- **commit f61f75b 包含了所有修改**，但端到端测试**尚未在重启后的新 thread 上运行**
- 之前的测试日志都是旧进程/旧 checkpoint 的残留

## 3. 未完成事项

### 必须做（P0）

- **端到端验证**：修复后还没有在干净环境下测试过
  ```bash
  cd /home/qiuyangwang/noldus-insight
  make stop
  rm -f packages/agent/backend/.deer-flow/checkpoints.db*
  rm -rf packages/agent/backend/.deer-flow/threads/*
  make dev
  ```
  然后在前端（localhost:2026）：
  1. 新建 thread
  2. 上传 5 个斑马鱼轨迹文件
  3. 发送 "分析这些斑马鱼 shoaling 数据，control=[Subject 1, Subject 2], treatment=[Subject 3, Subject 4, Subject 5]"
  4. 检查 `logs/langgraph.log` 确认：
     - lead agent 直接派遣 code-executor（不自己跑 bash）
     - code-executor 第一步调用 `get_analysis_template`
     - `bash("python analysis.py")` 成功（不报 `AttributeError`）
     - 派遣顺序正确：code-executor → data-analyst → report-writer（不并行）

### 应该做（P1）

- **Lead agent 并行派遣问题**：日志显示 lead agent 同时派遣了 code-executor 和 data-analyst，违反了 orchestration_guide 的顺序要求。可能需要进一步强化 prompt 或在框架层做限制。

- **Git commit 拆分**：当前所有改动在一个大 commit（f61f75b）中，考虑是否需要更细粒度的 commit 历史。

### 可以做（P2）

- **Handoff 机制可靠性**：data-analyst 和 report-writer 写 handoff 完全依赖模型遵守 prompt。可以考虑在 `SubagentExecutor` 层自动从最后一条 AI 消息提取结果写 handoff，减少对模型的依赖。

- **Pydantic 序列化警告**：`PydanticSerializationUnexpectedValue` 无害但刷日志，可以在 context 传递时做类型适配或在日志配置中过滤。

## 4. 关键上下文

### 根因链条

```
upstream DeerFlow 新版把 path_mappings 从 dict 改为 list[PathMapping]
  → Noldus 恢复代码在 execute_command() 中用 .items() 遍历 list
  → AttributeError: 'list' object has no attribute 'items'
  → 每次 bash tool call 都失败
  → code-executor 反复重试（ls、pwd、换路径执行）
  → 耗尽 recursion_limit
  → 触发 GraphRecursionError
```

### PathMapping 数据结构（新版）

```python
@dataclass(frozen=True)
class PathMapping:
    container_path: str   # 如 "/mnt/user-data/uploads"
    local_path: str       # 如 "/home/.../threads/{id}/user-data/uploads"
    read_only: bool = False
```

旧版是 `dict[str, str]`，key=container_path, value=local_path。

### 当前 Agent 角色分工

| 角色 | 职责 | 绝不做的事 |
|------|------|-----------|
| Lead Agent（调度员） | 理解需求 → 派遣专员 → 传达结果 | 自己跑代码、自己读数据文件、自己探索环境 |
| code-executor | 调用模板 → 执行分析脚本 | 探索文件系统、从头写代码 |
| data-analyst | 阅读分析结果 → 撰写专业解读 | 跑代码、画图 |
| report-writer | 阅读解读+数据 → 撰写科学报告 | 跑代码、重新分析 |

### 分析脚本手动验证

脚本本身**完全正常**，手动设置环境变量后可以成功运行：
```bash
cd /home/qiuyangwang/noldus-insight/packages/agent/backend
DEERFLOW_PATH_MNT_USER_DATA="./.deer-flow/threads/<thread_id>/user-data" \
DEERFLOW_PATH_MNT_USER_DATA_UPLOADS="...uploads" \
DEERFLOW_PATH_MNT_USER_DATA_OUTPUTS="...outputs" \
DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE="...workspace" \
.venv/bin/python .deer-flow/threads/<thread_id>/user-data/workspace/analysis.py
```

### GLM-5 API 行为

- 单次请求 5-60s 不等，间歇性慢
- code-executor 每轮 LLM 调用只发一个 tool call（不会批量发多个）
- 修复后理想流程约 4-5 轮 LLM 调用 × ~15s = 约 1-2 分钟

### 关键文件清单

| 文件 | 本 session 修改 |
|------|----------------|
| `packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py` | 修复 path_mappings.items() → for mapping in self.path_mappings |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | noldus_rules 重写 + subagent_reminder/thinking 简化 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` | 加严 disallowed_tools |
| `packages/agent/config.yaml` | 移除 web tool group |

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
3. 在前端新建 thread，上传斑马鱼数据，发起分析
4. 检查 `logs/langgraph.log`，确认 bash tool 不再报 `AttributeError`
5. 如果成功，观察整个 code-executor → data-analyst → report-writer 流程
6. 如果 lead agent 仍然并行派遣 subagent，需要进一步强化 orchestration_guide
