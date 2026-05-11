# 交接文档：DeerFlow Merge 恢复 — 第三阶段（Bug 修复 + Agent 边界 + Skill 限制）

> 写给下一位接手的 AI Agent。假设你无法访问之前的对话上下文。

## 1. 当前任务目标

**核心问题**：执行 `git subtree pull` 从上游 DeerFlow 拉取更新后，Noldus 专有代码被覆盖。前两个 session 恢复了代码，本 session 定位了端到端测试失败的根因并修复，同时定义了 agent 职责边界。

**预期产出**：
- `make dev` → 新建 thread → 上传斑马鱼数据 → 完整的 code-executor → data-analyst → report-writer 流程跑通

## 2. 当前进展

### ✅ 已提交（commit f61f75b）

以下改动在前两个 session 中完成并已 commit：

- sandbox.py / local_sandbox.py / tools.py — `extra_env` + `_build_path_env()` 恢复
- executor.py — `recursion_limit = self.config.max_turns * 3` 恢复
- builtins/__init__.py — 只注册 Noldus 三个 subagent
- prompt.py — Noldus orchestration_guide + 条件注入
- 前端 6 个文件品牌恢复
- .gitattributes 保护
- config.yaml — GLM-5 配置

### ✅ 已修改未提交（本 session）

#### 1. `local_sandbox.py` — path_mappings.items() TypeError 修复（根因）

**文件**: `packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py:248`

上游把 `path_mappings` 从 `dict[str, str]` 改为 `list[PathMapping]` dataclass，但恢复代码用 `.items()` 遍历 list → `AttributeError` → bash tool 每次都失败 → code-executor 反复重试 → recursion limit。

```python
# 旧（错误）
for container_path, local_path in self.path_mappings.items():
# 新（正确）
for mapping in self.path_mappings:
    env_key = "DEERFLOW_PATH_" + mapping.container_path.strip("/")...
    env[env_key] = mapping.local_path
```

**验证**: 手动运行 analysis.py 成功生成 7 张图 + metrics.csv + statistics.json。

#### 2. `prompt.py` — Lead Agent 角色边界重写

- `noldus_rules`: 英文 MUST NOT 规则 → 中文角色分工表
- `subagent_reminder`: "调度员模式：检测到数据分析需求时，按 orchestration_guide 派遣专员"
- `subagent_thinking`: "派遣优先：涉及数据分析、图表、报告时，直接派遣对应专员"

#### 3. `data_analyst.py` — disallowed_tools 加严

新增禁用：`web_search`、`web_fetch`、`image_search`

#### 4. `config.yaml` — 移除 web tools

移除 `web` tool group 及 `web_search`、`web_fetch`

#### 5. `executor.py` — Subagent 禁止读取 skill 文件

**文件**: `packages/agent/backend/packages/harness/deerflow/subagents/executor.py`

新增 `_restrict_read_file_for_subagent()` 函数：对 subagent 的 `read_file` tool 做 invoke 包装——路径匹配 `/mnt/skills/` 时返回"Skill 内容已注入到 system prompt 中"，不执行实际读取。

**背景**: DeerFlow 的 `task_tool.py`（lines 75-77）在派遣 subagent 时已自动把 `skills_section` 追加到 subagent 的 system_prompt。Subagent 不需要自己 `read_file` 读 skill 文件，但 GLM-5 会自作主张去读，浪费一轮 LLM 调用（日志中观察到 "Read ethoinsight skill for APA formatting guidance" 耗时很长）。

### ✅ 新增文档

- `docs/milestone/2026-04-07-deerflow-merge-restoration.md` — 三个阶段的完整记录
- `docs/handoffs/2026-04-07-phase3-bug-fixes-and-boundaries-handoff.md` — 上一轮交接文档

## 3. 未完成事项

### 必须做（P0）

- **Git commit**: 本 session 的修改尚未提交（executor.py + milestone 文档 + 本交接文档）
- **端到端验证**: 修复后还未在干净环境下测试
  ```bash
  cd /home/qiuyangwang/noldus-insight
  make stop
  rm -f packages/agent/backend/.deer-flow/checkpoints.db*
  rm -rf packages/agent/backend/.deer-flow/threads/*
  make dev
  ```
  新建 thread → 上传 5 个斑马鱼轨迹文件 → 检查 `logs/langgraph.log`：
  - bash tool 不再报 `AttributeError`
  - code-executor 第一步调用 `get_analysis_template`
  - 派遣顺序 code-executor → data-analyst → report-writer（不并行）
  - subagent 不出现读 `/mnt/skills/` 的 read_file 调用

### 应该做（P1）

- **Lead agent 并行派遣问题**: 日志显示 lead agent 同时派遣了 code-executor 和 data-analyst，违反顺序。可能需要进一步强化 prompt 或框架层限制
- **Handoff 机制可靠性**: data-analyst 和 report-writer 写 handoff 完全依赖模型遵守 prompt，考虑在 SubagentExecutor 层自动化

### 可选（P2）

- 评估是否将 ethoinsight skill 拆分为多个独立 skill（apa-reporting、paradigm-reference 等）
- Pydantic 序列化警告过滤

## 4. 关键上下文

### 根因链条

```
上游 path_mappings 从 dict 改为 list[PathMapping]
  → execute_command() 中 .items() 对 list 调用
  → AttributeError → bash tool 每次失败
  → code-executor 反复重试（ls/pwd/换路径）
  → recursion_limit 耗尽
  → GraphRecursionError
```

### DeerFlow Skill 加载机制

1. Lead agent system prompt 注入 `<skill_system>` 区块（`prompt.py:587-601`）
2. `task_tool.py:75-77` 派遣 subagent 时自动追加 `skills_section` 到 subagent system_prompt
3. Subagent **不应该**自己 read_file 读 skill 文件
4. `executor.py` 中的 `_restrict_read_file_for_subagent()` 强制执行此约束

### 当前 Agent 角色分工

| 角色 | 职责 | 绝不做的事 |
|------|------|-----------|
| Lead Agent（调度员） | 理解需求 → 派遣专员 → 传达结果 | 自己跑代码、自己读数据文件、自己探索环境 |
| code-executor | 调用模板 → 执行分析脚本 | 探索文件系统、从头写代码 |
| data-analyst | 阅读分析结果 → 撰写专业解读 | 跑代码、画图 |
| report-writer | 阅读解读+数据 → 撰写科学报告 | 跑代码、重新分析 |

### 关键文件路径

| 文件 | 本 session 改动 |
|------|----------------|
| `packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py` | path_mappings.items() 修复 |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | noldus_rules 中文重写 + reminder/thinking 简化 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` | disallowed_tools 加严 |
| `packages/agent/backend/packages/harness/deerflow/subagents/executor.py` | _restrict_read_file_for_subagent() |
| `packages/agent/config.yaml` | 移除 web tool group |
| `docs/milestone/2026-04-07-deerflow-merge-restoration.md` | 新增 |

### GLM-5 API 特点

- 单次请求 5-60s 不等，间歇性慢（handoff 文档提到过 2-13 分钟极端情况）
- 每轮 LLM 调用通常只发一个 tool call
- `request_timeout: 120.0` + `max_retries: 3` 已配置

## 5. 下一位 Agent 的第一步建议

1. 读取本文档
2. 先 commit 未提交的改动
3. 清理旧 checkpoint 并重启：
   ```bash
   cd /home/qiuyangwang/noldus-insight
   make stop
   rm -f packages/agent/backend/.deer-flow/checkpoints.db*
   rm -rf packages/agent/backend/.deer-flow/threads/*
   make dev
   ```
4. 在前端新建 thread，上传斑马鱼数据，发起分析
5. 检查 `logs/langgraph.log` 确认：
   - bash tool 不报 `AttributeError`
   - subagent 不读 `/mnt/skills/` 文件
   - 派遣顺序正确
6. 如果 lead agent 仍并行派遣 subagent，需要进一步强化 orchestration_guide
7. 详细的三阶段记录见 `docs/milestone/2026-04-07-deerflow-merge-restoration.md`
