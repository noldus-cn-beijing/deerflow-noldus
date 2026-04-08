# Milestone: DeerFlow 2.0 Merge 后 Noldus 专有功能恢复

**日期**: 2026-04-07
**状态**: 代码修复完成，待端到端验证

## 背景

执行 `git subtree pull` 从 deerflow-noldus 上游拉取了 232 个文件的更新（commit 4cd8829），覆盖了大量 Noldus 专有的代码定制。需要分阶段恢复被覆盖的内容并修复兼容性问题。

## 阶段一：基础设施恢复（commit 9c2f7be ~ bb187b4）

| 项目 | 状态 |
|------|------|
| ethoinsight 包加入 uv workspace | ✅ |
| ethoinsight symlink 恢复 | ✅ |
| 三个自定义 subagent 文件恢复 + 注册 | ✅ |
| `--allow-blocking` 加到 Makefile | ✅ |
| `.gitattributes` 保护 Noldus 专有文件 | ✅ |
| config.yaml 保留 GLM-5 配置 | ✅ |

## 阶段二：代码恢复（commit f61f75b）

### Agent 核心流程

| 文件 | 改动 |
|------|------|
| `sandbox/sandbox.py` | `execute_command` 基类签名添加 `extra_env` 参数 |
| `sandbox/local/local_sandbox.py` | `execute_command` 注入 venv PATH、`DEERFLOW_PATH_*` 环境变量、合并 `extra_env` |
| `sandbox/tools.py` | 添加 `_build_path_env()` 函数，bash_tool 传递 `extra_env=path_env` |
| `lead_agent/prompt.py` | Noldus 中文角色分工表、orchestration_guide、条件注入 subagent 描述 |

### 前端品牌恢复

| 文件 | 改动 |
|------|------|
| `layout.tsx` | title `DeerFlow` → `Noldus` |
| `workspace-header.tsx` | 折叠态/展开态品牌标识 |
| `header.tsx` / `hero.tsx` / `footer.tsx` | 品牌文字替换 |
| `about-content.ts` | 恢复 Noldus Insight 版本信息 |
| `artifact-file-list.tsx` | 添加图片内联预览（2列网格） |

### 保护机制

- `.gitattributes` 新增 5 个文件的 `merge=ours` 保护

## 阶段三：Bug 修复与边界定义（当前 session）

### Bug 修复

#### P0 — `path_mappings.items()` TypeError（根因）

- **文件**: `sandbox/local/local_sandbox.py:248`
- **问题**: 上游 DeerFlow 新版把 `path_mappings` 从 `dict[str, str]` 改为 `list[PathMapping]` dataclass，但 Noldus 恢复代码在 `execute_command()` 中仍用 `.items()` 遍历
- **症状**: 每次 bash tool call 报 `AttributeError: 'list' object has no attribute 'items'` → code-executor 反复重试 → 耗尽 recursion_limit → `GraphRecursionError`
- **修复**: `for container_path, local_path in self.path_mappings.items()` → `for mapping in self.path_mappings:` + `mapping.container_path` / `mapping.local_path`
- **验证**: 手动运行 analysis.py 脚本成功生成全部输出

#### P1 — `executor.py` recursion_limit 确认

- **文件**: `subagents/executor.py:236`
- **状态**: 已确认为 `self.config.max_turns * 3`（旧版逻辑正确恢复）

#### P1 — `builtins/__init__.py` subagent 注册

- **状态**: 已确认只注册 Noldus 三个 subagent（code-executor、data-analyst、report-writer），不含 general-purpose 和 bash

### Agent 职责边界

#### Lead Agent prompt 重写

- **文件**: `lead_agent/prompt.py`
- **策略**: 用"你是谁"定义边界，而非列一堆 MUST NOT 规则
- **改动**:
  - `noldus_rules`: 英文 MUST NOT 规则 → 中文角色分工表（调度员 / code-executor / data-analyst / report-writer）
  - `subagent_reminder`: 简化为"调度员模式：检测到数据分析需求时，按 orchestration_guide 派遣专员"
  - `subagent_thinking`: 简化为"派遣优先：涉及数据分析、图表、报告时，直接派遣对应专员"

#### data-analyst disallowed_tools 加严

- **文件**: `subagents/builtins/data_analyst.py`
- **新增禁用**: `web_search`、`web_fetch`、`image_search`

#### Subagent 禁止读取 skill 文件

- **文件**: `subagents/executor.py`
- **背景**: DeerFlow 的 `task_tool.py` 在派遣 subagent 时已自动把 `skills_section` 追加到 subagent 的 system_prompt 中。Subagent 不需要自己 `read_file("/mnt/skills/...")` 读取 skill 文件，但 GLM-5 会自作主张去读，浪费 LLM 轮次
- **修复**: 在 `SubagentExecutor.__init__` 中对 `read_file` tool 的 `invoke` 做包装——路径匹配 `/mnt/skills/` 时返回 "Skill 内容已注入到 system prompt 中" 的提示，不执行实际读取

### 配置清理

| 项目 | 改动 |
|------|------|
| config.yaml | 移除 `web` tool group 及 `web_search`、`web_fetch` 工具 |
| extensions_config.json | 保持现状（ethoinsight enabled，其他 skill 保留） |

## 关键架构决策

### 1. PathMapping dataclass（上游变更）

```python
# 旧版（Noldus 恢复代码假设的）
path_mappings: dict[str, str]  # {container_path: local_path}

# 新版（上游 DeerFlow 2.0）
@dataclass(frozen=True)
class PathMapping:
    container_path: str
    local_path: str
    read_only: bool = False
```

所有遍历 `path_mappings` 的代码都需要从 `.items()` 改为 `for mapping in` 访问。

### 2. Skill 加载机制

DeerFlow 的设计意图是 **lead agent 是唯一的 skill 管理者**：
- Lead agent 的 system prompt 中注入 `<skill_system>` 区块
- `task_tool.py` 派遣 subagent 时自动追加 `skills_section` 到 subagent 的 system_prompt
- Subagent 不应该自己读 skill 文件

### 3. Agent 角色分工

| 角色 | 职责 | 绝不做的事 |
|------|------|-----------|
| Lead Agent（调度员） | 理解需求 → 派遣专员 → 传达结果 | 自己跑代码、自己读数据文件 |
| code-executor | 调用 get_analysis_template → 执行分析脚本 | 探索文件系统、从头写代码 |
| data-analyst | 阅读分析结果 → 撰写专业解读 | 跑代码、画图 |
| report-writer | 阅读解读+数据 → 撰写科学报告 | 跑代码、重新分析 |

## 待办

- [ ] 清理 checkpoint 后端到端验证（`make stop` → 清理 → `make dev` → 新 thread 测试）
- [ ] 确认 lead agent 不再并行派遣 code-executor 和 data-analyst
- [ ] 考虑 handoff 机制的框架层自动化（当前依赖模型遵守 prompt 写 handoff JSON）
- [ ] 评估是否将 ethoinsight skill 拆分为多个独立 skill（apa-reporting、paradigm-reference 等）
