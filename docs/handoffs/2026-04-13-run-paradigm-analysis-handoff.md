# Phase 0c: run_paradigm_analysis + 工具隔离 + E2E 验证 — 交接文档

> 日期: 2026-04-13
> 上一份交接: `docs/handoffs/2026-04-09-statistics-guide-integration-handoff.md`

---

## 1. 当前任务目标

**本次会话完成了三件事**：

1. **产品方向设计** — 微调小模型替代 GLM-4/5 API 的方案（两层架构 + RAG）
2. **Phase 0c 核心实现** — `run_paradigm_analysis` 工具，一次 tool call 完成完整分析
3. **E2E 问题诊断与修复** — 手动测试发现 5 个问题，已修复 3 个核心问题

---

## 2. 当前进展

### 微调方向设计 ✅

| 产出 | 位置 |
|------|------|
| 设计文档 | `docs/plans/2026-04-13-fine-tuning-small-model-design.md` |
| Memory 记录 | `~/.claude/projects/-home-qiuyangwang/memory/project_finetune_direction.md` |

核心决策：两层架构（确定性规则引擎 + 微调 Qwen2.5-7B/Qwen3-8B）+ RAG（noldus-kb）。微调和 Phase 0c 并行推进。

### Phase 0c 实现

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | `run_paradigm_analysis_tool` + `run_paradigm_analysis_core` 函数实现 | ✅ |
| 2 | config.yaml 工具注册（group: `ethoinsight:executor`） | ✅ |
| 3 | code-executor prompt 精简（4 行） | ✅ |
| 4 | `ethoinsight-analysis` skill 创建（详细使用指南） | ✅ |
| 5 | lead agent 工具隔离（tool_groups 过滤） | ✅ |
| 6 | lead agent prompt 更新（不再引用 run_paradigm_analysis） | ✅ |
| 7 | extensions_config.json 启用 ethoinsight-analysis skill | ✅ |
| 8 | E2E 手动测试 | ✅ 已跑一轮，发现问题并修复 |
| 9 | 回归测试 | ✅ 32/32 pytest passed |
| 10 | 提交代码 | ❌ 未提交 |
| 11 | 修复后的第二轮 E2E 测试 | ❌ 未开始 |

---

## 3. 改动的文件

| 文件 | 改动类型 | 详情 |
|------|---------|------|
| `packages/ethoinsight/ethoinsight/templates/tool.py` | **+301 行** | 新增 `run_paradigm_analysis_core()`（纯函数）+ `run_paradigm_analysis_tool()`（@tool 包装 + ToolRuntime 路径解析）+ `_resolve_virtual_path()` |
| `packages/ethoinsight/ethoinsight/templates/__init__.py` | 更新导出 | 新增 `run_paradigm_analysis_tool` 和 `run_paradigm_analysis_core` |
| `packages/agent/config.yaml` | 工具注册 | 新增 `run_paradigm_analysis`，两个 ethoinsight tool 的 group 改为 `ethoinsight:executor` |
| `packages/agent/backend/.../lead_agent/agent.py` | **+13 行** | lead agent 默认用 config.yaml `tool_groups` 做过滤，排除 `ethoinsight:executor` |
| `packages/agent/backend/.../lead_agent/prompt.py` | 小改 | code-executor 描述和示例中移除 `run_paradigm_analysis` 引用 |
| `packages/agent/backend/.../code_executor.py` | **重写** | prompt 从 ~50 行精简到 4 行；max_turns 10→8 |
| `packages/agent/skills/custom/ethoinsight-analysis/SKILL.md` | **新文件** | code-executor 的分析执行指南（工具参数、统计方法、质量检查、fallback） |
| `packages/agent/extensions_config.json` | 更新 | 启用 `ethoinsight-analysis` skill；临时禁用 noldus-kb |
| `docs/plans/2026-04-13-fine-tuning-small-model-design.md` | **新文件** | 微调方向设计文档 |

---

## 4. 关键架构决策

### tool.py 分层设计

- `run_paradigm_analysis_core()` — **纯函数**，不依赖 langchain，接受物理路径，可独立测试
- `run_paradigm_analysis_tool()` — **@tool 包装**，用 `ToolRuntime` 解析虚拟路径后调用 core 函数
- `tool.py` 顶层 import `from langchain.tools import ToolRuntime, tool` — 只在 DeerFlow agent 进程中加载（langchain 1.2.3），ethoinsight 独立 pytest 不触发此 import

### 路径解析

`_resolve_virtual_path()` 处理 `/mnt/user-data/` → 物理路径：
- Local sandbox：从 ToolRuntime 的 `thread_data` 映射（workspace_path/uploads_path/outputs_path）
- Docker (aio sandbox)：`/mnt/user-data/` 直接是挂载路径，thread_data 为 None，原样返回

### 工具隔离（lead agent 看不到 subagent 专用工具）

**问题**：DeerFlow 的 lead agent 默认 `groups=None` 加载所有工具。

**解决方案**（利用框架已有能力）：
1. config.yaml 中 `run_paradigm_analysis` 和 `get_analysis_template` 的 group 改为 `ethoinsight:executor`
2. `tool_groups` 声明中不包含 `ethoinsight:executor`
3. `agent.py` 中默认 lead agent 用 config.yaml 的 `tool_groups` 做过滤

**修改位置**: `packages/agent/backend/.../lead_agent/agent.py` line 342-356

**效果验证**:
- Lead agent: 8 tools（不含 run_paradigm_analysis、get_analysis_template）
- Subagent 全量池: 9 tools（含两个 ethoinsight tool，code-executor 通过 allowlist filter 拿到）

### code-executor prompt 精简 + skill 注入

**问题**：code-executor 的 ~50 行 prompt 太长，GLM-5.1 忽略了关键指令（调用 `run_paradigm_analysis`），反而去 `bash file` 和 `bash iconv` 手动探索文件。

**解决方案**：
- system_prompt 精简到 4 行：只说"提取参数 → 调 run_paradigm_analysis → 返回结果"
- 详细知识（工具参数说明、统计方法选择、数据质量检查、fallback 流程、输出契约）移到 `ethoinsight-analysis` skill
- DeerFlow 框架自动将 enabled skills 注入到 subagent 的 system_prompt 后面（`task_tool.py` line 87-89）

---

## 5. E2E 测试发现的 5 个问题及修复状态

| 问题 | 严重度 | 状态 | 修复方式 |
|------|--------|------|---------|
| 1. code-executor 没调 run_paradigm_analysis，用 bash 手动探索 | 严重 | ✅ 已修复 | prompt 精简 + skill 注入 |
| 2. max_turns=5 太低，5 轮用完没做正事 | 严重 | ✅ 已修复 | 改为 8 |
| 3. lead agent 自己调了 run_paradigm_analysis | 中等 | ✅ 已修复 | tool_groups 过滤 + prompt 移除引用 |
| 4. GLM-5.1 API 不稳定（429 限流、超时） | 中等 | ❌ 用户解决 | 长期靠微调本地模型 |
| 5. MCP 初始化阻塞 83 秒（noldus-kb 503） | 低 | ⚠️ 临时禁用 | extensions_config.json enabled=false |

---

## 6. 未完成事项

### 高优先级

1. **第二轮 E2E 测试** — 重启 `make dev`，验证修复后 code-executor 是否正确调用 `run_paradigm_analysis`
2. **提交代码** — 所有改动尚未 git commit
3. **恢复 noldus-kb** — 等 `180.184.84.124:7001` 恢复后改 `extensions_config.json` 为 `"enabled": true`

### 中优先级

4. **微调工作启动** — 按 `docs/plans/2026-04-13-fine-tuning-small-model-design.md` 开始数据构建
5. **更多范式模板** — open_field 和 epm 有指标函数但 `run_paradigm_analysis_core` 需要验证
6. **`run_paradigm_analysis_core` 单元测试** — 用 demo 数据写 pytest

### 低优先级

7. **agent.py 的 tool_groups 改动** — 当前改动影响 DeerFlow 框架的默认行为（默认 lead agent 从 `groups=None` 变为用 `tool_groups` 过滤）。如果有其他项目基于 DeerFlow，需要确认兼容性。
8. **spec 表填写** — `docs/specs/paradigm-analysis-tools-spec.md` 仍空

---

## 7. 建议接手路径

### 如果要继续 E2E 测试

```bash
cd /home/qiuyangwang/noldus-insight

# 查看改动
git diff --stat

# 确认 noldus-kb 已禁用
grep '"enabled"' packages/agent/extensions_config.json | head -1

# 回归测试
cd packages/ethoinsight && uv run python -m pytest tests/test_statistics.py -v

# 验证 lead agent 工具隔离
cd /home/qiuyangwang/noldus-insight/packages/agent/backend && uv run python -c "
from deerflow.tools.tools import get_available_tools
from deerflow.config.app_config import get_app_config
app_config = get_app_config()
groups = [g.name for g in app_config.tool_groups]
tools = get_available_tools(groups=groups, subagent_enabled=True)
print([t.name for t in tools])
# 应该不包含 run_paradigm_analysis
"

# 启动服务
cd /home/qiuyangwang/noldus-insight && make dev

# 上传 demo-data/DemoData/斑马鱼鱼群行为/ 下的轨迹文件
# 观察 langgraph.log：code-executor 应该第一步就调 run_paradigm_analysis
```

### 如果要提交代码

```bash
cd /home/qiuyangwang/noldus-insight
git add packages/ethoinsight/ethoinsight/templates/tool.py \
       packages/ethoinsight/ethoinsight/templates/__init__.py \
       packages/agent/config.yaml \
       packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py \
       packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py \
       packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py \
       packages/agent/skills/custom/ethoinsight-analysis/SKILL.md \
       docs/plans/2026-04-13-fine-tuning-small-model-design.md
# 注意不要提交 extensions_config.json（noldus-kb 临时禁用）
```

---

## 8. 风险与注意事项

1. **代码尚未提交**: 5 个修改文件 + 2 个新文件在工作区中。

2. **noldus-kb 临时禁用**: `extensions_config.json` 中 `"enabled": false`。不要提交这个改动。

3. **agent.py tool_groups 改动影响框架行为**: 默认 lead agent 从加载全部工具变为只加载 `tool_groups` 中声明的 group。如果后续在 config.yaml 中添加新 tool group 但忘记在 `tool_groups` 中声明，lead agent 将看不到。这是一个有意识的 trade-off。

4. **第二轮 E2E 未验证**: 修复了 3 个核心问题但还没有重新测试。GLM-5.1 可能仍然不按指引调用工具。

5. **用户偏好**: 正面指令（memory: `feedback_positive_prompting.md`）、工程师思维、CWD 是 `/home/qiuyangwang`。

6. **DeerFlow skill 注入机制**: `task_tool.py` line 87-89 自动把所有 enabled skills 注入 subagent prompt。`ethoinsight-analysis` skill 会同时注入到 data-analyst 和 report-writer，但因为 skill 内容是关于 run_paradigm_analysis 工具的，这两个 subagent 看不到该工具，所以不会产生误导（它们没有这个工具在 allowlist 中）。

---

## 下一位 Agent 的第一步建议

1. 读取本文档了解全貌
2. 重启 `make dev`，进行第二轮 E2E 测试
3. 查看 `langgraph.log` 验证：
   - lead agent 的工具列表不包含 `run_paradigm_analysis`
   - code-executor 第一步就调用 `run_paradigm_analysis`
   - 完整流水线 code-executor → data-analyst → report-writer 跑通
4. 如果 GLM-5.1 仍然不按指引：考虑进一步减少 code-executor 的 tools 列表（去掉 bash），或在 skill 中用更强的正面指令
5. 测试通过后提交代码
