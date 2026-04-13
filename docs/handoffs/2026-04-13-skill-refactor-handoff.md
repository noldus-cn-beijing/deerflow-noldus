# Phase 0c 续: Custom Skill 重构 + Subagent 限制修复 — 交接文档

> 日期: 2026-04-13
> 上一份交接: `docs/handoffs/2026-04-13-run-paradigm-analysis-handoff.md`

---

## 1. 当前任务目标

**本次会话完成了两件事**：

1. **Custom Skill 重构** — 将 3 个 custom skill 从"单文件塞所有内容"改为渐进式披露结构（SKILL.md + references/ + templates/）
2. **修复 subagent `/mnt/skills/` 读取限制** — 删除了我们之前错误添加的 `_restrict_read_file_for_subagent`，恢复 DeerFlow 框架的渐进式披露设计

---

## 2. 当前进展

### Skill 重构 ✅

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | `ethoinsight` 拆分 5 个 references/ 文件 + 重写 SKILL.md | ✅ |
| 2 | `ethoinsight-analysis` 拆分 3 个 references/ + 1 个 templates/ + 重写 SKILL.md | ✅ |
| 3 | `ethoinsight-charts` 拆分 3 个 references/ + 1 个 templates/ + 重写 SKILL.md | ✅ |
| 4 | 删除 `_restrict_read_file_for_subagent` 及调用 | ✅ |
| 5 | 框架加载验证（3 个 skill enabled=True） | ✅ |
| 6 | 回归测试（93 passed） | ✅ |
| 7 | 代码提交 | ✅ commit `dd564d7` |
| 8 | 第二轮 E2E 测试（验证修复后流水线） | ❌ 未开始 |

---

## 3. 改动的文件

### 已提交（commit `dd564d7` 及之前）

| 文件 | 改动类型 | 详情 |
|------|---------|------|
| `packages/agent/backend/.../subagents/executor.py` | **-35 行** | 删除 `_restrict_read_file_for_subagent` 函数和 `_SKILLS_PREFIX` 常量，删除 `SubagentExecutor.__init__` 中的调用 |

### 在 skills/custom/ 下（gitignored，不在 git 中）

| 文件 | 改动类型 |
|------|---------|
| `skills/custom/ethoinsight/SKILL.md` | 重写为精简入口 |
| `skills/custom/ethoinsight/references/paradigm-interpretation.md` | 新文件 |
| `skills/custom/ethoinsight/references/confound-checklist.md` | 新文件 |
| `skills/custom/ethoinsight/references/effect-size-guide.md` | 新文件 |
| `skills/custom/ethoinsight/references/statistics-decision-tree.md` | 新文件 |
| `skills/custom/ethoinsight/references/apa-reporting-format.md` | 新文件 |
| `skills/custom/ethoinsight-analysis/SKILL.md` | 重写为精简入口 |
| `skills/custom/ethoinsight-analysis/references/run-paradigm-analysis-api.md` | 新文件 |
| `skills/custom/ethoinsight-analysis/references/data-quality-checks.md` | 新文件 |
| `skills/custom/ethoinsight-analysis/references/fallback-workflow.md` | 新文件 |
| `skills/custom/ethoinsight-analysis/templates/output-contract.md` | 新文件 |
| `skills/custom/ethoinsight-charts/SKILL.md` | 重写为精简入口 |
| `skills/custom/ethoinsight-charts/references/distribution-charts.md` | 新文件 |
| `skills/custom/ethoinsight-charts/references/association-charts.md` | 新文件 |
| `skills/custom/ethoinsight-charts/references/spatial-temporal-charts.md` | 新文件 |
| `skills/custom/ethoinsight-charts/templates/chart-combinations.md` | 新文件 |

---

## 4. 关键架构决策

### DeerFlow Skill 渐进式披露机制

通过本次会话深入研究，确认了 DeerFlow skill 系统的实际工作方式：

1. **启动时**：`loader.py` 扫描 `skills/{public,custom}/`，`parser.py` 只提取 frontmatter 的 name/description/license
2. **注入 prompt**：`prompt.py` 将所有 enabled skill 拼成 XML（name + description + location 路径），加上 Progressive Loading Pattern 指令
3. **agent 匹配**：LLM 根据用户请求和 skill description 判断是否需要该 skill
4. **按需读取**：agent 调 `read_file` 读 SKILL.md 正文 → 按正文指引读 references/ 子目录
5. **每一步 read_file 都是 LLM 自己决定的**，框架不自动加载

### 删除 `_restrict_read_file_for_subagent` 的原因

- 这个函数是我们在 commit `58b4c13`（2026-04-08）错误添加的
- 它阻止 subagent 读取 `/mnt/skills/`，破坏了渐进式披露设计
- 错误消息"Skill 内容已注入到你的 system prompt 中"是误导性的 — 实际只注入了 name+description+path
- DeerFlow 原生设计中 subagent 可以通过 `read_file` 访问 skill 文件

### DeerFlow Skill 格式现状 vs 文档设计

- **文档描述**的完整 YAML 格式（prompts/tools/constraints/dependencies）在当前代码中**未实现**
- **parser.py** 只解析 frontmatter 顶层 key-value，不支持嵌套 YAML 结构
- **没有 Skill Registry**，没有"从 skill 生成可调用 Tool"的逻辑
- **当前正确格式**：YAML frontmatter（name/description 必填，version/author/license/compatibility 可选）+ Markdown 正文 + references/templates/scripts/assets 子目录

### Skill frontmatter 支持的字段

来源：`validation.py` line 12

```python
ALLOWED_FRONTMATTER_PROPERTIES = {"name", "description", "license", "allowed-tools", "metadata", "compatibility", "version", "author"}
```

其中 `allowed-tools` 存在于白名单但**完全未实现**（没有代码读取或使用它）。

---

## 5. 关键发现

### DeerFlow Skill 系统总结

| 方面 | 现状 |
|------|------|
| SKILL.md 格式 | YAML frontmatter + Markdown 正文（参照 chart-visualization、data-analysis） |
| 子目录 | references/、templates/、scripts/、assets/（manager.py ALLOWED_SUPPORT_SUBDIRS） |
| 注入方式 | name + description + path 拼成 XML，加上 Progressive Loading 指令 |
| Agent 获取内容 | 自己调 read_file 读 SKILL.md，然后按需读 references/ |
| Subagent 访问 | 现在已恢复（删除了限制），可以正常读取 /mnt/skills/ |
| Tool 注册 | Skill 不注册 tool，tool 通过 config.yaml 注册 |

### 与前一份交接的关系

前一份交接（`2026-04-13-run-paradigm-analysis-handoff.md`）中的未完成事项：
- ❌ 第二轮 E2E 测试 — 仍未完成
- ✅ 提交代码 — 已在 commit `dd564d7` 前完成
- ❌ 恢复 noldus-kb — 仍等待 `180.184.84.124:7001` 恢复
- ✅ Skill 重构 — 本次完成

---

## 6. 未完成事项

### 高优先级

1. **第二轮 E2E 测试** — 重启 `make dev`，验证：
   - lead agent 工具列表不包含 `run_paradigm_analysis`（tool_groups 过滤）
   - code-executor 能通过 `read_file` 读到 ethoinsight-analysis skill
   - code-executor 第一步调用 `run_paradigm_analysis`
   - 完整流水线 code-executor → data-analyst → report-writer 跑通

2. **恢复 noldus-kb** — 等 `180.184.84.124:7001` 恢复后改 `extensions_config.json` 为 `"enabled": true`

### 中优先级

3. **微调工作启动** — 按 `docs/plans/2026-04-13-fine-tuning-small-model-design.md` 开始数据构建
4. **更多范式模板验证** — open_field 和 epm 有指标函数但 `run_paradigm_analysis_core` 需要验证
5. **`run_paradigm_analysis_core` 单元测试** — 用 demo 数据写 pytest

### 低优先级

6. **spec 表填写** — `docs/specs/paradigm-analysis-tools-spec.md` 仍空
7. **agent.py 的 tool_groups 改动兼容性确认** — 默认 lead agent 从 `groups=None` 变为用 `tool_groups` 过滤

---

## 7. 建议接手路径

### 如果要继续 E2E 测试

```bash
cd /home/qiuyangwang/noldus-insight

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

# 验证 subagent 不再有 /mnt/skills/ 限制
cd /home/qiuyangwang/noldus-insight/packages/agent/backend && grep -c "restrict_read_file" packages/harness/deerflow/subagents/executor.py
# 应该返回 0

# 启动服务
cd /home/qiuyangwang/noldus-insight && make dev

# 上传 demo-data/DemoData/斑马鱼鱼群行为/ 下的轨迹文件
# 观察 langgraph.log：
# 1. code-executor 应该先 read_file 读 ethoinsight-analysis skill
# 2. 然后调 run_paradigm_analysis
```

---

## 8. 风险与注意事项

1. **skills/custom/ 是 gitignored**: skill 文件不在 git 中，需要手动备份或取消 gitignore。

2. **noldus-kb 临时禁用**: `extensions_config.json` 中 `"enabled": false`。不要提交这个改动。

3. **executor.py 改动已提交**: 删除 `_restrict_read_file_for_subagent` 的改动在 commit `dd564d7` 中。

4. **第二轮 E2E 未验证**: 渐进式披露能否在 GLM-5.1 上正常工作（LLM 是否会主动 read_file skill）还没测试。如果不行，可能需要在 code-executor 的 system_prompt 中加更强的正面指令。

5. **DeerFlow Skill 格式**: 当前框架只支持 frontmatter + markdown 正文，不支持文档中描述的完整 YAML 格式（prompts/tools/constraints）。这是框架的实现现状，不是我们的问题。

6. **用户偏好**: 正面指令（memory: `feedback_positive_prompting.md`）、工程师思维、CWD 是 `/home/qiuyangwang`。

---

## 下一位 Agent 的第一步建议

1. 读取本文档了解全貌
2. 重启 `make dev`，进行第二轮 E2E 测试
3. 重点观察 `langgraph.log`：
   - code-executor 是否调 `read_file` 读取 `ethoinsight-analysis/SKILL.md`
   - 读到后是否正确调用 `run_paradigm_analysis`
   - 完整流水线是否跑通
4. 如果 GLM-5.1 仍然不读 skill 文件：考虑在 code-executor system_prompt 中加一条正面指令"遇到行为数据分析任务时，先读取 ethoinsight-analysis skill"
5. E2E 通过后，考虑把 `skills/custom/` 从 gitignore 中移除并提交
