# 交接文档：knowledge-assistant subagent + 数据质量关卡 + MCP 防护 + 高价值图表

> 日期: 2026-04-08
> 原始计划: `/home/qiuyangwang/noldus-insight/docs/plans/2026-04-08-knowledge-assistant-and-quality-gate.md`

---

## 1. 当前任务目标

1. 为 Lead Agent 新增 `knowledge-assistant` subagent，处理追问、领域知识问答、产品咨询
2. 在端到端流水线中增加数据质量校验关卡
3. 新增 3 种顶刊级图表类型 + 图表选择 skill

---

## 2. 当前进展

### Part A：knowledge-assistant + 路由 + 质量关卡 ✅

| # | 改动 | 文件 |
|---|------|------|
| 1 | 新建 knowledge-assistant subagent | `subagents/builtins/knowledge_assistant.py` |
| 2 | 注册到 BUILTIN_SUBAGENTS（现在 4 个） | `subagents/builtins/__init__.py` |
| 3 | prompt.py 5 处修改：noldus_descriptions、noldus_rules（路由判断）、subagent_reminder、subagent_thinking、has_noldus_agents×2 | `agents/lead_agent/prompt.py` |
| 4 | 启用 noldus-kb MCP server | `packages/agent/extensions_config.json` |
| 5 | orchestration_guide 插入 Step 1.5 数据质量校验 | `prompt.py` |
| 6 | code-executor 增加 Step 8.5 输出校验指令 | `subagents/builtins/code_executor.py` |

所有后端文件路径前缀：`packages/agent/backend/packages/harness/deerflow/`

### Part B：Bug 修复 ✅

| # | 问题 | 修复 | 文件 |
|---|------|------|------|
| 7 | `StructuredTool` Pydantic setattr 导致所有 subagent 崩溃 | `object.__setattr__(tool, "invoke", ...)` | `subagents/executor.py:136` |
| 8 | MCP 工具返回结果无大小限制 | 三层防护：prompt 约束 + max_turns=6 + 统一截断 4096 chars | `knowledge_assistant.py` + `mcp/tools.py:28-50,137-138` |

### Part C：高价值图表 + 图表选择 skill ✅

| # | 改动 | 文件 |
|---|------|------|
| 9 | 新增 raincloud_plot、beeswarm_plot、correlogram | `packages/ethoinsight/ethoinsight/charts.py` |
| 10 | 添加 seaborn 依赖 | `packages/ethoinsight/pyproject.toml` |
| 11 | 更新模板 CHART_TYPES 注释（新增 3 种类型） | `packages/ethoinsight/ethoinsight/templates/tool.py:79` + `templates/shoaling.py:36` |
| 12 | code-executor 参数化示例添加 raincloud_plot | `subagents/builtins/code_executor.py` |
| 13 | 新建 ethoinsight-charts skill（图表选择指南） | `packages/agent/skills/custom/ethoinsight-charts/SKILL.md` |
| 14 | 启用 ethoinsight-charts skill | `packages/agent/extensions_config.json` |
| 15 | code-executor system_prompt 添加 `<图表选择>` 引导 | `subagents/builtins/code_executor.py` |

### 已验证 ✅

- `object.__setattr__` 修复生效 — 日志无 ValueError
- noldus-kb MCP 6 个工具加载成功
- code-executor 正常完成端到端流水线（8 条 AI message，status: completed）
- 9 个图表函数全部可导入：`box_plot, bar_chart, violin_plot, trajectory_plot, timeseries_plot, raincloud_plot, beeswarm_plot, correlogram, add_significance_markers`
- seaborn 已安装到 backend venv（`uv pip install --python .venv/bin/python seaborn`）

---

## 3. 关键架构知识

### 4 个 BUILTIN_SUBAGENTS

| subagent | 职责 | 有 bash？ | 画图？ |
|----------|------|----------|-------|
| code-executor | 执行 Python 分析脚本 | ✅ | ✅ 唯一能画图的 |
| data-analyst | 解读分析结果 | ❌ | ❌ |
| report-writer | 撰写 APA 报告 | ❌ | ❌ |
| knowledge-assistant | 追问 + 知识问答 | ❌ | ❌ |

### 三种工具来源

| 类型 | 来源 | 运行方式 |
|------|------|---------|
| Tool | `config.yaml` → Python 函数 | 进程内执行 |
| MCP | `extensions_config.json` → 远程服务 | HTTP 调用 |
| Skill | `skills/` → Markdown | 注入 system prompt（不可调用） |

### Lead Agent 路由

```
<uploaded_files> 有新数据文件 + 要求分析？
├── 是 → code-executor → [Step 1.5 质量校验] → data-analyst → report-writer
└── 否 → knowledge-assistant
```

### 图表选择机制

- `ethoinsight-charts` skill 注入所有 subagent 的 system prompt（DeerFlow 不支持 per-subagent 过滤）
- 只有 code-executor 有 bash 工具能实际画图，其他 subagent 看到但无法执行
- code-executor 的 `<图表选择>` 指引：默认 raincloud_plot，小样本选 beeswarm_plot
- 模板通过 `getattr(charts, chart_type)` 动态调用，CHART_TYPES 变量可通过 str_replace 修改

### Skill 注入机制

`task_tool.py:75-77` 对所有 subagent 统一注入所有启用的 skill，无 per-subagent 过滤。这是框架限制，当前可接受（多几百 tokens），未来可优化。

---

## 4. 关键发现

1. **Pydantic 兼容性**：`StructuredTool` 不允许 `setattr`，必须 `object.__setattr__`
2. **MCP 无截断**：noldus-kb `get_paradigm` 单次 16K chars，已在 `mcp/tools.py` 添加 `_wrap_tool_with_truncation()` 统一截断
3. **code-executor 性能**：8 轮 LLM × ~20s/轮 ≈ 3 分钟，瓶颈是 GLM-5 推理而非工具执行
4. **图表函数签名**：分布类图表统一签名 `(metrics, metrics_to_plot, significance=None, output_path=None) -> str`；correlogram 签名不同：无 significance 参数

---

## 5. 未完成事项

### 高优先级

- [ ] **端到端功能验证**：前端 UI 验证计划文档的 7 个测试场景
- [ ] **knowledge-assistant 路由验证**：追问（场景 2）+ 纯知识问题（场景 3）
- [ ] **新图表类型端到端验证**：上传数据，确认 code-executor 自动选用 raincloud_plot
- [ ] **noldus-kb MCP 工具调用验证**：确认 knowledge-assistant 调用 search_knowledge 等工具

### 中优先级

- [ ] **code-executor 性能优化**（3 个方向未实施）：
  - A：get_analysis_template 直接写入文件
  - B：合并 ls + read_file + handoff 检查为模板自动完成
  - C：code-executor 用更快模型（GLM-4-flash）
- [ ] **数据质量关卡验证**：上传有问题数据，确认 data_quality_warnings

### 低优先级

- [ ] 更新 README.md / CLAUDE.md
- [ ] 评估 search_knowledge 的 500 错误（其他 5 个 noldus-kb 工具正常）
- [ ] 考虑更多图表类型：山脊图、桑基图、堆叠流线图

---

## 6. 建议接手路径

### 启动

```bash
cd /home/qiuyangwang/noldus-insight
make stop
rm -f packages/agent/backend/.deer-flow/checkpoints.db*
rm -rf packages/agent/backend/.deer-flow/threads/*
make dev
```

### 验证顺序

1. Ultra 模式 → 上传数据 → "请分析" → 确认端到端正常 + 图表用了 raincloud_plot
2. 同 thread → "这个 NND 的 p 值为什么不显著？" → 确认 knowledge-assistant
3. 新 thread → "什么是高架十字迷宫？" → 确认 knowledge-assistant
4. `tail -f packages/agent/logs/langgraph.log` 看路由行为

### 关键文件

| 文件 | 用途 |
|------|------|
| `docs/plans/2026-04-08-knowledge-assistant-and-quality-gate.md` | 原始实施计划（含 7 个验证场景） |
| `packages/agent/logs/langgraph.log` | 运行时日志 |
| `packages/ethoinsight/ethoinsight/charts.py` | 图表函数实现 |
| `packages/agent/skills/custom/ethoinsight-charts/SKILL.md` | 图表选择指南 skill |
| `packages/agent/backend/packages/harness/deerflow/mcp/tools.py` | MCP 截断层 |

---

## 7. 风险与注意事项

1. **不要改 `executor.py:136`** 的 `object.__setattr__` — 改回 `tool.invoke = ...` 会崩溃
2. **MCP 截断 4096 chars 可能太小** — 调 `mcp/tools.py:25` 的 `MCP_TOOL_RESULT_MAX_CHARS`
3. **has_noldus_agents 有两处** — prompt.py line 204 和 line 746，修改必须同步
4. **seaborn 需手动安装到 venv** — `uv pip install --python .venv/bin/python seaborn`
5. **ethoinsight-charts skill 全局注入** — 所有 subagent 都收到，但只有 code-executor 能画图，无风险
