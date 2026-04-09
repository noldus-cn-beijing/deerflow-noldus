# Code-Executor 架构重设计 + 范式分析需求规格 — 交接文档

> 日期: 2026-04-09
> 上一份交接: `docs/handoffs/2026-04-09-glm5-instruction-following-handoff.md`

---

## 1. 当前任务目标

**背景**: 上一轮会话完成了 GLM-5.1 指令遵循改进（prompt 精简 + 正面指令 + 代码层硬限制），但尚未验证。在本次会话中，用户决定不做验证测试，而是重新审视 code-executor 的整体架构设计。

**核心设计决策**: 将 code-executor 从"拿模板代码 → 多步执行"改为"一次 tool call 直接出结果"。

**预期产出**:
1. ✅ 新的架构设计方向（已确定）
2. ✅ 给行为学专家同事的需求收集文档（已写好）
3. 未开始：实现新架构的代码

---

## 2. 当前进展

### 架构设计讨论 ✅

经过多轮讨论，确定了以下架构：

#### 两层架构

**第一层：预制 Tool 层（code-executor subagent 调用）**
- **一个通用 tool** `run_paradigm_analysis(paradigm, file_pattern, groups, metrics?, chart_types?)`
- 通过 `paradigm` 参数区分范式（而非每个范式一个 tool）
- tool 内部直接执行完整流程：解析 → 计算指标 → 统计检验 → 生成图表 → 返回结构化 JSON
- **核心优势**: 一次 tool call 搞定，不再依赖 GLM-5.1 的多步执行能力

**第二层：自由编码 Subagent（新建）**
- 当预制 tool 返回"不支持此指标/范式"时，lead agent 降级到此 subagent
- 此 subagent 可以与用户交互，了解需求后参考预制模板的代码风格编写自定义分析代码
- 需要一个 DeerFlow skill 来约束其编码行为（必须用 ethoinsight 库、遵循模板结构、输出到正确路径）

#### 关键设计决定
1. **不是每个范式一个 tool** — 避免 tool 数量膨胀（12 范式 × N 不会是 60 个 tool）
2. **一个通用 tool + paradigm 参数** — 最终讨论到这个方向，用户尚未明确确认但倾向认同
3. **范式不支持的情况暂不考虑** — 用户认为 EthoVision 19 的实验模板基本覆盖了所有常见范式
4. **行为学指标由专家同事定义** — 用户不是行为学专家，不该由开发人员猜测指标

### 需求规格文档 ✅

**文件**: `docs/specs/paradigm-analysis-tools-spec.md`

这是一份面向**不懂代码的行为学专家**的需求收集表：
- 纯自然语言，无代码、无 JSON、无技术术语
- 用 `[请填写]` 和 `[请确认]` 标记需要专家回答的地方
- 图表选择用勾选框
- 覆盖 13 个范式 + 8 个待确认优先级的范式
- 最后请专家排出最常用的 3-5 个范式优先实现

用户的计划是让专家用自然语言回答，然后开发团队根据回答编码实现。

---

## 3. 关键上下文

### ethoinsight 库能力现状（完整审计结果）

| 能力 | shoaling | open_field | epm | 其他 17+ 范式 |
|------|----------|------------|-----|--------------|
| 范式自动检测 | ✅ | ✅ | ✅ | ✅ (utils.py PARADIGM_KEYWORDS) |
| 专用指标 | ✅ IID, NND, polarity | ✅ center_time, thigmotaxis | ✅ open_arm_time | ❌ |
| 通用指标 | ✅ distance, velocity | ✅ | ✅ | ✅ (只有这两个) |
| 统计检验 | ✅ | ✅ | ✅ | ✅ (通用 compare_groups) |
| 图表 (8种) | ✅ | ✅ | ✅ | ✅ |
| 域知识评估 | ✅ (比较型) | ✅ 阈值 | ✅ 阈值 | ❌ |
| 预制模板 | ✅ shoaling.py | ❌ | ❌ | ❌ |

**关键模块路径**:
- `packages/ethoinsight/ethoinsight/metrics.py` — `compute_paradigm_metrics()` 仅支持 shoaling/open_field/epm 三个范式分支
- `packages/ethoinsight/ethoinsight/statistics.py` — `compare_groups()` 通用，所有范式可用
- `packages/ethoinsight/ethoinsight/charts.py` — 8 种图表：box_plot, violin_plot, bar_chart, raincloud_plot, beeswarm_plot, trajectory_plot, timeseries_plot, correlogram
- `packages/ethoinsight/ethoinsight/assess.py` — 阈值仅覆盖 epm/open_field/o_maze
- `packages/ethoinsight/ethoinsight/templates/tool.py` — 当前的 `get_analysis_template` tool 实现
- `packages/ethoinsight/ethoinsight/templates/shoaling.py` — 唯一的预制模板
- `packages/ethoinsight/ethoinsight/utils.py` — `detect_paradigm()` 支持 21 个范式关键词

### 当前 code-executor 配置
- **文件**: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`
- **tools**: `["bash", "read_file", "write_file", "ls", "str_replace", "get_analysis_template"]`
- **max_turns**: 10
- **工作流**: 8 步 checklist（get_template → write_file → bash → ls → read_file → handoff）
- **问题**: GLM-5.1 经常不按 checklist 执行

### 上一轮已完成但未验证的工作
- prompt 精简 + 正面指令改写（见 `2026-04-09-glm5-instruction-following-handoff.md`）
- AI message 硬限制 + recursion_limit 修正
- **这些改动仍然有效**，但新架构可能会让部分改动变得不必要（code-executor 的 system prompt 会重写）

---

## 4. 关键发现

1. **"每个范式一个 tool"会导致 tool 爆炸** — 用户在讨论中意识到这个问题，同意改为一个通用 tool + paradigm 参数
2. **ethoinsight 库已有足够的基础设施** — 解析、统计、图表都是通用的，只缺各范式的专用指标计算
3. **指标定义必须由行为学专家提供** — 用户明确表示自己不是专家，不应该猜测指标
4. **专家不会写代码** — 需求文档必须用纯自然语言，表格用自然语言列
5. **自由编码 subagent 是兜底方案** — 当预制 tool 不够用时才启用，需要新建一个 subagent + skill

---

## 5. 未完成事项

### 高优先级（依赖专家输入）

1. **等待专家填写需求规格** — `docs/specs/paradigm-analysis-tools-spec.md` 交给专家后，等待回复
2. **根据专家回复实现各范式指标** — 在 `ethoinsight/metrics.py` 的 `compute_paradigm_metrics()` 中添加范式分支
3. **根据专家回复补充域知识阈值** — 在 `ethoinsight/assess.py` 的 `_DEFAULT_THRESHOLDS` 中添加

### 高优先级（不依赖专家）

4. **实现新的通用 tool `run_paradigm_analysis`** — 替代当前的 `get_analysis_template`
   - 位置: 可在 `ethoinsight/templates/tool.py` 中改造，或新建一个文件
   - 接口: `run_paradigm_analysis(paradigm, file_pattern, groups, metrics?, chart_types?)`
   - 内部流程: 直接调 `parse.parse_batch()` → `metrics.compute_paradigm_metrics()` → `statistics.compare_groups()` → `charts.*()` → `assess.assess_results()` → 返回 JSON
   - 关键变化: tool 内部直接 **执行** Python 代码，而不是 **返回** Python 代码让 subagent 去执行

5. **重写 code-executor subagent 配置** — `code_executor.py` 的 system prompt 大幅简化
   - 不再需要 8 步 checklist
   - tools 列表可能缩减为: `["run_paradigm_analysis", "bash", "read_file", "ls"]`
   - max_turns 可以降低（可能 5 就够了）

6. **确认通用 tool + paradigm 参数的方案** — 用户在讨论末尾倾向这个方向但对话中断，下一轮需要先确认

### 中优先级

7. **创建自由编码 subagent** — lead agent 可以在预制 tool 不够用时调用
   - 新建 subagent 配置文件（类似 code_executor.py）
   - 可以与用户交互（需要 `ask_clarification` tool）
   - 需要一个 DeerFlow skill 约束其编码行为

8. **创建行为学分析编码 skill** — 指导自由编码 subagent 的编码规范
   - 必须使用 ethoinsight 库
   - 遵循预制模板的代码结构
   - 输出到 `/mnt/user-data/outputs/`
   - 生成 handoff JSON

### 低优先级（来自上一轮交接）

9. **端到端验证测试** — 上一轮 prompt 改动的验证（如果新架构完全替换了 code-executor，这一步可能不再需要）
10. **citations 区块否定指令正面化** — `prompt.py:503-564`
11. **report-writer max_turns 调整** — 15 → 8

---

## 6. 建议接手路径

### 第一步：确认架构方案

和用户确认：
- 是否采用"一个通用 tool `run_paradigm_analysis` + paradigm 参数"的方案？
- 上一轮讨论到这里时对话结束，需要明确确认

### 第二步：用现有的 3 个范式做 MVP

不等专家回复，先用已实现的 shoaling / open_field / epm 三个范式做出 `run_paradigm_analysis` tool：
1. 改造 `ethoinsight/templates/tool.py`（或新建文件）
2. tool 内部执行完整流程，返回结构化 JSON
3. 更新 code-executor subagent 配置
4. 注册到 `config.yaml`

### 第三步：验证新架构

清理 checkpoint → 重启 → 上传旷场数据 → 检查：
- code-executor 是否只用了一次 `run_paradigm_analysis` tool call
- AI message 数是否显著减少
- 无 GraphRecursionError

### 第四步：等专家回复后扩展

根据专家填写的 `paradigm-analysis-tools-spec.md`，逐范式添加指标到 `metrics.py`

---

## 7. 风险与注意事项

1. **`run_paradigm_analysis` tool 的执行环境** — 当前 tool 在 DeerFlow 的 agent 进程中执行。如果分析脚本涉及大量计算（如大文件解析），可能需要在 sandbox 中执行。需要确认 tool 是否可以直接调用 sandbox 的 bash 来跑 Python，或者直接在进程内执行。

2. **虚拟路径翻译** — 现有的模板脚本（shoaling.py）使用 `_resolve_path()` 处理 `/mnt/user-data/` → 物理路径的映射（通过 `DEERFLOW_PATH_*` 环境变量）。新的通用 tool 内部也需要处理这个路径翻译，但执行环境可能不同——tool 在 agent 进程中执行时，sandbox 的环境变量可能不可用。

3. **图表生成需要 matplotlib** — 在 agent 进程中直接生成图表可能有 display backend 问题。当前模板使用 `Agg` backend（无头模式），需要确认 agent 进程中也能用。

4. **tool 内部 vs sandbox 执行的取舍** — 最简单的实现可能是：tool 内部组装好 Python 脚本，然后通过 sandbox 的 bash 执行（类似现在的模式，但自动化了 write_file + bash 的步骤，不再让 subagent 手动操作）。

5. **用户尚未最终确认"一个通用 tool"方案** — 讨论在这个方向结束但没有明确"就这么做"。下一轮需要先确认。

---

## 下一位 Agent 的第一步建议

1. 读取本文档了解全貌
2. 读取上一份交接 `docs/handoffs/2026-04-09-glm5-instruction-following-handoff.md` 了解 prompt 改动的背景
3. 和用户确认"一个通用 tool + paradigm 参数"的方案
4. 先用 shoaling/open_field/epm 三个已实现的范式做 MVP
5. 重写 code-executor subagent 配置
