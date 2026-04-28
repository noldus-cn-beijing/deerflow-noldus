# 质量门控点

在规划和执行的关键节点触发检查，必要时 `ask_clarification`。

## Gate 0: 实验范式确认（仅 manual 模式）

**触发时机**: 用户选择"端到端数据分析"后，派遣任何 subagent 之前

**检查**: 用户是否已通过两级 ask_clarification 明确选择了实验范式

| 状态 | 行动 |
|------|------|
| 用户已明确大类+细分范式（如"斑马鱼鱼群行为"） | 直接调用 `set_experiment_paradigm` tool 写入 experiment-context.json，跳过两级确认 |
| 用户只明确大类（如"焦虑迷宫"） | 只问细分范式那一级（1-6 个选项），确认后调用 `set_experiment_paradigm` |
| 用户未提供任何实验类型信息 | **两级 `ask_clarification`**：先选 7 大类（旷场及物体识别 / 焦虑迷宫 / 空间学习记忆迷宫 / 社会交互与偏好 / 抑郁绝望 / 恐惧条件化 / 斑马鱼行为），再选该大类下的细分范式（1-6 个） |

**选项来源**: ethoinsight.templates.list_categories() + list_paradigms(category=...)

**强制执行**: GateEnforcementMiddleware 在 task() 调度时检查 experiment-context.json 是否存在，不存在则拦截。

## Gate 1.5: 数据-范式一致性检查（Gate 0 完成后）

**触发时机**: 用户选择范式后，Gate 1 之前

**检查**: verify_paradigm_columns(paradigm_name, file_path)

| 状态 | 行动 |
|------|------|
| match=True | 流水线继续 |
| match=False | **`ask_clarification(type="risk_confirmation")`**：告知用户数据列与范式预期不完全匹配，**不阻止分析** |
| 文件不存在 | 静默跳过（老 thread 兼容） |

**注意**: 这是"软门控"——只提醒，不阻断。即使列不匹配，用户也可选择继续。

## Gate 1: 规划阶段 — 样本量检查

**触发时机**: Step 2 需求完整性检查之后，Step 5 输出计划之前

**检查**: 每组样本数（从 `<uploaded_files>` 数量和分组定义推断）

| 每组样本数 | 行动 |
|-----------|------|
| ≥ 5 | 标准流水线，无特殊处理 |
| 3-4 | 执行标准流水线，但在 data-analyst prompt 中注明"样本量偏小，谨慎解读" |
| < 3 | **`ask_clarification`**：告知用户样本量不足，询问：(a) 仅做描述性统计 (b) 先补数据 (c) 强行跑统计（不推荐）|

## Gate 2: code-executor 返回后 — 数据质量警告

**触发时机**: code-executor 完成，返回 handoff JSON

**检查**: handoff JSON 中的 `data_quality_warnings` 字段

| 状态 | 行动 |
|------|------|
| 空数组 | 继续流水线 |
| 非空 | **`ask_clarification`**：列出警告项，询问：(a) 排除异常个体并重算 (b) 保留并继续 (c) 查看详情 |

**强制执行**: 由 GateEnforcementMiddleware 在 task(data-analyst) 调度时检查 handoff_code_executor.json，存在 critical 且未在 experiment-context.json 中 acknowledge 则拦截。

**常见 warnings**:
- 轨迹中断（missing data > 10%）
- 采样频率不一致
- 某只动物的 mean_nnd 或象限分布明显偏离群体（shoaling 范式专属）→ 提示研究员检查该个体的生物学依据

## Gate 3: code-executor 失败

**触发时机**: code-executor 返回 status=failed 或超时

**行动**: 按失败类型分支，详见 `failure-recovery.md`

## Gate 4: data-analyst 超时/空返回

**触发时机**: data-analyst 完成后检查返回内容

| 状态 | 行动 |
|------|------|
| 返回正常解读 | 继续流水线，读 handoff_data_analyst.json 向用户呈现洞察 |
| 超时 | 跳过 report-writer，直接把 handoff_code_executor.json 的统计摘要展示给用户 |
| 返回空或仅重复统计 | 视为超时处理 |

## Gate 5: report-writer 超时/空返回

**触发时机**: report-writer 完成后检查返回内容

| 状态 | 行动 |
|------|------|
| 返回正常报告 | 用 present_files 展示给用户 |
| 超时 | 用 data-analyst 的 handoff_data_analyst.json 里的 key_findings 作为最终输出 |

## 通用原则

- 每个 Gate 的 `ask_clarification` 必须给 options（用户不用思考）
- 连续 2 个 Gate 失败 → 必须 `ask_clarification` 整体方向，不能继续盲目流水线
- Gate 触发的信息必须如实告诉用户，不要隐瞒质量问题
