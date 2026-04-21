# 质量门控点

在规划和执行的关键节点触发检查，必要时 `ask_clarification`。

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

**常见 warnings**:
- 某只动物总运动量异常偏高（> 对照组 200%）→ 可能运动亢进
- 某只动物总运动量异常偏低（< 对照组 50%）→ 可能运动障碍
- 轨迹中断（missing data > 10%）
- 采样频率不一致

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
