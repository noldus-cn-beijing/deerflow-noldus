# 质量门控点

> ✅ **2026-05-08 更新**：Gate 0 已实施 EV19 模板识别地基。旧「7 大类 18 范式」分类已删除，替换为 ethovision-paradigm-knowledge skill 体系。
>
> **新体系**：lead agent 通过 skill 渐进披露识别 EV19 模板 → 调 `set_experiment_paradigm(..., ev19_template)` 写入 experiment-context.json。
> `ev19_template` 必须在 62 变体白名单内（见 `ethovision-paradigm-knowledge` skill `references/_facts.md`）。
>
> **多层鲁棒性**：
> - L2 工具签名：`set_experiment_paradigm` 加 `ev19_template` 必填 + 白名单校验
> - L4 GuardrailMiddleware：拒绝 `ev19_template=null` 时的 `task("code-executor")` 派遣
> - L4 锁定：已设置后拒绝二次修改 `set_experiment_paradigm`（除非 `confirm_template_change=True`）
> - L6 默认值降级：反问失败时按 `default-template-fallback.md` 选默认变体
>
> GateEnforcementMiddleware 继续管 `paradigm` 字段，与 GuardrailMiddleware 职责正交。

---

在规划和执行的关键节点触发检查，必要时 `ask_clarification`。

## Gate 0: EV19 模板识别（必须）

**触发时机**: 用户上传数据并请求分析时，派遣任何 subagent 之前

**检查**: `experiment-context.json` 中 `ev19_template` 字段是否已设置

**流程**（详见 `ethovision-paradigm-knowledge` skill）:
1. agent 读 SKILL.md 决策树，综合用户文字 + 文件名推测候选
2. 候选 = 1 → 直接 `set_experiment_paradigm(paradigm, ..., ev19_template)`
3. 候选 2-3 → `ask_clarification` 给结构化选项（推荐项放第一位 + 默认值兜底）
4. 反问失败 → 查 `default-template-fallback.md` 选默认变体

**强制执行**: GuardrailMiddleware 在 `task("code-executor")` 时拦截 `ev19_template=null`

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

**分组落地（必读）**: 用户在 ask_clarification 答完分组（如"第一个是实验组"）之后，**必须**把答案翻译成 dict 传给 `prep_metric_plan(groups=...)`，详见 `ethoinsight-grouping` skill 的 [lead-translates-answer.md](../../ethoinsight-grouping/references/lead-translates-answer.md)。**不允许**在派遣 code-executor 的 prompt 里用自然语言描述分组而不传 groups dict——会导致 code-executor 幻觉脚本探测 drug 列（2026-05-28 thread 485a899d 实证）。

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
