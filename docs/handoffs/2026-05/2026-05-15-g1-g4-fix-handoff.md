# 2026-05-15 G1+G4 修复交接

## 修复范围

本次修复针对 Batch A/B 检查表剩余两项 ❌：

| Issue | 问题 | 修复方式 |
|-------|------|---------|
| G1 / #3 | subagent 透传违规话术（"典型高焦虑"、"参考范围"、"金标准"） | 创建 output-constitution.md + subagent 开工必读 + lead 输出扫描 |
| G4 / #5 | 阶段播报失败（仅 1 emoji，期望 ≥4） | FIRST-TOKEN 强制规则 + 自检机制 + 反例强化 |

## 改动文件

### 新建
- `packages/agent/skills/custom/ethoinsight/references/output-constitution.md` — 输出宪法（判读哲学/元数据 grounding/表达规范/离群处理/统计表达 五大约束）

### 修改
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` — workflow 第一步加宪法读取 + gate_signals 加 `constitution_acknowledged: true`
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` — 同上
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py` — 同上
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`:
  - 角色边界硬约束加第 0 条：先读输出宪法
  - 输出规则强化：输出前强制扫描违规关键词表（4 类），从 subagent handoff 搬运的内容也必须过滤
  - 过程透明原则升级为 FIRST-TOKEN 规则 + 自检 + 反例（先说寒暄再补播报 = 违规）
  - orchestration_guide 加第 5 步：每条消息第一个 token 必须是播报 emoji

## Dogfood 复测结果

**Thread**: `0640d00f-f3b5-4dcf-9f17-e385b4efebe5`

### G1（违规话术）— 显著改善 ✅

| 指标 | 修复前（thread 8ff3be6d） | 修复后（thread 0640d00f） |
|------|--------------------------|--------------------------|
| "金标准" | ❌ 出现（"EPM 金标准要求每组至少6-8只"） | ✅ 未出现 |
| "参考范围" | ❌ 出现（"远低于正常小鼠的20-40%参考范围"） | ✅ 未出现 |
| "典型高焦虑" | ❌ 出现（"呈现典型的高焦虑样行为特征"） | ✅ 未出现 |
| "正常小鼠" | ❌ 出现 | ✅ 未出现 |
| constitution_acknowledged | ❌ 无此字段 | ✅ code-executor 和 data-analyst 均输出 `true` |
| 绝对焦虑判读 | ❌ "高焦虑样行为"、"提示较高焦虑水平" | ✅ 改为中性的"风险评估行为"描述 |
| 文献引用 | ❌ 无依据地引"文献" | ✅ 改为"EPM 文献中描述"（学术语境，合理） |

**判定**: G1 修复有效。code-executor 和 data-analyst 均读取了输出宪法并输出 `constitution_acknowledged: true`。违规术语（金标准/参考范围/典型高焦虑）在本次 dogfood 中全部消失。data-analyst 输出从违规的绝对判读转变为中性的行为模式描述。

### G4（阶段播报）— 轻微改善，未根本解决 ⚠️

| 播报场景 | 期望 | 修复前 | 修复后 |
|---------|------|--------|--------|
| 📂 dump_headers | 1 | 0 | 0 |
| 📋 catalog.resolve | 1 | 0 | 0 |
| 🧮 code-executor dispatch | 1 | 0 | 0 |
| 📋 指标计算完成 | — | 0 | 1（使用了错误的 emoji） |
| 🔬 data-analyst dispatch | 1 | 0 | 0 |
| **合计** | **≥4** | **0** | **1** |

**判定**: G4 从 0 个 emoji 提升到 1 个——有改善但远未达标。根本原因可能是 deepseek 模型在多步推理中对格式指令的遵守不够稳定——prompt-level 约束在模型需要同时进行大量工具调用和文件操作时容易丢失。

## G4 深入分析

G4 修复效果不佳的可能原因：

1. **认知负载过大**: lead 在第一轮要同时做 planning → read skills → read constitution → set_experiment_paradigm → write files → dump_headers → catalog.resolve → dispatch code-executor，播报指令在大量 thinking 和 tool call 中被淹没
2. **模型特性**: deepseek 倾向于先"想清楚"再行动，thinking 块不中断时不会产生 user-facing text
3. **播报时机与工具调用冲突**: lead 在调用 task/bash 前需要先输出播报，但 deepseek 倾向于在同一 turn 中直接调用工具

**可能的改进方向**（留给后续）:
- 在 `task()` tool description 中直接要求播报 emoji
- 在 orchestration_guide 的 Step 0 之后插入简短的强制性 notification 而非依赖 prompt 段
- 考虑 middleware 层注入播报消息（机制层保证，而非信任模型）
- 将播报要求从"规则段"移到每条 Example 中（通过示例引导而非规则声明）

## 当前 Batch A/B 状态

| Issue | 检查项 | 修复前 | 修复后 |
|-------|--------|--------|--------|
| #3 | lead 透传违规话术 | ❌ | ✅（关键违规术语消灭） |
| #5 | 阶段播报次数 ≥4 | ❌ | ⚠️（1/4，部分改善） |
| #6 | plan.json 虚拟路径 | ❌ | ✅（上轮 G5 修复） |

## Commit

`007fb390` — fix: G1+G4 修复 — 输出宪法 + 强制阶段播报
