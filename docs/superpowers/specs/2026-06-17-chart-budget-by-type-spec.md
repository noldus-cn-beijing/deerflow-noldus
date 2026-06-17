# Spec 5 (P5): chart 预算限错维度 —— 按图类型而非数量，聚合图优先

> 日期：2026-06-17
> 类型：行为修复（红线一变体：限错维度导致有用产出被挤掉）
> 触发源：2026-06-17 EPM dogfood —— chart-maker "最多 4 图"预算把名额给了 4 张 per-subject 个体图（trajectory/heatmap），而组间对比图（box/bar，最有分析价值）即便可画也排不上（本轮还叠加 P3 被 skip）。
> 状态：待 review → 批准后 worktree 实施。**依赖 P3**（P3 修好后 box/bar 才不被 skip，本 spec 的"优先聚合图"才有对象）。
> **遵循工程实践**：[红线一可执行判据](../../refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md#红线一失败必须响亮禁止静默降级成空结果--成功状态) + 通用纪律"响亮优于沉默"（预算挤掉产出要 log）。

---

## 0. 症状
chart-maker 的 fallback-decision-tree.md:17 `bash budget ≤ 6 次 (resolve + 最多 4 plot...)`。本意防 chart-maker 跑 60 个 per-subject 图烧爆 turn（合理）。但它**限"画几张"，不限"画哪类"**：plan_charts 里 28 trajectory + 28 heatmap（per_subject）排在前面，4 张名额全被个体图吃掉，组间对比图（box_open_arm / open_arm_time_ratio_bar，aggregate）排不上。用户拿到 4 张"看不出组间差异"的个体轨迹图，而非本该有的组间对比。

## 1. 根因
预算是"数量上限"，不是"类型优先级"。而 PlanChart 本身**有类型字段** `output_mode: aggregate | per_subject`（resolve.py:953/1014）——aggregate 图（box/bar/rose，组间对比，数量少且最有用）和 per_subject 图（trajectory/heatmap，N×2 张，个体细节）语义完全不同，却被同一个"前 4 个"规则一刀切。

## 2. 修法（按类型定优先级，确定性规则）
chart-maker 执行 plan_charts 时，**按 output_mode 分桶 + 优先级排序**，而非按数组顺序取前 4：

1. **aggregate 图全画**（box/bar/rose/zone_distribution 等组间对比，通常 ≤ 5 张，是分析核心）。
2. **per_subject 图受预算限制**（trajectory/heatmap），在 aggregate 画完后用剩余预算画**代表性子集**（如每组首个 subject 各一张，而非全 28 张）。
3. 预算耗尽时，**log 出"跳过了哪些 per_subject 图、为什么"**（红线一：降级留指纹），并在 handoff 记 `remaining_charts[]`，让用户知道"还能画更多个体图"。

预算总量可适度上调（aggregate 全画 + 每组 1-2 张代表 per_subject），但**核心是优先级，不是单纯加预算**。

> 这是确定性规则（按 output_mode 排序），不靠 LLM 判断"哪张更重要"。chart-maker SKILL 改为"按 output_mode 优先级取"，而非"取前 4"。

## 3. 测试
- `test_aggregate_charts_prioritized`：plan_charts 含 5 aggregate + 56 per_subject，预算 N → 断言 5 张 aggregate 全部入选，per_subject 用剩余名额。
- `test_per_subject_budget_logged`：per_subject 被预算截断 → handoff `remaining_charts[]` 非空 + 有降级 log。
- `test_no_aggregate_still_picks_per_subject`：plan 无 aggregate（纯坐标数据）→ 退回按代表性取 per_subject（不空手）。

## 4. 风险边界
- 依赖 P3：P3 没修时 aggregate 图被 skip、根本不在 plan_charts 里，本 spec 的"优先 aggregate"无对象。P3→P5 顺序。
- output_mode 字段已存在，不新增 schema。
- 不改图脚本本身，只改 chart-maker 的选图逻辑（SKILL + 可能的选图工具）。
