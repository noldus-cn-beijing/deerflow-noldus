# Fallback 决策树(完整)

## Case A: catalog 命中非空
if plan_charts.charts: selected = plan_charts.charts; fallback_used = False

## Case B: catalog 空 + fallback 非空 + 用户语义明确
if not charts and fallback: selected = filter_by_intent(fallback, user_intent); fallback_used = True

## Case C: catalog 空 + fallback 非空 + 用户语义模糊
if not charts and fallback: selected = fallback (全选); fallback_used = True

## Case D: catalog 空 + fallback 空
write handoff_chart_maker.json status="failed"; let lead 反问用户

## 错误处理
- 脚本失败 → 重试 ≤ 2 次 → 仍失败 → 进 handoff.errors

## 预算 = 按图类型优先级，不是数量上限（P5 / spec 2026-06-17）

**默认全画（不限资源，有多少画多少）**。chart_budget 省略 = 不限制，per_subject 个体图
逐个 subject 全部画，不按子集截断。只有 lead 在派遣 prompt 中明确给定了一个预算数字时
才传 `chart_budget`（spec 2026-06-22-chart-budget-ask-user-not-auto-throttle）——lead 只在
用户主动表达「画几张就行/代表性/少画点/挑几个」时才给数字；派遣 prompt 未给预算（默认
情形）→ 省略 `chart_budget` = 全画。「画多少」是用户的决策：用户说要图就全画，用户主动
要少画才传预算。执行者只照搬 lead 给定的预算值，不自定数字。

旧规则「最多 4 plot」是数量上限，按数组顺序取前 N —— per_subject 个体图
（trajectory/heatmap，N×类 张）排在前面吃光名额，aggregate 组间对比图（box/bar，
最有分析价值）排不上。新规则**按 `output_mode` 分桶定优先级**（确定性，不靠你判断
"哪张更重要"）：

1. **aggregate 图全画**（`plan_charts.json.charts[]` 里 `output_mode=="aggregate"` 的，
   box/bar/rose 等组间对比，通常 ≤ 5 张，分析核心）—— 不受预算挤占。
2. **per_subject 图：默认全画；仅当 lead 给了预算数字时按代表性子集画**（`output_mode=="per_subject"` 的，trajectory/heatmap
   个体图）—— 全画模式下逐个 subject 全部画；**仅当 lead 在派遣 prompt 给了 chart_budget 时**，用剩余预算画**每组首个 subject 各一张**（有分组信息时按组轮转：各组首个
   subject 的各类图先画完，再各组第二个，保证子集对每组都有代表；无分组信息时退回
   subject_index 升序），而非全 N 张。

**这层优先级已由 `catalog.resolve --chart-budget <N>` 在 plan 阶段确定性完成**：
resolve 会把"aggregate 全画 + per_subject 代表子集"写进 `plan_charts.json.charts[]`，
被预算截断的 per_subject 图写进 `charts_budget_remaining[]`。**你只需按 `charts[]`
数组顺序执行即可**（aggregate 已排在前面，先画先得）。

### 预算怎么传给 plan
plan 现由 `prep_chart_plan` 工具生成（不再 bash 拼 catalog.resolve CLI）。把 lead 给定
的预算透传给工具入参：
```
prep_chart_plan(..., chart_budget=<N>)   # lead 派遣 prompt 给了数字 → 传它
prep_chart_plan(..., chart_budget=None)  # lead 说全画/未给预算 → 省略 = 不限制（全画，charts_budget_remaining=[]）
```
省略 `chart_budget`（None）时全画（适合 lead 转达用户「所有个体图」意愿，或单 subject 场景）。
catalog.resolve 的 `--chart-budget` 是 `prep_chart_plan` 工具内部调用的入参，你不需要
也不应该自己在 bash 里跑 `catalog.resolve` 传预算。

### 降级指纹（红线一：挤掉产出要留痕）
若 `charts_budget_remaining[]` 非空 → 说明预算挤掉了 per_subject 图。把它们透传进
handoff 的 `remaining_charts[]`（每条 `{chart_id, reason:"chart_budget_truncated"}`），
让 lead/用户知道"还能画更多个体图"。bash 预算本身（≈ 1 resolve + dump_headers + batch read
+ 绘图脚本次数）仍是硬约束，绘图脚本跑不完时按下面"失败硬规范"写 partial。
