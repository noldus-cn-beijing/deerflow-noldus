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
- bash budget ≤ 6 次 (resolve + 最多 4 plot + 文件操作)
