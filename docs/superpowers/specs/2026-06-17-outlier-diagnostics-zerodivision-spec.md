# Spec 1: compute_outlier_diagnostics 的 ZeroDivisionError —— median-ratio 漏 value==0 边界

> 日期：2026-06-17
> 类型：bug 修复（PR#144 回归，纯函数边界未穷举）
> 触发源：2026-06-17 EPM dogfood，真实数据含 `open_arm_time_ratio = 0.0` 的 subject（Trial 19）。
> 状态：待 review → 批准后 worktree 实施。
> 归属根病家族：**确定性函数未穷举边界值**（与"通道判据未穷举调用组合"同构）。
> **遵循工程实践**：[红线三 —— 测试 fixture 必须含真实数据边界值（0/空串/非标列名）](../../refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md#红线三测试-fixture-必须含真实数据的边界值0--空串--非标准列名禁止理想化合成数据)。**测试必须用 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28` 真实数据**（含 Trial 19 = open_arm_time_ratio 0.0，正是崩溃触发值），不得用理想化合成数据（PR#144 假绿正因用了 `[1,...,100]`）。
> **用户立场（红线一）**：ZeroDivision 不是"边界没穷举"的小事——是代码对真实数据合法值（0）做了非法假设。"用正确数据本就不该崩"，修复标准是代码本就该正确处理 0 值。

---

## 0. 症状

EPM dogfood：140 个 compute 全成功，但 statistics 段抛 `ZeroDivisionError: float division by zero` → statistics runner 失败 → `run_metric_plan` 的 `statistics_payload = None` → handoff `statistics={}` → data-analyst 走描述性 partial 路径（"仅描述性分析、未做推断检验"）。**用户拿不到任何组间检验**，尽管数据齐全、组充分（control=7 / treatment=21）。

`code-executor` 还误把它当无害："统计检验过程中遇到 ZeroDivisionError（推测为某个组内分子为 0），但不影响指标计算产物的完整性" —— 实际上它**摧毁了整个推断统计层**。

## 1. 根因：PR#144 我引入的 `compute_outlier_diagnostics` 漏了 `value==0` 这一支

`packages/ethoinsight/ethoinsight/statistics.py:421-426`：

```python
if grp_median != 0:
    ratio = max(value / grp_median, grp_median / value)   # ← value==0 时 grp_median/0 炸
elif value != 0:
    ratio = float("inf")
else:
    ratio = 1.0
```

median-ratio 判据的本意是"这个值偏离组中位数多少倍"。分支逻辑：
- `grp_median != 0` 且 `value != 0` → 双向比值，取最大（覆盖极大值和极小值两侧）—— **但代码没区分 value 是否为 0**。
- 当 `grp_median != 0` 且 **`value == 0`** → `grp_median / value` = `grp_median / 0` → ZeroDivisionError。

你的 Trial 19 `open_arm_time_ratio = 0.0`、treatment 组中位数 ≈ 0.042（非 0）→ 命中这条未守卫路径。

**语义上 value==0、median≠0 是极端偏离**（动物完全不进开放臂，组里别人都进）→ ratio 应为 `inf`（必然离群），而不是崩。

## 2. 为什么测试没抓到（假绿教训）

PR#144 的 `test_median_ratio_rule` 用 `[1,1,...,1,100]`（median=1，离群是 100）—— **离群值是极大值，不是 0**。所有测试数据都没有"value==0 且 median≠0"的组合。这是边界值未穷举的典型假绿：测了"极大值离群"，漏了"零值离群"。

## 3. 修法（治本：穷举 value/median 的 0 组合，零值离群→inf）

`statistics.py` 的 ratio 计算改为穷举四象限：

```python
# median-ratio：value 偏离组中位数的倍数。穷举 value/median 的 0 组合，
# 任一为 0 而另一非 0 = 极端偏离 = inf（必然离群）；都为 0 = 不偏离 = 1.0。
if value == 0 and grp_median == 0:
    ratio = 1.0                       # 都 0，不偏离
elif value == 0 or grp_median == 0:
    ratio = float("inf")              # 一方 0 一方非 0 = 极端偏离
else:
    ratio = max(value / grp_median, grp_median / value)   # 双向，覆盖两侧
```

> 注意：原代码 `grp_median != 0 → max(value/median, median/value)` 在 value==0 时崩。新代码先判"任一为 0"，只有两者都非 0 才做除法。`deviation_sd`（line 420）已有 `grp_std > 0` 守卫，不动。

**这是确定性数值修复，不碰 prompt。** statistics 层是 outlier/LOO 的 SSOT，它对零值健壮，data-analyst 才能拿到完整 statistics。

## 4. 测试（红→绿，穷举 0 组合）

`packages/ethoinsight/tests/test_outlier_diagnostics.py` 新增：

- `test_zero_value_nonzero_median_no_crash`（**修复前必红 = ZeroDivisionError**）：用真实 dogfood treatment 值（含 0.0：`[0.085, 0.090, 0.045, 0.0, 0.042, ...]`，median≠0）→ 断言不抛、value==0 的 subject 被标为离群（ratio==inf）。
- `test_zero_value_zero_median_ratio_one`：构造一组多数为 0（median=0）+ 个别非 0 → value==0 的 ratio=1.0（不离群），value≠0 的 ratio=inf（离群）。
- `test_all_zero_group_no_outlier`：整组全 0 → median=0、全部 value=0 → 无离群、不抛。
- 回归：原 `test_median_ratio_rule`（极大值离群）仍绿。
- **端到端**：`test_compare_groups_with_zero_value_subject` 用含 0 值的 group_summary 调 `compare_groups` → 断言返回含完整 comparisons + outlier_diagnostics，不抛（坐实 statistics 段不再因零值崩）。

红→绿拆 commit。

## 5. 风险边界

- 只改 ratio 的 0 守卫，不动 `deviation_sd`、不动 LOO、不动 compare_groups 主体。
- `inf` 进 `deviation_median_ratio` 字段 → JSON 序列化：`json.dump` 默认把 `float('inf')` 写成 `Infinity`（非法 JSON 但 Python 能读回）。**实施时确认落盘**：若 statistics.json 要严格 JSON，把 inf 在出口转成一个大哨兵值（如 `1e9`）或字符串 `"inf"`，并在 data-analyst 引用侧容忍。**实施第一步先测 `json.dumps({"r": float('inf')})` 在本仓库 save 路径的行为**，再决定是否需要哨兵化（这是本 spec 唯一的未定细节）。
- 不碰 statistics 检验逻辑（正态/方差/p 值）—— ZeroDivision 只在 outlier_diagnostics，正交。
