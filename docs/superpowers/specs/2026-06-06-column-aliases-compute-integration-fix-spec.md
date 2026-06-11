# Spec：Column Aliases → Compute 脚本集成修复

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-06
> 范围：修复 Sprint 1 `column_aliases` 在 resolve 层面通过校验、但 compute 脚本拿到 catalog 概念名而非物理列名导致指标全部 null 的集成缺口。**仅动 resolve.py 内部**，不改 compute 脚本、不改 experiment_context.py、不改 HITL 流程。
> 前序：Sprint 1 spec [`2026-06-05-column-semantics-alignment-sprint1-spec.md`](2026-06-05-column-semantics-alignment-sprint1-spec.md)（PR #99 已合 dev）
> 前置修复：[`2026-06-06-detect-ethovision-english-locale`](../../handoffs/2026-06/2026-06-06-dogfood-real-data-findings-handoff.md)（commit `f557980d` — `detect_ethovision` 英文 marker 支持，已合 dev）
> 问题来源：[`2026-06-06 dogfood 报告`](../../handoffs/2026-06/2026-06-06-dogfood-real-data-findings-handoff.md)
> Review：Opus agent `ad9414945dd4542ce`（2026-06-06）
> 铁律：`feedback_single_source_of_truth`、`feedback_pr_merge_must_run_full_suite_on_shared_logic`、`feedback_known_full_suite_test_pollution_4_tests`

---

## 0. 给实施 agent 的必读

### 0.1 一句话

Sprint 1 的 `column_aliases` 用 `_apply_aliases` 把 columns 列表里的物理列名（`center`）替换成 catalog 概念名（`in_zone_center`），resolve 校验通过了，但 `_compute_parameters_in_use` 拿到的 Oft catalog default `center_zone = "in_zone_center"` 传给了 compute 脚本 → 脚本去数据文件里找 `in_zone_center` 列 → 找不到 → 170 个脚本全部返回 null。

### 0.2 真根因（Opus review 确认）

**`center_zone = "in_zone_center"` 不是从 columns 列表派生的，是 `oft.yaml` 的 catalog default。** `_compute_parameters_in_use` 根本不接收 columns 或 column_aliases。`anonymous_zone_is` 之所以能工作，是因为它把物理列名通过 `anonymous_zone_override.target_param` 注入进了 `overrides`。`column_aliases` 缺了这段注入逻辑。

### 0.3 与 `anonymous_zone_is` 的精确对账

`anonymous_zone_is` 链路（工作正常，不要动）：

```
anonymous_zone_is = "in_zone"
  → _detect_anonymous_zone 检测到裸 in_zone
  → anonymous_zone_override.target_param = "center_zone"
  → 注入 overrides: {center_zone: "in_zone"}   ← 物理列名！
  → compute: _find_center_zone_column(df, hint="in_zone") → 精确匹配 ✅
```

`column_aliases` 链路（当前，坏掉的）：

```
column_aliases = {"center": "in_zone_center"}
  → _apply_aliases: columns["center"] → "in_zone_center"   ← 物理列名丢了
  → _missing_columns: 看到 "in_zone_center" → 匹配 in_zone_center_* ✅
  → _compute_parameters_in_use: center_zone = oft.yaml default "in_zone_center"
  → compute: _find_center_zone_column(df, hint="in_zone_center") → 找不到 → null ❌
```

**目标链路**（本次修复要实现的）：

```
column_aliases = {"center": "in_zone_center"}
  → columns 不改，保留 "center"
  → _missing_columns(columns, column_aliases): "center" 不在 requires_center_zone glob 里，
    但 aliases 说它映射到 "in_zone_center" → 匹配通过 ✅
  → 解析：OFT 的 zone 概念 → anonymous_zone_override.target_param = "center_zone"
  → 注入 overrides: {center_zone: "center"}   ← 物理列名！
  → compute: _find_center_zone_column(df, hint="center") → 精确匹配 ✅
```

### 0.4 不做

- 不改 compute 脚本（`metrics/oft.py`、`metrics/epm.py` 等）
- 不改 `experiment_context.py`（`_derive_column_aliases` 逻辑正确）
- 不改 HITL 流程（inspect / set_experiment_paradigm / guardrail）
- 不删 `anonymous_zone_is` 机制（独立兜底路径，保持解耦）
- 不处理 Sprint 2（结构聚合）

---

## 1. 设计

### 1.1 核心思路

`column_aliases` 是语义字典（"用户列 `center` 语义上是 `in_zone_center`"），应持久保留、多处查询，而不是一次性改 columns 列表。

**两处消费 `column_aliases`：**

1. **`_missing_columns`**：匹配时，若列名不在 requires glob 里，查 aliases → 用映射后的概念名匹配 → 决定是否 missing
2. **zone param override 注入**（`resolve_metrics` / `resolve_charts` 在调 `_compute_parameters_in_use` 之前）：aliases 中映射到本范式 zone 概念的列 → 找到 `anonymous_zone_override` → 把物理列名注入 `overrides`

复用已有的 `anonymous_zone_override` 基础设施（`resolve.py:988-994`），column_aliases 的 zone 列和 `anonymous_zone_is` 走同一条注入通路。

### 1.2 关键约束

- **`columns` 列表保持物理列名不动**——`_apply_aliases` 不再改写 columns
- **`_materialize_concept` 删除**——它产的合成名 `in_zone_center_aligned` 不存在于任何数据文件，是 compute-null 的直接原因
- **`__ignore__` 移除逻辑保留**——用户标记忽略的列必须从 columns 移除
- **zero_maze 的 `open_zones` 是 `list[str]`**——`target_param` 注入时注意 `wrap_list`，与 `anonymous_zone_is` 处理一致

### 1.3 两机制 precedence

当 `column_aliases` 和 `anonymous_zone_is` 同时存在并指向同一个 `target_param` 时，`anonymous_zone_is` 优先（显式用户指定 > 派生别名）。但实际上两者不太可能同时指向同一个 param，因为 `anonymous_zone_is` 要求 `in_zone` 列存在，而有 `column_aliases` 的场景通常没有 `in_zone`。

---

## 2. 实施改动（精确到文件:行）

### 2.1 `resolve.py` `_missing_columns` — 加 aliases 参数（~15 行改动）

文件：`packages/ethoinsight/ethoinsight/catalog/resolve.py`，约 L486

```python
def _missing_columns(
    metric: dict,
    available_columns: list[str],
    column_aliases: dict[str, str] | None = None,  # 新增
) -> list[str]:
    """返回 metric 需要但 available 中缺失的列。
    若 column_aliases 提供，对不在 requires glob 中的列，
    先查 aliases 映射后的概念名再匹配。
    """
    requires = metric.get("requires_columns", [])
    missing = []
    for col_pattern in requires:
        if not _any_column_matches(available_columns, col_pattern):
            # 列名不在 requires 里 → 查 aliases
            if column_aliases:
                if _matches_via_aliases(available_columns, col_pattern, column_aliases):
                    continue
            missing.append(col_pattern)
    return missing
```

实际实现时：在现有的列匹配循环里，匹配失败后查 aliases。不要引入新的 `_matches_via_aliases` 独立函数（除非逻辑确实复杂），inline 即可：

```python
# 伪代码：在现有匹配失败时
if not matched:
    if column_aliases:
        for col in available_columns:
            concept = column_aliases.get(col)
            if concept and fnmatch.fnmatch(concept, col_pattern):
                matched = True
                break
```

### 2.2 `resolve.py` — zone param override 注入（~30 行新增）

在 `resolve_metrics()` 和 `resolve_charts()` 中，调 `_compute_parameters_in_use` 之前，从 `column_aliases` 提取本范式 zone 概念的物理列名，注入 `overrides`。

位置：`resolve_metrics()` 约 L185-195（`overrides` 字典构建处），`resolve_charts()` 约 L380-390。

```python
# 新增函数（放在 _apply_aliases 附近，约 L540）
def _build_zone_aliases_overrides(
    column_aliases: dict[str, str] | None,
    anonymous_zone_override: dict | None,
) -> dict:
    """从 column_aliases 提取 zone param overrides。
    
    对 column_aliases 中每个映射到 catalog zone 概念的物理列，
    找到对应范式的 anonymous_zone_override.target_param，
    产出 {target_param: 物理列名} 的 overrides。
    
    若 anonymous_zone_is 已为同一 target_param 设值，不覆盖（显式优先）。
    """
    if not column_aliases or not anonymous_zone_override:
        return {}
    
    target_param = anonymous_zone_override.get("target_param")
    concept_glob = anonymous_zone_override.get("zone_concept")
    if not target_param or not concept_glob:
        return {}
    
    matches = []
    for physical_col, concept in column_aliases.items():
        if fnmatch.fnmatch(concept, concept_glob):
            matches.append(physical_col)
    
    if not matches:
        return {}
    
    # target_param 是 list 类型（zero_maze 的 open_zones）→ 传列表
    # target_param 是 str 类型（OFT center_zone、LDB light_zone）→ 传字符串
    value = matches if len(matches) > 1 or target_param == "open_zones" else matches[0]
    return {target_param: value}
```

注意 `target_param` 的类型来自 catalog `anonymous_zone_override`：
- OFT: `target_param: "center_zone"` → str
- zero_maze: `target_param: "open_zones"` → list
- LDB: `target_param: "light_zone"` → str

### 2.3 `resolve.py` `_apply_aliases` — 精简为只处理 `__ignore__`（~10 行改动）

```python
def _apply_aliases(columns: list[str], aliases: dict[str, str]) -> list[str]:
    """移除用户标记为忽略的列。不再改写列为概念名。"""
    return [c for c in columns if aliases.get(c) not in (None, "__ignore__")]
```

`_materialize_concept` 删除。`_zone_concept_map` 删除。

### 2.4 `resolve.py` 所有调用点 — 传 `column_aliases`

需要改 4 处 `_missing_columns` 调用 + 2 处 overrides 注入：

| 位置 | 改动 |
|------|------|
| `resolve_metrics` 主循环 `:209` | `_missing_columns(metric, columns)` → 传 `column_aliases` |
| `resolve_metrics` include 循环 `:265` | 同上 |
| `resolve_charts` 主循环 `:404` | 同上 |
| `resolve_charts` fallback 循环 `:433` | 同上 |
| `resolve_metrics` overrides 构建处 `:185-195` | 调 `_build_zone_aliases_overrides` 注入 |
| `resolve_charts` overrides 构建处 | 同上 |

`resolve_metrics` 已有 `column_aliases` 参数（Sprint 1 加的），直接传。检查 `resolve_charts` 是否也有——如果没有，加参数并传到 `_missing_columns`。

### 2.5 `resolve.py` `load_catalog` 之后 — columns 不再被 `_apply_aliases` 改写

当前代码（`resolve_metrics` 约 L164-166）：

```python
if column_aliases:
    columns = _apply_aliases(columns, column_aliases)
```

改为：

```python
if column_aliases:
    columns = [c for c in columns if column_aliases.get(c) not in (None, "__ignore__")]
```

（即把 `_apply_aliases` 的内容 inline，只保留 ignore 移除。）

---

## 3. 不改的部分

- `experiment_context.py` `_derive_column_aliases`：逻辑正确，column_aliases 产出的 `{物理列: catalog概念}` 映射不变
- `experiment_context.py` `set_experiment_paradigm_tool`：column_semantics 写盘逻辑不变
- `inspect_uploaded_file_tool.py`：column_assessment 生成不变
- `prep_metric_plan_tool.py`：读 column_aliases 传 resolve 的流程不变，只是 resolve 内部消费方式变了
- `ev19_template_provider.py`：guardrail 逻辑不变
- 所有 compute 脚本（`metrics/oft.py`、`metrics/epm.py`、`metrics/zero_maze.py` 等）
- 所有 chart 脚本
- `anonymous_zone_is` 全路径

---

## 4. 测试（~30 行新增 + ~20 行改动）

### 4.1 新增：`parameters_in_use` 断言（red anchor → pass）

文件：`packages/ethoinsight/tests/test_column_semantics.py`

```python
def test_column_aliases_sets_physical_zone_param_in_parameters_in_use(self):
    """column_aliases 产出的 parameters_in_use 必须是物理列名，不是 catalog 概念名。"""
    # 用真实 34 文件 fixture 或合成 fixture（列含 center / 边缘区）
    plan = resolve_metrics(
        columns=["trial_time", "recording_time", "x_center", "y_center",
                 "distance_moved", "velocity", "center", "边缘区", "result_1"],
        paradigm="oft",
        column_aliases={"center": "in_zone_center", "边缘区": "in_zone_border"}
    )
    # 关键断言：center_zone 是物理列名 "center"，不是 "in_zone_center"
    center_metric = [m for m in plan.metrics if m.id == "center_time_ratio"][0]
    assert center_metric.parameters_in_use["center_zone"] == "center"
```

### 4.2 新增：zero_maze list-type param

```python
def test_column_aliases_list_param_for_zero_maze(self):
    """zero_maze 的 open_zones 是 list[str]，注入时保持列表类型。"""
    plan = resolve_metrics(
        columns=["trial_time", "x_center", "y_center", "distance_moved",
                 "velocity", "open1", "open2"],
        paradigm="zero_maze",
        column_aliases={"open1": "in_zone_open_arms_1", "open2": "in_zone_open_arms_2"}
    )
    # open_zones 参数是物理列名列表
    zm_metric = [m for m in plan.metrics if m.id == "open_time_ratio"][0]
    assert set(zm_metric.parameters_in_use.get("open_zones", [])) == {"open1", "open2"}
```

### 4.3 改动：现有 aliases 测试保留但调整断言

`test_resolve_with_column_aliases` / `test_with_concept_keyword_aliases_resolves` 等现有测试：保留 `metric.id in plan.metrics` 断言，**新增** `parameters_in_use` 断言（物理列名而非概念名）。不要删旧测试——它们是回归锚点。

### 4.4 改动：`test_apply_aliases`

如果 `_apply_aliases` 不再改写 columns（只移除 ignore），更新测试。ignore 移除的测试保留。

---

## 5. 验收标准

1. **OFT dogfood 端到端**：34 文件 → HITL 列语义确认 → `center_zone = "center"` 出现在 parameters_in_use → compute 脚本产出非 null 值 → 指标有效 → data-analyst 可消费
2. **newdemodata OFT 不回归**：`in_zone` 列 → `anonymous_zone_is` 路径 → `center_zone = "in_zone"` → 指标正常（现有行为不变）
3. **ignore 列仍然被移除**：`边缘区到center → __ignore__` → 不在 columns 中 → 不参与匹配
4. **zero_maze / LDB 的 zone 别名路径通**（如有对应测试数据）
5. **全量 ethoinsight 测试绿**（除已知 4 污染）
6. **parameters_in_use 断言落在物理列名上**

---

## 6. 合并前检查

- `cd packages/ethoinsight && pytest` 全量（改 resolve.py = 6 范式承重墙）
- `cd packages/agent/backend && make test`（column_semantics 相关测试）
- 用真实 OFT 34 文件手动跑一次 `prep_metric_plan`，检查 `parameters_in_use["center_zone"]`

---

## 7. 附带清理

以下 Sprint 1 引入、本次修复后不再需要的代码，一并删除：

| 代码 | 位置 | 原因 |
|------|------|------|
| `_materialize_concept` | `resolve.py:526` | 产合成名 `in_zone_center_aligned` — 不存在于任何数据文件 |
| `_zone_concept_map` | `resolve.py:508` | 只为 `_materialize_concept` 服务 |
| `_apply_aliases` 的 column renaming 分支 | `resolve.py:542` | 改为只处理 `__ignore__`，不重命名列 |
