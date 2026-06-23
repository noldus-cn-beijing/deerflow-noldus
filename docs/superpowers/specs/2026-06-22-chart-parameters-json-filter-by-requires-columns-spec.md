# Spec：chart `--parameters-json` 按 chart 的 requires_columns 裁剪 —— 修 `plot_open_arm_time_ratio_bar` 全失败

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-22
> 性质：结构性根治（catalog chart 层）。`resolve_charts` 给每张图注入的 `--parameters-json` 是**全范式 zone 参数的并集**（`open_arm_zones` + `closed_arm_zones`），不按该 chart 实际用到的 zone 概念裁剪。底层 compute 函数签名严格的图（`compute_open_arm_time_ratio` 只收 `open_arm_zones`）被强塞了它不认识的 `closed_arm_zones` → `TypeError` → chart-maker 靠手删参数重跑 5 次兜底。**每次 EPM（任何用 HITL 列对齐的范式）都会触发。**
> 关联：
> - 直接前序：`docs/superpowers/specs/2026-06-18-plot-scripts-zone-param-alignment-spec.md`（让 per-subject plot 脚本接收并透传 zone 参数——**那份 spec 解决了「plot 拿不到参数」，却同时埋下了「拿到的参数过宽」**：它要求注入「对齐后的 zone 参数」全量，未要求按 chart 裁剪。本 spec 是它的收尾补丁）
> - 姊妹问题（不同根因）：`docs/handoffs/2026-06/2026-06-22-ecs-dogfood-chart-maker-turn-exhaustion-per-subject-path-handoff.md`（chart-maker turn 耗尽，per-subject 路径）
> - 调研来源：`docs/problems/2026-06-22-thread-3a41e483-four-issues-investigation.md` B 段
> - SSOT 纪律：memory `feedback_single_source_of_truth`、`feedback_ssot_lives_in_review_packages`
> - 列对齐铁律：memory `feedback_lead_inverts_fewzones_vs_nozones_by_column_name`、`feedback_seal_failure_third_root_cause_list_zone_params_reject_scalar_schema`

---

## 〇、给实施 agent 的一句话

`resolve_charts`（[resolve.py:444](../../../packages/ethoinsight/ethoinsight/catalog/resolve.py#L444)）调一次 `_build_zone_aliases_overrides(column_aliases, cat, {})` 得到**一份全局 zone_overrides**（如 `{"open_arm_zones": ["open"], "closed_arm_zones": ["closed"]}`），然后 L463 把**同一份**传给每个 `_chart_to_plan`；`_chart_to_plan`（L1075/L1102）无条件 `json.dumps(zone_overrides)` 进 `--parameters-json`。**每张图收到的是全范式 zone 参数并集，而非该图关联概念子集**。`plot_open_arm_time_ratio_bar` 底层只要 `open_arm_zones`，被塞了 `closed_arm_zones` → TypeError。

治本：**chart 的 `requires_columns` 就是权威裁剪依据**。按该 chart 的 `requires_columns` 里匹配到的 zone 概念（`in_zone_open_arms_*` → open 概念；`in_zone_closed_arms_*` → closed 概念），只注入这些概念对应的 zone param。不需要新建 chart↔metric 关联表——`requires_columns` 是 chart 已显式声明、loader 校验过的字段（[schema.py:89](../../../packages/ethoinsight/ethoinsight/catalog/schema.py#L89)）。

---

## 一、根因（逐字节实证，dogfood thread `3a41e483`）

### 1.1 现象

chart-maker 第一次并行跑 10 张图（前端 `Experimental paradigm(1).txt` 1035-1098 行）：

| 图 | output_mode | 底层 compute | 结果 |
|---|---|---|---|
| `plot_box_open_arm` | aggregate | 需 open+closed | ✅ |
| `plot_trajectory_s0-s3` | per_subject | 不调 zone metric | ✅ |
| `plot_open_arm_time_ratio_bar_s0-s4` | per_subject | `compute_open_arm_time_ratio`（**只收 `open_arm_zones`**） | ❌ ×5 全失败 |

5 张失败的错误：`TypeError: compute_open_arm_time_ratio() got an unexpected keyword argument 'closed_arm_zones'`。

chart-maker 自己诊断出 `--parameters-json` 里塞了 `{"closed_arm_zones": ["closed"], "open_arm_zones": ["open"]}`，手动改成只带 `open_arm_zones` 重跑 5 次才成功（前端 1098-1178）。

### 1.2 因果链（代码坐实）

```
resolve_charts (resolve.py:444)
  └─ zone_aliases_overrides = _build_zone_aliases_overrides(column_aliases, cat, {})
       → 返回 {"open_arm_zones": ["open"], "closed_arm_zones": ["closed"]}   ← 全范式并集，正确
  └─ for ch in charts:
       _chart_to_plan(ch, ..., zone_overrides=zone_aliases_overrides)         ← 同一份传给每张图 (L463)
         └─ if zone_overrides:
              args.extend(["--parameters-json", json.dumps(zone_overrides)])  ← 无条件全量注入 (L1075/L1102)
```

`_chart_to_plan` **不看 `ch.requires_columns`**，把整份 zone_overrides 序列化进每张图的 args。

### 1.3 为什么 `box_open_arm` 没炸、`open_arm_time_ratio_bar` 炸

- `box_open_arm` 底层要 open+closed 两个 zone → 全量参数恰好都认识 → 不炸
- `open_arm_time_ratio_bar` 底层 `compute_open_arm_time_ratio(open_arm_zones=...)` **严格签名**（epm.py，只列 `open_arm_zones` 参数）→ 收到不认识的 `closed_arm_zones` kwarg → TypeError

本质：**chart A 需要的参数集 ⊃ chart B 需要的参数集，系统给每张图都发「并集」而非「该图自己的子集」**。签名宽容的函数（`**kwargs` 或显式列全）容错，签名严格的炸。

### 1.4 catalog yaml 里 chart 已声明了它需要哪些 zone 概念

`epm.yaml` 的 charts 段（实证）：

```yaml
box_open_arm:              requires_columns: ['in_zone_open_arms_*']                         # 只要 open
open_arm_time_ratio_bar:   requires_columns: ['in_zone_open_arms_*']                         # 只要 open
zone_entry_distribution:   requires_columns: ['in_zone_open_arms_*', 'in_zone_closed_arms_*'] # open+closed
trajectory:                requires_columns: ['x_center', 'y_center']                        # 不需 zone
```

`requires_columns` 里的 `in_zone_<concept>_*` glob 就是该 chart 用到的 zone 概念的权威声明。`box_open_arm` 和 `open_arm_time_ratio_bar` 都只声明了 open → 它们都**只该**收到 `open_arm_zones`。当前却收到了 open+closed。

**这是裁剪的最干净依据**：复用 chart 自己声明的 `requires_columns`，不需要新建 chart↔metric 关联（避免动 SSOT、避免拍映射）。

---

## 二、设计

### 2.1 裁剪依据：chart 的 requires_columns → zone 概念 → zone param

新增一个 helper（复用 `_build_zone_aliases_overrides` 内部已有的 concept→param 映射逻辑）：

```
chart.requires_columns 里的 in_zone_<concept>_* glob
  → 提取 concept 关键词（open / closed / center / ...，复用 _extract_concept_keyword）
  → 查 cat.resolved_zone_concepts 得 (concept → param, wrap_list)
  → 只保留这些 concept 对应的 param
```

例：
- `open_arm_time_ratio_bar`（`requires_columns: ['in_zone_open_arms_*']`）→ concept=open → 只留 `open_arm_zones`
- `zone_entry_distribution`（open+closed）→ 留 `open_arm_zones` + `closed_arm_zones`
- `trajectory`（无 in_zone glob）→ 留空 → 不注入 `--parameters-json`

### 2.2 注入点：`_chart_to_plan` 收到的是裁剪后的子集

两种实现位置（实施 agent 选一，推荐 A）：

**A. 在 `resolve_charts` 调 `_chart_to_plan` 前，按 chart 裁剪**（推荐——裁剪逻辑集中、`_chart_to_plan` 签名不变）：

```python
# resolve.py resolve_charts 内，L463 附近
for ch in cat.charts:
    if not _chart_columns_available(ch, available_columns):
        skipped.append(...); continue
    chart_zone_overrides = _filter_zone_overrides_for_chart(ch, zone_aliases_overrides, cat)
    charts.extend(_chart_to_plan(ch, raw_files, workspace_dir, ..., zone_overrides=chart_zone_overrides))
```

**B. 把 cat 传进 `_chart_to_plan`，在内部裁剪**——签名变重，不推荐。

### 2.3 `_filter_zone_overrides_for_chart` 设计

```python
def _filter_zone_overrides_for_chart(
    ch: ChartEntry,
    zone_overrides: dict[str, Any],
    cat: Catalog,
) -> dict[str, Any]:
    """按 chart.requires_columns 声明的 zone 概念，裁剪 zone_overrides 到该图子集。

    chart 没声明任何 in_zone_* 概念（如 trajectory）→ 返回 {}（不注入）。
    chart 声明了 open → 只留 open 对应的 param。
    """
    if not zone_overrides:
        return {}
    # 1. 从 ch.requires_columns 提取该 chart 用到的 zone concept 关键词
    chart_concepts: set[str] = set()
    for pat in _flatten_requires_columns(getattr(ch, "requires_columns", [])):
        kw = _extract_concept_keyword(pat)  # "in_zone_open_arms_*" → "open"
        if kw:
            chart_concepts.add(kw)
    if not chart_concepts:
        return {}  # trajectory/heatmap 等不依赖 zone 的图
    # 2. 这些 concept → 该留哪些 param（复用 resolved_zone_concepts）
    allowed_params: set[str] = set()
    for concept_key, rc in cat.resolved_zone_concepts.items():
        if rc.binding is None:
            continue
        # concept_key 形如 "open" / "closed"；命中 chart 声明的 concept 之一即保留其 param
        if concept_key in chart_concepts or any(concept_key == c or concept_key.startswith(c) for c in chart_concepts):
            allowed_params.add(rc.binding.param)
    # 3. 只留 allowed_params 里的 override
    return {k: v for k, v in zone_overrides.items() if k in allowed_params}
```

> ⚠️ concept_key 与 requires_columns 关键词的匹配规则需对齐 `_build_zone_aliases_overrides` / `_concept_matches_pattern` 已有的归一化逻辑（EPM 的 concept 是 `open`/`closed`，requires_columns 是 `in_zone_open_arms_*`/`in_zone_closed_arms_*`）。**实施时先 grep `_build_zone_aliases_overrides` 的 Step 3（concept→pattern 匹配）确认归一化规则，复用同一套，不要新写匹配**（守 SSOT）。

### 2.4 不改 metric 路径

`resolve_metrics`（L216）的 zone_overrides 注入给 `compute_*` 脚本——**那条路径不动**。compute 脚本由 `plan_metrics.json` 的 `parameters_in_use` 驱动（已按 metric 裁剪，[resolve.py:258](../../../packages/ethoinsight/ethoinsight/catalog/resolve.py#L258)），本来就准。本 spec 只修 chart 路径。

### 2.5 plot 脚本侧不加 `**kwargs` 兜底（治标，不做）

调研文档 B.1 提的「compute 函数加 `**_` 吞多余 kwargs」是治标——会让「参数注入错误」变成静默错误（chart 收到它根本不该收的参数却不报错），掩盖未来 catalog 配置 bug。**本 spec 不采纳**。治本在注入端裁剪，让每张图只收到它声明的参数；底层函数保持严格签名，配置错了就响亮地炸。

---

## 三、改动清单

### 3.1 `resolve.py` —— 新增 `_filter_zone_overrides_for_chart` + 在 `resolve_charts` 调用

- 新增 `_filter_zone_overrides_for_chart(ch, zone_overrides, cat)`（§2.3）。
- `resolve_charts`（L444-500 附近）：拿到 `zone_aliases_overrides` 后，对每个 chart 先 `_filter_zone_overrides_for_chart(ch, zone_aliases_overrides, cat)` 再传给 `_chart_to_plan`。
- **fallback 路径同改**：L493 附近的 `fallback.extend(_chart_to_plan(...))` 也要走裁剪（守两条路径一致）。
- 复用 `_flatten_requires_columns`、`_extract_concept_keyword`、`cat.resolved_zone_concepts`，**不新写匹配逻辑**。

### 3.2 不改 `_chart_to_plan` 签名

`_chart_to_plan` 仍收 `zone_overrides: dict | None`，只是调用方传入的是裁剪后的子集。L1075/L1102 的注入逻辑不变。

### 3.3 不改 catalog yaml、不改 plot 脚本、不改 metric 函数

`requires_columns` 已正确声明，无需动 yaml。plot 脚本接收逻辑（06-18 spec 已修）不变。metric 函数签名不变。

---

## 四、测试（红→绿坐实，TDD 强制）

新建 `packages/ethoinsight/tests/test_chart_zone_overrides_filtered.py`：

1. **`test_open_arm_bar_gets_only_open_zone_param`**（红线，复现 dogfood）：
   构造 EPM catalog + column_aliases `{open: open, closed: closed}`，调 `resolve_charts`，断言 `open_arm_time_ratio_bar` 生成的 PlanChart.args 里 `--parameters-json` **只含 `open_arm_zones`**、**不含 `closed_arm_zones`**。改动前该断言红（现状是两个都在）。

2. **`test_zone_entry_distribution_gets_both_zone_params`**：
   同 catalog，断言 `zone_entry_distribution`（requires_columns open+closed）的 `--parameters-json` **同时含** `open_arm_zones` + `closed_arm_zones`。

3. **`test_trajectory_gets_no_parameters_json`**：
   断言 `trajectory`（requires_columns 是坐标列）的 PlanChart.args **不含** `--parameters-json`。

4. **`test_box_open_arm_gets_only_open_zone_param`**（防回归 + 验证 aggregate 路径）：
   断言 `box_open_arm`（只要 open）也只收 `open_arm_zones`。dogfood 里它「碰巧没炸」是因为底层签名宽容，但裁剪后它也只该收 open。

5. **`test_no_column_aliases_no_injection`**：
   column_aliases 为空时，所有 chart 都不注入 `--parameters-json`（`_build_zone_aliases_overrides` 返回 {} → 裁剪后仍 {}）。

6. **多范式回归**：跑 OFT / Zero Maze / LDB 的 catalog（它们也用 zone_overrides 注入），断言裁剪逻辑不破坏它们的 chart 注入（zero_maze 的 open/closed、OFT 的 center/periphery）。

全量回归：`cd packages/ethoinsight && pytest tests/`（按 memory `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib` 注意 worktree venv 指向；有独立 venv 用它跑）。

---

## 五、验收标准

1. **dogfood EPM 复跑**（thread `3a41e483` 同款数据，列名 `open`/`closed`，HITL 对齐）：chart-maker 的 `plot_open_arm_time_ratio_bar` 对所有 subject **首次并行即成功**，不再 TypeError、不再靠 chart-maker 手删参数重跑。
2. chart 生成的 `--parameters-json` 只含该 chart `requires_columns` 声明的 zone 概念对应的 param。
3. aggregate 图（box/entry_distribution）按各自 requires_columns 收到正确子集。
4. 无 zone 依赖的图（trajectory/heatmap）不注入 `--parameters-json`。
5. ethoinsight 全量测试绿；多范式（OFT/ZM/LDB）chart 注入不回归。

---

## 六、风险与注意事项

1. **守 SSOT：concept 关键词匹配复用 `_build_zone_aliases_overrides` 已有归一化**，不要新写一套 concept→pattern 匹配——否则两套匹配逻辑会漂移（memory `feedback_single_source_of_truth`）。实施前先读 `_build_zone_aliases_overrides` Step 3 + `_concept_matches_pattern` + `_extract_concept_keyword`，确保裁剪用的「chart concept」与注入用的「column_aliases concept」是同一套归一化。
2. **守 review-packages 纪律**：动 catalog chart 注入逻辑前，grep 同事的 review-packages（`docs/review-packages/`）确认 chart↔zone 概念映射没在别处定义（memory `feedback_ssot_lives_in_review_packages`）。`requires_columns` 是 loader 校验过的权威字段，以它为准。
3. **两条路径都要改**：`resolve_charts` 的主路径（L463）和 fallback 路径（L493）都要走裁剪，别只改一处（守一致）。
4. **不加 `**kwargs` 兜底**（§2.5）：保持 metric 函数严格签名，配置错误要响亮炸。裁剪在注入端做。
5. **zone param 是 list 型**（`open_arm_zones=['open']`）——裁剪只过滤 key 不动 value，list/scalar 类型不变（memory `feedback_seal_failure_third_root_cause_list_zone_params_reject_scalar_schema` 同源教训）。
6. **与 06-18 spec 的关系**：06-18 spec 让 plot 脚本能接收参数，本 spec 让它收到的是**对的**参数子集。两者正交，都落地后 EPM chart 链路才完整（拿得到 + 拿得对）。

---

## milestone 建议

本 spec 让「harness 鲁棒性 / dogfood 根因治理」track 再进一步：第三轮 EPM dogfood（thread `3a41e483`）暴露 chart 层第二条独立根因——**`--parameters-json` 注入过宽，不按 chart 裁剪**。与 06-18「plot 脚本拿不到参数」（注入缺失）是同一条注入链的两端：那端补注入、这端裁剪注入。建议在该 milestone 记录：① 此根因 + 本 spec；② **可复用教训**：catalog chart 的 `requires_columns` 是 chart↔zone 概念的权威映射，参数注入应按它裁剪而非发全集并集——「签名宽容的函数容错、签名严格的炸」是参数注入过宽的典型症状；③ 06-18 spec（注入）+ 本 spec（裁剪）合起来才是 chart zone 参数注入的完整治本。
