# Spec：per-subject plot 脚本承接列对齐参数 —— 修 chart-maker「open arm 不存在」

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-18
> 性质：结构性根治。HITL 列语义对齐（用户把 `open` 列确认为开臂）对 compute 脚本生效、但对 per-subject plot 脚本**未生效**——plot 脚本重算指标时不传 `open_arm_zones` 参数，metric 函数走列名模式 fallback 找不到 `open` 列，返回 None → chart-maker 报「could not compute open_arm_time_ratio — no open-arm zone columns」。数据里开臂**存在**，只是列名经对齐后叫 `open`。
> 关联：
> - 前序（只修了 prep_chart_plan 自读对齐、没修 plot 脚本重算）：`docs/superpowers/specs/2026-06-17-charts-column-alignment-self-read-spec.md`（P3）
> - 列语义对齐 milestone：`docs/milestone/column-semantics-alignment.md`
> - SSOT 铁律：memory `feedback_single_source_of_truth`
> - 列对齐铁律：memory `feedback_lead_inverts_fewzones_vs_nozones_by_column_name`（看划区不看列名）、`feedback_seal_failure_third_root_cause_list_zone_params_reject_scalar_schema`（zone 参数是 list 型）

---

## 〇、给实施 agent 的一句话

`compute_open_arm_time_ratio.py` 成功、`plot_open_arm_time_ratio_bar.py` 失败，差别只有一行：compute 调 `parse_parameters(args)` 把对齐后的 `open_arm_zones=['open']` 传进 metric 函数；plot 调 `compute_open_arm_time_ratio(df)` **一个参数都不传**，于是 metric 函数走 `_get_open_zone_cols(df)` 模式匹配（找 `in_zone.*open_arm`），匹配不上用户列名 `open` → 返回 None。**让 per-subject plot 脚本也接收并透传 zone 参数**——和 compute 脚本同源同语义。

**核心约束**：参数来源必须是同一个 SSOT（plan/context 里已对齐的 `parameters_used`），不能让 plot 脚本自己猜列名（那会再造一个漂移点）。

---

## 一、根因（实证）

dogfood thread `3bcbee10`：用户上传 EPM，列名 `open`/`closed`，经 HITL 确认 `open`=开臂、`closed`=闭臂。compute 全部成功（140/140），但 chart-maker 跑 `plot_open_arm_time_ratio_bar` 对每个 subject 都失败。

对比两个脚本：

| | compute_open_arm_time_ratio.py | plot_open_arm_time_ratio_bar.py |
|---|---|---|
| 取参数 | `parameters = parse_parameters(args)`（L34） | **无** |
| 调 metric | `compute_open_arm_time_ratio(df, **parameters)`（L35） | `compute_open_arm_time_ratio(df)`（L36，**裸调**） |
| 结果 | `open_arm_zones=['open']` 命中 → 有值 | 走 fallback `_get_open_zone_cols(df)` → 找 `in_zone.*open_arm` → 列名是 `open` 匹配不上 → None |

metric 函数 [`compute_open_arm_time_ratio`](../../../packages/ethoinsight/ethoinsight/metrics/epm.py)（epm.py:83-104）逻辑：

```python
if open_arm_zones:
    cols = [c for c in open_arm_zones if c in df.columns]   # 传了参数 → 命中 'open'
else:
    cols = _get_open_zone_cols(df)                          # 没传 → 模式匹配 → 失败
if not cols:
    return None                                             # plot 脚本走到这里
```

`make_plot_parser`（[`_cli.py`](../../../packages/ethoinsight/ethoinsight/scripts/_cli.py) L333）**根本没有 `--parameters` / `--parameters-json` 参数**，所以 plot 脚本物理上拿不到对齐结果——即便 plan 里有 `parameters_used`，也没有通道传进来。

**这是 P3 的盲区**：P3 修的是 `prep_chart_plan` 自读列对齐、把 zone 参数写进 plan_charts.json 的 entry.args（aggregate 图如 `box_open_arm` 因此能拿到 `--groups` 等）。但 **per-subject plot 脚本重算单值指标时走的是另一条路**——它读单个 xlsx、裸调 metric 函数，从未接过 zone 参数。

---

## 二、设计

### 2.1 plot 脚本必须像 compute 脚本一样接收 zone 参数

凡是「重算指标」的 per-subject plot 脚本（如 `plot_open_arm_time_ratio_bar`，以及任何调 `compute_*` 的 epm/oft/... plot 脚本），都要：
1. parser 支持 `--parameters-json`（与 compute 脚本同一个 `parse_parameters` 入口）。
2. 把解析出的参数透传给 metric 函数：`compute_open_arm_time_ratio(df, **parameters)`。

这与 code-executor 的并行 compute 调用形态完全一致（[`ethoinsight-code/SKILL.md:66`](../../../packages/agent/skills/custom/ethoinsight-code/SKILL.md) 示例已展示 `--parameters-json '{"center_zone":"in_zone"}'`）。

### 2.2 参数从哪来：prep_chart_plan 必须把 zone 参数写进 per-subject plot entry 的 args

参数的 SSOT 是 experiment-context.json 的 `column_aliases` / plan_metrics.json 的 `parameters_used`（HITL 对齐结果）。`prep_chart_plan` / `resolve_charts` 在生成 per-subject plot entry 时，必须像给 compute / aggregate 图注入 zone 参数一样，把 `--parameters-json '{"open_arm_zones": ["open"], ...}'` 拼进 entry.args。

**关键**：不要让 plot 脚本自己去读 context（那是第二个 SSOT 副本，会漂移）。由 `prep_chart_plan`（已自读 context，P3 能力）统一把参数投影进 entry.args，plot 脚本只负责接收 + 透传。这与 §2.1 的「plot 脚本不猜列名」一致。

### 2.3 fallback 行为保留但降级为"无对齐信息时的尽力而为"

metric 函数的 `else: _get_open_zone_cols(df)` 分支**保留**（无参数的旧调用、无对齐场景仍可用），但 plot 脚本在拿到对齐参数时**必须优先用参数**。若 plan 没给参数（老 plan / 无对齐数据），plot 脚本裸调 metric → 走 fallback → 可能失败，此时脚本应 emit 清晰错误（区分「真无开臂列」vs「有列但没传对齐参数」），便于诊断。

---

## 三、改动清单

### 3.1 `_cli.py` —— make_plot_parser 支持 --parameters-json

`make_plot_parser`（L333）加 `--parameters-json` 参数（与 `make_compute_parser` 同款），并确保 `parse_parameters` 能从 plot parser 的 namespace 取到它。**复用现有 `parse_parameters`，不要新写解析逻辑**（SSOT）。

### 3.2 per-subject plot 脚本 —— 接收并透传参数

至少修 `plot_open_arm_time_ratio_bar.py`（dogfood 实证失败的那个）：

```python
# 之前
args = make_plot_parser(description=__doc__, supports_groups=False).parse_args(argv)
...
value = compute_open_arm_time_ratio(df)

# 之后
parser = make_plot_parser(description=__doc__, supports_groups=False)
args = parser.parse_args(argv)
parameters = parse_parameters(args)   # 与 compute 脚本同源
...
value = compute_open_arm_time_ratio(df, **parameters)
if value is None:
    # 区分两种失败：传了参数仍 None（真无该列）vs 没传参数（缺对齐）
    hint = "（已传 zone 参数仍无值，疑似该 subject 数据无开臂列）" if parameters else "（未收到 zone 对齐参数，疑似 prep_chart_plan 未注入 --parameters-json）"
    print(f"error: could not compute open_arm_time_ratio — no open-arm zone columns {hint}", file=sys.stderr)
    return 1
```

**扫描所有 per-subject plot 脚本**：`grep -rln "compute_" packages/ethoinsight/ethoinsight/scripts/*/plot_*.py`，凡是裸调 `compute_*(df)`（无参数）且该 metric 函数签名含 `*_zones` 参数的，都按本模式修。aggregate 图（如 `plot_box_open_arm`）若已通过 `--groups`/`--inputs` 路径拿到对齐，确认其内部也用对齐列（一并核查，不漏）。

### 3.3 `resolve.py` / prep_chart_plan —— per-subject plot entry 注入 zone 参数

在 `resolve_charts` 生成 per-subject plot entry 的 args 时，从已自读的 context（column_aliases / zone 概念映射）拼出 `--parameters-json '{"open_arm_zones": [...], ...}'` 加入 entry.args。**参数投影逻辑应复用 compute/aggregate 路径已有的 zone→参数映射**（P3 已建立的列对齐投影），不要为 plot 单独写一套。

> 定位：`packages/ethoinsight/ethoinsight/catalog/resolve.py`（P3 改过 `_build_groups_payload`，列对齐投影逻辑应在附近）。

---

## 四、测试（红→绿坐实，TDD 强制）

新建 `packages/ethoinsight/tests/test_plot_scripts_receive_zone_params.py`：

1. **`test_plot_open_arm_bar_with_aliased_column`**（红线）：构造一个列名为 `open`（非 `in_zone_open_arm`）的轨迹 DataFrame/xlsx，跑 `plot_open_arm_time_ratio_bar --input ... --parameters-json '{"open_arm_zones": ["open"]}'`，断言 exit 0 + 输出 png 存在 + emit 的 value 非 None。
   - **坐实红**：去掉 `--parameters-json`（裸调），断言 exit 1 + stderr 含「no open-arm zone columns」—— 证明列名 `open` 在无参数时确实失败（复现 dogfood）。
2. **`test_make_plot_parser_accepts_parameters_json`**：单测 parser 能解析 `--parameters-json` 且 `parse_parameters` 返回 `{"open_arm_zones": ["open"]}`。
3. **`test_resolve_charts_injects_zone_params_into_per_subject_plot`**：给 `resolve_charts` 喂带 column_aliases 的 context，断言生成的 per-subject plot entry.args 含 `--parameters-json` 且内容是对齐后的 zone 参数。
4. **`test_plot_error_message_distinguishes_missing_param_vs_missing_column`**：传参数仍 None 的 stderr 与没传参数的 stderr 文案不同（便于诊断）。

全量回归：`cd packages/ethoinsight && pytest tests/`（改的是 ethoinsight 库，按 memory `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib` 注意 worktree venv 指向）。

---

## 五、验收标准

1. dogfood EPM（列名 `open`/`closed`，HITL 对齐）复跑：chart-maker 的 `plot_open_arm_time_ratio_bar` 对所有 subject 成功出图，不再报「no open-arm zone columns」。
2. per-subject plot 脚本与 compute 脚本对同一 subject 同一指标，用同一组 zone 参数算出**同一个值**（一致性）。
3. 无对齐参数的老 plan 仍走 fallback（不硬崩，错误信息清晰）。
4. ethoinsight 全量测试绿。

---

## 六、风险与注意事项

1. **别让 plot 脚本读 context**：参数必须由 prep_chart_plan 投影进 args（单一注入点），plot 脚本只接收。否则又造一个列对齐 SSOT 副本（违反 `feedback_single_source_of_truth`）。
2. **zone 参数是 list 型**（`open_arm_zones=['open']` 不是标量）——`parse_parameters` / `--parameters-json` 必须正确解析 list 值（memory `feedback_seal_failure_third_root_cause_list_zone_params_reject_scalar_schema` 同源教训：标量 schema 拒 list）。
3. **扫全所有 per-subject plot 脚本**，别只修 dogfood 撞到的那一个——其他范式（oft/zero_maze/...）的同类 plot 脚本有同样隐患，一并修 + 测。
4. **aggregate 图核查**：确认 `plot_box_open_arm` 等已通过 P3 路径拿到对齐列（dogfood 里 `box_open_arm` 成功了，说明 aggregate 路径 OK，但仍核查避免遗漏）。
