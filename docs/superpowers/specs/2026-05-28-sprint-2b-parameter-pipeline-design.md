# Sprint 2b 实施 spec — 参数管线 5 跳封闭（管线端）

**关联**：[2026-05-28 SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 2b
**估期**：2 周
**前置**：**Sprint 2a 必须先合 main**（Sprint 2b 用到 Sprint 2a 定义的 `ParamSpec` / `MetricEntry.parameters_ref` / `Catalog.paradigm_parameters` / `CommonCatalog.shared_parameters` / `validate_catalog_consistency` / `load_all_catalogs`）
**执行者**：交给独立 agent 执行

---

## 1. 背景与目标

### 5 跳参数链路（grill A 早绑定 + b JSON 文件锁定）

Sprint 2a 把参数定义搬进 catalog YAML，但执行管线 5 跳每一跳都需要让参数透传。完整链路：

```
跳 1: experiment-context.json.parameter_overrides (Sprint 4.5 写入,Sprint 2b 读取)
        ↓
跳 2: catalog.resolve --overrides-file <path>  →  PlanMetric.parameters_in_use 固化
        ↓ (early binding,这一跳之后 plan.json 自含全套参数)
跳 3: code-executor bash python -m ethoinsight.scripts.fst.compute_immobility_time
       --parameters-json <inline JSON>
        ↓
跳 4: metric script argparse → compute_immobility_time_fst(df, **parameters)
        ↓
跳 5: seal_code_executor_handoff → MetricStat.parameters_used 回写
```

### Sprint 2a 之后 Sprint 2b 之前的状态

- catalog YAML 含参数定义 ✅（Sprint 2a 完）
- PlanMetric 有 `parameters_in_use` 字段但默认空 dict ✅（Sprint 2a 占位）
- MetricStat 有 `parameters_used` 字段但默认空 dict ✅（Sprint 0 占位）
- experiment-context.json 有 `parameter_overrides` 字段（Sprint 4.5 占位，Sprint 2b 实读）
- metric script 仍走硬编码常量 ❌
- metric 函数（如 `compute_immobility_time_fst`）签名不接受参数 ❌
- code-executor 派遣脚本时不传参数 ❌

### Sprint 2b 的 4 件事

1. **resolve.py 加 `--overrides-file` + 计算 parameters_in_use**：catalog default + override 合并 → 写入 PlanMetric
2. **metric script CLI 加统一 `--parameters-json` 入口**：通过 `_cli.py.make_compute_parser` 添加一个共享 arg
3. **metric 函数签名扩展**：`_common.py` / `_pendulum.py` 函数接受关键字参数；**删除模块常量**（Sprint 2a 已让 catalog 知道这些参数）
4. **code-executor 把 PlanMetric.parameters_in_use → bash 参数**：seal 时把实际用的 parameters 回写到 MetricStat.parameters_used

**注**：experiment-context.json 的 `parameter_overrides` 字段读取，Sprint 2b 实现读取通路（resolve.py 接受 overrides-file），但**实际写入是 Sprint 4.5**（set_experiment_paradigm tool 加 parameter_overrides 参数）。Sprint 2b 验证时手动写一个 overrides.json 模拟。

---

## 2. 文件改动清单

### 2.1 改动 `catalog/resolve.py`

**位置**：`packages/ethoinsight/ethoinsight/catalog/resolve.py`

#### a) `resolve_metrics()` 接受 overrides 参数

```python
def resolve_metrics(
    catalog: Catalog,
    *,
    # ... existing args ...
    overrides: dict[str, float | int | str] | None = None,  # === Sprint 2b 新增 ===
    common_catalog: CommonCatalog | None = None,  # === Sprint 2b 新增,提供 shared_parameters ===
) -> tuple[list[PlanMetric], list[PlanSkipped]]:
    """...

    Args (Sprint 2b 新增):
        overrides: 用户参数覆盖。key=参数名,value=覆盖值。
            范围验证(valid_range)在调用前完成,本函数信任。
        common_catalog: 含 shared_parameters,用于解析 metric.parameters_ref。
            若为 None,fall back 到只用 metric.parameters (不解析 ref)。
    """
    overrides = overrides or {}
    shared_params: dict[str, ParamSpec] = (
        common_catalog.shared_parameters.parameters if common_catalog else {}
    )

    plan_metrics = []
    for m in selected_metrics:
        # === Sprint 2b 新增:计算 parameters_in_use ===
        params_in_use = _compute_parameters_in_use(
            metric=m,
            shared_params=shared_params,
            paradigm_params=catalog.paradigm_parameters.parameters,
            overrides=overrides,
        )

        plan_metrics.append(
            PlanMetric(
                # ... existing fields ...
                parameters_in_use=params_in_use,  # === Sprint 2b 填充 ===
            )
        )

    return plan_metrics, skipped


def _compute_parameters_in_use(
    metric: MetricEntry,
    shared_params: dict[str, ParamSpec],
    paradigm_params: dict[str, ParamSpec],
    overrides: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    """合并 catalog default + override → 实际生效的参数集合。

    优先级 (低 → 高):
    1. shared_params (来自 _common.yaml,通过 metric.parameters_ref 引用)
    2. paradigm_params (来自 <paradigm>.yaml.paradigm_parameters,范式级共用)
    3. metric.parameters (单 metric 独有,本 sprint 阶段通常为空)
    4. overrides (用户覆盖,key 与上述任一名字匹配则替换 default)

    Returns:
        合并后的 dict: {param_name: actual_value},包含所有适用于此 metric 的参数。
    """
    result: dict[str, float | int | str] = {}

    # 1. shared (via parameters_ref)
    for ref_name in metric.parameters_ref:
        spec = shared_params.get(ref_name)
        if spec is None:
            # 已被 Sprint 2a validate_catalog_consistency 拦截,此处实际不应该发生
            continue
        result[ref_name] = spec.default

    # 2. paradigm_params (range:适用所有 metric)
    # Sprint 2b 阶段:dispatcher 用的 paradigm_parameters (如 sample_size_threshold)
    # 不进 metric 的 parameters_in_use; dispatcher 自己从 catalog 读 (Sprint 2a §2.6)
    # 这里仅注入与 metric 计算相关的 paradigm-scoped 参数(如 fst.yaml 的 pendulum_*)
    # 实施判断:若 metric.script 在 fst/tst,注入 pendulum_*; 其他范式跳过
    if _metric_uses_pendulum(metric):
        for pname, pspec in paradigm_params.items():
            if pname.startswith("pendulum_"):
                result[pname] = pspec.default

    # 3. metric.parameters (单 metric 独有)
    for pname, pspec in metric.parameters.items():
        result[pname] = pspec.default

    # 4. overrides (最高优先级)
    for pname, override_val in overrides.items():
        if pname in result:
            result[pname] = override_val
        # 否则 override 是一个不适用于本 metric 的参数,silently skip
        # (不报错:同一个 overrides dict 可能跨多个 metric 共用)

    return result


def _metric_uses_pendulum(metric: MetricEntry) -> bool:
    """启发式:metric.script 含 'fst' 或 'tst' 且 metric.id 含 'immobility' / 'struggle' / 'pendulum'。"""
    script_lower = metric.script.lower()
    id_lower = metric.id.lower()
    is_swim_test = ".fst." in script_lower or ".tst." in script_lower
    is_pendulum_metric = any(kw in id_lower for kw in ["immobility", "struggle", "pendulum", "activity_intensity"])
    return is_swim_test and is_pendulum_metric
```

#### b) `resolve.py` CLI 加 `--overrides-file`

resolve.py 当前已有 CLI 入口（被 prep_metric_plan_tool 调用）。在 argparse 加：

```python
def _build_argparser():
    ap = argparse.ArgumentParser(...)
    # ... existing args (paradigm, mode, output, etc) ...
    ap.add_argument(
        "--overrides-file",
        default=None,
        help=(
            "Optional JSON file with parameter overrides, e.g. "
            "{\"velocity_threshold\": 5.0, \"pendulum_periodicity_threshold\": 0.6}. "
            "Sprint 2b: read by resolve_metrics() to populate parameters_in_use."
        ),
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    # === Sprint 2b 新增:读 overrides ===
    overrides = {}
    if args.overrides_file:
        with open(args.overrides_file, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        if not isinstance(overrides, dict):
            print(f"--overrides-file content must be a JSON dict, got {type(overrides).__name__}", file=sys.stderr)
            return 1

    # === Sprint 2b 新增:加载 common catalog ===
    from ethoinsight.catalog.loader import load_common_catalog
    common = load_common_catalog()

    # ... existing logic, 在调用 resolve_metrics 处增加 overrides/common_catalog 参数 ...
    plan_metrics, skipped = resolve_metrics(
        catalog,
        # ... existing kwargs ...
        overrides=overrides,
        common_catalog=common,
    )
    # ... 写 plan.json 一切如旧 ...
```

#### c) 生成 args 时把 parameters_in_use → CLI args

`resolve.py` 现有 `_build_metric_args()` / 类似函数（参考 line 516 `paradigm` 参数注入逻辑）—— 在 metric 的 CLI args 列表中**注入 `--parameters-json '<json>'`**：

```python
def _build_metric_args(metric: PlanMetric, paradigm: str) -> list[str]:
    args = ["--input", metric.input, "--output", metric.output]
    # ... existing --paradigm injection ...

    # === Sprint 2b 新增:注入 parameters JSON ===
    if metric.parameters_in_use:
        params_json = json.dumps(metric.parameters_in_use, ensure_ascii=False, sort_keys=True)
        args.extend(["--parameters-json", params_json])

    return args
```

PlanMetric.args（在 plan.json 中）会包含 `--parameters-json {...}`，code-executor 执行 bash 时原样传给 script。

### 2.2 改动 `scripts/_cli.py`

**位置**：`packages/ethoinsight/ethoinsight/scripts/_cli.py`

```python
def make_compute_parser(description: str) -> argparse.ArgumentParser:
    """Argparse for ``compute_*.py``: --input + --output + (Sprint 2b) --parameters-json."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument(
        "--input", required=True, help="Path to a single EthoVision trajectory file"
    )
    ap.add_argument("--output", required=True, help="Path to write the metric JSON")
    # === Sprint 2b 新增 ===
    ap.add_argument(
        "--parameters-json",
        default="{}",
        help=(
            "JSON dict of parameters (e.g. velocity_threshold, pendulum_*). "
            "Empty dict means use script-side defaults. "
            "Populated by catalog.resolve from PlanMetric.parameters_in_use."
        ),
    )
    return ap


def parse_parameters(args_namespace) -> dict[str, float | int | str]:
    """Parse args.parameters_json into dict. Returns {} on empty/invalid (with stderr warning).

    Sprint 2b 新增:供所有 compute_*.py 脚本调用,统一参数解析逻辑。
    """
    raw = getattr(args_namespace, "parameters_json", "") or "{}"
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[warning] --parameters-json parse failed ({e}); falling back to script defaults", file=sys.stderr)
        return {}
    if not isinstance(result, dict):
        print(f"[warning] --parameters-json must be a JSON object, got {type(result).__name__}", file=sys.stderr)
        return {}
    return result
```

对应 `make_plot_parser` 和 `make_stats_parser` 是否也加 `--parameters-json`？**Sprint 2b 不加**（plot/stats 当前没有参数化需求；YAGNI）。

### 2.3 改动 `metrics/_common.py`

**位置**：`packages/ethoinsight/ethoinsight/metrics/_common.py`

#### a) 删除模块常量

```python
# Sprint 2b 删除以下两行:
# _VELOCITY_THRESHOLD_MM_S = 30.0
# _VELOCITY_MIN_DURATION = 25
```

#### b) `_resolve_immobile_from_velocity()` 函数签名加参数

```python
def _resolve_immobile_from_velocity(
    df: pd.DataFrame,
    *,
    velocity_threshold: float = 30.0,   # === Sprint 2b 加入参数 ===
    velocity_min_duration: int = 25,
) -> tuple[pd.Series, int] | None:
    """Derive immobility from center-point velocity (Noldus Non-movement bouts).

    Args:
        df: 含 x_center/y_center 列的轨迹 DataFrame
        velocity_threshold: ≤ 此速度判为 immobile (mm/s)。Sprint 2b 从 catalog 传入,
            默认 30.0 仅供本地 unit test 使用,生产路径必须传参。
        velocity_min_duration: 至少持续此样本数才计入 immobility (samples @ 25fps baseline)。
    """
    # ... existing logic, 但把所有 _VELOCITY_THRESHOLD_MM_S → velocity_threshold ...
    # ... 把所有 _VELOCITY_MIN_DURATION → velocity_min_duration ...
```

**关键设计**：函数签名 default 值与 catalog default 一致（30.0 / 25），保证：

- unit test 不传参 → 与 Sprint 2a 之前完全等价
- catalog 路径 → 透传 catalog default 也是 30.0 / 25，bit-identical

#### c) 所有内部调用 `_resolve_immobile_from_velocity()` 的点加参数传递

```python
# 在调用此 helper 的所有函数（如 compute_immobility_time_fst 等）签名加参数:
def compute_immobility_time_fst(
    df: pd.DataFrame,
    *,
    velocity_threshold: float = 30.0,
    velocity_min_duration: int = 25,
    # pendulum 参数(若用)
    pendulum_periodicity_threshold: float = 0.55,
    # ... 其他 pendulum_* 参数 ...
) -> float:
    """..."""
    # ... 内部调用时传递 ...
    series, label = _resolve_immobile_from_velocity(
        df,
        velocity_threshold=velocity_threshold,
        velocity_min_duration=velocity_min_duration,
    )
```

### 2.4 改动 `metrics/_pendulum.py`

类似 _common.py：

```python
# Sprint 2b 删除以下模块常量:
# SMOOTH_WINDOW = 1
# ANALYSIS_WINDOW = 25
# PERIOD_MIN = 4
# PERIOD_MAX = 12
# PERIODICITY_THRESHOLD = 0.55
# ACTIVITY_STRUGGLE_THRESHOLD = 2.0
# MIN_STILL_ACTIVITY = 0.3
# MODERATE_ACTIVITY_THRESHOLD = 1.0
# MIN_STATE_DURATION = 25
# PENDULUM_GRACE_PERIOD = 20


def pendulum_immobility_series(
    activity: np.ndarray,
    dt: float,
    *,
    smooth_window: int = 1,
    analysis_window: int = 25,
    period_min: int = 4,
    period_max: int = 12,
    periodicity_threshold: float = 0.55,
    activity_struggle_threshold: float = 2.0,
    min_still_activity: float = 0.3,
    moderate_activity_threshold: float = 1.0,
    min_state_duration: int = 25,
    pendulum_grace_period: int = 20,
) -> np.ndarray:
    """..."""
    # ... existing logic, 所有 PERIODICITY_THRESHOLD → periodicity_threshold 等 ...
```

参数名与 catalog yaml 中的 key 一致（去掉 PENDULUM_ 前缀），便于 catalog override JSON 直接 mapping。**注**：catalog 中是 `pendulum_periodicity_threshold`（有 `pendulum_` 前缀），但 Python 函数签名内的关键字参数**不带前缀**。所以中间需要 mapping 或者两边对齐 —— 选择**两边都带 `pendulum_` 前缀**（函数签名也写 `pendulum_periodicity_threshold` 避免 mapping 复杂度）：

```python
def pendulum_immobility_series(
    activity: np.ndarray,
    dt: float,
    *,
    pendulum_smooth_window: int = 1,
    pendulum_analysis_window: int = 25,
    pendulum_period_min: int = 4,
    pendulum_period_max: int = 12,
    pendulum_periodicity_threshold: float = 0.55,
    pendulum_activity_struggle_threshold: float = 2.0,
    pendulum_min_still_activity: float = 0.3,
    pendulum_moderate_activity_threshold: float = 1.0,
    pendulum_min_state_duration: int = 25,
    pendulum_grace_period: int = 20,
) -> np.ndarray:
```

这样 `**parameters_in_use` 直接 unpack 到函数签名，无需任何 key 转换。

### 2.5 改动 metric script（共 6+ 个）

**位置**：`packages/ethoinsight/ethoinsight/scripts/{fst,tst,epm,oft,ldb,zero_maze}/compute_*.py`

修改所有 `compute_*.py` 主函数，从 args 解析 parameters 后传给 metric 函数：

```python
# 示例:fst/compute_immobility_time.py
from ethoinsight.scripts._cli import (
    emit_result,
    make_compute_parser,
    parse_parameters,  # === Sprint 2b 新增 import ===
    save_output_json,
)


def main(argv: list[str] | None = None) -> int:
    parser = make_compute_parser(description=__doc__)
    args = parser.parse_args(argv)

    df = parse_trajectory(args.input)
    parameters = parse_parameters(args)  # === Sprint 2b 新增 ===
    value = compute_immobility_time_fst(df, **parameters)  # === Sprint 2b 传参 ===

    payload = {
        "metric": METRIC_NAME,
        "value": value,
        "parameters_used": parameters,  # === Sprint 2b 新增:回写实际参数 ===
    }
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0
```

**关键**：metric script 输出 JSON 中加 `parameters_used` 字段——这是 Sprint 0 在 MetricStat 中预留的字段，Sprint 2b 实际填入。

**注**：`**parameters` 是 kwargs unpack —— 函数签名带 `**kwargs` 兼容性 (`def compute_immobility_time_fst(df, **kwargs)`)，或显式列出所有参数 default。**推荐显式列出**（IDE 自动补全 + 类型检查友好）；如果 parameters dict 有函数签名中不存在的 key，Python 会抛 TypeError —— 这是好事，迫使 resolve.py / catalog 端字段名严格对齐。

### 2.6 改动 `subagents/builtins/code_executor.py`

**位置**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`

#### a) prompt 修改：不再"自己拼参数"

code-executor 当前 prompt 教 LLM "按 metric.args 跑 bash"。Sprint 2b 之后 metric.args 已经包含 `--parameters-json '...'`，code-executor 不需要做任何额外工作。**确认 prompt 没有"拼参数"教导即可**。

#### b) seal_code_executor_handoff 时回写 parameters_used

当 code-executor 收集所有 metric 结果时，每个 metric 输出 JSON 里有 `parameters_used` 字段（Sprint 2b §2.5 加入）。subagent 把它放进 metrics_summary 嵌套结构：

```
# subagent prompt 增加一行:
# 在收集 metric 结果时,把每个 metric 输出 JSON 的 "parameters_used" 字段
# 填入 metrics_summary[group][metric_id].parameters_used (MetricStat 字段)
```

实际实施：code-executor 通过 bash 跑 `python -m ethoinsight.scripts.X.Y` 拿到 stdout 中的 `[result] {json}` 行，里面已含 parameters_used。code-executor 调 `seal_code_executor_handoff` 时把这个 dict 透传到对应 MetricStat。

**Sprint 0 已经给 MetricStat 加了 `parameters_used: dict` 字段（默认空），Sprint 2b 实际填值。**

### 2.7 单元测试

新建 `packages/ethoinsight/tests/test_resolve_parameters.py`：

| 测试 | 期望 |
|---|---|
| `test_parameters_in_use_default_no_overrides` | overrides 为空 → PlanMetric.parameters_in_use 含 catalog default |
| `test_parameters_in_use_with_override` | overrides={"velocity_threshold": 5.0} → 该字段被覆盖，其他保持 default |
| `test_parameters_in_use_unrelated_override_silently_skipped` | overrides 含 metric 不需要的 key → 不报错，不进 parameters_in_use |
| `test_parameters_in_use_pendulum_only_for_swim_test` | EPM metric 不应含 pendulum_* 参数；FST immobility metric 应含 |
| `test_resolve_cli_overrides_file_loaded` | catalog.resolve CLI 跑 `--overrides-file overrides.json` → 生成的 plan.json 含 overrides 后的 parameters_in_use |
| `test_resolve_cli_overrides_file_invalid_json` | --overrides-file 指向非 JSON 文件 → exit 1 + stderr 错误 |
| `test_metric_cli_args_include_parameters_json` | PlanMetric.args 包含 `--parameters-json '...'` |

新建 `packages/ethoinsight/tests/test_metrics_parameter_passing.py`：

| 测试 | 期望 |
|---|---|
| `test_immobility_time_with_custom_velocity_threshold` | compute_immobility_time_fst(df, velocity_threshold=5.0) 与 default 30.0 结果不同（用人工数据：5-30 mm/s 速度的帧将被重新分类） |
| `test_pendulum_with_custom_periodicity` | pendulum_immobility_series(activity, dt, pendulum_periodicity_threshold=0.8) 比 default 0.55 严格 |
| `test_metric_script_outputs_parameters_used` | 跑 `python -m ethoinsight.scripts.fst.compute_immobility_time --input ... --output ... --parameters-json '{"velocity_threshold": 5.0}'` → 输出 JSON 含 `parameters_used: {"velocity_threshold": 5.0}` |
| `test_metric_script_no_parameters_uses_defaults` | --parameters-json 省略 → 输出 JSON 的 parameters_used 为空 dict（function default 生效，但 script 不知道 default 是什么不写） |
| `test_metric_script_extra_param_raises` | --parameters-json='{"fake_param": 999}' → metric 函数 TypeError（unexpected keyword） |

新建 `packages/agent/backend/tests/test_code_executor_parameters_used.py`：

| 测试 | 期望 |
|---|---|
| `test_seal_metricstat_parameters_used_populated` | code-executor 跑完 metric 后 seal handoff → metrics_summary[group][metric].parameters_used 等于 plan_metrics.parameters_in_use |

### 2.8 集成测试

| 测试 | 期望 |
|---|---|
| `test_dogfood_fst_bit_identical_when_no_overrides` | overrides 为空（或不存在 overrides.json）→ 跑 fst 输出 handoff bit-identical 于 Sprint 2a 之前 |
| `test_dogfood_fst_with_velocity_override_changes_result` | overrides={"velocity_threshold": 5.0} → immobility_time 值与默认 30.0 不同；handoff_code_executor.json 内 MetricStat.parameters_used 含 {"velocity_threshold": 5.0} |
| `test_e2e_overrides_round_trip` | 写 overrides.json `{"velocity_threshold": 10}` → 触发 prep_metric_plan → plan.json 含 parameters_in_use={"velocity_threshold": 10} → bash 跑 metric script → script JSON 输出 parameters_used={"velocity_threshold": 10} → handoff MetricStat.parameters_used={"velocity_threshold": 10}（**5 跳全闭环**）|

---

## 3. 实施顺序（task 拆分）

| Task | 内容 | 估时 |
|---|---|---|
| T1 | resolve.py 加 `_compute_parameters_in_use` + `_metric_uses_pendulum` helper | 1 天 |
| T2 | resolve.py CLI 加 `--overrides-file` + 改 `resolve_metrics` 接受 overrides | 0.5 天 |
| T3 | resolve.py `_build_metric_args` 注入 `--parameters-json` | 0.25 天 |
| T4 | T1-T3 单元测试（test_resolve_parameters.py 7 个测试）| 1 天 |
| T5 | scripts/_cli.py 加 `--parameters-json` arg + `parse_parameters` helper | 0.5 天 |
| T6 | metrics/_common.py 删常量 + `_resolve_immobile_from_velocity` 加参数 + 所有调用方传参（compute_immobility_time_fst 等）| 1.5 天 |
| T7 | metrics/_pendulum.py 删常量 + `pendulum_immobility_series` 加参数 + 所有调用方传参 | 1 天 |
| T8 | 所有 metric script `compute_*.py` 改用 `parse_parameters` + 输出 JSON 加 parameters_used | 1 天 |
| T9 | T5-T8 单元测试（test_metrics_parameter_passing.py 5 个测试）| 1 天 |
| T10 | code_executor.py prompt review + 必要时微调（多半不用改，因为 args 已包含 --parameters-json）| 0.25 天 |
| T11 | code_executor seal 时回写 parameters_used + 单元测试 | 0.5 天 |
| T12 | 集成测试（§2.8 三个，含 5 跳全闭环 e2e）| 1 天 |
| T13 | 全量回归测试 + 修复退化（缓冲）| 0.5 天 |
| **合计** | | **9.5 天 ≈ 2 周** |

---

## 4. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 删 `_common.py` / `_pendulum.py` 常量后某些不在 metric script 路径的代码（如 ipython notebook、test 旧代码）找不到符号 | T6/T7 前 grep `_VELOCITY_THRESHOLD_MM_S\|PERIODICITY_THRESHOLD` 全仓库；找到外部引用 → 改为函数 default 或 catalog 读取 |
| 函数签名 default = catalog default 写错（不一致）| T6/T7 实施时**逐项 grep 对照** _pendulum.py 当前值；T12 集成测试 dogfood bit-identical 验证 |
| --parameters-json '...' 在 bash 中转义问题（含双引号、空格等）| 在 resolve.py 用 `json.dumps(..., ensure_ascii=False, sort_keys=True)` + code-executor bash 调用走 subprocess 直接传 argv 数组（不走 shell 字符串），避免转义；T9 add test 验证含 unicode 参数能正确 round-trip |
| metric 函数收到 `**parameters_in_use` unpack 包含函数签名不存在的 key → TypeError | 这是 feature 不是 bug：暴露 catalog/函数签名不一致；T9 加 test_metric_script_extra_param_raises 验证 |
| `_metric_uses_pendulum` 启发式漏判某个 fst/tst metric | T1 时仔细 grep 实际 fst/tst metric.id 列表，列入 keyword 集；若漏判 → metric 收到空 pendulum 参数走函数 default 30/0.55 等 = bit-identical 老行为 |
| Sprint 2b 完工时 experiment-context.json 还没有 parameter_overrides 字段（Sprint 4.5 才写）| Sprint 2b 不依赖 set_experiment_paradigm 真写入；test 用手写 overrides.json 模拟；prep_metric_plan_tool 在 Sprint 2b 期间从 experiment-context.json 读 parameter_overrides 字段（若字段不存在则空 dict） |

---

## 5. 验收 checklist

实施完成时确认下列全部通过：

- [ ] resolve.py 含 `_compute_parameters_in_use` + `_metric_uses_pendulum` helper
- [ ] resolve.py CLI 接受 `--overrides-file`，content 必须是 JSON dict 否则 exit 1
- [ ] PlanMetric.parameters_in_use 在 Sprint 2b 之后**非空**（含 catalog default）
- [ ] PlanMetric.args 包含 `--parameters-json '<JSON>'`
- [ ] scripts/_cli.py `make_compute_parser` 加 `--parameters-json`
- [ ] scripts/_cli.py 有 `parse_parameters(args)` helper
- [ ] metrics/_common.py 删除 `_VELOCITY_THRESHOLD_MM_S` / `_VELOCITY_MIN_DURATION` 模块常量
- [ ] metrics/_pendulum.py 删除 9 个 PENDULUM_* 模块常量
- [ ] `compute_immobility_time_fst()` 等函数签名接受 velocity_threshold / velocity_min_duration / pendulum_* 参数
- [ ] `pendulum_immobility_series()` 函数签名接受 9 个 pendulum_* 参数
- [ ] 所有 `compute_*.py` 主函数调用 `parse_parameters(args)` + 传 `**parameters` 给 metric 函数
- [ ] metric script 输出 JSON 含 `parameters_used` 字段
- [ ] code-executor seal handoff 时把 metric 的 parameters_used 写入 MetricStat
- [ ] 单元测试通过（约 13 个测试新增）
- [ ] 集成测试 e2e overrides round-trip 通过（5 跳全闭环）
- [ ] dogfood FST bit-identical（无 overrides 时）
- [ ] dogfood FST 改 overrides.json velocity_threshold=5 → immobility_time 值变化 & handoff parameters_used={"velocity_threshold": 5}
- [ ] `git grep "_VELOCITY_THRESHOLD_MM_S\|^PERIODICITY_THRESHOLD" packages/ethoinsight/` 返回空（除测试文件中明确预期外）

---

## 6. 不在 Sprint 2b 范围

- ❌ set_experiment_paradigm tool 接受 parameter_overrides 参数（Sprint 4.5 做）
- ❌ analysis_config_id 真实计算（Sprint 4.5 做；Sprint 2b 阶段 handoff 仍占位 "PENDING_SPRINT_4.5"）
- ❌ data-analyst 参数审计（Sprint 3 做：data-analyst 比对 parameters_used vs 数据分布生成建议）
- ❌ 调参指南 md（Sprint 4 做）
- ❌ DataQualityGuardrail 拦截（Sprint 5 做）
- ❌ Lineage 封印 hash 验证（Sprint 5.5 做；Sprint 0 已经在 seal 时写 manifest）
- ❌ plot_* / stats_* 脚本参数化（YAGNI，dogfood 没需求）

---

## 7. 参考

- [SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) — Sprint 2b 章节
- [Sprint 2a 实施 spec](2026-05-28-sprint-2a-catalog-parameters-design.md) — catalog 端基础
- catalog resolve：`packages/ethoinsight/ethoinsight/catalog/resolve.py`
- scripts CLI：`packages/ethoinsight/ethoinsight/scripts/_cli.py`
- metrics common：`packages/ethoinsight/ethoinsight/metrics/_common.py`
- metrics pendulum：`packages/ethoinsight/ethoinsight/metrics/_pendulum.py`
- metric scripts：`packages/ethoinsight/ethoinsight/scripts/{fst,tst,epm,oft,ldb,zero_maze}/compute_*.py`
- code-executor：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`
- prep_metric_plan_tool：`packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py`
