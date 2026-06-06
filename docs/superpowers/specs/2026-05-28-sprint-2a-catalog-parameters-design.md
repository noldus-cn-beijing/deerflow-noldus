# Sprint 2a 实施 spec — 参数下沉 catalog（catalog 端）

**关联**：[2026-05-28 SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) Sprint 2a
**估期**：2.5 周
**前置**：Sprint 0 + Sprint 1 已合 main（Sprint 1 dispatcher.py 用到的 `_ZM_LOW_DISTANCE_THRESHOLD` 会在本 sprint 搬到 catalog）
**执行者**：交给独立 agent 执行

---

## 1. 背景与目标

### 现状

- 12 个可调常量散落在 ethoinsight 代码里（grep 实测）：
  - `metrics/_common.py`：`_VELOCITY_THRESHOLD_MM_S=30.0` / `_VELOCITY_MIN_DURATION=25`
  - `metrics/_pendulum.py`：9 个 PENDULUM_* 常量
  - `metrics/dispatcher.py:235`：`_ZM_LOW_DISTANCE_THRESHOLD=10.0`（Sprint 1 加注释保留为变量读）
- `catalog/schema.py.MetricEntry` 没有 `parameters` 字段，是"半 SSOT"
- `catalog/_common.yaml` 只有 `common_charts`，没有 `shared_parameters`
- `catalog/loader.py` 用 dataclass + 手写校验，不依赖 Pydantic（保持依赖最小）

### Sprint 2a 范围（grill 锁定 C 混合）

Sprint 2a 只做 **catalog 端**——把参数定义搬进 YAML，不动执行管线（管线端是 Sprint 2b）。具体三件事：

1. **schema 扩展**：新增 `ParamSpec` dataclass + `MetricEntry.parameters` + `MetricEntry.parameters_ref` 字段 + `PlanMetric.parameters_in_use` 字段
2. **catalog YAML 搬迁**：跨范式 shared 进 `_common.yaml.shared_parameters`；per-paradigm 独有进各自 yaml
3. **loader 强化**：解析 parameters/parameters_ref + valid_range 校验 + **重复参数自动检测**

### SSOT 设计（grill C 混合锁定）

| 类别 | 放哪 | 例子 |
|---|---|---|
| **shared**（跨范式共享）| `_common.yaml.shared_parameters` | `velocity_threshold` / `velocity_min_duration` |
| **per-paradigm**（范式独有）| 各自 `<paradigm>.yaml` | pendulum 9 个 → fst.yaml + tst.yaml；`zm_low_distance_threshold` → zero_maze.yaml |
| **范式特定 sample size 阈值**（各范式不同）| 各自 `<paradigm>.yaml` | epm n<5、ldb transition_count<4 等 |

**重复参数自动检测**（防退化的硬保障）：loader 在加载所有 yaml 后，扫描相同参数名 + 相同 default 出现在 2+ paradigm yaml 中 → 直接 raise CatalogError 报"应提到 `_common.yaml`"。

**Sprint 2a 不做**：
- ❌ 修改 metric script 函数签名（Sprint 2b 做）
- ❌ 写 overrides JSON 机制（Sprint 4.5 做）
- ❌ 删除 `_common.py` / `_pendulum.py` 中的硬编码常量（Sprint 2b 删；本 sprint 只让 catalog 知道这些参数存在）

---

## 2. 文件改动清单

### 2.1 改动 `catalog/schema.py`

**位置**：`packages/ethoinsight/ethoinsight/catalog/schema.py`

```python
# === Sprint 2a 新增 ===

@dataclass(frozen=True)
class ParamSpec:
    """单个参数的定义。

    - default: 参数默认值 (float/int/str)
    - unit: 单位描述,如 'mm/s' / 'samples' / 'count'
    - description: 一句话说明
    - tunable_by_user: 用户是否可在 ask_clarification 阶段调整
    - valid_range: 合法值范围 [min, max] (对 numeric 参数);str 参数为 None
    """
    default: float | int | str
    unit: str
    description: str
    tunable_by_user: bool
    valid_range: list[float | int] | None  # [min, max] for numeric; None for str

```

修改 `MetricEntry`：

```python
@dataclass(frozen=True)
class MetricEntry:
    id: str
    script: str
    requires_columns: list[str]
    output_unit: str
    display_name_zh: str
    unit_zh: str
    one_liner: str
    direction_for_anxiety: str | None
    statistical_default: str
    # === Sprint 2a 新增 ===
    parameters: dict[str, ParamSpec] = field(default_factory=dict)
    parameters_ref: list[str] = field(default_factory=list)
    # parameters_ref 内 ID 必须在 _common.yaml.shared_parameters 中存在
    # parameters 是本 metric 独有,parameters_ref 是引用 shared
```

修改 `PlanMetric`：

```python
@dataclass
class PlanMetric:
    # ... existing fields ...
    # === Sprint 2a 新增（Sprint 2b 才填充实际值） ===
    parameters_in_use: dict[str, float | int | str] = field(default_factory=dict)
    # Sprint 2a 阶段 resolve.py 不填这个字段（默认空 dict）;
    # Sprint 2b resolve.py 才合并 catalog default + overrides 写入
```

新增 `SharedParameters` dataclass（_common.yaml 顶层结构）：

```python
@dataclass(frozen=True)
class SharedParameters:
    """跨范式共享的参数集合 (_common.yaml.shared_parameters)。"""
    parameters: dict[str, ParamSpec]
```

修改 `CommonCatalog` dataclass（loader.py 末尾的现有 dataclass，把 SharedParameters 加进去）：

```python
@dataclass(frozen=True)
class CommonCatalog:
    common_charts: list[ChartEntry]
    shared_parameters: SharedParameters  # === Sprint 2a 新增 ===
```

### 2.2 改动 `catalog/_common.yaml`

**位置**：`packages/ethoinsight/ethoinsight/catalog/_common.yaml`

在 `common_charts` 段之外，**新增顶层 `shared_parameters` 段**：

```yaml
# Sprint 2a 新增:跨范式共享的可调参数。
# 改这里 = 改所有引用此参数的范式;不要在 <paradigm>.yaml 重复定义。
# loader 会自动校验重复 (同名 + 同 default 出现在 2+ paradigm yaml → CatalogError)。
shared_parameters:
  velocity_threshold:
    default: 30.0
    unit: mm/s
    description: "Noldus 默认: ≤ 此速度判为 immobile (静止)。跨范式共用 (EPM/OFT/LDB/ZM/FST 的 velocity-based immobility)。"
    tunable_by_user: true
    valid_range: [1.0, 100.0]

  velocity_min_duration:
    default: 25
    unit: samples
    description: "Noldus 默认: 至少持续此样本数才计入 immobility (25 帧 @ 25fps = 1s)。跨范式共用。"
    tunable_by_user: true
    valid_range: [5, 250]

common_charts:
  # ... existing ...
```

### 2.3 改动 6 个 paradigm yaml — 加 parameters / parameters_ref

**位置**：`packages/ethoinsight/ethoinsight/catalog/{epm,oft,ldb,zero_maze,fst,tst}.yaml`

#### a) `epm.yaml` / `oft.yaml` / `ldb.yaml` — 加 sample size 阈值 + 引用 velocity shared

各范式 sample size 阈值（来自 dispatcher.py Sprint 1 之后的 evidence dict）：

```yaml
# epm.yaml 末尾加（在 default_metrics / optional_metrics 之后）
paradigm_parameters:
  sample_size_underpowered_threshold:
    default: 5
    unit: count
    description: "EPM: 组内样本数 < 此值时,触发 SAMPLE.UNDERPOWERED 警告。"
    tunable_by_user: true
    valid_range: [2, 30]

  motor_low_entries_threshold:
    default: 8
    unit: count
    description: "EPM: subject 总进臂次数 < 此值时,触发 MOTOR.LOW_ENTRIES 警告 (运动抑制嫌疑)。"
    tunable_by_user: true
    valid_range: [1, 50]
```

对于会用 velocity-based immobility 的 metric，**加 `parameters_ref`**（如 epm 的 open_arm_immobility 等 — 检查范式实际有没有这种 metric；本 sprint 仅在确认引用关系存在的 metric 上加 ref）：

```yaml
default_metrics:
  - id: open_arm_immobility_time
    script: ethoinsight.scripts.epm.compute_open_arm_immobility_time
    requires_columns:
      - velocity
    output_unit: seconds
    display_name_zh: 开放臂不动时间
    # ... existing fields ...
    # === Sprint 2a 新增 ===
    parameters_ref:
      - velocity_threshold
      - velocity_min_duration
```

**注**：实施时**先 grep 实际有哪些 metric 用 velocity-based immobility**，按实际加 ref。**不在没用到这两个参数的 metric 上瞎加**——避免无效引用。

#### b) `zero_maze.yaml` — 加 zm_low_distance_threshold + sample size + velocity ref

```yaml
paradigm_parameters:
  sample_size_underpowered_threshold:
    default: 5
    unit: count
    description: "Zero maze: 组内样本数 < 此值时,触发 SAMPLE.UNDERPOWERED 警告。"
    tunable_by_user: true
    valid_range: [2, 30]

  zm_low_distance_threshold:
    default: 10.0
    unit: cm
    description: "Zero maze: subject 总移动距离 < 此值时,触发 MOTOR.LOW_DISTANCE 警告 (运动抑制嫌疑)。"
    tunable_by_user: true
    valid_range: [0.1, 200.0]
```

velocity 相关 metric（如有）按 a 节加 `parameters_ref`。

#### c) `ldb.yaml` — 加 sample size + transition + velocity ref

```yaml
paradigm_parameters:
  sample_size_underpowered_threshold:
    default: 5
    unit: count
    description: "LDB: 组内样本数 < 此值时,触发 SAMPLE.UNDERPOWERED 警告。"
    tunable_by_user: true
    valid_range: [2, 30]

  signal_low_transition_threshold:
    default: 4
    unit: count
    description: "LDB: subject 穿梭次数 < 此值时,触发 SIGNAL.LOW_TRANSITION_COUNT 警告 (探索动机不足嫌疑)。"
    tunable_by_user: true
    valid_range: [1, 50]
```

#### d) `fst.yaml` + `tst.yaml` — 加 pendulum 9 参数 + sample size + velocity ref

```yaml
# fst.yaml 末尾
paradigm_parameters:
  sample_size_underpowered_threshold:
    default: 5
    unit: count
    description: "FST: 组内样本数 < 此值时,触发 SAMPLE.UNDERPOWERED 警告。"
    tunable_by_user: true
    valid_range: [2, 30]

  pendulum_smooth_window:
    default: 1
    unit: frames
    description: "钟摆检测:预处理平滑窗口,1 = 不平滑。"
    tunable_by_user: true
    valid_range: [1, 25]

  pendulum_analysis_window:
    default: 25
    unit: frames
    description: "钟摆检测:自相关分析窗口,覆盖 3~5 个钟摆周期。"
    tunable_by_user: true
    valid_range: [5, 100]

  pendulum_period_min:
    default: 4
    unit: frames
    description: "钟摆检测:最短搜索周期,对应约 0.16s @ 25fps。"
    tunable_by_user: false
    valid_range: [1, 25]

  pendulum_period_max:
    default: 12
    unit: frames
    description: "钟摆检测:最长搜索周期,对应约 0.48s @ 25fps。"
    tunable_by_user: false
    valid_range: [2, 50]

  pendulum_periodicity_threshold:
    default: 0.55
    unit: ratio
    description: "钟摆检测:周期性强度阈值,钟摆通常 > 0.5。"
    tunable_by_user: true
    valid_range: [0.1, 1.0]

  pendulum_activity_struggle_threshold:
    default: 2.0
    unit: noldus_activity
    description: "钟摆检测:高 Activity 直接判挣扎。"
    tunable_by_user: true
    valid_range: [0.5, 10.0]

  pendulum_min_still_activity:
    default: 0.3
    unit: noldus_activity
    description: "钟摆检测:极低 Activity 直接判静止。"
    tunable_by_user: true
    valid_range: [0.0, 2.0]

  pendulum_moderate_activity_threshold:
    default: 1.0
    unit: noldus_activity
    description: "钟摆检测:中等 Activity 无周期性 → 挣扎。"
    tunable_by_user: true
    valid_range: [0.1, 5.0]

  pendulum_min_state_duration:
    default: 25
    unit: frames
    description: "钟摆检测:状态最短持续帧数 (25fps 下 = 1s)。"
    tunable_by_user: true
    valid_range: [5, 250]

  pendulum_grace_period:
    default: 20
    unit: frames
    description: "钟摆检测:宽容期帧数 (25fps 下)。"
    tunable_by_user: true
    valid_range: [0, 100]
```

**tst.yaml 同上**——TST 也用 pendulum（参考 Sprint 1 spec：`paradigm in ("forced_swim", "tail_suspension")`）。

immobility 类 metric 都加 `parameters_ref: [velocity_threshold, velocity_min_duration]`（实际 grep 哪些 metric 用 velocity）。

#### e) **新增 schema 字段说明**

`paradigm_parameters` 是 **paradigm 级**（不绑定到某个具体 metric），与 metric 内 `parameters` 区分：

| 字段名 | 作用域 | 例子 |
|---|---|---|
| `MetricEntry.parameters` | 单 metric 独有 | （v0.1 暂时没用到，预留位）|
| `MetricEntry.parameters_ref` | 单 metric 引用 shared 参数 | `[velocity_threshold]` |
| `paradigm.yaml: paradigm_parameters` | 整范式共用（如 dispatcher 用） | `sample_size_underpowered_threshold` |
| `_common.yaml: shared_parameters` | 跨范式共用 | `velocity_threshold` / `velocity_min_duration` |

Sprint 2b 会让 resolve.py 在生成 PlanMetric 时正确合并这四个来源。Sprint 2a 阶段，只在 schema/loader 层把它们都能读出来即可。

### 2.4 改动 `catalog/schema.py` 增 ParadigmParameters dataclass + Catalog 加字段

```python
@dataclass(frozen=True)
class ParadigmParameters:
    """范式级共用参数 (各 <paradigm>.yaml 的 paradigm_parameters 段)。"""
    parameters: dict[str, ParamSpec]


@dataclass(frozen=True)
class Catalog:
    paradigm: str
    ev19_templates: list[str]
    default_metrics: list[MetricEntry]
    optional_metrics: list[MetricEntry]
    charts: list[ChartEntry]
    statistics_default: StatisticsEntry | None
    # === Sprint 2a 新增 ===
    paradigm_parameters: ParadigmParameters = field(
        default_factory=lambda: ParadigmParameters(parameters={})
    )
```

### 2.5 改动 `catalog/loader.py` — 解析 parameters + 重复校验

#### a) 新增 `_parse_param_spec` helper

```python
def _parse_param_spec(item: dict, where: str, source: Path) -> ParamSpec:
    """解析单个 ParamSpec yaml 字段。"""
    def req(field_name: str, expected_type) -> Any:
        if field_name not in item:
            raise CatalogError(f"{source} {where}: missing '{field_name}'")
        v = item[field_name]
        if not isinstance(v, expected_type):
            raise CatalogError(
                f"{source} {where}: '{field_name}' must be {expected_type}, "
                f"got {type(v).__name__}"
            )
        return v

    default = req("default", (int, float, str))
    unit = req("unit", str)
    description = req("description", str)
    tunable = req("tunable_by_user", bool)

    valid_range = item.get("valid_range", None)
    if valid_range is not None:
        if not isinstance(valid_range, list) or len(valid_range) != 2:
            raise CatalogError(
                f"{source} {where}: 'valid_range' must be [min, max] or null"
            )
        lo, hi = valid_range
        if not all(isinstance(x, (int, float)) for x in [lo, hi]):
            raise CatalogError(
                f"{source} {where}: 'valid_range' members must be numeric"
            )
        if lo > hi:
            raise CatalogError(
                f"{source} {where}: 'valid_range' min ({lo}) > max ({hi})"
            )
        # 验 default 在 range 内 (numeric 参数)
        if isinstance(default, (int, float)) and not (lo <= default <= hi):
            raise CatalogError(
                f"{source} {where}: default {default} outside valid_range [{lo}, {hi}]"
            )

    return ParamSpec(
        default=default,
        unit=unit,
        description=description,
        tunable_by_user=tunable,
        valid_range=valid_range,
    )


def _parse_param_block(raw: dict, key: str, source: Path) -> dict[str, ParamSpec]:
    """解析 parameters / shared_parameters / paradigm_parameters 这类 dict 块。"""
    block = raw.get(key, {}) or {}
    if not isinstance(block, dict):
        raise CatalogError(f"{source}: '{key}' must be a mapping")
    result: dict[str, ParamSpec] = {}
    for pname, pdict in block.items():
        if not isinstance(pdict, dict):
            raise CatalogError(f"{source}: '{key}.{pname}' must be a mapping")
        result[pname] = _parse_param_spec(pdict, where=f"{key}.{pname}", source=source)
    return result
```

#### b) 修改 `_parse_metric_entry` 解析 parameters / parameters_ref

```python
def _parse_metric_entry(item: dict, where: str, source: Path) -> MetricEntry:
    # ... existing parsing ...

    # === Sprint 2a 新增 ===
    parameters = _parse_param_block(item, "parameters", source) if "parameters" in item else {}
    parameters_ref = item.get("parameters_ref", []) or []
    if not isinstance(parameters_ref, list) or not all(isinstance(x, str) for x in parameters_ref):
        raise CatalogError(f"{source} {where}: 'parameters_ref' must be list[str]")

    return MetricEntry(
        # ... existing fields ...
        parameters=parameters,
        parameters_ref=list(parameters_ref),
    )
```

#### c) 修改 `_parse_catalog` 解析 paradigm_parameters

```python
def _parse_catalog(raw: dict[str, Any], source: Path) -> Catalog:
    # ... existing parsing ...

    # === Sprint 2a 新增 ===
    paradigm_params_block = _parse_param_block(raw, "paradigm_parameters", source)

    catalog = Catalog(
        # ... existing fields ...
        paradigm_parameters=ParadigmParameters(parameters=paradigm_params_block),
    )
    return catalog
```

#### d) 修改 `load_common_catalog` 解析 shared_parameters

```python
def load_common_catalog(catalog_dir: str | Path | None = None) -> CommonCatalog:
    # ... existing parsing of common_charts ...

    # === Sprint 2a 新增 ===
    shared_params = _parse_param_block(raw, "shared_parameters", yaml_path)

    return CommonCatalog(
        common_charts=common_charts,
        shared_parameters=SharedParameters(parameters=shared_params),
    )
```

#### e) 新增 `validate_catalog_consistency` — 重复参数自动检测

**这是 Sprint 2a 防退化的关键保障**。Sprint 4 调参指南、Sprint 2b 管线、Sprint 4.5 hash 计算都依赖参数定义无重复。

```python
def validate_catalog_consistency(
    common: CommonCatalog,
    paradigm_catalogs: list[tuple[str, Catalog]],  # [(paradigm_name, Catalog), ...]
) -> None:
    """跨范式 catalog 一致性校验。Sprint 2a 引入。

    校验项:
    1. parameters_ref 中所有 ID 在 _common.yaml.shared_parameters 中存在
    2. **重复参数检测**:同名 + 相同 default 出现在 2+ paradigm yaml 的
       paradigm_parameters 中 → 应提到 _common.yaml,raise CatalogError

    Args:
        common: load_common_catalog() 的结果
        paradigm_catalogs: 已加载的 6 个范式 Catalog (按 (paradigm_name, Catalog) 配对)

    Raises:
        CatalogError: 校验失败,含详细诊断信息
    """
    # 1. 验 parameters_ref 引用合法
    shared_keys = set(common.shared_parameters.parameters.keys())
    for paradigm_name, cat in paradigm_catalogs:
        for m in cat.default_metrics + cat.optional_metrics:
            for ref in m.parameters_ref:
                if ref not in shared_keys:
                    raise CatalogError(
                        f"{paradigm_name}.yaml: metric '{m.id}' parameters_ref includes "
                        f"'{ref}' which is not in _common.yaml.shared_parameters. "
                        f"Available shared parameters: {sorted(shared_keys)}"
                    )

    # 2. 跨范式重复参数检测 (paradigm_parameters 层)
    # 收集 (param_name, default_value) -> [出现的 paradigm 列表]
    occurrences: dict[tuple[str, Any], list[str]] = {}
    for paradigm_name, cat in paradigm_catalogs:
        for pname, pspec in cat.paradigm_parameters.parameters.items():
            key = (pname, pspec.default)
            occurrences.setdefault(key, []).append(paradigm_name)

    duplicates = [
        (pname, default, paradigms)
        for (pname, default), paradigms in occurrences.items()
        if len(paradigms) >= 2
    ]
    if duplicates:
        msgs = [
            f"  - '{pname}' (default={default}) appears in: {paradigms}; "
            f"should be promoted to _common.yaml.shared_parameters"
            for pname, default, paradigms in duplicates
        ]
        raise CatalogError(
            "Duplicate parameters detected across paradigm yamls "
            "(same name + same default → should be shared):\n"
            + "\n".join(msgs)
        )


def load_all_catalogs(catalog_dir: str | Path | None = None) -> tuple[CommonCatalog, list[tuple[str, Catalog]]]:
    """Sprint 2a 新增:加载 _common.yaml + 全部 6 个 paradigm yaml,跑一致性校验。

    用于:
    - test_catalog_consistency.py (本 sprint 增强版)
    - catalog.resolve CLI 入口 (Sprint 2b 用)
    """
    common = load_common_catalog(catalog_dir)
    paradigm_names = ["epm", "open_field", "light_dark_box", "forced_swim", "tail_suspension", "zero_maze"]
    paradigm_catalogs = []
    for pname in paradigm_names:
        try:
            cat = load_catalog(pname, catalog_dir)
            paradigm_catalogs.append((pname, cat))
        except CatalogError as e:
            # 范式 yaml 不存在则跳过 (v0.1 阶段某些范式未实现)
            if "not found" in str(e).lower():
                continue
            raise

    validate_catalog_consistency(common, paradigm_catalogs)
    return common, paradigm_catalogs
```

### 2.6 dispatcher.py 改用 catalog 读取阈值

**位置**：`packages/ethoinsight/ethoinsight/metrics/dispatcher.py`

Sprint 1 把 evidence dict 里的 threshold 用变量读（`_ZM_LOW_DISTANCE_THRESHOLD`），Sprint 2a 把这个变量改为**catalog 读取**：

```python
# 文件顶部加（替换原 _ZM_LOW_DISTANCE_THRESHOLD = 10.0 的局部赋值）
from ethoinsight.catalog.loader import load_catalog, CatalogError


def _get_paradigm_threshold(paradigm: str, param_name: str, default: Any) -> Any:
    """从 catalog 读取范式阈值,失败时回退到 default 并 warn。"""
    try:
        cat = load_catalog(paradigm)
        spec = cat.paradigm_parameters.parameters.get(param_name)
        if spec is None:
            return default
        return spec.default
    except CatalogError:
        return default


# 然后在 zero_maze 段使用:
_zm_threshold = _get_paradigm_threshold("zero_maze", "zm_low_distance_threshold", 10.0)
# evidence={"threshold": _zm_threshold, ...}
```

**同理处理**：epm/oft/ldb/fst/tst 的 sample_size_underpowered_threshold、epm 的 motor_low_entries_threshold、ldb 的 signal_low_transition_threshold——dispatcher.py 全改为 catalog 读取。

**注**：这是 Sprint 2a 范围（让 dispatcher 端从 catalog 读阈值），不是 Sprint 2b（Sprint 2b 是让 metric script 改函数签名 + override JSON）。

### 2.7 单元测试

新建 `packages/ethoinsight/tests/test_catalog_parameters.py`：

| 测试 | 期望 |
|---|---|
| `test_param_spec_required_fields` | 缺 default/unit/description/tunable_by_user 任一 → CatalogError |
| `test_param_spec_valid_range_consistency` | valid_range=[100, 50] (min > max) → CatalogError |
| `test_param_spec_default_within_range` | default=200, valid_range=[1, 100] → CatalogError |
| `test_load_common_catalog_with_shared_params` | _common.yaml 加 shared_parameters → 能解析出 velocity_threshold 等 |
| `test_metric_entry_parameters_ref` | epm.yaml metric 含 parameters_ref → MetricEntry.parameters_ref 列表非空 |
| `test_paradigm_parameters_block` | epm.yaml 含 paradigm_parameters → Catalog.paradigm_parameters 解析正确 |
| `test_validate_parameters_ref_unknown_id` | metric parameters_ref=["fake_param"] → CatalogError 含 "not in shared_parameters" |
| `test_validate_duplicate_parameters_across_paradigms` | mock 两个范式都定义 `foo_threshold` default=10 → CatalogError 含 "should be promoted" |
| `test_validate_different_defaults_not_duplicate` | 两范式都定义 `foo_threshold` 但 default 不同 → 通过（不是同义重复）|
| `test_load_all_catalogs_e2e` | 跑 load_all_catalogs() 实际 6 yaml → 不抛异常（验证当前 catalog 一致性）|

新建 `packages/ethoinsight/tests/test_dispatcher_catalog_thresholds.py`：

| 测试 | 期望 |
|---|---|
| `test_dispatcher_reads_zm_threshold_from_catalog` | 改 zero_maze.yaml.paradigm_parameters.zm_low_distance_threshold = 5.0 → dispatcher warning evidence threshold=5.0 |
| `test_dispatcher_fallback_when_catalog_missing` | mock load_catalog 抛 CatalogError → dispatcher 用 hardcoded fallback 不挂 |
| `test_dispatcher_epm_sample_size_from_catalog` | 改 epm.yaml.paradigm_parameters.sample_size_underpowered_threshold = 8 → epm 组内 n=7 触发 SAMPLE.UNDERPOWERED |

### 2.8 集成测试

| 测试 | 期望 |
|---|---|
| `test_dogfood_fst_unchanged_behavior_with_catalog_thresholds` | Sprint 2a 上线后跑同一份 FST 数据 → handoff_code_executor.json 输出 bit-identical（因为 catalog default 与 Sprint 1 之前的硬编码值一致）|
| `test_load_all_catalogs_consistency_after_sprint_2a` | 6 个 yaml 全部加载 + validate_catalog_consistency → 不抛异常 |

---

## 3. 实施顺序（task 拆分）

| Task | 内容 | 估时 |
|---|---|---|
| T1 | catalog/schema.py 加 ParamSpec / SharedParameters / ParadigmParameters dataclass + MetricEntry/Catalog/CommonCatalog 加字段 | 1 天 |
| T2 | catalog/loader.py 加 `_parse_param_spec` / `_parse_param_block` / `_parse_metric_entry` 改造 / `_parse_catalog` 加 paradigm_parameters / `load_common_catalog` 加 shared_parameters | 1.5 天 |
| T3 | catalog/loader.py 加 `validate_catalog_consistency` + `load_all_catalogs` | 1 天 |
| T4 | T1-T3 的单元测试（test_catalog_parameters.py 10 个测试）| 1.5 天 |
| T5 | _common.yaml 加 shared_parameters 段（velocity_threshold + velocity_min_duration）| 0.25 天 |
| T6 | epm/oft/ldb/zero_maze 4 个 yaml 加 paradigm_parameters 段 + metric parameters_ref | 1 天 |
| T7 | fst.yaml + tst.yaml 加 paradigm_parameters（含 pendulum 9 参数）+ metric parameters_ref | 1 天 |
| T8 | dispatcher.py 改用 catalog 读取阈值（§2.6）+ 单元测试 | 1.5 天 |
| T9 | 集成测试（§2.8 dogfood FST 输出 bit-identical）| 0.5 天 |
| T10 | 全量回归测试 + 修复退化（缓冲）| 0.75 天 |
| T11 | 跑 `load_all_catalogs()` 确认 6 yaml 全过一致性校验 | 0.25 天 |
| **合计** | | **10.25 天 ≈ 2.5 周** |

---

## 4. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 改 schema.py dataclass 加字段后 ChartEntry / 其他下游代码字段位置错 | dataclass 必填字段都加 `default_factory` 或具体默认值，向后兼容；T4 全量回归测试 |
| yaml 中 paradigm_parameters 拼写错 | T11 跑 load_all_catalogs 一致性校验；先在 dev 跑通再合 main |
| 重复参数检测过严 / 误报 | duplicate 检测**要求 name 相同 + default 相同**，不同 default 算"范式独有"不报；test_validate_different_defaults_not_duplicate 验证 |
| dispatcher.py 改 catalog 读取后跑得慢（每次 load_catalog）| dispatcher 内部对 `_get_paradigm_threshold` 加 functools.lru_cache，避免重复 IO |
| FST 范式的 pendulum 9 参数 default 写错（与硬编码不一致）| T7 实施时**逐项 grep 对照** `_pendulum.py` 当前值；T9 集成测试验证 handoff bit-identical |
| Sprint 2b 期间发现需要在 Sprint 2a 没建好的字段上挂东西 | Sprint 2b spec 时再 grep 实测，必要时补 Sprint 2a 漏掉的 schema 字段 |

---

## 5. 验收 checklist

实施完成时确认下列全部通过：

- [ ] schema.py 含 ParamSpec / SharedParameters / ParadigmParameters dataclass
- [ ] MetricEntry 有 `parameters` 和 `parameters_ref` 字段（默认空）
- [ ] PlanMetric 有 `parameters_in_use` 字段（默认空，Sprint 2b 才填）
- [ ] Catalog 有 `paradigm_parameters` 字段
- [ ] CommonCatalog 有 `shared_parameters` 字段
- [ ] `_common.yaml` 含 `shared_parameters` 段，至少 velocity_threshold + velocity_min_duration
- [ ] 6 个 paradigm yaml（epm/oft/ldb/zero_maze/fst/tst）含 `paradigm_parameters` 段
- [ ] fst.yaml + tst.yaml 含 pendulum 9 个参数
- [ ] zero_maze.yaml 含 zm_low_distance_threshold
- [ ] loader.py 含 `_parse_param_spec` / `_parse_param_block` / `validate_catalog_consistency` / `load_all_catalogs`
- [ ] dispatcher.py 阈值从 catalog 读取（带 functools.lru_cache）
- [ ] 单元测试通过（test_catalog_parameters.py 10+ tests）
- [ ] 集成测试通过（dogfood FST handoff bit-identical）
- [ ] `load_all_catalogs()` 不抛异常（一致性校验通过）
- [ ] 全量回归测试（ethoinsight ≥455 + agent backend 不变）
- [ ] **重复参数检测真生效**：mock 两范式同名同 default → CatalogError

---

## 6. 不在 Sprint 2a 范围

- ❌ 修改 `metrics/_common.py` / `_pendulum.py` 中的硬编码常量（Sprint 2b 改函数签名时删）
- ❌ 修改 metric script CLI 参数（Sprint 2b 做）
- ❌ 写 overrides JSON 机制（Sprint 4.5 做）
- ❌ catalog.resolve 改用新 schema（Sprint 2b 做）
- ❌ PlanMetric.parameters_in_use 实际填值（Sprint 2b 做）
- ❌ 加非 v0.1 dogfood 用到的参数（如 timeseries 平滑参数等）—— YAGNI

---

## 7. 参考

- [SOTA agent 路线图 v2](../../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md) — Sprint 2a 章节
- [Sprint 1 实施 spec](2026-05-28-sprint-1-data-quality-structured-design.md) — dispatcher.py warning evidence 字段名延续
- catalog schema：`packages/ethoinsight/ethoinsight/catalog/schema.py`
- catalog loader：`packages/ethoinsight/ethoinsight/catalog/loader.py`
- 现有 _common.yaml：`packages/ethoinsight/ethoinsight/catalog/_common.yaml`
- 12 个硬编码常量（grep 实测）：
  - `metrics/_common.py:219-220`：velocity 套 2 个
  - `metrics/_pendulum.py:18-27`：pendulum 9 个
  - `metrics/dispatcher.py:235`：zm low distance
- 6 个 paradigm yaml：`packages/ethoinsight/ethoinsight/catalog/{epm,oft,ldb,zero_maze,fst,tst}.yaml`
