# Metric Catalog 架构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 ethoinsight 范式→指标知识沉到库内 YAML catalog（single source of truth），新增 `ethoinsight.catalog.resolve` CLI 把"用户意图 + 列名 + 范式"翻译成确定性 `metric_plan.json`，让 lead 触发 bash 生成 plan、code-executor 读 plan 逐条执行。

**Architecture:** 数据沉到 `packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml`，由 `catalog.resolve()` 函数 + CLI 消费；agent skill 极薄、只承担"如何读 catalog + 按 role 关注什么字段"指引；lead 持有 bash 触发 dump_headers + resolve、code-executor 读 plan.json 逐条执行；data-analyst / report-writer 按 metric id 直读 catalog YAML 取自己需要的字段。

**Tech Stack:** Python 3.10+、PyYAML、pydantic 或 dataclasses、pytest、argparse；YAML 为数据载体；JSON 为 agent 间 facts file；deerflow harness 不动。

**前置 spec:** [docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md](../specs/2026-05-13-metric-catalog-architecture-design.md)（已 approved）

---

## 文件结构总览

```
packages/ethoinsight/ethoinsight/
├── catalog/                    [NEW] 范式 → 指标 catalog 模块
│   ├── __init__.py             public: load_catalog, resolve, ResolveError
│   ├── schema.py               dataclass: MetricEntry / Catalog / Plan
│   ├── loader.py               YAML → Catalog + 一致性校验
│   ├── resolve.py              主逻辑：catalog + columns + 用户偏差 → Plan
│   ├── cli.py                  python -m ethoinsight.catalog.resolve 入口
│   ├── epm.yaml                EPM 5 个默认指标（Q6 白名单对齐）
│   ├── oft.yaml                OFT 5 个默认指标（Q6 白名单对齐）
│   ├── fst.yaml                FST 3 个默认指标（Q6 白名单对齐）
│   ├── tst.yaml                TST 3 个默认指标
│   ├── ldb.yaml                LDB 3 个默认指标
│   ├── zero_maze.yaml          Zero Maze 4 个默认指标
│   └── shoaling.yaml           Shoaling 3 个默认指标
├── parse/                      [REFACTOR] parse.py → 包；保持向后兼容
│   ├── __init__.py             re-export parse_header/parse_trajectory/parse_batch
│   ├── _core.py                现 parse.py 内容（移过来）
│   └── dump_headers.py         [NEW] CLI: python -m ethoinsight.parse.dump_headers
├── metrics/oft.py              [MOD] 删 _find_center_zone_column 裸 in_zone fallback；新增 AmbiguousZoneError
├── metrics/epm.py              [MOD] 加 ev19_template 兼容性常量
├── metrics/oft.py              [MOD] 加缺失指标 compute_center_time / compute_center_distance
└── scripts/oft/                [MOD] 加 compute_center_time.py / compute_center_distance.py

packages/ethoinsight/tests/
├── test_catalog.py             [NEW] catalog 模块全套测试
├── test_dump_headers.py        [NEW] CLI 行为测试
└── test_q6_whitelist.py        [NEW] 反退化测试：catalog 与同事 Q6 白名单严格对齐

packages/agent/skills/custom/
├── ethoinsight-metric-catalog/ [NEW] 领域知识接口 skill
│   ├── SKILL.md
│   └── references/
│       ├── resolve-cli.md      lead 专用：resolve CLI 参数 + 错误码 → 反问话术映射
│       └── field-guide.md      各 role 关注的字段映射表
├── ethoinsight-code/SKILL.md   [MOD] workflow 段重写为 read plan.json
├── ethoinsight-code/references/by-paradigm/*.md  [DELETE] × 7 份
├── ethoinsight-code/references/error-recovery.md [MOD] 删第 46-51 行（assess_and_handoff 旧引用）
└── ethoinsight-code/references/quality-checks.md [MOD] 删第 44 行（同上）

packages/agent/backend/packages/harness/deerflow/
├── subagents/builtins/code_executor.py    [MOD] system_prompt workflow 段改写
├── subagents/builtins/data_analyst.py     [MOD] skills 列表加 metric-catalog
├── subagents/builtins/report_writer.py    [MOD] skills 列表加 metric-catalog
└── agents/lead_agent/prompt.py            [MOD] skill description 段 + Gate 2 工作流提示

CLAUDE.md                                  [MOD] §7 流水线描述更新
```

---

## Task 1: catalog 模块骨架 + schema

奠基。这一步完成后 `from ethoinsight.catalog import Catalog, MetricEntry, Plan` 可工作，但还没有真数据。

**Files:**
- Create: `packages/ethoinsight/ethoinsight/catalog/__init__.py`
- Create: `packages/ethoinsight/ethoinsight/catalog/schema.py`
- Create: `packages/ethoinsight/tests/test_catalog.py`

- [ ] **Step 1: 写失败测试 - schema 类可导入并接受字段**

`packages/ethoinsight/tests/test_catalog.py`：

```python
"""Tests for ethoinsight.catalog module."""

from __future__ import annotations

import pytest


def test_schema_classes_importable():
    from ethoinsight.catalog import schema
    assert hasattr(schema, "MetricEntry")
    assert hasattr(schema, "Catalog")
    assert hasattr(schema, "Plan")


def test_metric_entry_minimal_construction():
    from ethoinsight.catalog.schema import MetricEntry
    m = MetricEntry(
        id="open_arm_time_ratio",
        script="ethoinsight.scripts.epm.compute_open_arm_time_ratio",
        requires_columns=["in_zone_open_arms_*"],
        output_unit="ratio",
        display_name_zh="开放臂时间比例",
        unit_zh="比例",
        one_liner="动物在开放臂中停留时间占总时长的比例",
        direction_for_anxiety="lower_is_anxious",
        statistical_default="groupwise_compare",
    )
    assert m.id == "open_arm_time_ratio"
    assert m.direction_for_anxiety == "lower_is_anxious"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'ethoinsight.catalog'`）

- [ ] **Step 3: 写 catalog 模块 __init__.py**

`packages/ethoinsight/ethoinsight/catalog/__init__.py`：

```python
"""ethoinsight.catalog — 范式 → 指标 catalog 模块.

承载 single source of truth：每个 paradigm 一份 YAML 文件，定义默认指标清单 +
脚本路径 + 列要求 + 展示元数据 + 判读方向性。被 lead / data-analyst /
report-writer 多方共读、被 dispatcher / 单测 / golden-case 共消费。

设计 spec: docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md
"""

from __future__ import annotations

from ethoinsight.catalog.schema import (
    Catalog,
    MetricEntry,
    Plan,
    PlanMetric,
    PlanSkipped,
    StatisticsEntry,
    ChartEntry,
)

__all__ = [
    "Catalog",
    "MetricEntry",
    "Plan",
    "PlanMetric",
    "PlanSkipped",
    "StatisticsEntry",
    "ChartEntry",
]
```

- [ ] **Step 4: 写 schema.py**

`packages/ethoinsight/ethoinsight/catalog/schema.py`：

```python
"""Catalog 数据模型（dataclass）。

为什么不用 pydantic：保持 ethoinsight 库的依赖最小化（参见
pyproject.toml）。YAML 校验由 loader.py 手工做。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DirectionEnum = Literal["lower_is_anxious", "higher_is_anxious"] | None
StatDefault = Literal["groupwise_compare", "paired_compare"]
ChartCondition = str  # e.g. "always", "n_per_group >= 3"

ALLOWED_DIRECTIONS: frozenset[str | None] = frozenset({
    "lower_is_anxious", "higher_is_anxious", None,
})
ALLOWED_STAT_DEFAULTS: frozenset[str] = frozenset({
    "groupwise_compare", "paired_compare",
})


@dataclass(frozen=True)
class MetricEntry:
    id: str
    script: str
    requires_columns: list[str]
    output_unit: str
    display_name_zh: str
    unit_zh: str
    one_liner: str
    direction_for_anxiety: str | None  # validated against ALLOWED_DIRECTIONS
    statistical_default: str           # validated against ALLOWED_STAT_DEFAULTS


@dataclass(frozen=True)
class ChartEntry:
    id: str
    script: str
    when: ChartCondition  # "always" | "n_per_group >= K" | "n_groups >= K"


@dataclass(frozen=True)
class StatisticsEntry:
    id: str
    script: str
    when: ChartCondition


@dataclass(frozen=True)
class Catalog:
    paradigm: str
    ev19_templates: list[str]
    default_metrics: list[MetricEntry]
    optional_metrics: list[MetricEntry]
    charts: list[ChartEntry]
    statistics_default: StatisticsEntry | None


# ============================================================================
# Plan (输出结构) — metric_plan.json schema
# ============================================================================


PlanReasonEnum = Literal[
    "paradigm.default", "paradigm.required",
    "user.include", "paradigm.optional.applicable",
]
SkippedReasonEnum = Literal[
    "user.exclude", "columns.missing",
    "paradigm.not_applicable", "catalog.unknown",
]


@dataclass
class PlanMetric:
    id: str
    script: str
    input: str
    output: str
    required: bool
    reason: str  # PlanReasonEnum


@dataclass
class PlanSkipped:
    id: str
    reason: str  # SkippedReasonEnum
    detail: str


@dataclass
class PlanStatistics:
    id: str
    script: str
    input: str
    output: str
    skip_reason: str | None  # None = 跑；非空字符串 = 跳过原因


@dataclass
class PlanChart:
    id: str
    script: str
    input: str
    output: str


@dataclass
class PlanInputs:
    raw_files: list[str]
    groups_file: str | None
    columns_file: str | None


@dataclass
class Plan:
    schema_version: str
    paradigm: str
    ev19_template: str | None
    generated_at: str
    inputs: PlanInputs
    metrics: list[PlanMetric]
    statistics: PlanStatistics | None
    charts: list[PlanChart]
    skipped: list[PlanSkipped]
    notes: list[str]
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v`
Expected: PASS（2 个测试）

- [ ] **Step 6: Commit**

```bash
git add packages/ethoinsight/ethoinsight/catalog/__init__.py \
        packages/ethoinsight/ethoinsight/catalog/schema.py \
        packages/ethoinsight/tests/test_catalog.py
git commit -m "feat(catalog): 添加 catalog 模块骨架 + schema dataclass

为 metric catalog 架构（spec 2026-05-13）奠基。schema.py 用 dataclass
定义 Catalog / MetricEntry / Plan 等数据模型，不引入 pydantic 依赖。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: catalog YAML loader + 一致性校验

完成后 `load_catalog("epm")` 能读 YAML 并返回 `Catalog` 实例；YAML 损坏 / 必填字段缺失 / enum 越界都报清晰错误。

**Files:**
- Create: `packages/ethoinsight/ethoinsight/catalog/loader.py`
- Modify: `packages/ethoinsight/ethoinsight/catalog/__init__.py`（export `load_catalog`、`CatalogError`）
- Modify: `packages/ethoinsight/tests/test_catalog.py`（追加测试）
- Create: `packages/ethoinsight/tests/fixtures/catalog/test_paradigm.yaml`（测试用 fixture）

- [ ] **Step 1: 写失败测试 - load_catalog 接受 paradigm 名返回 Catalog 实例**

追加到 `tests/test_catalog.py`：

```python
def test_load_catalog_returns_catalog_instance(tmp_path):
    from ethoinsight.catalog.loader import load_catalog
    from ethoinsight.catalog.schema import Catalog

    # 临时写一份 minimal 合法 YAML
    yaml_content = """
paradigm: epm_test
ev19_templates:
  - Elevated Plus Maze XT190
default_metrics:
  - id: open_arm_time_ratio
    script: ethoinsight.scripts.epm.compute_open_arm_time_ratio
    requires_columns: [in_zone_open_arms_*]
    output_unit: ratio
    display_name_zh: 开放臂时间比例
    unit_zh: 比例
    one_liner: 动物在开放臂中停留时间占总时长的比例
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare
optional_metrics: []
charts: []
statistics:
  default:
    id: groupwise_compare
    script: ethoinsight.scripts.epm.run_groupwise_stats
    when: n_per_group >= 2 and n_groups >= 2
"""
    yaml_path = tmp_path / "epm_test.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    cat = load_catalog("epm_test", catalog_dir=str(tmp_path))
    assert isinstance(cat, Catalog)
    assert cat.paradigm == "epm_test"
    assert len(cat.default_metrics) == 1
    assert cat.default_metrics[0].id == "open_arm_time_ratio"
    assert cat.statistics_default is not None
    assert cat.statistics_default.id == "groupwise_compare"


def test_load_catalog_unknown_paradigm_raises(tmp_path):
    from ethoinsight.catalog.loader import load_catalog, CatalogError

    with pytest.raises(CatalogError) as exc:
        load_catalog("totally_made_up", catalog_dir=str(tmp_path))
    assert "totally_made_up" in str(exc.value)


def test_load_catalog_rejects_invalid_direction_enum(tmp_path):
    from ethoinsight.catalog.loader import load_catalog, CatalogError

    bad = """
paradigm: epm_test
ev19_templates: []
default_metrics:
  - id: foo
    script: ethoinsight.scripts.epm.compute_open_arm_time_ratio
    requires_columns: []
    output_unit: ratio
    display_name_zh: 测试
    unit_zh: 单位
    one_liner: 描述
    direction_for_anxiety: very_weirdly_anxious   # 非法 enum
    statistical_default: groupwise_compare
optional_metrics: []
charts: []
statistics: null
"""
    (tmp_path / "epm_test.yaml").write_text(bad, encoding="utf-8")
    with pytest.raises(CatalogError) as exc:
        load_catalog("epm_test", catalog_dir=str(tmp_path))
    assert "direction_for_anxiety" in str(exc.value)


def test_load_catalog_rejects_duplicate_metric_ids(tmp_path):
    from ethoinsight.catalog.loader import load_catalog, CatalogError

    bad = """
paradigm: epm_test
ev19_templates: []
default_metrics:
  - id: dup_id
    script: ethoinsight.scripts.epm.compute_open_arm_time_ratio
    requires_columns: []
    output_unit: ratio
    display_name_zh: 一
    unit_zh: u
    one_liner: a
    direction_for_anxiety: null
    statistical_default: groupwise_compare
  - id: dup_id
    script: ethoinsight.scripts.epm.compute_open_arm_time
    requires_columns: []
    output_unit: seconds
    display_name_zh: 二
    unit_zh: u
    one_liner: b
    direction_for_anxiety: null
    statistical_default: groupwise_compare
optional_metrics: []
charts: []
statistics: null
"""
    (tmp_path / "epm_test.yaml").write_text(bad, encoding="utf-8")
    with pytest.raises(CatalogError) as exc:
        load_catalog("epm_test", catalog_dir=str(tmp_path))
    assert "dup_id" in str(exc.value)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v -k "load_catalog"`
Expected: FAIL（`ImportError: cannot import name 'load_catalog'`）

- [ ] **Step 3: 写 loader.py**

`packages/ethoinsight/ethoinsight/catalog/loader.py`：

```python
"""Catalog YAML loader + 一致性校验。

加载 packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml 并构造
Catalog dataclass。校验失败一律抛 CatalogError，含 paradigm + 问题点。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ethoinsight.catalog.schema import (
    ALLOWED_DIRECTIONS,
    ALLOWED_STAT_DEFAULTS,
    Catalog,
    ChartEntry,
    MetricEntry,
    StatisticsEntry,
)


class CatalogError(Exception):
    """Catalog YAML 加载 / 校验失败。"""


_DEFAULT_CATALOG_DIR = Path(__file__).parent


def load_catalog(paradigm: str, catalog_dir: str | Path | None = None) -> Catalog:
    """加载 <catalog_dir>/<paradigm>.yaml 并校验返回 Catalog。

    Args:
        paradigm: 范式 key（如 "epm"、"oft"）
        catalog_dir: catalog YAML 目录；默认为本模块所在目录

    Raises:
        CatalogError: 文件不存在 / 必填字段缺失 / enum 越界 / id 重复 等
    """
    catalog_dir = Path(catalog_dir) if catalog_dir else _DEFAULT_CATALOG_DIR
    yaml_path = catalog_dir / f"{paradigm}.yaml"
    if not yaml_path.is_file():
        raise CatalogError(f"Catalog file not found for paradigm '{paradigm}': {yaml_path}")
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise CatalogError(f"YAML parse failed for {yaml_path}: {e}") from e

    if not isinstance(raw, dict):
        raise CatalogError(f"{yaml_path}: top-level must be a mapping, got {type(raw).__name__}")

    return _parse_catalog(raw, source=yaml_path)


def _parse_catalog(raw: dict[str, Any], source: Path) -> Catalog:
    paradigm = _require_str(raw, "paradigm", source)
    ev19_templates = _require_list_of_str(raw, "ev19_templates", source)

    default_metrics = _parse_metric_list(raw, "default_metrics", source)
    optional_metrics = _parse_metric_list(raw, "optional_metrics", source)

    # ID 在 default + optional 全集内必须唯一
    seen_ids: set[str] = set()
    for m in default_metrics + optional_metrics:
        if m.id in seen_ids:
            raise CatalogError(
                f"{source}: duplicate metric id '{m.id}' in default_metrics + optional_metrics"
            )
        seen_ids.add(m.id)

    charts = _parse_chart_list(raw, source)
    statistics_default = _parse_statistics(raw, source)

    return Catalog(
        paradigm=paradigm,
        ev19_templates=ev19_templates,
        default_metrics=default_metrics,
        optional_metrics=optional_metrics,
        charts=charts,
        statistics_default=statistics_default,
    )


def _parse_metric_list(raw: dict, key: str, source: Path) -> list[MetricEntry]:
    items = raw.get(key, []) or []
    if not isinstance(items, list):
        raise CatalogError(f"{source}: '{key}' must be a list, got {type(items).__name__}")
    result: list[MetricEntry] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise CatalogError(f"{source}: {key}[{i}] must be a mapping")
        result.append(_parse_metric_entry(item, where=f"{key}[{i}]", source=source))
    return result


def _parse_metric_entry(item: dict, where: str, source: Path) -> MetricEntry:
    def req(field: str, expected_type: type | tuple) -> Any:
        if field not in item:
            raise CatalogError(f"{source} {where}: missing required field '{field}'")
        v = item[field]
        if not isinstance(v, expected_type):
            raise CatalogError(
                f"{source} {where}: field '{field}' must be {expected_type}, got {type(v).__name__}"
            )
        return v

    mid = req("id", str)
    script = req("script", str)
    if "requires_columns" not in item:
        raise CatalogError(f"{source} {where}: missing 'requires_columns' (use empty list if none)")
    requires_columns = item["requires_columns"]
    if not isinstance(requires_columns, list) or not all(isinstance(c, str) for c in requires_columns):
        raise CatalogError(f"{source} {where}: 'requires_columns' must be list[str]")

    output_unit = req("output_unit", str)
    display_name_zh = req("display_name_zh", str)
    unit_zh = req("unit_zh", str)
    one_liner = req("one_liner", str)

    direction = item.get("direction_for_anxiety", None)
    if direction not in ALLOWED_DIRECTIONS:
        raise CatalogError(
            f"{source} {where}: 'direction_for_anxiety' must be one of "
            f"{sorted(str(x) for x in ALLOWED_DIRECTIONS)}, got {direction!r}"
        )

    stat_default = req("statistical_default", str)
    if stat_default not in ALLOWED_STAT_DEFAULTS:
        raise CatalogError(
            f"{source} {where}: 'statistical_default' must be one of "
            f"{sorted(ALLOWED_STAT_DEFAULTS)}, got {stat_default!r}"
        )

    return MetricEntry(
        id=mid,
        script=script,
        requires_columns=list(requires_columns),
        output_unit=output_unit,
        display_name_zh=display_name_zh,
        unit_zh=unit_zh,
        one_liner=one_liner,
        direction_for_anxiety=direction,
        statistical_default=stat_default,
    )


def _parse_chart_list(raw: dict, source: Path) -> list[ChartEntry]:
    items = raw.get("charts", []) or []
    if not isinstance(items, list):
        raise CatalogError(f"{source}: 'charts' must be a list")
    out: list[ChartEntry] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise CatalogError(f"{source}: charts[{i}] must be a mapping")
        for f in ("id", "script", "when"):
            if f not in it:
                raise CatalogError(f"{source}: charts[{i}] missing '{f}'")
        out.append(ChartEntry(id=it["id"], script=it["script"], when=it["when"]))
    return out


def _parse_statistics(raw: dict, source: Path) -> StatisticsEntry | None:
    block = raw.get("statistics")
    if block is None:
        return None
    if not isinstance(block, dict) or "default" not in block:
        raise CatalogError(f"{source}: 'statistics' must contain 'default' mapping or be null")
    d = block["default"]
    for f in ("id", "script", "when"):
        if f not in d:
            raise CatalogError(f"{source}: statistics.default missing '{f}'")
    return StatisticsEntry(id=d["id"], script=d["script"], when=d["when"])


def _require_str(raw: dict, key: str, source: Path) -> str:
    if key not in raw or not isinstance(raw[key], str):
        raise CatalogError(f"{source}: missing or non-string '{key}'")
    return raw[key]


def _require_list_of_str(raw: dict, key: str, source: Path) -> list[str]:
    v = raw.get(key, []) or []
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise CatalogError(f"{source}: '{key}' must be list[str]")
    return list(v)
```

- [ ] **Step 4: 导出 load_catalog 和 CatalogError**

修改 `packages/ethoinsight/ethoinsight/catalog/__init__.py` 末尾：

```python
from ethoinsight.catalog.loader import CatalogError, load_catalog

__all__ = [
    "Catalog",
    "CatalogError",
    "ChartEntry",
    "MetricEntry",
    "Plan",
    "PlanMetric",
    "PlanSkipped",
    "StatisticsEntry",
    "load_catalog",
]
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v -k "load_catalog"`
Expected: PASS（4 个测试）

- [ ] **Step 6: 确认 PyYAML 已是依赖**

Run: `cd packages/ethoinsight && python -c "import yaml; print(yaml.__version__)"`
Expected: 输出版本号（assess.py 已用 yaml，应已是依赖）

如果失败：在 `packages/ethoinsight/pyproject.toml` 的 dependencies 加 `pyyaml`，然后 `uv sync` 或 `pip install -e .`

- [ ] **Step 7: Commit**

```bash
git add packages/ethoinsight/ethoinsight/catalog/__init__.py \
        packages/ethoinsight/ethoinsight/catalog/loader.py \
        packages/ethoinsight/tests/test_catalog.py
git commit -m "feat(catalog): YAML loader + schema 校验

load_catalog(paradigm) 读 packages/ethoinsight/ethoinsight/catalog/
<paradigm>.yaml，校验：必填字段、enum 取值（direction_for_anxiety、
statistical_default）、metric id 唯一性。失败统一抛 CatalogError。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: EPM / OFT / FST 三份 catalog YAML（Q6 白名单对齐）

完成后 `load_catalog("epm")`、`load_catalog("oft")`、`load_catalog("fst")` 能加载真实数据。这三个范式是同事 5-13 反馈里有白名单的，优先做。

**Files:**
- Create: `packages/ethoinsight/ethoinsight/catalog/epm.yaml`
- Create: `packages/ethoinsight/ethoinsight/catalog/oft.yaml`
- Create: `packages/ethoinsight/ethoinsight/catalog/fst.yaml`
- Modify: `packages/ethoinsight/tests/test_catalog.py`（追加 Q6 alignment 测试）

- [ ] **Step 1: 写失败测试 - 三份 YAML 加载 + Q6 白名单对齐**

追加到 `tests/test_catalog.py`：

```python
# Q6 白名单来源：docs/review-packages/2026-05-12-feedback.md
EPM_Q6_WHITELIST = {
    "open_arm_time_ratio",
    "open_arm_time",
    "open_arm_entry_count",
    "open_arm_entry_ratio",
    "total_entry_count",
}

OFT_Q6_WHITELIST = {
    "center_time_ratio",
    "center_distance_ratio",
    "center_entry_count",
    "center_time",
    "center_distance",
}

FST_Q6_WHITELIST = {
    "immobility_time",
    "immobility_latency",
    "immobility_bout_count",
}


@pytest.mark.parametrize("paradigm,whitelist", [
    ("epm", EPM_Q6_WHITELIST),
    ("oft", OFT_Q6_WHITELIST),
    ("fst", FST_Q6_WHITELIST),
])
def test_catalog_default_metrics_match_q6_whitelist(paradigm, whitelist):
    """反退化测试：catalog default_metrics 必须严格等于同事 5-13 Q6 白名单。

    Source: docs/review-packages/2026-05-12-feedback.md
    任何 catalog 改动如果碰这三个范式的 default 集合，必须显式同步
    本测试 + Q6 白名单文档（不允许偷偷改）。
    """
    from ethoinsight.catalog import load_catalog
    cat = load_catalog(paradigm)
    catalog_ids = {m.id for m in cat.default_metrics}
    assert catalog_ids == whitelist, (
        f"{paradigm} default_metrics 与 Q6 白名单偏差：\n"
        f"  catalog 有但 Q6 没: {catalog_ids - whitelist}\n"
        f"  Q6 有但 catalog 没: {whitelist - catalog_ids}"
    )


@pytest.mark.parametrize("paradigm", ["epm", "oft", "fst"])
def test_catalog_loads_real_yaml(paradigm):
    from ethoinsight.catalog import load_catalog
    cat = load_catalog(paradigm)
    assert cat.paradigm == paradigm
    assert len(cat.default_metrics) > 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v -k "q6_whitelist or loads_real_yaml"`
Expected: FAIL（CatalogError: file not found）

- [ ] **Step 3: 写 epm.yaml**

`packages/ethoinsight/ethoinsight/catalog/epm.yaml`：

```yaml
paradigm: epm
ev19_templates:
  - Elevated Plus Maze XT190

default_metrics:
  - id: open_arm_time_ratio
    script: ethoinsight.scripts.epm.compute_open_arm_time_ratio
    requires_columns:
      - in_zone_open_arms_*
    output_unit: ratio
    display_name_zh: 开放臂时间比例
    unit_zh: 比例
    one_liner: 动物在开放臂中停留时间占总时长的比例，用于评估焦虑样回避行为
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: open_arm_time
    script: ethoinsight.scripts.epm.compute_open_arm_time
    requires_columns:
      - in_zone_open_arms_*
    output_unit: seconds
    display_name_zh: 开放臂时间
    unit_zh: 秒
    one_liner: 动物在开放臂内的累计停留时间
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: open_arm_entry_count
    script: ethoinsight.scripts.epm.compute_open_arm_entry_count
    requires_columns:
      - in_zone_open_arms_*
    output_unit: count
    display_name_zh: 开放臂进入次数
    unit_zh: 次
    one_liner: 进入开放臂的累计次数
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: open_arm_entry_ratio
    script: ethoinsight.scripts.epm.compute_open_arm_entry_ratio
    requires_columns:
      - in_zone_open_arms_*
      - in_zone_closed_arms_*
    output_unit: ratio
    display_name_zh: 开放臂进入比例
    unit_zh: 比例
    one_liner: 进入开放臂次数占总进入次数的比例
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: total_entry_count
    script: ethoinsight.scripts.epm.compute_total_entry_count
    requires_columns:
      - in_zone_open_arms_*
      - in_zone_closed_arms_*
    output_unit: count
    display_name_zh: 总进入次数
    unit_zh: 次
    one_liner: 进入所有臂的累计次数，反映动物的整体探索活动量
    direction_for_anxiety: null
    statistical_default: groupwise_compare

optional_metrics: []

charts:
  - id: box_open_arm
    script: ethoinsight.scripts.epm.plot_box_open_arm
    when: n_per_group >= 3

statistics:
  default:
    id: groupwise_compare
    script: ethoinsight.scripts.epm.run_groupwise_stats
    when: n_per_group >= 2 and n_groups >= 2
```

- [ ] **Step 4: 写 oft.yaml**

`packages/ethoinsight/ethoinsight/catalog/oft.yaml`：

```yaml
paradigm: oft
ev19_templates:
  - Open Field XT190
  - 旷场_小鼠_三点

default_metrics:
  - id: center_time_ratio
    script: ethoinsight.scripts.oft.compute_center_time_ratio
    requires_columns:
      - in_zone_center_*
    output_unit: ratio
    display_name_zh: 中心区时间比例
    unit_zh: 比例
    one_liner: 动物在中心区停留时间占总时长的比例，center 越低焦虑样回避越明显
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: center_distance_ratio
    script: ethoinsight.scripts.oft.compute_center_distance_ratio
    requires_columns:
      - in_zone_center_*
      - distance_moved
    output_unit: ratio
    display_name_zh: 中心区移动距离比例
    unit_zh: 比例
    one_liner: 中心区移动距离占总移动距离的比例
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: center_entry_count
    script: ethoinsight.scripts.oft.compute_center_entry_count
    requires_columns:
      - in_zone_center_*
    output_unit: count
    display_name_zh: 中心区进入次数
    unit_zh: 次
    one_liner: 进入中心区的累计次数
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: center_time
    script: ethoinsight.scripts.oft.compute_center_time
    requires_columns:
      - in_zone_center_*
    output_unit: seconds
    display_name_zh: 中心区时间
    unit_zh: 秒
    one_liner: 动物在中心区内的累计停留时间
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: center_distance
    script: ethoinsight.scripts.oft.compute_center_distance
    requires_columns:
      - in_zone_center_*
      - distance_moved
    output_unit: cm
    display_name_zh: 中心区移动距离
    unit_zh: cm
    one_liner: 动物在中心区内的累计移动距离
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

optional_metrics: []

charts:
  - id: box_center
    script: ethoinsight.scripts.oft.plot_box_center
    when: n_per_group >= 3

statistics:
  default:
    id: groupwise_compare
    script: ethoinsight.scripts.oft.run_groupwise_stats
    when: n_per_group >= 2 and n_groups >= 2
```

- [ ] **Step 5: 写 fst.yaml**

`packages/ethoinsight/ethoinsight/catalog/fst.yaml`：

```yaml
paradigm: fst
ev19_templates:
  - Forced Swim Test XT190
  - 强迫游泳_大鼠

default_metrics:
  - id: immobility_time
    script: ethoinsight.scripts.fst.compute_immobility_time
    requires_columns:
      - mobility_state*
    output_unit: seconds
    display_name_zh: 不动时间
    unit_zh: 秒
    one_liner: 动物在测试期间累计不动的时间，FST 抑郁样行为的核心指标
    direction_for_anxiety: null
    statistical_default: groupwise_compare

  - id: immobility_latency
    script: ethoinsight.scripts.fst.compute_immobility_latency
    requires_columns:
      - mobility_state*
    output_unit: seconds
    display_name_zh: 首次不动潜伏期
    unit_zh: 秒
    one_liner: 从测试开始到首次出现不动的时间
    direction_for_anxiety: null
    statistical_default: groupwise_compare

  - id: immobility_bout_count
    script: ethoinsight.scripts.fst.compute_immobility_bout_count
    requires_columns:
      - mobility_state*
    output_unit: count
    display_name_zh: 不动次数
    unit_zh: 次
    one_liner: 测试期间不动行为发生的累计次数（run-length encoding）
    direction_for_anxiety: null
    statistical_default: groupwise_compare

optional_metrics: []

charts:
  - id: box_immobility
    script: ethoinsight.scripts.fst.plot_box_immobility
    when: n_per_group >= 3

statistics:
  default:
    id: groupwise_compare
    script: ethoinsight.scripts.fst.run_groupwise_stats
    when: n_per_group >= 2 and n_groups >= 2
```

- [ ] **Step 6: 跑测试确认通过**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v -k "q6_whitelist or loads_real_yaml"`
Expected: PASS（6 个测试）。如果 fail，对照报错信息修 YAML（90% 是字段拼写或字段缺失）。

- [ ] **Step 7: Commit**

```bash
git add packages/ethoinsight/ethoinsight/catalog/epm.yaml \
        packages/ethoinsight/ethoinsight/catalog/oft.yaml \
        packages/ethoinsight/ethoinsight/catalog/fst.yaml \
        packages/ethoinsight/tests/test_catalog.py
git commit -m "feat(catalog): EPM / OFT / FST catalog YAML（Q6 白名单对齐）

按同事 2026-05-12 review feedback Q6 提供的指标白名单填三份 catalog：
- EPM: 5 个默认指标（open_arm_time_ratio/_time/_entry_count/_entry_ratio
  + total_entry_count）
- OFT: 5 个默认指标，含同事新要求的 center_time + center_distance
- FST: 3 个默认指标，命名收口为 immobility_time/_latency/_bout_count

Q6 白名单严格对齐用反退化测试 test_catalog_default_metrics_match_q6_whitelist
锁死，未来改 catalog 必须显式同步该测试。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: 补齐 OFT 缺失的脚本（compute_center_time + compute_center_distance）

Task 3 的 OFT YAML 引用了两个脚本，但实际 `packages/ethoinsight/ethoinsight/scripts/oft/` 没有它们。先补脚本，否则后续 Task 6 的 `script_importable` 测试会挂。

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/metrics/oft.py`（加 `compute_center_time` + `compute_center_distance` 函数）
- Create: `packages/ethoinsight/ethoinsight/scripts/oft/compute_center_time.py`
- Create: `packages/ethoinsight/ethoinsight/scripts/oft/compute_center_distance.py`
- Modify: `packages/ethoinsight/tests/test_metrics_oft.py`（追加新指标测试）

- [ ] **Step 1: 看现有 OFT 指标的实现模式**

Run: `cd packages/ethoinsight && cat ethoinsight/metrics/oft.py | head -100`

看 `compute_center_time_ratio` 怎么写的、用了哪些列、返回什么类型。下面的实现要照搬这个模式。

- [ ] **Step 2: 写失败测试**

追加到 `packages/ethoinsight/tests/test_metrics_oft.py`：

```python
def test_compute_center_time_returns_seconds():
    """center_time = center_time_ratio * total_duration"""
    import pandas as pd
    from ethoinsight.metrics.oft import compute_center_time

    df = pd.DataFrame({
        "time": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "in_zone_center_center_point": [1, 1, 0, 0, 1, 0],
    })
    result = compute_center_time(df)
    # 6 frames * (3/6 ratio) * (0.1 frame interval) = 0.3s 误差 ε
    # 简化：直接断言不为 None 且 > 0
    assert result is not None
    assert result > 0


def test_compute_center_distance_returns_cm():
    import pandas as pd
    from ethoinsight.metrics.oft import compute_center_distance

    df = pd.DataFrame({
        "time": [0.0, 0.1, 0.2, 0.3],
        "in_zone_center_center_point": [1, 1, 0, 1],
        "distance_moved": [0.0, 1.5, 2.0, 0.5],
    })
    result = compute_center_distance(df)
    # 仅累加 in_zone_center=1 时的 distance_moved
    # = 1.5（第二帧的 cumulative 增量）+ 0.5（第四帧）= 2.0 cm
    assert result is not None
    assert 1.8 <= result <= 2.2  # 留点容差
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd packages/ethoinsight && pytest tests/test_metrics_oft.py -v -k "center_time or center_distance"`
Expected: FAIL（ImportError 或 AttributeError）

- [ ] **Step 4: 实现两个 metric 函数**

追加到 `packages/ethoinsight/ethoinsight/metrics/oft.py`（保留原有代码、仅追加）：

```python
def compute_center_time(df: pd.DataFrame) -> float | None:
    """Total time the subject spent in center zone (seconds).

    = center_time_ratio * total_duration

    Returns None if center column cannot be resolved (raises AmbiguousZoneError
    in upstream once that path lands).
    """
    ratio = compute_center_time_ratio(df)
    if ratio is None:
        return None
    if "time" not in df.columns:
        return None
    duration = float(df["time"].iloc[-1] - df["time"].iloc[0])
    return ratio * duration


def compute_center_distance(df: pd.DataFrame) -> float | None:
    """Total distance moved while in center zone (cm).

    Accumulates ``distance_moved`` only at frames where the center-zone indicator is 1.
    Returns None if either column is missing.
    """
    if "distance_moved" not in df.columns:
        return None
    center_col = _find_center_zone_column(df)
    if center_col is None:
        return None
    mask = df[center_col].fillna(0) > 0
    return float(df.loc[mask, "distance_moved"].sum())
```

**Note:** `_find_center_zone_column` 函数在现有 oft.py 里已有；本任务还不动它（Task 7 才退役 silent fallback）。

- [ ] **Step 5: 跑测试确认通过**

Run: `cd packages/ethoinsight && pytest tests/test_metrics_oft.py -v -k "center_time or center_distance"`
Expected: PASS（2 个测试）

- [ ] **Step 6: 写两个 CLI 脚本（照搬 compute_center_time_ratio.py 的模板）**

`packages/ethoinsight/ethoinsight/scripts/oft/compute_center_time.py`：

```python
"""OFT: 中心区累计停留时间 (center_time, 秒)。

CLI: python -m ethoinsight.scripts.oft.compute_center_time \\
       --input <轨迹文件> --output <metric.json>

输出 JSON: {"metric": "center_time", "value": <float or null>}
stdout 末尾打印 [result] {json} 行供 subagent 抓取。
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.oft import compute_center_time
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import (
    emit_result,
    make_compute_parser,
    save_output_json,
)


METRIC_NAME = "center_time"


def main(argv: list[str] | None = None) -> int:
    parser = make_compute_parser(description=__doc__)
    args = parser.parse_args(argv)

    df = parse_trajectory(args.input)
    value = compute_center_time(df)

    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`packages/ethoinsight/ethoinsight/scripts/oft/compute_center_distance.py`：

```python
"""OFT: 中心区累计移动距离 (center_distance, cm)。

CLI: python -m ethoinsight.scripts.oft.compute_center_distance \\
       --input <轨迹文件> --output <metric.json>

输出 JSON: {"metric": "center_distance", "value": <float or null>}
stdout 末尾打印 [result] {json} 行供 subagent 抓取。
"""

from __future__ import annotations

import sys

from ethoinsight.metrics.oft import compute_center_distance
from ethoinsight.parse import parse_trajectory
from ethoinsight.scripts._cli import (
    emit_result,
    make_compute_parser,
    save_output_json,
)


METRIC_NAME = "center_distance"


def main(argv: list[str] | None = None) -> int:
    parser = make_compute_parser(description=__doc__)
    args = parser.parse_args(argv)

    df = parse_trajectory(args.input)
    value = compute_center_distance(df)

    payload = {"metric": METRIC_NAME, "value": value}
    save_output_json(args.output, payload)
    emit_result(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: 冒烟测试两个脚本可被 import + main 可调用**

Run:
```bash
cd packages/ethoinsight && python -c "
from ethoinsight.scripts.oft import compute_center_time, compute_center_distance
assert callable(compute_center_time.main)
assert callable(compute_center_distance.main)
print('OK')
"
```
Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add packages/ethoinsight/ethoinsight/metrics/oft.py \
        packages/ethoinsight/ethoinsight/scripts/oft/compute_center_time.py \
        packages/ethoinsight/ethoinsight/scripts/oft/compute_center_distance.py \
        packages/ethoinsight/tests/test_metrics_oft.py
git commit -m "feat(metrics/oft): 补 compute_center_time + compute_center_distance

同事 5-13 review feedback Q6 要求 OFT 报告呈现 center_time 和
center_distance。本提交补齐对应的 metric 函数 + scripts CLI 包装。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: resolve 函数 + 错误体系

完成后 `from ethoinsight.catalog import resolve; resolve("epm", columns=[...], raw_files=[...])` 返回 `Plan`；非法输入抛 `ResolveError` 带结构化 code。

**Files:**
- Create: `packages/ethoinsight/ethoinsight/catalog/resolve.py`
- Modify: `packages/ethoinsight/ethoinsight/catalog/__init__.py`（export `resolve`、`ResolveError`）
- Modify: `packages/ethoinsight/tests/test_catalog.py`（追加 resolve 测试）

- [ ] **Step 1: 写失败测试 - happy path + 6 种错误**

追加到 `tests/test_catalog.py`：

```python
def _epm_columns_full() -> list[str]:
    """Returns column list that satisfies all EPM default metrics' requires_columns."""
    return [
        "time", "x_center", "y_center",
        "in_zone_open_arms_center_point",
        "in_zone_closed_arms_center_point",
        "distance_moved",
    ]


def test_resolve_epm_default_path():
    from ethoinsight.catalog import resolve

    plan = resolve(
        paradigm="epm",
        columns=_epm_columns_full(),
        raw_files=["/tmp/raw.txt"],
        workspace_dir="/tmp/workspace",
    )
    assert plan.paradigm == "epm"
    metric_ids = {m.id for m in plan.metrics}
    assert metric_ids == EPM_Q6_WHITELIST
    for m in plan.metrics:
        assert m.input == "/tmp/raw.txt"
        assert m.output.startswith("/tmp/workspace/")
        assert m.required is True
        assert m.reason == "paradigm.default"


def test_resolve_user_include_outside_catalog_raises():
    from ethoinsight.catalog import resolve, ResolveError

    with pytest.raises(ResolveError) as exc:
        resolve(
            paradigm="epm",
            columns=_epm_columns_full(),
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
            include=["nonexistent_metric_xyz"],
        )
    assert exc.value.code == "unknown_metric"


def test_resolve_user_exclude_marks_skipped():
    from ethoinsight.catalog import resolve

    plan = resolve(
        paradigm="epm",
        columns=_epm_columns_full(),
        raw_files=["/tmp/raw.txt"],
        workspace_dir="/tmp/workspace",
        exclude=["open_arm_time"],
    )
    metric_ids = {m.id for m in plan.metrics}
    assert "open_arm_time" not in metric_ids
    skipped_ids = {s.id for s in plan.skipped}
    assert "open_arm_time" in skipped_ids
    excluded = next(s for s in plan.skipped if s.id == "open_arm_time")
    assert excluded.reason == "user.exclude"


def test_resolve_missing_required_columns_raises():
    from ethoinsight.catalog import resolve, ResolveError

    # 故意缺 in_zone_open_arms_*
    cols = ["time", "x_center", "y_center", "in_zone_closed_arms_center_point"]
    with pytest.raises(ResolveError) as exc:
        resolve(
            paradigm="epm",
            columns=cols,
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
        )
    assert exc.value.code == "columns_missing"
    assert "in_zone_open_arms" in str(exc.value)


def test_resolve_unknown_paradigm_raises():
    from ethoinsight.catalog import resolve, ResolveError

    with pytest.raises(ResolveError) as exc:
        resolve(
            paradigm="totally_made_up_paradigm",
            columns=[],
            raw_files=["/tmp/raw.txt"],
            workspace_dir="/tmp/workspace",
        )
    assert exc.value.code == "unknown_paradigm"


def test_resolve_statistics_skipped_when_n_per_group_too_small():
    from ethoinsight.catalog import resolve

    plan = resolve(
        paradigm="epm",
        columns=_epm_columns_full(),
        raw_files=["/tmp/raw.txt"],
        workspace_dir="/tmp/workspace",
        n_per_group=1, n_groups=1,
    )
    assert plan.statistics is not None
    assert plan.statistics.skip_reason is not None
    assert "n_per_group" in plan.statistics.skip_reason or "n_groups" in plan.statistics.skip_reason


def test_resolve_plan_to_dict_serializable():
    """Plan 必须能 json.dumps（lead bash 之后写盘）"""
    from ethoinsight.catalog import resolve, plan_to_dict
    import json

    plan = resolve(
        paradigm="epm",
        columns=_epm_columns_full(),
        raw_files=["/tmp/raw.txt"],
        workspace_dir="/tmp/workspace",
    )
    d = plan_to_dict(plan)
    s = json.dumps(d, ensure_ascii=False, indent=2)
    assert '"paradigm": "epm"' in s
    assert "schema_version" in d
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v -k "resolve"`
Expected: FAIL（`ImportError: cannot import name 'resolve'`）

- [ ] **Step 3: 写 resolve.py**

`packages/ethoinsight/ethoinsight/catalog/resolve.py`：

```python
"""Catalog resolver: catalog + columns + 用户偏差 → Plan.

输入：
  paradigm  : 范式 key（来自 lead 的 Gate 1 识别）
  columns   : raw 数据真实列名（来自 dump_headers）
  raw_files : raw 文件绝对路径列表
  include   : 用户额外要的 metric ids
  exclude   : 用户排除的 metric ids
  n_per_group / n_groups : 用于 charts/statistics 的 when 条件评估

输出：Plan dataclass（含 metrics / statistics / charts / skipped / notes）
       由 plan_to_dict() 序列化为 metric_plan.json schema_version="1.0"
"""

from __future__ import annotations

import datetime as dt
import fnmatch
from pathlib import Path
from typing import Iterable

from ethoinsight.catalog.loader import CatalogError, load_catalog
from ethoinsight.catalog.schema import (
    Catalog,
    ChartEntry,
    MetricEntry,
    Plan,
    PlanChart,
    PlanInputs,
    PlanMetric,
    PlanSkipped,
    PlanStatistics,
    StatisticsEntry,
)

SCHEMA_VERSION = "1.0"


class ResolveError(Exception):
    """Resolver 失败，含结构化 code 供 lead 反问。

    code 枚举:
      unknown_paradigm  - paradigm 不在 catalog 目录
      unknown_metric    - include 里的 id 不在 catalog
      columns_missing   - default 或 user-include 指标缺必需列
      empty_plan        - 所有 default + include 都被剪光
      schema_violation  - catalog YAML 损坏（fallback 自 CatalogError）
    """

    def __init__(self, code: str, message: str, details: dict | None = None) -> None:
        self.code = code
        self.details = details or {}
        super().__init__(message)


def resolve(
    paradigm: str,
    columns: list[str],
    raw_files: list[str],
    workspace_dir: str,
    *,
    include: list[str] | tuple[str, ...] = (),
    exclude: list[str] | tuple[str, ...] = (),
    n_per_group: int | None = None,
    n_groups: int | None = None,
    groups_file: str | None = None,
    columns_file: str | None = None,
    ev19_template: str | None = None,
) -> Plan:
    """生成 Plan dataclass。失败抛 ResolveError。"""
    try:
        cat = load_catalog(paradigm)
    except CatalogError as e:
        # 区分"paradigm 不存在"和"catalog 损坏"
        if "file not found" in str(e).lower() or "not found for paradigm" in str(e).lower():
            raise ResolveError(
                code="unknown_paradigm",
                message=f"Unknown paradigm '{paradigm}'.",
                details={"requested": paradigm},
            ) from e
        raise ResolveError(
            code="schema_violation",
            message=f"Catalog YAML for '{paradigm}' is malformed: {e}",
            details={"paradigm": paradigm},
        ) from e

    include_set = set(include or ())
    exclude_set = set(exclude or ())

    # 校验 include 全部在 catalog（default + optional）
    all_known_ids = {m.id for m in cat.default_metrics} | {m.id for m in cat.optional_metrics}
    unknown = include_set - all_known_ids
    if unknown:
        raise ResolveError(
            code="unknown_metric",
            message=f"Metric(s) not found in {paradigm} catalog: {sorted(unknown)}",
            details={"requested": sorted(unknown), "available": sorted(all_known_ids)},
        )

    plan_metrics: list[PlanMetric] = []
    skipped: list[PlanSkipped] = []

    # 步骤 1: 处理 default_metrics
    for m in cat.default_metrics:
        if m.id in exclude_set:
            skipped.append(PlanSkipped(
                id=m.id, reason="user.exclude",
                detail=f"User explicitly excluded {m.id}.",
            ))
            continue
        missing = _missing_columns(m.requires_columns, columns)
        if missing:
            raise ResolveError(
                code="columns_missing",
                message=(
                    f"Required column(s) for metric '{m.id}' not found in data: "
                    f"{missing}. Default metrics must always run; if the data "
                    f"truly lacks these columns, ask the user before excluding."
                ),
                details={
                    "metric": m.id, "missing_patterns": missing,
                    "available_columns": columns,
                },
            )
        plan_metrics.append(_metric_to_plan(m, raw_files, workspace_dir, required=True, reason="paradigm.default"))

    # 步骤 2: 处理 user-include（来自 default+optional 集）
    for inc_id in include_set:
        if inc_id in exclude_set:
            # 用户自相矛盾（同时 include + exclude），优先 exclude，加 note
            skipped.append(PlanSkipped(
                id=inc_id, reason="user.exclude",
                detail=f"User specified both include and exclude for {inc_id}; honoring exclude.",
            ))
            continue
        if inc_id in {m.id for m in cat.default_metrics}:
            continue  # default 已加，不重复
        m = next(m for m in cat.optional_metrics if m.id == inc_id)
        missing = _missing_columns(m.requires_columns, columns)
        if missing:
            skipped.append(PlanSkipped(
                id=m.id, reason="columns.missing",
                detail=f"User-included metric {m.id} skipped: missing columns {missing}.",
            ))
            continue
        plan_metrics.append(_metric_to_plan(m, raw_files, workspace_dir, required=False, reason="user.include"))

    # 步骤 3: 处理 optional_metrics（用户没显式提，但 catalog 标 applicable）
    # 当前版本不自动启用 optional；只在用户 include 时入；以后可加 auto 规则。

    if not plan_metrics:
        raise ResolveError(
            code="empty_plan",
            message=f"All metrics for paradigm '{paradigm}' were excluded or unavailable.",
            details={"paradigm": paradigm},
        )

    # 步骤 4: charts
    plan_charts: list[PlanChart] = []
    for ch in cat.charts:
        if _evaluate_when(ch.when, n_per_group=n_per_group, n_groups=n_groups):
            plan_charts.append(_chart_to_plan(ch, raw_files, workspace_dir))

    # 步骤 5: statistics
    plan_stats: PlanStatistics | None = None
    if cat.statistics_default is not None:
        if _evaluate_when(cat.statistics_default.when, n_per_group=n_per_group, n_groups=n_groups):
            plan_stats = _stats_to_plan(cat.statistics_default, workspace_dir, skip_reason=None)
        else:
            plan_stats = _stats_to_plan(
                cat.statistics_default, workspace_dir,
                skip_reason=f"condition '{cat.statistics_default.when}' not met (n_per_group={n_per_group}, n_groups={n_groups})",
            )

    notes: list[str] = []
    if include_set:
        notes.append(f"User include: {sorted(include_set)}")
    if exclude_set:
        notes.append(f"User exclude: {sorted(exclude_set)}")
    notes.append(f"Data columns: {len(columns)}; metrics planned: {len(plan_metrics)}; skipped: {len(skipped)}")

    return Plan(
        schema_version=SCHEMA_VERSION,
        paradigm=cat.paradigm,
        ev19_template=ev19_template,
        generated_at=_utcnow_iso(),
        inputs=PlanInputs(raw_files=list(raw_files), groups_file=groups_file, columns_file=columns_file),
        metrics=plan_metrics,
        statistics=plan_stats,
        charts=plan_charts,
        skipped=skipped,
        notes=notes,
    )


# ============================================================================
# Helpers
# ============================================================================


def _missing_columns(patterns: list[str], available: list[str]) -> list[str]:
    """对 requires_columns 中的每个 glob pattern，检查是否 ≥1 列匹配；返回未匹配的 pattern 集。"""
    missing: list[str] = []
    for pat in patterns:
        if not any(fnmatch.fnmatchcase(col, pat) for col in available):
            missing.append(pat)
    return missing


def _metric_to_plan(m: MetricEntry, raw_files: list[str], workspace_dir: str, *, required: bool, reason: str) -> PlanMetric:
    # 当前简化：每个 metric 用 raw_files[0]（单文件输入；shoaling 多文件后续扩展）
    input_path = raw_files[0]
    output_path = str(Path(workspace_dir) / f"m_{m.id}.json")
    return PlanMetric(
        id=m.id, script=m.script,
        input=input_path, output=output_path,
        required=required, reason=reason,
    )


def _chart_to_plan(ch: ChartEntry, raw_files: list[str], workspace_dir: str) -> PlanChart:
    return PlanChart(
        id=ch.id, script=ch.script,
        input=raw_files[0],
        output=str(Path(workspace_dir) / f"plot_{ch.id}.png"),
    )


def _stats_to_plan(st: StatisticsEntry, workspace_dir: str, skip_reason: str | None) -> PlanStatistics:
    return PlanStatistics(
        id=st.id, script=st.script,
        input=str(Path(workspace_dir) / "handoff_code_executor.json"),
        output=str(Path(workspace_dir) / "stats.json"),
        skip_reason=skip_reason,
    )


def _evaluate_when(condition: str, *, n_per_group: int | None, n_groups: int | None) -> bool:
    """评估 chart/statistics 的 when 条件字符串。

    支持的语法：
      "always"
      "n_per_group >= K"     K 是正整数
      "n_groups >= K"
      "n_per_group >= K and n_groups >= K"
    其他形式直接拒绝（保守）。
    """
    cond = condition.strip()
    if cond == "always":
        return True

    parts = [p.strip() for p in cond.split(" and ")]
    for part in parts:
        if not _evaluate_atomic_when(part, n_per_group=n_per_group, n_groups=n_groups):
            return False
    return True


def _evaluate_atomic_when(part: str, *, n_per_group: int | None, n_groups: int | None) -> bool:
    # 形如 "n_per_group >= 3"
    tokens = part.split()
    if len(tokens) != 3:
        return False
    var, op, val_str = tokens
    if op != ">=":
        return False
    try:
        val = int(val_str)
    except ValueError:
        return False
    if var == "n_per_group":
        return n_per_group is not None and n_per_group >= val
    if var == "n_groups":
        return n_groups is not None and n_groups >= val
    return False


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ============================================================================
# Serialization
# ============================================================================


def plan_to_dict(plan: Plan) -> dict:
    """Plan dataclass → JSON-serializable dict."""
    return {
        "schema_version": plan.schema_version,
        "paradigm": plan.paradigm,
        "ev19_template": plan.ev19_template,
        "generated_at": plan.generated_at,
        "inputs": {
            "raw_files": plan.inputs.raw_files,
            "groups_file": plan.inputs.groups_file,
            "columns_file": plan.inputs.columns_file,
        },
        "metrics": [
            {
                "id": m.id, "script": m.script,
                "input": m.input, "output": m.output,
                "required": m.required, "reason": m.reason,
            } for m in plan.metrics
        ],
        "statistics": (
            None if plan.statistics is None else {
                "id": plan.statistics.id, "script": plan.statistics.script,
                "input": plan.statistics.input, "output": plan.statistics.output,
                "skip_reason": plan.statistics.skip_reason,
            }
        ),
        "charts": [
            {"id": c.id, "script": c.script, "input": c.input, "output": c.output}
            for c in plan.charts
        ],
        "skipped": [
            {"id": s.id, "reason": s.reason, "detail": s.detail}
            for s in plan.skipped
        ],
        "notes": plan.notes,
    }
```

- [ ] **Step 4: 导出 resolve / ResolveError / plan_to_dict**

修改 `packages/ethoinsight/ethoinsight/catalog/__init__.py`：

```python
"""ethoinsight.catalog — 范式 → 指标 catalog 模块.

承载 single source of truth：每个 paradigm 一份 YAML 文件...
"""

from __future__ import annotations

from ethoinsight.catalog.loader import CatalogError, load_catalog
from ethoinsight.catalog.resolve import ResolveError, plan_to_dict, resolve
from ethoinsight.catalog.schema import (
    Catalog,
    ChartEntry,
    MetricEntry,
    Plan,
    PlanChart,
    PlanInputs,
    PlanMetric,
    PlanSkipped,
    PlanStatistics,
    StatisticsEntry,
)

__all__ = [
    "Catalog",
    "CatalogError",
    "ChartEntry",
    "MetricEntry",
    "Plan",
    "PlanChart",
    "PlanInputs",
    "PlanMetric",
    "PlanSkipped",
    "PlanStatistics",
    "ResolveError",
    "StatisticsEntry",
    "load_catalog",
    "plan_to_dict",
    "resolve",
]
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v -k "resolve"`
Expected: PASS（7 个测试）

- [ ] **Step 6: Commit**

```bash
git add packages/ethoinsight/ethoinsight/catalog/__init__.py \
        packages/ethoinsight/ethoinsight/catalog/resolve.py \
        packages/ethoinsight/tests/test_catalog.py
git commit -m "feat(catalog): resolve 函数 + 结构化错误体系

resolve(paradigm, columns, raw_files, workspace_dir, ...) 把 catalog
+ 列名 + 用户偏差 翻译为 Plan dataclass，失败抛 ResolveError 含
code 枚举（unknown_paradigm / unknown_metric / columns_missing /
empty_plan / schema_violation）。Plan 由 plan_to_dict 序列化为
metric_plan.json schema_version='1.0'。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: catalog CLI（python -m ethoinsight.catalog.resolve）

让 agent 能从 bash 触发 resolve。完成后 `python -m ethoinsight.catalog.resolve --paradigm epm ...` 写 plan.json；错误时 exit 1 + stderr 写结构化 JSON。

**Files:**
- Create: `packages/ethoinsight/ethoinsight/catalog/cli.py`
- Create: `packages/ethoinsight/ethoinsight/catalog/__main__.py`
- Modify: `packages/ethoinsight/tests/test_catalog.py`（追加 CLI 测试）

- [ ] **Step 1: 写失败测试 - CLI happy path + 错误退出**

追加到 `tests/test_catalog.py`：

```python
import json
import subprocess
import sys


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    """Run `python -m ethoinsight.catalog.resolve <args>` and capture."""
    proc = subprocess.run(
        [sys.executable, "-m", "ethoinsight.catalog.resolve", *args],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_happy_path(tmp_path):
    columns_file = tmp_path / "columns.json"
    columns_file.write_text(json.dumps({
        "file": "raw.txt",
        "columns": _epm_columns_full(),
        "n_subjects": 1,
        "duration_s": 300.0,
    }), encoding="utf-8")

    raw_files_json = tmp_path / "raw_files.json"
    raw_files_json.write_text(json.dumps(["/tmp/raw.txt"]), encoding="utf-8")

    output = tmp_path / "plan.json"

    rc, stdout, stderr = _run_cli([
        "--paradigm", "epm",
        "--columns-file", str(columns_file),
        "--raw-files-json", str(raw_files_json),
        "--workspace-dir", str(tmp_path),
        "--output", str(output),
    ])
    assert rc == 0, f"stderr: {stderr}"
    plan = json.loads(output.read_text())
    assert plan["paradigm"] == "epm"
    assert plan["schema_version"] == "1.0"
    assert len(plan["metrics"]) == 5  # EPM Q6 白名单


def test_cli_unknown_paradigm_exit1_with_json(tmp_path):
    columns_file = tmp_path / "columns.json"
    columns_file.write_text(json.dumps({"columns": []}), encoding="utf-8")
    raw_files_json = tmp_path / "raw_files.json"
    raw_files_json.write_text(json.dumps(["/tmp/raw.txt"]), encoding="utf-8")

    rc, stdout, stderr = _run_cli([
        "--paradigm", "totally_made_up",
        "--columns-file", str(columns_file),
        "--raw-files-json", str(raw_files_json),
        "--workspace-dir", str(tmp_path),
        "--output", str(tmp_path / "plan.json"),
    ])
    assert rc == 1
    err = json.loads(stderr.strip().splitlines()[-1])  # stderr 最后一行是结构化 JSON
    assert err["code"] == "unknown_paradigm"


def test_cli_user_include_unknown_exit1(tmp_path):
    columns_file = tmp_path / "columns.json"
    columns_file.write_text(json.dumps({"columns": _epm_columns_full()}), encoding="utf-8")
    raw_files_json = tmp_path / "raw_files.json"
    raw_files_json.write_text(json.dumps(["/tmp/raw.txt"]), encoding="utf-8")

    rc, _, stderr = _run_cli([
        "--paradigm", "epm",
        "--columns-file", str(columns_file),
        "--raw-files-json", str(raw_files_json),
        "--workspace-dir", str(tmp_path),
        "--output", str(tmp_path / "plan.json"),
        "--include", "nonexistent_metric_xyz",
    ])
    assert rc == 1
    err = json.loads(stderr.strip().splitlines()[-1])
    assert err["code"] == "unknown_metric"
    assert "nonexistent_metric_xyz" in err["details"]["requested"]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v -k "cli_"`
Expected: FAIL（`No module named 'ethoinsight.catalog.__main__'` 或 nonzero return code 解析失败）

- [ ] **Step 3: 写 cli.py**

`packages/ethoinsight/ethoinsight/catalog/cli.py`：

```python
"""Catalog CLI entry — python -m ethoinsight.catalog.resolve.

参数:
  --paradigm PARADIGM            必填
  --columns-file PATH            必填（dump_headers 产物）
  --raw-files-json PATH          必填（指向 JSON 数组）
  --groups-file PATH             可选
  --workspace-dir PATH           必填
  --include METRIC_ID            可重复
  --exclude METRIC_ID            可重复
  --n-per-group INT              可选
  --n-groups INT                 可选
  --ev19-template TEMPLATE_ID    可选；从 lead 透传作为 plan.ev19_template 字段
  --output PATH                  必填，写 plan.json 路径

行为:
  成功: exit 0 + 写 plan.json
  失败: exit 1 + stderr 最后一行写 {"code": "...", "message": "...", "details": {...}}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ethoinsight.catalog.resolve import ResolveError, plan_to_dict, resolve


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ethoinsight.catalog.resolve")
    p.add_argument("--paradigm", required=True)
    p.add_argument("--columns-file", required=True)
    p.add_argument("--raw-files-json", required=True)
    p.add_argument("--workspace-dir", required=True)
    p.add_argument("--groups-file", default=None)
    p.add_argument("--include", action="append", default=[])
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--n-per-group", type=int, default=None)
    p.add_argument("--n-groups", type=int, default=None)
    p.add_argument("--ev19-template", default=None)
    p.add_argument("--output", required=True)
    return p


def _emit_error(code: str, message: str, details: dict | None = None) -> int:
    payload = {"code": code, "message": message, "details": details or {}}
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # 读 columns.json
    try:
        columns_data = json.loads(Path(args.columns_file).read_text(encoding="utf-8"))
        columns = columns_data.get("columns", [])
        if not isinstance(columns, list):
            return _emit_error(
                "schema_violation",
                f"columns-file does not contain a 'columns' list: {args.columns_file}",
                {"path": args.columns_file},
            )
    except (OSError, json.JSONDecodeError) as e:
        return _emit_error(
            "schema_violation",
            f"Cannot read columns-file: {e}",
            {"path": args.columns_file},
        )

    # 读 raw-files-json
    try:
        raw_files = json.loads(Path(args.raw_files_json).read_text(encoding="utf-8"))
        if not isinstance(raw_files, list) or not all(isinstance(p, str) for p in raw_files):
            return _emit_error(
                "schema_violation",
                f"raw-files-json must be a JSON array of strings: {args.raw_files_json}",
                {"path": args.raw_files_json},
            )
    except (OSError, json.JSONDecodeError) as e:
        return _emit_error(
            "schema_violation",
            f"Cannot read raw-files-json: {e}",
            {"path": args.raw_files_json},
        )

    try:
        plan = resolve(
            paradigm=args.paradigm,
            columns=columns,
            raw_files=raw_files,
            workspace_dir=args.workspace_dir,
            include=args.include,
            exclude=args.exclude,
            n_per_group=args.n_per_group,
            n_groups=args.n_groups,
            groups_file=args.groups_file,
            columns_file=args.columns_file,
            ev19_template=args.ev19_template,
        )
    except ResolveError as e:
        return _emit_error(e.code, str(e), e.details)
    except Exception as e:  # noqa: BLE001
        return _emit_error("schema_violation", f"Unexpected resolver failure: {e}", {})

    # 写 plan.json
    plan_dict = plan_to_dict(plan)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    # 成功也在 stdout 写一行人类可读摘要供 lead 在派遣 prompt 中引用
    summary = (
        f"Plan written to {args.output}: paradigm={plan.paradigm}, "
        f"metrics={len(plan.metrics)}, charts={len(plan.charts)}, "
        f"skipped={len(plan.skipped)}, statistics="
        f"{'skip' if (plan.statistics and plan.statistics.skip_reason) else 'run' if plan.statistics else 'none'}"
    )
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: 写 __main__.py**

`packages/ethoinsight/ethoinsight/catalog/__main__.py`：

```python
"""Make `python -m ethoinsight.catalog.resolve` work via package __main__.

NOTE: Python's -m loader runs <pkg>/__main__.py when called as `python -m <pkg>`.
We expose `resolve` as a submodule under catalog/ so that
`python -m ethoinsight.catalog.resolve` works (handled by resolve.py's
own __main__ block at module load).
"""

from ethoinsight.catalog.cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

**Note**：实际命令是 `python -m ethoinsight.catalog.resolve` —— 这要求 resolve.py 本身可被作为 `-m` 入口。所以在 resolve.py 末尾加：

修改 `packages/ethoinsight/ethoinsight/catalog/resolve.py` 末尾追加：

```python


if __name__ == "__main__":
    import sys
    from ethoinsight.catalog.cli import main
    sys.exit(main())
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v -k "cli_"`
Expected: PASS（3 个测试）

- [ ] **Step 6: 跑完整 catalog 测试集**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v`
Expected: 全部 PASS（之前累计的所有测试 + 本任务的 3 个）

- [ ] **Step 7: 加一个反退化测试 - catalog 中每个 script 可 import**

追加到 `tests/test_catalog.py`：

```python
@pytest.mark.parametrize("paradigm", ["epm", "oft", "fst"])
def test_all_catalog_scripts_are_importable(paradigm):
    """catalog 里声明的 script dotted path 必须真的能 import 到一个有 main() 的模块。

    防御现状：YAML 改了引用脚本但脚本没建（或反之）→ 测试 fail。
    """
    import importlib
    from ethoinsight.catalog import load_catalog

    cat = load_catalog(paradigm)
    scripts = (
        [m.script for m in cat.default_metrics]
        + [m.script for m in cat.optional_metrics]
        + [c.script for c in cat.charts]
        + ([cat.statistics_default.script] if cat.statistics_default else [])
    )
    for dotted in scripts:
        try:
            mod = importlib.import_module(dotted)
        except ImportError as e:
            pytest.fail(f"Catalog references non-importable script '{dotted}': {e}")
        assert hasattr(mod, "main"), f"Script {dotted} has no main() entry"
        assert callable(mod.main), f"Script {dotted}.main is not callable"
```

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v -k "scripts_are_importable"`
Expected: PASS（3 个范式 × 1 个测试 = 3 个 case）。如果 fail，对照报错补齐缺失脚本（Task 4 已补 OFT 两个；其他 fail 项需立即补脚本壳子）。

- [ ] **Step 8: Commit**

```bash
git add packages/ethoinsight/ethoinsight/catalog/cli.py \
        packages/ethoinsight/ethoinsight/catalog/__main__.py \
        packages/ethoinsight/ethoinsight/catalog/resolve.py \
        packages/ethoinsight/tests/test_catalog.py
git commit -m "feat(catalog): CLI 入口 python -m ethoinsight.catalog.resolve

成功 → exit 0 + 写 plan.json + stdout 一行摘要；失败 → exit 1 +
stderr JSON {code, message, details}。lead 通过 stderr 解析 code
决定 ask_clarification 话术。

追加反退化测试 test_all_catalog_scripts_are_importable，未来改 YAML
若引用不存在脚本会立刻 fail。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: dump_headers CLI

让 lead 能从 bash 提取 raw 数据列名清单写到 columns.json。完成后 `python -m ethoinsight.parse.dump_headers --input <raw.txt> --output <columns.json>` 工作。

**Files:**
- Refactor: `packages/ethoinsight/ethoinsight/parse.py` → `packages/ethoinsight/ethoinsight/parse/__init__.py`（保持公共 API 完全向后兼容）
- Create: `packages/ethoinsight/ethoinsight/parse/_core.py`（现 parse.py 内容）
- Create: `packages/ethoinsight/ethoinsight/parse/dump_headers.py`
- Create: `packages/ethoinsight/tests/test_dump_headers.py`

- [ ] **Step 1: 先确认 parse.py 当前公共 API**

Run: `cd packages/ethoinsight && python -c "
from ethoinsight import parse
print([n for n in dir(parse) if not n.startswith('_')])
"`

记录输出（典型应含 `detect_ethovision`、`parse_header`、`parse_trajectory`、`parse_batch` 等）。后续 __init__.py 必须 re-export 这些名字。

- [ ] **Step 2: 把 parse.py 移到 parse/_core.py（不改内容）**

```bash
cd packages/ethoinsight/ethoinsight && mkdir -p parse && mv parse.py parse/_core.py
```

- [ ] **Step 3: 写 parse/__init__.py（re-export 全部公共 API）**

`packages/ethoinsight/ethoinsight/parse/__init__.py`：

```python
"""ethoinsight.parse — EthoVision XT data file parser.

Backward compat shim: re-exports public API from parse._core so that
`from ethoinsight.parse import parse_header, ...` continues to work.

新增子模块:
  - dump_headers : CLI 工具，输出 raw 文件列名清单到 JSON
"""

from __future__ import annotations

from ethoinsight.parse._core import (  # noqa: F401
    detect_ethovision,
    parse_header,
    parse_trajectory,
    parse_batch,
)
```

**Note**：上面的 `from ... import` 列表必须跟 Step 1 输出的 public API 完全一致。如果 parse.py 还导出了其他名字（比如 `normalize_columns`、`_slugify` 公共别名），都要加进来。再跑一次 Step 1 那段命令、对比新旧输出（移动后再跑）。

- [ ] **Step 4: 跑现有测试确认向后兼容没破**

Run: `cd packages/ethoinsight && pytest tests/test_parse.py tests/test_metrics_epm.py tests/test_metrics_oft.py tests/test_metrics_fst.py -v`
Expected: 全部 PASS（如果有失败，多半是 __init__.py re-export 列表漏了名字）

- [ ] **Step 5: 写 dump_headers 测试**

`packages/ethoinsight/tests/test_dump_headers.py`：

```python
"""Tests for ethoinsight.parse.dump_headers CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _fixtures_dir() -> Path:
    # 复用现有 parse 测试 fixture
    return Path(__file__).parent / "fixtures"


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "ethoinsight.parse.dump_headers", *args],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_dump_headers_writes_expected_keys(tmp_path):
    """需要一份真 EthoVision 文件 fixture；如不存在，标 xfail 让 reviewer 决策。"""
    fixtures = _fixtures_dir()
    candidates = list(fixtures.glob("*.txt")) if fixtures.exists() else []
    if not candidates:
        pytest.skip("No EthoVision fixture file under tests/fixtures/")
    raw = candidates[0]

    out = tmp_path / "columns.json"
    rc, _, stderr = _run_cli(["--input", str(raw), "--output", str(out)])
    assert rc == 0, f"stderr: {stderr}"
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "file" in data
    assert "columns" in data
    assert isinstance(data["columns"], list)
    assert len(data["columns"]) > 0


def test_dump_headers_file_not_found(tmp_path):
    out = tmp_path / "columns.json"
    rc, _, stderr = _run_cli([
        "--input", "/nonexistent/path/raw.txt", "--output", str(out),
    ])
    assert rc == 1
    err = json.loads(stderr.strip().splitlines()[-1])
    assert err["code"] == "file.not_found"


def test_dump_headers_format_unrecognized(tmp_path):
    """非 EthoVision 文件应返回 format.unrecognized."""
    bad = tmp_path / "not_ethovision.txt"
    bad.write_text("this is not an EthoVision file", encoding="utf-8")
    out = tmp_path / "columns.json"
    rc, _, stderr = _run_cli(["--input", str(bad), "--output", str(out)])
    assert rc == 1
    err = json.loads(stderr.strip().splitlines()[-1])
    assert err["code"] == "format.unrecognized"
```

- [ ] **Step 6: 跑测试确认失败**

Run: `cd packages/ethoinsight && pytest tests/test_dump_headers.py -v`
Expected: FAIL（`No module named 'ethoinsight.parse.dump_headers'`）

- [ ] **Step 7: 写 dump_headers.py**

`packages/ethoinsight/ethoinsight/parse/dump_headers.py`：

```python
"""ethoinsight.parse.dump_headers — CLI: 提取 raw 数据列名清单到 JSON.

CLI: python -m ethoinsight.parse.dump_headers \\
       --input <raw_file.txt> --output <columns.json>

成功: exit 0, 写 JSON {file, columns, n_subjects?, duration_s?}
失败: exit 1, stderr 最后一行写 {"code": "...", "message": "...", "details": {...}}
  code 枚举: file.not_found / header.parse_failed / format.unrecognized
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ethoinsight.parse._core import detect_ethovision, parse_header, parse_trajectory


def _emit_error(code: str, message: str, details: dict | None = None) -> int:
    payload = {"code": code, "message": message, "details": details or {}}
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
    return 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m ethoinsight.parse.dump_headers")
    p.add_argument("--input", required=True, help="raw EthoVision trajectory file")
    p.add_argument("--output", required=True, help="path to write columns.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    raw_path = Path(args.input)

    if not raw_path.is_file():
        return _emit_error("file.not_found", f"Input not found: {raw_path}", {"path": str(raw_path)})

    if not detect_ethovision(str(raw_path)):
        return _emit_error(
            "format.unrecognized",
            f"File does not look like an EthoVision export: {raw_path}",
            {"path": str(raw_path)},
        )

    try:
        header = parse_header(str(raw_path))
    except Exception as e:  # noqa: BLE001
        return _emit_error(
            "header.parse_failed",
            f"parse_header failed: {e}",
            {"path": str(raw_path)},
        )

    # 取列名：通过 parse_trajectory 拿 dataframe 列名（normalize 后）
    try:
        df = parse_trajectory(str(raw_path))
        columns = list(df.columns)
    except Exception as e:  # noqa: BLE001
        return _emit_error(
            "header.parse_failed",
            f"parse_trajectory failed: {e}",
            {"path": str(raw_path)},
        )

    payload = {
        "file": str(raw_path),
        "columns": columns,
        "n_subjects": header.get("number_of_subjects"),
        "duration_s": header.get("trial_duration"),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(columns)} column names to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8: 跑测试确认通过**

Run: `cd packages/ethoinsight && pytest tests/test_dump_headers.py -v`
Expected: 2 个 PASS（file_not_found + format_unrecognized）；happy path 可能 SKIP（取决于是否有真 fixture）。所有失败都需修。

- [ ] **Step 9: Commit**

```bash
git add packages/ethoinsight/ethoinsight/parse/ \
        packages/ethoinsight/tests/test_dump_headers.py
git rm packages/ethoinsight/ethoinsight/parse.py 2>/dev/null || true
git commit -m "feat(parse): 添加 dump_headers CLI；parse.py → parse/ 包

parse.py 重构为 parse/ 包，原内容移至 parse/_core.py；__init__.py
re-export 全部公共 API（向后兼容、所有现有测试不受影响）。

新增 parse/dump_headers.py 提供 'python -m ethoinsight.parse.dump_headers
--input <raw> --output <columns.json>' CLI，给 lead 在派遣 code-executor
之前做列名预检用。错误码：file.not_found / header.parse_failed /
format.unrecognized。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: 新建 ethoinsight-metric-catalog skill

完成后 skill 文件可被 deerflow 加载、内容按 role 分段告诉各 agent 怎么读 catalog。

**Files:**
- Create: `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md`
- Create: `packages/agent/skills/custom/ethoinsight-metric-catalog/references/resolve-cli.md`
- Create: `packages/agent/skills/custom/ethoinsight-metric-catalog/references/field-guide.md`
- Modify: `packages/agent/extensions_config.json`（启用新 skill）

- [ ] **Step 1: 看现有 skill 的 frontmatter 写法**

Run: `head -15 packages/agent/skills/custom/ethoinsight-code/SKILL.md`

参照它的 frontmatter 格式（name / description / type / version / author）。

- [ ] **Step 2: 写 SKILL.md**

`packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md`：

```markdown
---
name: ethoinsight-metric-catalog
description: >
  EthoInsight 范式指标 catalog 读取手册。lead 用 resolve CLI 生成 plan；
  data-analyst 取 direction_for_anxiety / statistical_default 做判读；
  report-writer 取 display_name_zh / unit_zh / one_liner 翻译展示。
  catalog 是 single source of truth：范式→指标清单 + 展示元数据 + 判读
  方向性集中在 packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml。
type: knowledge
version: 0.1.0
author: noldus-insight
---

# EthoInsight Metric Catalog 入口

## catalog 物理位置

```bash
python -c "import ethoinsight.catalog as c; print(c.__file__)"
# 输出: .../ethoinsight/catalog/__init__.py
# YAML 在同目录: epm.yaml / oft.yaml / fst.yaml / tst.yaml / ldb.yaml / zero_maze.yaml / shoaling.yaml
```

## 你是哪种 role？

### lead

在派遣 code-executor **之前** 走以下两步 bash：

**Step 1: 列名预检**

```bash
python -m ethoinsight.parse.dump_headers \
    --input /mnt/user-data/uploads/<raw_file>.txt \
    --output /mnt/user-data/workspace/columns.json
```

失败时 stderr 最后一行是 JSON，含 `code` 字段：

| code | 含义 | 怎么反问 |
|------|------|----------|
| `file.not_found` | 文件路径错 | 让用户重新提供路径或确认上传成功 |
| `format.unrecognized` | 文件不是 EthoVision 导出 | "这个文件看起来不像 EthoVision XT 导出，能确认下吗？" |
| `header.parse_failed` | header 损坏 | 让用户检查文件完整性 |

**Step 2: 生成执行计划**

```bash
python -m ethoinsight.catalog.resolve \
    --paradigm <epm|oft|fst|...> \
    --columns-file /mnt/user-data/workspace/columns.json \
    --raw-files-json /mnt/user-data/workspace/raw_files.json \
    --workspace-dir /mnt/user-data/workspace \
    --groups-file /mnt/user-data/workspace/groups.json \
    --output /mnt/user-data/workspace/metric_plan.json \
    [--include METRIC_ID]* \
    [--exclude METRIC_ID]* \
    [--n-per-group N] \
    [--n-groups N] \
    [--ev19-template TEMPLATE_ID]
```

完整参数文档见 [`references/resolve-cli.md`](references/resolve-cli.md)。

失败时按 stderr JSON 的 `code` 字段反问用户：

| code | 反问话术（中文） |
|------|------------------|
| `unknown_paradigm` | （内部错误，不应发生 —— Gate 1 已识别） |
| `unknown_metric` | "您要的指标 `<id>` 我们目前没有预制脚本，要不咱们换 `<details.available>` 中的一个？" |
| `columns_missing` | "您的数据里缺 `<details.missing_patterns>` 这些列，没法跑 `<details.metric>` 指标。能确认下数据是否完整、或者咱们跳过这个指标？" |
| `empty_plan` | "按这个组合一项指标都跑不了，咱们是不是哪里搞错了？" |
| `schema_violation` | （内部错误，向用户致歉、把 details 报给开发者） |

**Step 3: 派遣 code-executor**

派遣 prompt 中只需要：

```
范式：{paradigm}
plan 路径：/mnt/user-data/workspace/metric_plan.json
分组：/mnt/user-data/workspace/groups.json

请按 plan.metrics[]、plan.statistics、plan.charts[] 逐条 bash 执行
对应 script，聚合写 handoff_code_executor.json，输出 [gate_signals]。
```

**不要把指标清单展开写进派遣 prompt** —— plan.json 已经在那里，code-executor 自己 read。

### code-executor

你不读 catalog YAML。你的所有指令都在 `plan.json` 里：

1. `read_file /mnt/user-data/workspace/metric_plan.json` 解析结构
2. for entry in `plan.metrics`: bash `python -m <entry.script> --input <entry.input> --output <entry.output>`
3. if `plan.statistics.skip_reason` is null: bash `python -m <plan.statistics.script> --inputs ... --groups ... --output ...`
4. for chart in `plan.charts`: bash `python -m <chart.script> --input ... --output ...`
5. 聚合所有 outputs → write `handoff_code_executor.json`
6. 输出 `[gate_signals]` 块

详见 `ethoinsight-code` skill 的 workflow 段。

### data-analyst

判读 handoff 时按 metric id 查 catalog 字段：

```bash
# 取整个 paradigm 的指标元数据（最简单粗暴）
cat $(python -c "import ethoinsight.catalog as c, os; print(os.path.dirname(c.__file__))")/{paradigm}.yaml
```

或用 Python 一次性加载：

```bash
python -c "
from ethoinsight.catalog import load_catalog
import json
cat = load_catalog('{paradigm}')
for m in cat.default_metrics:
    print(json.dumps({
        'id': m.id,
        'direction_for_anxiety': m.direction_for_anxiety,
        'statistical_default': m.statistical_default,
    }, ensure_ascii=False))
"
```

关注字段：
- `direction_for_anxiety`：判断"显著差异是否符合实验假设"。值：`lower_is_anxious` / `higher_is_anxious` / `null`（方向无关）
- `statistical_default`：验证 code-executor 用了正确的统计入口

详见 [`references/field-guide.md`](references/field-guide.md)。

### report-writer

写"Results / Discussion"段时按 metric id 翻译展示元数据：

```bash
python -c "
from ethoinsight.catalog import load_catalog
cat = load_catalog('{paradigm}')
for m in cat.default_metrics:
    print(f'{m.id} → {m.display_name_zh}（{m.unit_zh}）：{m.one_liner}')
"
```

关注字段：
- `display_name_zh`：中文展示名
- `unit_zh`：单位
- `one_liner`：一句话领域解释（仅首次提及该指标时引用，不在每段都重复）

**禁止**自行造一份指标中文名 / 单位映射，全部走 catalog。

详见 [`references/field-guide.md`](references/field-guide.md)。

## 不在本 skill 范围

- 范式识别 / EV19 模板判别：见 `ethovision-paradigm-knowledge` skill
- 指标算法实现：见 `packages/ethoinsight/ethoinsight/metrics/<paradigm>.py` Python docstring
- 范式分析模板（生命周期、组合规则）：catalog 本身就是这份知识的承载，不再有独立的 markdown
```

- [ ] **Step 3: 写 references/resolve-cli.md**

`packages/agent/skills/custom/ethoinsight-metric-catalog/references/resolve-cli.md`：

```markdown
# resolve CLI 完整参数参考

```
python -m ethoinsight.catalog.resolve \\
    --paradigm PARADIGM \\
    --columns-file PATH \\
    --raw-files-json PATH \\
    --workspace-dir PATH \\
    --output PATH \\
    [--groups-file PATH] \\
    [--include METRIC_ID]* \\
    [--exclude METRIC_ID]* \\
    [--n-per-group INT] \\
    [--n-groups INT] \\
    [--ev19-template TEMPLATE_ID]
```

## 必填参数

| 参数 | 含义 | 来源 |
|------|------|------|
| `--paradigm` | 范式 key（epm/oft/fst/tst/ldb/zero_maze/shoaling） | lead Gate 1 识别 |
| `--columns-file` | dump_headers 产物 columns.json 路径 | 前一步 bash 写 |
| `--raw-files-json` | JSON 数组文件，元素是 raw 文件绝对路径。单文件场景也用单元素数组。 | lead 现 write_file 一份 |
| `--workspace-dir` | metrics[].output / charts[].output 等路径的 base | 通常是 /mnt/user-data/workspace |
| `--output` | plan.json 写盘路径 | 通常是 /mnt/user-data/workspace/metric_plan.json |

## 可选参数

| 参数 | 含义 | 何时用 |
|------|------|--------|
| `--groups-file` | groups.json 路径（如 {"control": ["s1","s2"], "drug": ["s3","s4"]}） | 分组场景；不分组省略 |
| `--include METRIC_ID` | 用户额外要的指标 id（可重复） | 用户在 Gate 2 提出定制需求 |
| `--exclude METRIC_ID` | 用户排除的指标 id（可重复） | 同上 |
| `--n-per-group` | 每组样本数（charts/statistics when 条件评估） | lead 从分组信息推导 |
| `--n-groups` | 分组数量 | 同上 |
| `--ev19-template` | EV19 模板 id 透传到 plan.ev19_template 字段 | lead Gate 1 识别后透传 |

## 退出码

| exit code | 含义 |
|-----------|------|
| 0 | 成功，plan.json 已写；stdout 输出一行人类可读摘要 |
| 1 | 失败，stderr **最后一行** 是结构化错误 JSON |

## 错误码完整清单

| code | message 模板 | details 字段 | lead 应反问什么 |
|------|---------------|---------------|------------------|
| `unknown_paradigm` | Unknown paradigm '...' | `{requested}` | 内部错误（Gate 1 已识别），转开发 |
| `unknown_metric` | Metric(s) not found in {paradigm} catalog: [...] | `{requested, available}` | "您要的 X 我们没有，可选 [available] 中的" |
| `columns_missing` | Required column(s) for metric '...' not found in data | `{metric, missing_patterns, available_columns}` | "数据里缺 [missing]，没法跑 [metric]，跳过还是确认数据？" |
| `empty_plan` | All metrics for paradigm '...' were excluded or unavailable | `{paradigm}` | "按这个组合啥都跑不了，是不是哪里搞错了？" |
| `schema_violation` | （多种触发：columns-file/raw-files-json 损坏或 catalog YAML 损坏） | `{path}` 或 `{paradigm}` | 内部错误，转开发 |
```

- [ ] **Step 4: 写 references/field-guide.md**

`packages/agent/skills/custom/ethoinsight-metric-catalog/references/field-guide.md`：

```markdown
# Catalog 字段按 role 分配

## 字段全集

每个 `default_metrics[i]` 含以下字段（YAML key）：

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | str | 指标唯一标识，handoff key |
| `script` | str | dotted import path（code-executor 用） |
| `requires_columns` | list[str (glob)] | 必需的数据列模式 |
| `output_unit` | str | 机器可读单位 (`ratio`/`seconds`/`count`/`cm`) |
| `display_name_zh` | str | 中文展示名 |
| `unit_zh` | str | 中文单位 |
| `one_liner` | str | 一句话领域解释 |
| `direction_for_anxiety` | enum/null | `lower_is_anxious` / `higher_is_anxious` / null |
| `statistical_default` | enum | `groupwise_compare` / `paired_compare` |

## 按 role 关注子集

| Role | 字段 | 用途 |
|------|------|------|
| **lead** | 不直接读，通过 resolve 函数消费 | 触发 resolve CLI |
| **code-executor** | `id`, `script`, `requires_columns`, `output_unit` | 仅 plan.json 里间接消费；不直接读 catalog |
| **data-analyst** | `id`, `direction_for_anxiety`, `statistical_default` | 判读差异方向、验证统计方法 |
| **report-writer** | `id`, `display_name_zh`, `unit_zh`, `one_liner`, `output_unit` | 翻译数字 → 中文展示 |

## 反 single-source 反例（**不要做**）

- ❌ 在 data-analyst 系统 prompt 里硬编码"EPM 的 open_arm_time_ratio 越低焦虑越高" → 同事改 catalog 后这里不会更新
- ❌ 在 report-writer 系统 prompt 里硬编码"open_arm_time_ratio 翻译成开放臂时间比例" → 同上
- ❌ 在 handoff_code_executor.json 内嵌 display_name_zh —— 让 catalog 是唯一源
- ✅ 任何与指标相关的展示 / 判读字段，**临用临读 catalog**
```

- [ ] **Step 5: 启用 skill**

修改 `packages/agent/extensions_config.json`，把 `ethoinsight-metric-catalog` 加入 skills 段并启用：

Run: `cat packages/agent/extensions_config.json | head -40`

按现有 skills 块的结构（如 `"skills": {"ethoinsight": {"enabled": true}, ...}`），追加：

```json
"ethoinsight-metric-catalog": { "enabled": true }
```

如果文件里没有 skills 段（旧格式），按现有 skill 注册风格补一份。**不允许把 noldus-kb 等其他 enabled:false 的项改为 true**（CLAUDE.md §第 2 条规定 noldus-kb 当前禁用）。

- [ ] **Step 6: 冒烟测试 skill 可被 deerflow 加载**

Run（从 deerflow backend 目录跑）：
```bash
cd packages/agent/backend && source .venv/bin/activate && python -c "
from deerflow.skills import load_skills
skills = load_skills()
names = [s.name for s in skills]
assert 'ethoinsight-metric-catalog' in names, f'skill not found in {names}'
print('OK')
"
```
Expected: `OK`。如果失败：检查 SKILL.md 的 frontmatter 是否合法（看 deerflow 的 skills/loader.py 报错）。

- [ ] **Step 7: Commit**

```bash
git add packages/agent/skills/custom/ethoinsight-metric-catalog/ \
        packages/agent/extensions_config.json
git commit -m "feat(skill): 新建 ethoinsight-metric-catalog skill

按 role（lead / data-analyst / report-writer）分段给出 catalog 读取
指引。lead 通过 resolve CLI 消费 catalog；data-analyst 取
direction_for_anxiety、statistical_default 判读差异方向；report-writer
取 display_name_zh / unit_zh / one_liner 翻译展示。catalog 是单一
事实源，禁止在任何 subagent prompt 里硬编码这些字段。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: agent 端 SubagentConfig 改造

把 catalog skill 挂到 lead / data-analyst / report-writer；改 code-executor 工作流 prompt 从"读 by-paradigm md"改为"读 plan.json"；不挂 metric-catalog 给 code-executor。

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- Modify: `packages/agent/backend/tests/test_ethoinsight_code_skill.py`（如有断言 skill 列表）

- [ ] **Step 1: 看现有断言哪些会因为 skills 列表改动而 fail**

Run: `grep -rn "skills=\|skills *=\|\"ethoinsight-code\"\|\"ethoinsight-charts\"" packages/agent/backend/tests/ | head -20`

记录所有需要同步改动的测试位置。

- [ ] **Step 2: 写 code_executor.py 的失败测试（workflow 段改写）**

`packages/agent/backend/tests/test_code_executor_workflow.py`（新建）：

```python
"""Verify code-executor workflow prompt references metric_plan.json, not by-paradigm md."""

from __future__ import annotations


def test_code_executor_workflow_reads_plan_json():
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
    sp = CODE_EXECUTOR_CONFIG.system_prompt

    # 必须出现的关键词
    assert "metric_plan.json" in sp, "workflow must reference plan.json"
    assert "plan.metrics" in sp or "metrics 数组" in sp, "must describe iterating metrics"

    # 必须不再出现的旧引用
    assert "by-paradigm" not in sp, "should no longer read by-paradigm md"
    assert "决策树" not in sp or "选脚本" not in sp, "decision tree responsibility moved to lead"


def test_code_executor_skills_list_unchanged():
    """code-executor 不挂 metric-catalog（保持执行纯净）。"""
    from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
    assert "ethoinsight-metric-catalog" not in (CODE_EXECUTOR_CONFIG.skills or [])
    assert "ethoinsight-code" in (CODE_EXECUTOR_CONFIG.skills or [])
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd packages/agent/backend && pytest tests/test_code_executor_workflow.py -v`
Expected: FAIL（旧 prompt 不含 "metric_plan.json"）

- [ ] **Step 4: 改 code_executor.py system_prompt 的 workflow 段**

打开 `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`，把 `<workflow>` 段从：

```
<workflow>
1. read `ethoinsight-code/references/by-paradigm/<paradigm>.md` — 看本范式可用脚本清单 + 实验设计决策树
2. 根据 lead 给的实验信息（范式、n、分组、用户特殊需求），决定要跑哪些脚本
3. （如需多文件聚合）write_file 生成 inputs.json 和 groups.json
4. for script in 选中列表:
     bash `python -m ethoinsight.scripts.<paradigm>.<script_name> --input ... --output ...`
5. 收集各脚本输出（JSON 文件 + stdout 的 [result] 行），构造 handoff JSON
6. write_file `${workspace_path}/handoff_code_executor.json`
</workflow>
```

改写为：

```
<workflow>
1. read `${workspace_path}/metric_plan.json` — 这是 lead 已经生成好的施工单，含 paradigm、metrics[]、statistics、charts[]、skipped[]
2. for entry in plan.metrics:
     bash `python -m <entry.script> --input <entry.input> --output <entry.output>`
   每个脚本 stdout 末尾会有 `[result] {json}` 行，抓出来留作聚合用。
3. if plan.statistics is not null and plan.statistics.skip_reason is null:
     bash `python -m <plan.statistics.script> --inputs ... --groups ... --output ...`
   注意：如果 plan.statistics.skip_reason 非空，跳过统计这一步（不报错）。
4. for chart in plan.charts:
     bash `python -m <chart.script> --input ... --output ...`
5. 聚合：把所有 metrics[].output 的 JSON 内容 + charts 路径 + statistics 输出（如有）合并构造 handoff_code_executor.json，schema 见 ethoinsight-code skill 的 templates/output-contract.md
6. write_file `${workspace_path}/handoff_code_executor.json`
</workflow>
```

并把 `<bash_constraints>` 段尾的"可用脚本清单见 by-paradigm/<范式>.md"改为"可用脚本由 lead 通过 metric_plan.json 提供，不需要你自己查"。

- [ ] **Step 5: 跑测试确认通过**

Run: `cd packages/agent/backend && pytest tests/test_code_executor_workflow.py -v`
Expected: PASS（2 个测试）

- [ ] **Step 6: 改 data_analyst.py 加 catalog skill**

打开 `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`，把 `skills=[...]` 改为：

```python
skills=["ethoinsight", "ethoinsight-metric-catalog"],
```

并在 system_prompt 适当位置（比如"工作方式"或"使用资源"段）追加一段：

```
## 指标元数据查询

判读某个指标时，按 metric id 查 catalog：

bash:
    python -c "from ethoinsight.catalog import load_catalog; c=load_catalog('<paradigm>'); [print(m.id, m.direction_for_anxiety, m.statistical_default) for m in c.default_metrics]"

关注字段：
- direction_for_anxiety: 判断"显著差异方向是否符合实验假设"。lower_is_anxious 意味着该指标越低焦虑样行为越明显
- statistical_default: 验证 code-executor 用了正确的统计入口

禁止在本 prompt 内硬编码任何指标的判读方向 —— 全部走 catalog。
```

**Note**：data-analyst 当前 `disallowed_tools` 含 `bash`（参见 data_analyst.py），所以上面的 bash 示例实际是 **read_file catalog 的 YAML 文件**。改示例代码为：

```
## 指标元数据查询

判读某个指标时，read catalog YAML：

read_file:
    /path/to/ethoinsight/catalog/<paradigm>.yaml

（catalog 物理路径由 lead 提供给你，或从 ethoinsight-metric-catalog skill 的 SKILL.md 顶部读取定位方法）

关注字段：
- direction_for_anxiety
- statistical_default
```

- [ ] **Step 7: 改 report_writer.py 加 catalog skill + prompt 指引**

打开 `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`，把 `skills` 改为：

```python
skills=["ethoinsight", "ethoinsight-metric-catalog"],
```

在 system prompt 适当位置追加：

```
## 指标展示元数据查询

写"Results / Discussion"段时，按 metric id read catalog YAML 取展示字段：

read_file:
    /path/to/ethoinsight/catalog/<paradigm>.yaml

按 metric id 查：
- display_name_zh: 中文展示名
- unit_zh: 单位
- one_liner: 一句话解释（仅首次提及该指标时引用，不要在每段重复）

禁止在本 prompt 内硬编码任何指标的中文名或单位 —— 全部走 catalog。
```

- [ ] **Step 8: 改 lead_agent/prompt.py 加 catalog skill description + Gate 2 工作流提示**

打开 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`。

**插入位置 A — skill 描述段**：grep 现有 skill 描述的锚点：

```bash
grep -n "ethoinsight-planning\|ethovision-paradigm-knowledge" \
  packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
```

在描述这两个 skill 的同一区块（多半是一个 markdown 列表，每个 skill 一行）内追加：

```
- **ethoinsight-metric-catalog**: 范式指标 catalog 读取手册。**在派遣 code-executor 之前**，按 SKILL.md 中 lead role 段的指引：(1) bash dump_headers 提取列名 (2) bash catalog.resolve 生成 metric_plan.json。失败时按 stderr JSON 的 code 字段 ask_clarification。
```

**插入位置 B — Gate 2 数据质量检查段**：grep 锚点：

```bash
grep -n "Gate 2\|派遣 code-executor\|code_executor" \
  packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py | head -10
```

在"派遣 code-executor"的指令前（或 Gate 2 通过后的工作流描述里）插入：

```
**派遣 code-executor 前必做**（详见 ethoinsight-metric-catalog skill）：

1. bash dump_headers 提取数据列名到 /mnt/user-data/workspace/columns.json
2. write_file /mnt/user-data/workspace/raw_files.json（JSON 数组含 raw 文件路径）
3. bash catalog.resolve 生成 /mnt/user-data/workspace/metric_plan.json
4. 派遣 prompt 仅需告诉 code-executor plan.json 路径，**不要展开指标清单**

resolve 失败时（stderr JSON 含 code 字段）按 skill 的话术映射反问用户。
```

若找不到明确的 Gate 2 锚点（prompt.py 结构因上游 sync 改变），追加到 prompt.py 末尾即可 —— deerflow 的 lead system_prompt 是按段拼接的、位置不是强校验。

**改完后 grep 验证插入成功**：

```bash
grep -n "metric-catalog\|metric_plan.json" \
  packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
```
Expected: 至少 4 处匹配（skill 描述 1 处 + Gate 2 工作流 3 处）


- [ ] **Step 9: 跑所有 subagent 相关测试**

Run: `cd packages/agent/backend && pytest tests/ -k "subagent or code_executor or data_analyst or report_writer or lead" -v`
Expected: 大部分 PASS；若有失败检查是否是 skill 列表硬编码断言（同步更新断言）。

- [ ] **Step 10: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py \
        packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py \
        packages/agent/backend/tests/test_code_executor_workflow.py
git commit -m "feat(agent): subagent skill 切分 + workflow 改造（catalog 架构）

- code-executor: workflow 改为 read metric_plan.json 逐条 bash 执行；
  skills 不变（不挂 catalog）
- data-analyst: skills 加 ethoinsight-metric-catalog；prompt 加'判读
  时 read catalog YAML 取 direction_for_anxiety / statistical_default'
- report-writer: skills 加 ethoinsight-metric-catalog；prompt 加'写
  展示时 read catalog YAML 取 display_name_zh / unit_zh / one_liner'
- lead_agent prompt: skill 描述段 + Gate 2 派遣前工作流提示

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: 现状清扫 - 删 by-paradigm md + 旧 assess_and_handoff 引用

完成后旧的范式 md 文档撤场，agent 不会再读到过时的"决策树 markdown"。

**Files:**
- Delete: `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/*.md`
- Modify: `packages/agent/skills/custom/ethoinsight-code/SKILL.md`（workflow 段重写）
- Modify: `packages/agent/skills/custom/ethoinsight-code/references/error-recovery.md`（删第 46-51 行）
- Modify: `packages/agent/skills/custom/ethoinsight-code/references/quality-checks.md`（删第 44 行）

- [ ] **Step 1: 看 ethoinsight-code SKILL.md 当前 workflow 段**

Run: `cat packages/agent/skills/custom/ethoinsight-code/SKILL.md`

- [ ] **Step 2: 改 SKILL.md workflow 段为 read plan.json 模式**

把"工作流（脚本即指标架构）"段重写为：

```markdown
## 工作流（catalog → plan → execute 架构）

code-executor 的工作流程：

1. **read** `${workspace_path}/metric_plan.json` —— lead 已生成的施工单
2. **for each entry in plan.metrics**：
   ```
   python -m <entry.script> --input <entry.input> --output <entry.output>
   ```
3. **if plan.statistics 非空且 skip_reason is null**：
   ```
   python -m <plan.statistics.script> --inputs ... --groups ... --output ...
   ```
4. **for each chart in plan.charts**：
   ```
   python -m <chart.script> --input ... --output ...
   ```
5. **聚合**：把所有 metrics[].output 的 JSON + charts 路径 + statistics 输出（如有）合并构造 handoff JSON
6. **write handoff**：`write_file ${workspace_path}/handoff_code_executor.json`
7. **输出 `[gate_signals]` 块**给 lead

### 重要约束

- 不要写胶水脚本拼接代码 —— 所有指标 + 统计 + 绘图都在脚本里
- 不要读 catalog YAML —— plan.json 已经把你需要的执行字段（script、input、output）展开
- bash 命令必须是脚本调用（`python -m ethoinsight.scripts.*`）或文件操作（mkdir / cp / mv / ls / cat / grep / head / tail）。其他形式的 bash（包括 `python -c`、`pip install`）会被运行时拦截
- 遇到脚本报错：读 stderr → 查 `references/error-recovery.md` → 决定重试 / 跳过 / 反问 lead
- 如果 plan.metrics[i].required is true 且脚本失败：必须停下来报 lead；required is false 时记 warning 继续
```

把"范式渐进披露入口"段（含 EPM / OFT / Shoaling / Zero Maze / LDB / TST / FST 链接）整段删除（不再有 by-paradigm md）。

- [ ] **Step 3: 删 7 份 by-paradigm md**

```bash
cd packages/agent/skills/custom/ethoinsight-code/references && \
  rm -rf by-paradigm/
```

- [ ] **Step 4: 删 error-recovery.md 第 46-51 行**

Run: `sed -n '40,60p' packages/agent/skills/custom/ethoinsight-code/references/error-recovery.md`

确认第 46-51 行包含 `assess_and_handoff` 旧引用。然后：

```bash
cd /home/wangqiuyang/noldus-insight && \
  python -c "
import pathlib
p = pathlib.Path('packages/agent/skills/custom/ethoinsight-code/references/error-recovery.md')
text = p.read_text(encoding='utf-8').splitlines(keepends=True)
# 删除 46-51 行（0-indexed 45-50）；同时清掉前后可能的孤立空行
new = text[:45] + text[51:]
p.write_text(''.join(new), encoding='utf-8')
print('Updated')
"
```

肉眼检查 `cat packages/agent/skills/custom/ethoinsight-code/references/error-recovery.md` 确认无悬挂引用。

- [ ] **Step 5: 删 quality-checks.md 第 44 行**

Run: `sed -n '40,50p' packages/agent/skills/custom/ethoinsight-code/references/quality-checks.md`

确认第 44 行是 `assess_and_handoff` 旧引用。然后：

```bash
cd /home/wangqiuyang/noldus-insight && \
  python -c "
import pathlib
p = pathlib.Path('packages/agent/skills/custom/ethoinsight-code/references/quality-checks.md')
text = p.read_text(encoding='utf-8').splitlines(keepends=True)
new = text[:43] + text[44:]
p.write_text(''.join(new), encoding='utf-8')
print('Updated')
"
```

肉眼检查 `cat packages/agent/skills/custom/ethoinsight-code/references/quality-checks.md` 确认无悬挂引用。

- [ ] **Step 6: 冒烟 skill loader 仍能正常加载（无悬挂引用）**

Run（backend 目录）：
```bash
cd packages/agent/backend && source .venv/bin/activate && python -c "
from deerflow.skills import load_skills
skills = load_skills()
names = [s.name for s in skills]
assert 'ethoinsight-code' in names
assert 'ethoinsight-metric-catalog' in names
print('OK')
"
```
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add packages/agent/skills/custom/ethoinsight-code/
git commit -m "refactor(skill): 清扫 ethoinsight-code 旧范式 md + assess_and_handoff 引用

删除 references/by-paradigm/*.md × 7（范式知识已沉到 catalog YAML
single source）。SKILL.md workflow 重写为 read metric_plan.json 模式，
不再让 code-executor 自己挑脚本 / 读决策树。

error-recovery.md 第 46-51 行 + quality-checks.md 第 44 行的
assess_and_handoff 旧引用清除（assess_and_handoff 已是 v0.1 前的
历史包袱，新架构走 catalog → plan → script）。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: 清扫 assess.py 死阈值代码

按 spec §8.1 删 `_DEFAULT_THRESHOLDS`、`_load_thresholds`、`_assess_thresholds`、`assess_results` 中阈值分支；保留组间比较主流程 + `_infer_phenotype`。

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/assess.py`
- Modify: `packages/ethoinsight/tests/test_statistics.py`（追加一条反退化测试）

- [ ] **Step 1: 写反退化测试 - assess_results 不再引用绝对阈值**

追加到 `packages/ethoinsight/tests/test_statistics.py`（在 `class TestAssessResults` 内）：

```python
    def test_assess_does_not_emit_reference_range(self, shoaling_metrics):
        """同事 5-13 反馈硬要求：不参考常模/基线/normal_range。

        断言 assess_results 返回的 findings 中不再出现任何
        'Reference range' / 'normal_range' / 'below threshold' / 'above threshold'
        这类绝对阈值语言。
        """
        from ethoinsight import assess, statistics

        stat_results = statistics.compare_groups(shoaling_metrics)
        result = assess.assess_results(
            stat_results, "epm",
            metrics_result=shoaling_metrics,
        )
        all_evidence = " ".join(
            f.get("evidence", "") + " " + f.get("finding", "")
            for f in result.get("findings", [])
        )
        for forbidden in ("Reference range", "normal_range", "below threshold", "above threshold"):
            assert forbidden not in all_evidence, (
                f"assess_results 仍引用绝对阈值语言: '{forbidden}' 出现在 {all_evidence!r}"
            )
```

- [ ] **Step 2: 跑测试确认当前会失败**

Run: `cd packages/ethoinsight && pytest tests/test_statistics.py -v -k "does_not_emit_reference_range"`
Expected: 该测试可能 PASS 也可能 FAIL（取决于 shoaling 路径是否触发阈值分支）—— 但**意图是它必须永远 PASS**，目标是在删完代码后仍 PASS。先记录当前结果。

- [ ] **Step 3: 修改 assess.py**

打开 `packages/ethoinsight/ethoinsight/assess.py`。具体改动：

1. **删除** `_DEFAULT_THRESHOLDS`（约第 20-53 行整个 dict）
2. **删除** `_load_thresholds` 函数（约第 174-184 行）
3. **删除** `_assess_thresholds` 函数（约第 205-246 行）
4. **改 `assess_results` 主流程**：
   - 删第 92-93 行 `thresholds = _load_thresholds(...)` 调用
   - 删第 128-133 行 `if metrics_result and thresholds: threshold_findings = _assess_thresholds(...)` 整块
   - 删第 155-157 行 `if paradigm in ("epm", "open_field", "o_maze"): recommendations.append("For anxiety phenotyping, cross-validate...")` 整块
5. **改 module docstring**（第 1-6 行）：

   把原 docstring（"interprets ... normal ranges, anxiety/depression thresholds, and insight frameworks ..."）改为：

   ```python
   """ethoinsight.assess — result assessment based on between-group statistical comparison.

   判读统计结果时仅基于组间显著差异 + 效应量方向，**不参考绝对阈值 / 常模 /
   文献基线**（2026-05-13 同事反馈硬要求 + CLAUDE.md §9）。

   保留 _infer_phenotype 用于把"显著差异 + 效应方向"映射到表型标签（如
   "Anxiety-like phenotype (EPM open arm avoidance)"），但映射本身依赖
   显著性，不依赖绝对值大小。
   """
   ```

6. **`yaml` import 如果只被已删函数用**：删 `import yaml` 那一行（避免 ruff warning）。Run `grep "yaml" packages/ethoinsight/ethoinsight/assess.py` 确认其他地方不再用。

- [ ] **Step 4: 跑全部 assess 相关测试**

Run: `cd packages/ethoinsight && pytest tests/test_statistics.py -v -k "TestAssessResults"`
Expected: 4 个测试全部 PASS（原 3 个 + 新增 1 个 does_not_emit_reference_range）

- [ ] **Step 5: 跑 ethoinsight 全测试集冒烟**

Run: `cd packages/ethoinsight && pytest tests/ -v`
Expected: 全部 PASS。若有失败可能是导入了已删函数的别处代码 —— grep 修。

- [ ] **Step 6: Commit**

```bash
git add packages/ethoinsight/ethoinsight/assess.py \
        packages/ethoinsight/tests/test_statistics.py
git commit -m "refactor(assess): 删 _DEFAULT_THRESHOLDS 死代码 + 阈值判读分支

同事 5-13 反馈硬要求：'不要参考常模或基线，所有指标和对照组比'
（feedback.md Q1/Q4）。assess.py 中的 _DEFAULT_THRESHOLDS（EPM/OFT/
O-maze 的 normal_range + high_anxiety / hypoactivity 等绝对阈值）和
_assess_thresholds 死分支删除；assess_results 仅保留基于
comparisons + effect_size 的组间判读主流程。

反退化测试 test_assess_does_not_emit_reference_range 锁死：未来
assess_results 输出不允许再含'Reference range'/'normal_range'/
'below threshold'/'above threshold' 等绝对阈值语言。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: 清扫 oft.py silent fallback + parse 自动分组重写

完成 spec §8.1 列的剩余两个清扫项。

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/metrics/oft.py`（删 `_find_center_zone_column` 裸 in_zone fallback）
- Modify: `packages/ethoinsight/ethoinsight/parse/_core.py`（重写自动分组推断）
- Modify: `packages/ethoinsight/tests/test_metrics_oft.py`、`test_parse.py`（追加测试）

- [ ] **Step 1: 看 oft.py 当前 `_find_center_zone_column` 实现**

Run: `grep -n -A 25 "_find_center_zone_column" packages/ethoinsight/ethoinsight/metrics/oft.py | head -30`

记录现行的 fallback 顺序（多半是 1.显式 center 列 → 2. 裸 in_zone → 3. None）。

- [ ] **Step 2: 写失败测试 - 裸 in_zone 时不应 silent fallback**

追加到 `packages/ethoinsight/tests/test_metrics_oft.py`：

```python
def test_find_center_zone_does_not_silently_fallback_to_bare_in_zone():
    """同事 5-13 反馈 Q2：列名歧义时不要猜要问。

    裸 in_zone（无后缀指明是 center 还是 periphery）应返回 None，让
    上层（resolve 函数的 requires_columns 检查 + lead）触发反问。
    """
    import pandas as pd
    from ethoinsight.metrics.oft import _find_center_zone_column

    df = pd.DataFrame({"time": [0.0], "in_zone": [1]})
    assert _find_center_zone_column(df) is None


def test_find_center_zone_still_finds_explicit_center_columns():
    """显式 center 列仍应被找到（不影响 happy path）."""
    import pandas as pd
    from ethoinsight.metrics.oft import _find_center_zone_column

    df = pd.DataFrame({"time": [0.0], "in_zone_center_center_point": [1]})
    assert _find_center_zone_column(df) == "in_zone_center_center_point"
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd packages/ethoinsight && pytest tests/test_metrics_oft.py -v -k "find_center_zone"`
Expected: FAIL（test_find_center_zone_does_not_silently_fallback_to_bare_in_zone 失败，因 fallback 还在）

- [ ] **Step 4: 改 oft.py 删 silent fallback**

打开 `packages/ethoinsight/ethoinsight/metrics/oft.py`，在 `_find_center_zone_column` 函数里：

- 保留所有"明确 center / centre"列的匹配分支
- **删除**那条裸 `in_zone`（无 `_center` / `_centre` 后缀）的 fallback 分支
- 函数文档 docstring 加一段：

```python
"""Find the column representing the center zone.

Returns:
    Column name if an explicit center-zone column exists.
    None otherwise. Bare `in_zone` (without `_center` / `_centre` suffix)
    is NOT treated as center by default — list-name ambiguity should
    trigger an upstream user clarification (see 2026-05-13 feedback Q2).
"""
```

- [ ] **Step 5: 跑测试确认通过**

Run: `cd packages/ethoinsight && pytest tests/test_metrics_oft.py -v -k "find_center_zone or center_time or center_distance"`
Expected: PASS（4 个测试）。

注意：删 silent fallback 后，**之前依赖该 fallback 的真数据可能不再产指标**。这是预期行为 —— 在 catalog 架构里，resolve 函数会先检测列缺失并报 `columns_missing` 让 lead 反问。

- [ ] **Step 6: 写 parse 自动分组测试**

追加到 `packages/ethoinsight/tests/test_parse.py`：

```python
def test_parse_groups_from_result_block_name():
    """同事 5-13 反馈 Q3：raw 最后一列是'数据选择配置的结果块命名'（不是
    user-defined variable）。规范用户会命名为 'Drug' / 'Saline' 等体现分组。

    默认 'Result 1' 等占位名应被识别为'未分组'。
    """
    from ethoinsight.parse._core import infer_groups_from_result_block

    # 显式命名分组
    result = infer_groups_from_result_block(
        subjects=[
            {"name": "A1", "result_block_name": "Drug"},
            {"name": "A2", "result_block_name": "Drug"},
            {"name": "A3", "result_block_name": "Saline"},
            {"name": "A4", "result_block_name": "Saline"},
        ]
    )
    assert result == {"Drug": ["A1", "A2"], "Saline": ["A3", "A4"]}

    # 默认占位名 → 视为未分组
    result_default = infer_groups_from_result_block(
        subjects=[
            {"name": "A1", "result_block_name": "Result 1"},
            {"name": "A2", "result_block_name": "Result 1"},
        ]
    )
    assert result_default is None  # 调用方应当 fall back 到 "all"


def test_parse_groups_handles_missing_field():
    from ethoinsight.parse._core import infer_groups_from_result_block

    result = infer_groups_from_result_block(
        subjects=[{"name": "A1"}, {"name": "A2"}],
    )
    assert result is None
```

- [ ] **Step 7: 跑测试确认失败**

Run: `cd packages/ethoinsight && pytest tests/test_parse.py -v -k "result_block"`
Expected: FAIL（`AttributeError: module ... has no attribute 'infer_groups_from_result_block'`）

- [ ] **Step 8: 实现 infer_groups_from_result_block**

追加到 `packages/ethoinsight/ethoinsight/parse/_core.py`：

```python
def infer_groups_from_result_block(subjects: list[dict]) -> dict[str, list[str]] | None:
    """Infer groupings from EthoVision 'result block name'.

    EthoVision 数据选择配置的"结果块命名"出现在 raw 文件末列。
    - 默认占位名（"Result 1", "Result 2", ...）→ 视为未分组，返回 None
    - 规范命名（"Drug" / "Saline" / "Control" 等）→ 按命名聚合

    Source: 2026-05-13 同事反馈 Q3。

    Args:
        subjects: list of dicts with keys 'name' (subject name) and optional
                  'result_block_name' (str)

    Returns:
        {group_name: [subject_name, ...]} 或 None（未分组）
    """
    import re

    if not subjects:
        return None

    block_names: dict[str, list[str]] = {}
    for s in subjects:
        block = s.get("result_block_name")
        if not block:
            return None
        # 默认占位名："Result 1" / "Result 2" / ...
        if re.fullmatch(r"Result\s+\d+", block.strip()):
            return None
        block_names.setdefault(block, []).append(s["name"])

    # 只有一个 block 也算未分组
    if len(block_names) < 2:
        return None
    return block_names
```

- [ ] **Step 9: 跑测试确认通过**

Run: `cd packages/ethoinsight && pytest tests/test_parse.py -v -k "result_block"`
Expected: PASS（2 个测试）

- [ ] **Step 10: 跑全量测试**

Run: `cd packages/ethoinsight && pytest tests/ -v`
Expected: 全部 PASS。

- [ ] **Step 11: Commit**

```bash
git add packages/ethoinsight/ethoinsight/metrics/oft.py \
        packages/ethoinsight/ethoinsight/parse/_core.py \
        packages/ethoinsight/tests/test_metrics_oft.py \
        packages/ethoinsight/tests/test_parse.py
git commit -m "refactor: 退役 OFT silent fallback + 重写 parse 自动分组推断

按 2026-05-13 同事反馈：
- Q2: OFT 列名歧义时'要问不要猜'。删 _find_center_zone_column 中
  裸 in_zone 当 center 的 silent fallback；该路径以后由 resolve
  的 requires_columns 检测 + lead ask_clarification 接管。
- Q3: raw 最后一列是'数据选择配置的结果块命名'（不是 user-defined
  variable）。新增 infer_groups_from_result_block：默认占位名
  'Result 1' 等视为未分组、命名规范者按命名聚合。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: 补齐剩余 4 个范式 catalog YAML（TST / LDB / Zero Maze / Shoaling）

EPM/OFT/FST 在 Task 3 已做；本任务补齐另外 4 个。每份 YAML 按现有 `metrics/<paradigm>.py` 已有的指标 + `scripts/<paradigm>/compute_*.py` 已有的脚本填即可。

**Files:**
- Create: `packages/ethoinsight/ethoinsight/catalog/tst.yaml`
- Create: `packages/ethoinsight/ethoinsight/catalog/ldb.yaml`
- Create: `packages/ethoinsight/ethoinsight/catalog/zero_maze.yaml`
- Create: `packages/ethoinsight/ethoinsight/catalog/shoaling.yaml`
- Modify: `packages/ethoinsight/tests/test_catalog.py`（参数化测试加 4 个范式）

- [ ] **Step 1: 看每个范式现有的 metric + script 清单**

Run:
```bash
cd packages/ethoinsight/ethoinsight && \
  for p in tst ldb zero_maze shoaling; do
    echo "=== $p ==="
    ls scripts/$p/compute_*.py 2>/dev/null
    echo "--- run_groupwise_stats ---"
    ls scripts/$p/run_groupwise_stats.py 2>/dev/null
    echo "--- charts ---"
    ls scripts/$p/plot_*.py 2>/dev/null
  done
```

记录每个范式的实际脚本清单。这些是 catalog YAML 的 default_metrics + charts + statistics 的来源。

- [ ] **Step 2: 写 tst.yaml**

`packages/ethoinsight/ethoinsight/catalog/tst.yaml`：

```yaml
paradigm: tst
ev19_templates:
  - Tail Suspension Test XT190

default_metrics:
  - id: immobility_time
    script: ethoinsight.scripts.tst.compute_immobility_time
    requires_columns:
      - mobility_state*
    output_unit: seconds
    display_name_zh: 不动时间
    unit_zh: 秒
    one_liner: 动物在测试期间累计不动的时间，TST 抑郁样行为的核心指标
    direction_for_anxiety: null
    statistical_default: groupwise_compare

  - id: immobility_latency
    script: ethoinsight.scripts.tst.compute_immobility_latency
    requires_columns:
      - mobility_state*
    output_unit: seconds
    display_name_zh: 首次不动潜伏期
    unit_zh: 秒
    one_liner: 从测试开始到首次出现不动的时间
    direction_for_anxiety: null
    statistical_default: groupwise_compare

  - id: immobility_bout_count
    script: ethoinsight.scripts.tst.compute_immobility_bout_count
    requires_columns:
      - mobility_state*
    output_unit: count
    display_name_zh: 不动次数
    unit_zh: 次
    one_liner: 测试期间不动行为发生的累计次数
    direction_for_anxiety: null
    statistical_default: groupwise_compare

optional_metrics: []

charts:
  - id: box_immobility
    script: ethoinsight.scripts.tst.plot_box_immobility
    when: n_per_group >= 3

statistics:
  default:
    id: groupwise_compare
    script: ethoinsight.scripts.tst.run_groupwise_stats
    when: n_per_group >= 2 and n_groups >= 2
```

- [ ] **Step 3: 写 ldb.yaml**

`packages/ethoinsight/ethoinsight/catalog/ldb.yaml`：

```yaml
paradigm: ldb
ev19_templates:
  - Light Dark Box XT190

default_metrics:
  - id: light_time_ratio
    script: ethoinsight.scripts.ldb.compute_light_time_ratio
    requires_columns:
      - in_zone_light*
    output_unit: ratio
    display_name_zh: 亮室时间比例
    unit_zh: 比例
    one_liner: 动物在亮室停留时间占总时长的比例，LDB 焦虑样回避评估的核心指标
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: transition_count
    script: ethoinsight.scripts.ldb.compute_transition_count
    requires_columns:
      - in_zone_light*
    output_unit: count
    display_name_zh: 跨室次数
    unit_zh: 次
    one_liner: 动物在亮室与暗室之间穿梭的累计次数
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: light_latency
    script: ethoinsight.scripts.ldb.compute_light_latency
    requires_columns:
      - in_zone_light*
    output_unit: seconds
    display_name_zh: 首次进入亮室潜伏期
    unit_zh: 秒
    one_liner: 从测试开始到首次进入亮室的时间
    direction_for_anxiety: higher_is_anxious
    statistical_default: groupwise_compare

optional_metrics: []

charts: []

statistics:
  default:
    id: groupwise_compare
    script: ethoinsight.scripts.ldb.run_groupwise_stats
    when: n_per_group >= 2 and n_groups >= 2
```

- [ ] **Step 4: 写 zero_maze.yaml**

`packages/ethoinsight/ethoinsight/catalog/zero_maze.yaml`：

```yaml
paradigm: zero_maze
ev19_templates:
  - Zero Maze XT190
  - O-Maze XT190

default_metrics:
  - id: open_zone_time_ratio
    script: ethoinsight.scripts.zero_maze.compute_open_zone_time_ratio
    requires_columns:
      - in_zone_open*
    output_unit: ratio
    display_name_zh: 开放区时间比例
    unit_zh: 比例
    one_liner: 动物在开放区停留时间占总时长的比例
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: open_zone_time
    script: ethoinsight.scripts.zero_maze.compute_open_zone_time
    requires_columns:
      - in_zone_open*
    output_unit: seconds
    display_name_zh: 开放区时间
    unit_zh: 秒
    one_liner: 动物在开放区的累计停留时间
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: open_zone_distance
    script: ethoinsight.scripts.zero_maze.compute_open_zone_distance
    requires_columns:
      - in_zone_open*
      - distance_moved
    output_unit: cm
    display_name_zh: 开放区移动距离
    unit_zh: cm
    one_liner: 动物在开放区的累计移动距离
    direction_for_anxiety: lower_is_anxious
    statistical_default: groupwise_compare

  - id: hesitation_count
    script: ethoinsight.scripts.zero_maze.compute_hesitation_count
    requires_columns:
      - in_zone_open*
    output_unit: count
    display_name_zh: 犹豫次数
    unit_zh: 次
    one_liner: 接近边界但未跨入开放区的犹豫行为次数
    direction_for_anxiety: higher_is_anxious
    statistical_default: groupwise_compare

optional_metrics: []

charts: []

statistics:
  default:
    id: groupwise_compare
    script: ethoinsight.scripts.zero_maze.run_groupwise_stats
    when: n_per_group >= 2 and n_groups >= 2
```

- [ ] **Step 5: 写 shoaling.yaml**

`packages/ethoinsight/ethoinsight/catalog/shoaling.yaml`：

```yaml
paradigm: shoaling
ev19_templates:
  - Social Interaction Three Chamber XT190

default_metrics:
  - id: mean_iid
    script: ethoinsight.scripts.shoaling.compute_inter_individual_distance
    requires_columns:
      - x_center
      - y_center
    output_unit: cm
    display_name_zh: 平均个体间距
    unit_zh: cm
    one_liner: 鱼群中两两个体的平均距离，反映群体凝聚度
    direction_for_anxiety: null
    statistical_default: groupwise_compare

  - id: mean_nnd
    script: ethoinsight.scripts.shoaling.compute_nearest_neighbor_distance
    requires_columns:
      - x_center
      - y_center
    output_unit: cm
    display_name_zh: 最近邻平均距离
    unit_zh: cm
    one_liner: 每条鱼到最近同伴的平均距离，反映群体紧密度
    direction_for_anxiety: null
    statistical_default: groupwise_compare

  - id: mean_polarity
    script: ethoinsight.scripts.shoaling.compute_group_polarity
    requires_columns:
      - x_center
      - y_center
    output_unit: ratio
    display_name_zh: 群体极性
    unit_zh: 比例
    one_liner: 鱼群移动方向的一致性 [0,1]，反映集体运动协调性
    direction_for_anxiety: null
    statistical_default: groupwise_compare

optional_metrics: []

charts: []

statistics:
  default:
    id: groupwise_compare
    script: ethoinsight.scripts.shoaling.run_groupwise_stats
    when: n_per_group >= 2 and n_groups >= 2
```

- [ ] **Step 6: 更新 catalog 测试参数化覆盖所有 7 个范式**

修改 `tests/test_catalog.py` 中 `test_all_catalog_scripts_are_importable` 的 `@pytest.mark.parametrize` 装饰器：

```python
@pytest.mark.parametrize("paradigm", ["epm", "oft", "fst", "tst", "ldb", "zero_maze", "shoaling"])
def test_all_catalog_scripts_are_importable(paradigm):
    ...
```

- [ ] **Step 7: 跑全量 catalog 测试**

Run: `cd packages/ethoinsight && pytest tests/test_catalog.py -v`
Expected: 全部 PASS。若 fail，多半是某个范式的脚本不存在 → 检查 `ls packages/ethoinsight/ethoinsight/scripts/<paradigm>/`，缺哪个脚本就要么补脚本壳子（不在本任务范围）、要么从 YAML 删该 metric。

- [ ] **Step 8: Commit**

```bash
git add packages/ethoinsight/ethoinsight/catalog/tst.yaml \
        packages/ethoinsight/ethoinsight/catalog/ldb.yaml \
        packages/ethoinsight/ethoinsight/catalog/zero_maze.yaml \
        packages/ethoinsight/ethoinsight/catalog/shoaling.yaml \
        packages/ethoinsight/tests/test_catalog.py
git commit -m "feat(catalog): 补齐 TST / LDB / Zero Maze / Shoaling catalog YAML

完成 7 个范式 catalog 全覆盖。这 4 个范式的 default_metrics 对应
当前 packages/ethoinsight/ethoinsight/scripts/<paradigm>/ 下已有的
compute_*.py 脚本一对一映射。

test_all_catalog_scripts_are_importable 现覆盖全部 7 个范式。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: 更新 CLAUDE.md §7 流水线描述

完成后 CLAUDE.md 不再描述过时的"code-executor 依次调用 5 个细粒度 tool"。

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: 看 §7 当前描述**

Run: `grep -n "code-executor\|parse_trajectories\|run_statistics\|assess_and_handoff" CLAUDE.md | head -20`

- [ ] **Step 2: 改 §7（架构核心 / Agent 分析流水线）**

把"code-executor（按 ethoinsight-code skill 依次调用 5 个细粒度 tool：parse_trajectories → compute_metrics → run_statistics → generate_charts → assess_and_handoff，中间状态经 /mnt/user-data/workspace/ 文件传递）"

改写为：

```
code-executor（按 lead 生成的 metric_plan.json 逐条 bash 调用对应
脚本：python -m ethoinsight.scripts.<paradigm>.<script_name>。指标
清单来自 packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml
（single source of truth），由 lead 通过 catalog.resolve CLI 在派
遣 code-executor 之前生成 plan.json。中间状态经 /mnt/user-data/
workspace/ 文件传递。详见 docs/superpowers/specs/2026-05-13-metric-
catalog-architecture-design.md）
```

也可以在 §7 末尾加一句指向新 spec："**2026-05-13 catalog 架构上线后**，code-executor 不再自主读 by-paradigm md 选脚本；指标决策权在 lead 通过 catalog.resolve 函数确定性生成。"

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md §7 流水线描述更新为 catalog 架构

去掉过时的'code-executor 依次调用 5 个细粒度 tool（parse_trajectories
→ compute_metrics → run_statistics → generate_charts → assess_and_handoff）'
描述。新流程：lead 通过 catalog.resolve CLI 生成 metric_plan.json，
code-executor 按 plan 逐条 bash python -m ethoinsight.scripts.* 执行。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: 端到端冒烟测试

最后一步：用真数据跑通 lead → catalog → plan → code-executor → handoff → data-analyst → report-writer 全链路。

**Files:**
- 仅运行 + 观察，不写代码（若发现 bug 则在本任务新增 micro-fix 提交）

- [ ] **Step 1: 跑 ethoinsight 全测试集**

Run: `cd packages/ethoinsight && pytest tests/ -v`
Expected: 全部 PASS（前置任务的所有测试 + 老测试）

- [ ] **Step 2: 跑 agent backend 测试集**

Run: `cd packages/agent/backend && pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 3: 手工跑一次 dump_headers + resolve 链路（用真数据）**

Run:
```bash
WS=/tmp/catalog_smoke && mkdir -p $WS

# 取 5-12 同事提供的 EPM 真数据
RAW=$(ls /home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/*.txt | head -1)

cd packages/ethoinsight

# 1. dump_headers
python -m ethoinsight.parse.dump_headers \
    --input "$RAW" \
    --output $WS/columns.json
cat $WS/columns.json | head -20

# 2. 写 raw_files.json
echo "[\"$RAW\"]" > $WS/raw_files.json

# 3. catalog.resolve
python -m ethoinsight.catalog.resolve \
    --paradigm epm \
    --columns-file $WS/columns.json \
    --raw-files-json $WS/raw_files.json \
    --workspace-dir $WS \
    --output $WS/metric_plan.json

# 4. 看 plan
cat $WS/metric_plan.json
```

Expected: plan.json 含 5 个 EPM 指标（Q6 白名单）；每个 metric 的 script 字段非空；statistics.skip_reason 非空（因为没分组）。

- [ ] **Step 4: 手工跑一遍 plan 里的 5 个 script**

Run:
```bash
WS=/tmp/catalog_smoke
cd packages/ethoinsight

for SCRIPT in \
  ethoinsight.scripts.epm.compute_open_arm_time_ratio \
  ethoinsight.scripts.epm.compute_open_arm_time \
  ethoinsight.scripts.epm.compute_open_arm_entry_count \
  ethoinsight.scripts.epm.compute_open_arm_entry_ratio \
  ethoinsight.scripts.epm.compute_total_entry_count
do
  python -m $SCRIPT \
      --input "$RAW" \
      --output $WS/m_$(echo $SCRIPT | rev | cut -d. -f1 | rev).json
done

ls $WS/m_*.json
cat $WS/m_compute_open_arm_time_ratio.json
```

Expected: 5 个 JSON 文件存在，每个含 `{"metric": "...", "value": <number_or_null>}`。

- [ ] **Step 5: 如果上面发现问题（脚本报错 / catalog 字段不一致 / plan 路径错），在本任务追加 micro-fix commit**

每个 fix 单独 commit、commit message 形如：
```
fix(catalog): <具体描述>

Discovered via Task 15 smoke test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

- [ ] **Step 6: 写交接文档**

`docs/handoffs/2026-05/2026-05-13-metric-catalog-implementation-handoff.md`：

```markdown
# 2026-05-13 Metric Catalog 架构实施完成交接

## 背景
- 前置 spec: docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md
- 前置 handoff: docs/handoffs/2026-05/2026-05-12-real-data-metrics-verified-handoff.md
- 前置 feedback: docs/review-packages/2026-05-12-feedback.md

## 已完成

### 库层（packages/ethoinsight/）
- catalog/ 模块：schema.py + loader.py + resolve.py + cli.py + 7 个范式 YAML
- parse/ 包：原 parse.py → parse/_core.py + 新 dump_headers.py CLI
- metrics/oft.py: 加 compute_center_time + compute_center_distance；删 silent fallback
- parse/_core.py: 加 infer_groups_from_result_block（按 result block 命名分组）
- assess.py: 删 _DEFAULT_THRESHOLDS + 阈值判读分支

### Agent 层（packages/agent/）
- 新建 ethoinsight-metric-catalog skill（lead / data-analyst / report-writer 共享）
- ethoinsight-code skill 瘦身：删 7 份 by-paradigm md；workflow 改 read plan.json
- code_executor / data_analyst / report_writer SubagentConfig 改造
- lead_agent prompt 加 Gate 2 catalog 工作流提示

### 测试覆盖
- catalog 模块全套（schema / loader / resolve / cli / Q6 白名单反退化 / script importable）
- dump_headers CLI
- parse 自动分组
- assess reference-range 反退化

## 端到端冒烟结果
（填 Task 15 Step 3-4 的实际输出）

## 未完成 / 已知限制
- shoaling 多文件场景：resolve 当前用 raw_files[0]，多文件 wrapper JSON 待 v0.2 扩展
- catalog i18n：仅有中文展示字段，英文待加
- catalog hot reload：改 YAML 需重启 agent，v0.1 不做

## 下一位 Agent 的第一步建议
- 跑 make dev 实测 EPM/OFT/FST 三个范式的真数据端到端
- 收集行为学同事对新流程的二次 review

## Commit 历史（按时间）
（git log --oneline 该任务范围）
```

- [ ] **Step 7: 跑 ruff 格式化**

Run: `cd packages/ethoinsight && ruff check . --fix && ruff format .`
Run: `cd packages/agent/backend && ruff check . --fix && ruff format .`

Commit 任何 ruff 整理的小动作。

- [ ] **Step 8: Final commit**

```bash
git add docs/handoffs/2026-05/2026-05-13-metric-catalog-implementation-handoff.md
git commit -m "docs(handoff): 2026-05-13 metric catalog 架构实施完成交接

记录本次实施的所有改动、端到端冒烟结果、未完成项目和下一位 agent 的
建议。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## 实施完成验证清单

最后过一遍，确认全部就位：

- [ ] `cd packages/ethoinsight && pytest tests/ -v` 全绿
- [ ] `cd packages/agent/backend && pytest tests/ -v` 全绿
- [ ] `python -m ethoinsight.catalog.resolve --paradigm epm ... --output plan.json` 实测产 plan.json
- [ ] `python -m ethoinsight.parse.dump_headers --input <real_ethovision.txt> --output columns.json` 实测产 columns.json
- [ ] `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/` 目录不存在
- [ ] `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md` 存在
- [ ] `grep -r "assess_and_handoff" packages/agent/skills/custom/ethoinsight-code/` 无结果
- [ ] `grep -r "_DEFAULT_THRESHOLDS\|_assess_thresholds" packages/ethoinsight/` 无结果
- [ ] CLAUDE.md §7 不再含 "依次调用 5 个细粒度 tool"
- [ ] git log 至少 14 个 commit（每个 Task 一个，加上 Task 15 micro-fix 若干）

---

## 设计决策快速回顾（防执行偏离）

| 决策 | 理由 |
|------|------|
| catalog YAML 在 ethoinsight 库内 | single source of truth，agent + 单测 + golden-case 共消费 |
| lead 自己 bash 调 dump_headers + resolve（不派遣 subagent 勘察） | A1 < A2 = 8:2（files-are-facts、Guardrail 零改动、1 跳反问、Qwen3-8B 友好） |
| code-executor 不挂 metric-catalog skill | 保持执行纯净、防 LLM 偷懒空间 |
| 统计算法决策树留在 statistics.py 硬编码 | 学界标准事实、不属于业务配置 |
| catalog 字段含 direction_for_anxiety + statistical_default | 判读和统计入口指针，data-analyst 用 |
| catalog 字段含 display_name_zh + unit_zh + one_liner | 展示元数据，report-writer 用，禁止在 prompt 硬编码 |
| metric_plan.json 不嵌完整 bash 命令、只放 script dotted path | 避免 shell 转义噩梦，code-executor 自行 format |
| metric_plan.json 不嵌 display_* / direction_* 字段 | 这些是 catalog 字段，避免双源 |
