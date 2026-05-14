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
        raise CatalogError(
            f"Catalog file not found for paradigm '{paradigm}': {yaml_path}"
        )
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise CatalogError(f"YAML parse failed for {yaml_path}: {e}") from e

    if not isinstance(raw, dict):
        raise CatalogError(
            f"{yaml_path}: top-level must be a mapping, got {type(raw).__name__}"
        )

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
        raise CatalogError(
            f"{source}: '{key}' must be a list, got {type(items).__name__}"
        )
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
                f"{source} {where}: field '{field}' must be {expected_type}, "
                f"got {type(v).__name__}"
            )
        return v

    mid = req("id", str)
    script = req("script", str)
    if "requires_columns" not in item:
        raise CatalogError(
            f"{source} {where}: missing 'requires_columns' (use empty list if none)"
        )
    requires_columns = item["requires_columns"]
    if not isinstance(requires_columns, list) or not all(
        isinstance(c, str) for c in requires_columns
    ):
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
        raise CatalogError(
            f"{source}: 'statistics' must contain 'default' mapping or be null"
        )
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
