"""加载 packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml 并构造
Catalog dataclass。校验失败一律抛 CatalogError，含 paradigm + 问题点。

Canonical paradigm key policy (2026-05-25 → 2026-05-26):
  The system uses ACADEMIC NAMES as canonical paradigm keys.
  v0.1 仅支持 5 个范式 (catalog/<paradigm>.yaml 实际存在):
    - epm           (file: epm.yaml)
    - open_field    (file: oft.yaml)
    - light_dark_box (file: ldb.yaml)
    - forced_swim   (file: fst.yaml)
    - zero_maze     (file: zero_maze.yaml)

  历史还在 _PARADIGM_ALIASES 中的 paradigm key (如 tail_suspension/shoaling) 仍能
  通过 alias 解析路径, 但对应 YAML 文件已删除会抛 file-not-found 错误。lead 应在
  identify_ev19_template 看到 status=unsupported 时反问用户, 不要走到 load_catalog。

  ``load_catalog`` accepts either the academic name (preferred) or the
  filename stem (legacy) and resolves to the correct YAML via _PARADIGM_ALIASES.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ethoinsight.catalog.schema import (
    ALLOWED_DIRECTIONS,
    ALLOWED_STAT_DEFAULTS,
    AnonymousZoneOverride,
    Catalog,
    ChartEntry,
    CommonCatalog,
    MetricEntry,
    ParamSpec,
    ParadigmParameters,
    SharedParameters,
    StatisticsEntry,
    ZoneConceptParam,
)


class CatalogError(Exception):
    """Catalog YAML 加载 / 校验失败。"""


_DEFAULT_CATALOG_DIR = Path(__file__).parent

# Canonical paradigm key (academic name) → catalog YAML filename stem.
# Upstream code (identify_ev19_template, prep_metric_plan, metrics dispatcher,
# skill docs, experiment_context) ALL use the academic name as the
# canonical paradigm key. Catalog YAML filenames historically use shortened
# abbreviations (fst / tst / oft / ldb); the alias map below preserves that
# physical layout without exposing the inconsistency upstream.
#
# Filename-style keys (e.g. "fst") are also accepted for backward
# compatibility with existing scripts (plot_timeseries.py, test_catalog.py)
# that still pass abbreviations directly.
_PARADIGM_ALIASES: dict[str, str] = {
    # academic name → filename stem
    "forced_swim": "fst",
    "tail_suspension": "tst",
    "open_field": "oft",
    "light_dark_box": "ldb",
    # already aligned (no aliasing needed but listed for clarity)
    "epm": "epm",
    "zero_maze": "zero_maze",
}


def load_catalog(paradigm: str, catalog_dir: str | Path | None = None) -> Catalog:
    """加载 <catalog_dir>/<paradigm>.yaml 并校验返回 Catalog。

    Args:
        paradigm: Canonical paradigm key (academic name, e.g. "forced_swim",
            "open_field", "epm"). Filename-style abbreviations (e.g. "fst",
            "oft") are also accepted for backward compatibility.
        catalog_dir: catalog YAML 目录；默认为本模块所在目录

    Raises:
        CatalogError: 文件不存在 / 必填字段缺失 / enum 越界 / id 重复 等
    """
    catalog_dir = Path(catalog_dir) if catalog_dir else _DEFAULT_CATALOG_DIR
    # Resolve canonical academic name → filename stem. Accept both directions:
    # if input is already the filename stem (no entry in alias map), use as-is.
    filename_stem = _PARADIGM_ALIASES.get(paradigm, paradigm)
    yaml_path = catalog_dir / f"{filename_stem}.yaml"
    if not yaml_path.is_file():
        raise CatalogError(
            f"Catalog file not found for paradigm '{paradigm}' "
            f"(resolved to '{filename_stem}.yaml'): {yaml_path}"
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

    paradigm_params_block = _parse_param_block(raw, "paradigm_parameters", source)

    # Parse anonymous_zone_override translation rule (2026-06-04: three-paradigm unification)
    anonymous_zone_override: AnonymousZoneOverride | None = None
    azo_raw = raw.get("anonymous_zone_override")
    if azo_raw is not None:
        if not isinstance(azo_raw, dict):
            raise CatalogError(
                f"{source}: 'anonymous_zone_override' must be a mapping, "
                f"got {type(azo_raw).__name__}"
            )
        target_param = azo_raw.get("target_param")
        if not isinstance(target_param, str) or not target_param:
            raise CatalogError(
                f"{source}: 'anonymous_zone_override.target_param' "
                f"must be a non-empty string"
            )
        wrap_list = azo_raw.get("wrap_list", False)
        if not isinstance(wrap_list, bool):
            raise CatalogError(
                f"{source}: 'anonymous_zone_override.wrap_list' "
                f"must be bool, got {type(wrap_list).__name__}"
            )
        anonymous_zone_override = AnonymousZoneOverride(
            target_param=target_param,
            wrap_list=wrap_list,
        )

    # Parse zone_concept_params — paradigm-level concept→param mapping
    zone_concept_params: dict[str, ZoneConceptParam] = {}
    for concept_key, mapping in raw.get("zone_concept_params", {}).items():
        if not isinstance(mapping, dict):
            raise CatalogError(
                f"{source}: zone_concept_params.{concept_key}: must be a dict, "
                f"got {type(mapping).__name__}"
            )
        param = mapping.get("param", "")
        if not isinstance(param, str) or not param:
            raise CatalogError(
                f"{source}: zone_concept_params.{concept_key}.param: "
                f"must be a non-empty string"
            )
        wrap_list = mapping.get("wrap_list", False)
        if not isinstance(wrap_list, bool):
            raise CatalogError(
                f"{source}: zone_concept_params.{concept_key}.wrap_list: "
                f"must be bool, got {type(wrap_list).__name__}"
            )
        zone_concept_params[concept_key] = ZoneConceptParam(
            param=param,
            wrap_list=wrap_list,
        )

    return Catalog(
        paradigm=paradigm,
        ev19_templates=ev19_templates,
        default_metrics=default_metrics,
        optional_metrics=optional_metrics,
        charts=charts,
        statistics_default=statistics_default,
        paradigm_parameters=ParadigmParameters(parameters=paradigm_params_block),
        anonymous_zone_override=anonymous_zone_override,
        zone_concept_params=zone_concept_params,
    )


def _parse_param_spec(item: dict, where: str, source: Path) -> ParamSpec:
    """解析单个 ParamSpec yaml 字段。"""
    def req(field_name: str, expected_type: type | tuple) -> Any:
        if field_name not in item:
            raise CatalogError(f"{source} {where}: missing '{field_name}'")
        v = item[field_name]
        if not isinstance(v, expected_type):
            raise CatalogError(
                f"{source} {where}: '{field_name}' must be {expected_type}, "
                f"got {type(v).__name__}"
            )
        return v

    default = req("default", (int, float, str, list))
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


def _is_cnf_requires_columns(value: object) -> bool:
    """Returns True if value is a valid CNF requires_columns: list of non-empty str,
    or non-empty list-of-(non-empty str)."""
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, str):
            if not item:
                return False
        elif isinstance(item, list):
            if not item or not all(isinstance(s, str) and s for s in item):
                return False
        else:
            return False
    return True


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
    if not _is_cnf_requires_columns(requires_columns):
        raise CatalogError(
            f"{source} {where}: 'requires_columns' must be list of str or list-of-str groups"
        )

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

    parameters = _parse_param_block(item, "parameters", source) if "parameters" in item else {}
    parameters_ref = item.get("parameters_ref", []) or []
    if not isinstance(parameters_ref, list) or not all(isinstance(x, str) for x in parameters_ref):
        raise CatalogError(f"{source} {where}: 'parameters_ref' must be list[str]")

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
        parameters=parameters,
        parameters_ref=list(parameters_ref),
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
        # 1.1: display_name_zh 必填
        display_name_zh = it.get("display_name_zh", "")
        if not display_name_zh:
            raise CatalogError(
                f"{source}: charts[{i}] missing 'display_name_zh'"
            )
        accepts_paradigm = bool(it.get("accepts_paradigm", False))
        output_mode = str(it.get("output_mode", "per_subject"))
        if output_mode not in ("per_subject", "aggregate"):
            raise CatalogError(
                f"{source}: charts[{i}] output_mode must be 'per_subject' or 'aggregate', got {output_mode!r}"
            )
        needs_groups = bool(it.get("needs_groups", False))
        requires_columns = it.get("requires_columns", []) or []
        if not _is_cnf_requires_columns(requires_columns):
            raise CatalogError(
                f"{source}: charts[{i}] 'requires_columns' must be list of str or list-of-str groups if present"
            )
        out.append(
            ChartEntry(
                id=it["id"],
                script=it["script"],
                when=it["when"],
                display_name_zh=display_name_zh,
                accepts_paradigm=accepts_paradigm,
                output_mode=output_mode,
                needs_groups=needs_groups,
                requires_columns=list(requires_columns),
            )
        )
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


# ============================================================================
# Common catalog — paradigm-agnostic fallback charts (W3)
# ============================================================================


def load_common_catalog(catalog_dir: str | Path | None = None) -> CommonCatalog:
    """Load _common.yaml from catalog directory."""
    catalog_dir = Path(catalog_dir) if catalog_dir else _DEFAULT_CATALOG_DIR
    yaml_path = catalog_dir / "_common.yaml"
    if not yaml_path.is_file():
        raise CatalogError(
            f"_common.yaml not found in catalog directory: {yaml_path}"
        )
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise CatalogError(f"YAML parse failed for {yaml_path}: {e}") from e

    if not isinstance(raw, dict):
        raise CatalogError(
            f"{yaml_path}: top-level must be a mapping, got {type(raw).__name__}"
        )

    common_charts = _parse_chart_list_under_key(raw, "common_charts", yaml_path)
    shared_params = _parse_param_block(raw, "shared_parameters", yaml_path)
    return CommonCatalog(
        common_charts=common_charts,
        shared_parameters=SharedParameters(parameters=shared_params),
    )


def _parse_chart_list_under_key(raw: dict, key: str, source: Path) -> list[ChartEntry]:
    """Variant of _parse_chart_list that uses configurable key (charts vs common_charts)."""
    items = raw.get(key, []) or []
    if not isinstance(items, list):
        raise CatalogError(f"{source}: '{key}' must be a list")
    out: list[ChartEntry] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise CatalogError(f"{source}: {key}[{i}] must be a mapping")
        for f in ("id", "script", "when"):
            if f not in it:
                raise CatalogError(f"{source}: {key}[{i}] missing '{f}'")
        # 1.1: display_name_zh 必填
        display_name_zh = it.get("display_name_zh", "")
        if not display_name_zh:
            raise CatalogError(
                f"{source}: {key}[{i}] missing 'display_name_zh'"
            )
        accepts_paradigm = bool(it.get("accepts_paradigm", False))
        output_mode = str(it.get("output_mode", "per_subject"))
        if output_mode not in ("per_subject", "aggregate"):
            raise CatalogError(
                f"{source}: {key}[{i}] output_mode must be 'per_subject' or 'aggregate', got {output_mode!r}"
            )
        needs_groups = bool(it.get("needs_groups", False))
        requires_columns = it.get("requires_columns", []) or []
        if not _is_cnf_requires_columns(requires_columns):
            raise CatalogError(
                f"{source}: {key}[{i}] 'requires_columns' must be list of str or list-of-str groups if present"
            )
        out.append(
            ChartEntry(
                id=it["id"],
                script=it["script"],
                when=it["when"],
                display_name_zh=display_name_zh,
                accepts_paradigm=accepts_paradigm,
                output_mode=output_mode,
                needs_groups=needs_groups,
                requires_columns=list(requires_columns),
            )
        )
    return out


# ============================================================================
# Sprint 2a: cross-catalog consistency validation
# ============================================================================


def validate_catalog_consistency(
    common: CommonCatalog,
    paradigm_catalogs: list[tuple[str, Catalog]],
) -> None:
    """跨范式 catalog 一致性校验。Sprint 2a 引入。

    校验项:
    1. parameters_ref 中所有 ID 在 _common.yaml.shared_parameters 中存在
    2. 重复参数检测: 同名 + 相同 default 出现在 2+ paradigm yaml 的
       paradigm_parameters 中 → 应提到 _common.yaml, raise CatalogError

    Raises:
        CatalogError: 校验失败
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


def load_all_catalogs(
    catalog_dir: str | Path | None = None,
) -> tuple[CommonCatalog, list[tuple[str, Catalog]]]:
    """加载 _common.yaml + 全部 paradigm yaml, 跑一致性校验。"""
    common = load_common_catalog(catalog_dir)
    paradigm_names = [
        "epm", "open_field", "light_dark_box",
        "forced_swim", "tail_suspension", "zero_maze",
    ]
    paradigm_catalogs: list[tuple[str, Catalog]] = []
    for pname in paradigm_names:
        try:
            cat = load_catalog(pname, catalog_dir)
            paradigm_catalogs.append((pname, cat))
        except CatalogError as e:
            if "not found" in str(e).lower():
                continue
            raise

    validate_catalog_consistency(common, paradigm_catalogs)
    return common, paradigm_catalogs
