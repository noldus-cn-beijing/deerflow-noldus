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
import json
from pathlib import Path
from typing import Any

from ethoinsight.catalog.loader import CatalogError, CommonCatalog, load_catalog, load_common_catalog
from ethoinsight.catalog.schema import (
    Catalog,
    ChartEntry,
    MetricEntry,
    Plan,
    PlanChart,
    PlanCharts,
    PlanInputs,
    PlanMetric,
    PlanMetrics,
    PlanSkipped,
    PlanStatistics,
    StatisticsEntry,
)

SCHEMA_VERSION = "1.1"


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
    virtual_workspace_dir: str | None = None,
    column_aliases: dict[str, str] | None = None,
) -> Plan:
    """生成 Plan dataclass。失败抛 ResolveError。

    workspace_dir: 真实物理路径（用于脚本实际执行时的 input 引用等）。
    virtual_workspace_dir: plan.json output 字段使用的虚拟路径（面向 downstream subagent）。
                           未提供时兜底使用 workspace_dir（兼容旧调用方）。

    Note: 旧 backward-compat wrapper — 内部调 resolve_metrics() + 保留 charts=[]。
    W22 dogfood 完成后删。
    """
    pm = resolve_metrics(
        paradigm=paradigm,
        columns=columns,
        raw_files=raw_files,
        workspace_dir=workspace_dir,
        include=include,
        exclude=exclude,
        n_per_group=n_per_group,
        n_groups=n_groups,
        groups_file=groups_file,
        columns_file=columns_file,
        ev19_template=ev19_template,
        virtual_workspace_dir=virtual_workspace_dir,
        column_aliases=column_aliases,
    )
    return Plan(
        schema_version=pm.schema_version,
        paradigm=pm.paradigm,
        ev19_template=pm.ev19_template,
        generated_at=pm.generated_at,
        inputs=pm.inputs,
        metrics=pm.metrics,
        statistics=pm.statistics,
        charts=[],
        skipped=pm.skipped,
        notes=pm.notes,
    )


def resolve_metrics(
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
    virtual_workspace_dir: str | None = None,
    column_aliases: dict[str, str] | None = None,
) -> PlanMetrics:
    """生成 PlanMetrics dataclass（不含 charts）。失败抛 ResolveError。

    workspace_dir: 真实物理路径（用于脚本实际执行时的 input 引用等）。
    virtual_workspace_dir: plan.json output 字段使用的虚拟路径（面向 downstream subagent）。
                           未提供时兜底使用 workspace_dir（兼容旧调用方）。
    column_aliases: {原始列 or normalized: catalog 概念} — Sprint 1 通用列别名表。
                    在 columns 入口单点重映射后喂给所有下游消费入口。
                    含 alias → None/__ignore__ 的列从 columns 移除。
    """
    try:
        cat = load_catalog(paradigm)
    except CatalogError as e:
        if (
            "file not found" in str(e).lower()
            or "not found for paradigm" in str(e).lower()
        ):
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

    all_known_ids = {m.id for m in cat.default_metrics} | {
        m.id for m in cat.optional_metrics
    }
    unknown = include_set - all_known_ids
    if unknown:
        raise ResolveError(
            code="unknown_metric",
            message=f"Metric(s) not found in {paradigm} catalog: {sorted(unknown)}",
            details={"requested": sorted(unknown), "available": sorted(all_known_ids)},
        )

    plan_metrics: list[PlanMetric] = []
    skipped: list[PlanSkipped] = []

    # Sprint 1 column-semantics: apply alias remapping before any downstream
    # consumer touches columns.  _missing_columns / _compute_parameters_in_use
    # / chart resolution ALL consume the post-alias columns list.
    if column_aliases:
        columns = _apply_aliases(columns, column_aliases, _zone_concept_map(cat))

    # Step 1: process default_metrics
    for m in cat.default_metrics:
        if m.id in exclude_set:
            skipped.append(
                PlanSkipped(
                    id=m.id,
                    reason="user.exclude",
                    detail=f"User explicitly excluded {m.id}.",
                )
            )
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
                    "metric": m.id,
                    "missing_patterns": missing,
                    "available_columns": columns,
                },
            )
        plan_metrics.extend(
            _metric_to_plan(
                m, raw_files, workspace_dir, required=True, reason="paradigm.default",
                virtual_workspace_dir=virtual_workspace_dir,
            )
        )

    # Step 2: process user-include
    for inc_id in include_set:
        if inc_id in exclude_set:
            skipped.append(
                PlanSkipped(
                    id=inc_id,
                    reason="user.exclude",
                    detail=f"User specified both include and exclude for {inc_id}; honoring exclude.",
                )
            )
            continue
        if inc_id in {m.id for m in cat.default_metrics}:
            continue
        m = next(m for m in cat.optional_metrics if m.id == inc_id)
        missing = _missing_columns(m.requires_columns, columns)
        if missing:
            skipped.append(
                PlanSkipped(
                    id=m.id,
                    reason="columns.missing",
                    detail=f"User-included metric {m.id} skipped: missing columns {missing}.",
                )
            )
            continue
        plan_metrics.extend(
            _metric_to_plan(
                m, raw_files, workspace_dir, required=False, reason="user.include",
                virtual_workspace_dir=virtual_workspace_dir,
            )
        )

    if not plan_metrics:
        raise ResolveError(
            code="empty_plan",
            message=f"All metrics for paradigm '{paradigm}' were excluded or unavailable.",
            details={"paradigm": paradigm},
        )

    # Step 3: statistics
    plan_stats: PlanStatistics | None = None
    if cat.statistics_default is not None:
        if _evaluate_when(
            cat.statistics_default.when, n_per_group=n_per_group, n_groups=n_groups
        ):
            plan_stats = _stats_to_plan(
                cat.statistics_default, workspace_dir, skip_reason=None,
                virtual_workspace_dir=virtual_workspace_dir,
            )
        else:
            plan_stats = _stats_to_plan(
                cat.statistics_default,
                workspace_dir,
                skip_reason=f"condition '{cat.statistics_default.when}' not met (n_per_group={n_per_group}, n_groups={n_groups})",
                virtual_workspace_dir=virtual_workspace_dir,
            )

    notes: list[str] = []
    if include_set:
        notes.append(f"User include: {sorted(include_set)}")
    if exclude_set:
        notes.append(f"User exclude: {sorted(exclude_set)}")
    notes.append(
        f"Data columns: {len(columns)}; metrics planned: {len(plan_metrics)}; skipped: {len(skipped)}"
    )

    return PlanMetrics(
        schema_version=SCHEMA_VERSION,
        paradigm=cat.paradigm,
        ev19_template=ev19_template,
        generated_at=_utcnow_iso(),
        inputs=PlanInputs(
            raw_files=list(raw_files),
            groups_file=groups_file,
            columns_file=columns_file,
        ),
        metrics=plan_metrics,
        statistics=plan_stats,
        skipped=skipped,
        notes=notes,
    )


def resolve_charts(
    paradigm: str,
    columns: list[str],
    raw_files: list[str],
    workspace_dir: str,
    *,
    user_intent: str | None = None,
    total_subjects: int | None = None,
    n_per_group: int | None = None,
    n_groups: int | None = None,
    total_duration_seconds: float | None = None,
    groups_file: str | None = None,
    groups: dict[str, Any] | None = None,
    columns_file: str | None = None,
    ev19_template: str | None = None,
    virtual_workspace_dir: str | None = None,
    column_aliases: dict[str, str] | None = None,
) -> PlanCharts:
    """生成 PlanCharts dataclass。失败抛 ResolveError。

    主路径：for ch in cat.charts: if _evaluate_when → charts.append
    Fallback 触发：if len(charts)==0 → load_common_catalog() → for ch in common_charts: if _evaluate_when → fallback.append

    1.2: ``groups`` (canonical {arena: group_name} or {group_name: [subject, ...]}) is
    threaded into ``_chart_to_plan`` so aggregate plots with ``needs_groups: true`` get a
    materialised groups.json passed via --groups.

    1.2: ``user_intent`` is now consulted to narrow catalog charts when the user uses
    common chart-type names (e.g. "箱线图" -> id contains "box", "时序图" -> id contains
    "activity" / "timeseries" / "time"). Filter is opt-in: when no recognised keyword
    is found, all catalog charts are returned unchanged.
    """
    try:
        cat = load_catalog(paradigm)
    except CatalogError as e:
        if (
            "file not found" in str(e).lower()
            or "not found for paradigm" in str(e).lower()
        ):
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

    # Apply user_intent filter — keep entries whose id matches the intent keyword(s).
    candidate_charts = _filter_charts_by_user_intent(cat.charts, user_intent)

    # Sprint 1 column-semantics: apply alias remapping before chart _missing_columns
    # checks, identical to resolve_metrics — custom zone columns must remap to catalog
    # concepts or zone-dependent charts silently drop.
    if column_aliases:
        columns = _apply_aliases(columns, column_aliases, _zone_concept_map(cat))

    charts: list[PlanChart] = []
    skipped: list[PlanSkipped] = []
    for ch in candidate_charts:
        missing = _missing_columns(ch.requires_columns, columns)
        if missing:
            skipped.append(
                PlanSkipped(
                    id=ch.id,
                    reason="columns.missing",
                    detail=(
                        f"Chart {ch.id} skipped: missing columns {missing} "
                        f"(available: {sorted(columns)[:8]}{'...' if len(columns) > 8 else ''})."
                    ),
                )
            )
            continue
        if _evaluate_when(ch.when, n_per_group=n_per_group, n_groups=n_groups, total_subjects=total_subjects, total_duration_seconds=total_duration_seconds):
            charts.extend(_chart_to_plan(
                ch, raw_files, workspace_dir, paradigm=cat.paradigm,
                virtual_workspace_dir=virtual_workspace_dir,
                groups=groups,
            ))

    fallback: list[PlanChart] = []
    if not charts:
        try:
            common = load_common_catalog()
        except CatalogError:
            common = CommonCatalog(common_charts=[])
        # Fallback also respects user_intent so "轨迹图" prefers trajectory over heatmap.
        candidate_fallback = _filter_charts_by_user_intent(common.common_charts, user_intent)
        for ch in candidate_fallback:
            missing = _missing_columns(ch.requires_columns, columns)
            if missing:
                skipped.append(
                    PlanSkipped(
                        id=ch.id,
                        reason="columns.missing",
                        detail=(
                            f"Fallback chart {ch.id} skipped: missing columns {missing} "
                            f"(available: {sorted(columns)[:8]}{'...' if len(columns) > 8 else ''})."
                        ),
                    )
                )
                continue
            if _evaluate_when(ch.when, n_per_group=n_per_group, n_groups=n_groups, total_subjects=total_subjects, total_duration_seconds=total_duration_seconds):
                fallback.extend(_chart_to_plan(
                    ch, raw_files, workspace_dir, paradigm=cat.paradigm,
                    virtual_workspace_dir=virtual_workspace_dir,
                    groups=groups,
                ))

    notes: list[str] = []
    if charts:
        notes.append(f"Generated {len(charts)} catalog charts")
    elif fallback:
        notes.append(f"Fallback path: {len(fallback)} common charts available")
    else:
        notes.append("No charts matched; chart-maker should ask user")
    if skipped:
        notes.append(
            f"Skipped {len(skipped)} chart(s) due to missing columns: "
            f"{', '.join(s.id for s in skipped)}"
        )
    if user_intent:
        notes.append(f"user_intent filter applied: {user_intent!r}")

    return PlanCharts(
        paradigm=cat.paradigm,
        ev19_template=ev19_template,
        generated_at=_utcnow_iso(),
        inputs=PlanInputs(raw_files=list(raw_files), groups_file=groups_file, columns_file=columns_file),
        charts=charts,
        charts_fallback_available=fallback,
        skipped=skipped,
        user_intent=user_intent,
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


def _zone_concept_map(cat: Catalog) -> dict[str, str]:
    """Build {concept_keyword → glob_pattern} from a catalog's zone requires_columns.

    Sprint 1 列语义对齐: the LLM writes a concept keyword (e.g. "center", "open_arms")
    as ``resolves_to`` — NOT a machine column name with a specific suffix. This map lets
    ``_apply_aliases`` translate that concept into a column name that actually satisfies
    the catalog glob, so the LLM never has to know suffixes like ``_point``/``_aligned``.

    The catalog YAML is the single source of truth (no second knowledge store): we scan
    every metric/chart ``requires_columns`` for ``in_zone_<concept>*`` patterns and key
    them by ``<concept>`` (the part between the ``in_zone_`` prefix and the glob ``*``).

    Example for OFT:
      requires_columns ``in_zone_center_*`` → {"center": "in_zone_center_*"}
    """
    concept_map: dict[str, str] = {}
    entries = list(cat.default_metrics) + list(cat.optional_metrics) + list(cat.charts)
    for entry in entries:
        for pat in getattr(entry, "requires_columns", []) or []:
            if not pat.startswith("in_zone_") or "*" not in pat:
                continue
            # Strip the in_zone_ prefix and the trailing glob to get the concept keyword.
            # "in_zone_center_*" → "center"; "in_zone_open_arms_*" → "open_arms";
            # "in_zone_light*" → "light"; "in_zone_open*" → "open".
            core = pat[len("in_zone_"):]
            core = core.rstrip("*").rstrip("_")
            if core:
                concept_map.setdefault(core, pat)
    return concept_map


def _materialize_concept(concept: str, glob_pattern: str) -> str:
    """Turn a concept keyword into a concrete column name that satisfies ``glob_pattern``.

    "center" + "in_zone_center_*" → "in_zone_center_aligned" (matches the glob).
    If the concept already looks like a full column name (starts with in_zone_ and
    matches the glob), it is returned unchanged.
    """
    if concept.startswith("in_zone_") and fnmatch.fnmatchcase(concept, glob_pattern):
        return concept
    # Replace the trailing glob with a synthetic, deterministic suffix.
    base = glob_pattern.rstrip("*")
    if base.endswith("_"):
        return f"{base}aligned"
    return f"{base}_aligned"


def _apply_aliases(
    columns: list[str],
    aliases: dict[str, str],
    concept_map: dict[str, str] | None = None,
) -> list[str]:
    """Apply column alias remapping: {raw_or_normalized → catalog concept}.

    The alias *target* may be either:
      * a concept keyword (e.g. "center") — translated via ``concept_map`` into a
        column name that satisfies the catalog glob (preferred; LLM writes this); or
      * already a concrete column name (e.g. "in_zone_center_point") — used as-is.

    Columns whose alias value is None or "__ignore__" are removed
    (user confirmed the column is irrelevant — D4).

    Returns a new list; the input is not mutated.
    """
    ignore_values = {None, "__ignore__"}
    concept_map = concept_map or {}
    result: list[str] = []
    for col in columns:
        if col in aliases:
            target = aliases[col]
            if target in ignore_values:
                continue
            # Translate concept keyword → matchable column name when recognised.
            if target in concept_map:
                target = _materialize_concept(target, concept_map[target])
            result.append(target)
        else:
            result.append(col)
    return result


def _metric_to_plan(
    m: MetricEntry,
    raw_files: list[str],
    workspace_dir: str,
    *,
    required: bool,
    reason: str,
    virtual_workspace_dir: str | None = None,
) -> list[PlanMetric]:
    """Expand one MetricEntry into N PlanMetric (one per raw_file).

    Fix 2026-05-20 (FST E2E): 之前只生成单个 PlanMetric 且 input=raw_files[0],
    用户上传多文件时除第一个外的 subject 全部丢失。现在按 raw_files 展开,
    每个 subject 一个 PlanMetric,output 用 subject_index 后缀避免覆盖。
    单文件(len==1)保持 output 名 m_<id>.json 兼容现有产物。

    W27 (2026-05-27): 透传 catalog 判读 / 展示字段到 PlanMetric,subagent 不再 read catalog YAML。
    详见 docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md
    """
    if not raw_files:
        return []
    effective_workspace = virtual_workspace_dir or workspace_dir
    multi = len(raw_files) > 1
    plans: list[PlanMetric] = []
    for idx, raw_file in enumerate(raw_files):
        suffix = f"_s{idx}" if multi else ""
        output_path = str(Path(effective_workspace) / f"m_{m.id}{suffix}.json")
        plans.append(
            PlanMetric(
                id=m.id,
                script=m.script,
                input=raw_file,
                output=output_path,
                required=required,
                reason=reason,
                subject_index=idx,
                display_name_zh=m.display_name_zh,
                unit_zh=m.unit_zh,
                one_liner=m.one_liner,
                output_unit=m.output_unit,
                direction_for_anxiety=m.direction_for_anxiety,
                statistical_default=m.statistical_default,
            )
        )
    return plans


def _filter_charts_by_user_intent(charts: list[ChartEntry], user_intent: str | None) -> list[ChartEntry]:
    """Narrow the candidate chart list when user_intent contains a recognised chart-type keyword.

    Maps user-spoken chart-type names (Chinese & English) to substring patterns on chart id:

      箱线图 / box plot         -> 'box'
      柱状图 / bar              -> 'bar'
      轨迹图 / trajectory       -> 'trajectory'
      热力图 / heatmap          -> 'heatmap'
      时序图 / timeseries / time-series / 时间进程 -> 'time' or 'activity_intensity'
      分布图 / distribution     -> 'distribution'

    Multi-keyword intents (e.g. "箱线图、轨迹图、时序图") accumulate matches across all hits.
    Filter is *additive*: when no keyword matches, all charts pass through (no-op). This keeps
    the legacy behaviour for ASKVIZ "A. 画图" choices that don't pin a specific chart type.
    """
    if not user_intent:
        return list(charts)
    intent_lower = user_intent.lower()
    # Each rule maps {keyword fragments} -> {id substrings to match}.
    rules: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
        (("箱线图", "box plot", "box-plot"), ("box",)),
        (("柱状图", "bar chart", "bar plot", "bar-plot"), ("bar",)),
        (("轨迹图", "轨迹", "trajectory"), ("trajectory",)),
        (("热力图", "heatmap", "heat map"), ("heatmap",)),
        (("时序图", "时间进程", "timeseries", "time-series", "time series"), ("time", "activity_intensity", "timeseries")),
        (("分布图", "distribution", "分布"), ("distribution",)),
    ]
    wanted_id_fragments: set[str] = set()
    for keywords, id_fragments in rules:
        if any(kw in user_intent or kw in intent_lower for kw in keywords):
            wanted_id_fragments.update(id_fragments)
    if not wanted_id_fragments:
        return list(charts)
    filtered = [ch for ch in charts if any(frag in ch.id for frag in wanted_id_fragments)]
    # Don't strip everything to zero — if the intent maps to nothing in this paradigm's catalog,
    # fall back to the full list so user still sees something (better than an empty plan).
    return filtered if filtered else list(charts)


def _chart_to_plan(
    ch: ChartEntry, raw_files: list[str], workspace_dir: str,
    paradigm: str = "",                          # 1.1: 范式名，用于 --paradigm
    virtual_workspace_dir: str | None = None,
    virtual_outputs_dir: str | None = None,       # 1.1: outputs 虚拟路径
    groups: dict[str, Any] | None = None,         # 1.2: optional group labels for aggregate plots
) -> list[PlanChart]:
    """Build PlanCharts from a ChartEntry, materializing inputs.json (and optional groups.json).

    1.2: Uniform CLI contract — every plot script is invoked with ``--inputs <json_file>``,
    never ``--input <single_file>``. The catalog yaml decides how many PlanCharts are
    produced and what gets written into each inputs.json:

      - ``output_mode: per_subject`` (default): expand to N PlanCharts (one per raw_file).
        Each PlanChart materializes its own ``inputs_<chart_id>_s<idx>.json`` with a
        single-element array. Scripts read ``paths[0]`` via ``resolve_per_subject_input``.

      - ``output_mode: aggregate``: collapse to one PlanChart with ``inputs_<chart_id>.json``
        containing all raw_files. Used for cross-subject comparisons (box / bar /
        struggle distribution).

    When ``ch.needs_groups`` is True and groups are available, ``groups_<chart_id>.json`` is
    materialised alongside and added to args as ``--groups <path>``. The group JSON is in the
    canonical ``{group_name: [subject_path, ...]}`` shape expected by ``read_groups_json``.
    """
    if not raw_files:
        return []
    effective_outputs = virtual_outputs_dir or "/mnt/user-data/outputs"
    effective_workspace = virtual_workspace_dir or workspace_dir
    # Physical paths for files we actually write (the virtual variants go into args).
    physical_workspace = Path(workspace_dir)
    physical_workspace.mkdir(parents=True, exist_ok=True)

    def _materialise_inputs(json_name: str, paths: list[str]) -> str:
        """Write inputs JSON to physical workspace, return its virtual path for args."""
        physical_path = physical_workspace / json_name
        physical_path.write_text(json.dumps(paths, ensure_ascii=False), encoding="utf-8")
        return str(Path(effective_workspace) / json_name)

    def _materialise_groups(json_name: str, mapping: dict[str, list[str]]) -> str:
        physical_path = physical_workspace / json_name
        physical_path.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
        return str(Path(effective_workspace) / json_name)

    def _build_groups_payload(paths: list[str]) -> dict[str, list[str]] | None:
        """Reshape catalog/groups dict into {group_name: [subject_path, ...]}.

        Accepts two upstream shapes:
          - {arena_key: group_name}: assigns each raw_file to its group via arena_key match.
          - {group_name: [subject_id, ...]}: passed through verbatim (legacy).
        Returns None when groups can't be confidently mapped (avoids guessing).
        """
        if not groups:
            return None
        # Detect shape: values are str → {arena: group_name}; values are list → already final.
        if all(isinstance(v, str) for v in groups.values()):
            mapping: dict[str, list[str]] = {}
            # Heuristic: each raw_file path likely contains the arena key (e.g. "Arena 1").
            for arena_key, group_name in groups.items():
                bucket = mapping.setdefault(group_name, [])
                for p in paths:
                    if arena_key in Path(p).name:
                        bucket.append(p)
            # If nothing matched, fall back to None to avoid silent miscategorisation.
            if not any(mapping.values()):
                return None
            # Trim empty groups (defensive).
            return {k: v for k, v in mapping.items() if v}
        if all(isinstance(v, list) for v in groups.values()):
            return {k: [str(x) for x in v] for k, v in groups.items()}
        return None

    plans: list[PlanChart] = []

    if ch.output_mode == "aggregate":
        # One PlanChart aggregating all raw_files.
        inputs_name = f"inputs_{ch.id}.json"
        inputs_virtual = _materialise_inputs(inputs_name, raw_files)
        output_path = str(Path(effective_outputs) / f"plot_{ch.id}.png")
        args = ["--inputs", inputs_virtual]
        if ch.needs_groups:
            groups_payload = _build_groups_payload(raw_files)
            if groups_payload:
                groups_virtual = _materialise_groups(f"groups_{ch.id}.json", groups_payload)
                args.extend(["--groups", groups_virtual])
        args.extend(["--output", output_path])
        if ch.accepts_paradigm and paradigm:
            args.extend(["--paradigm", paradigm])
        plans.append(
            PlanChart(
                id=ch.id,
                script=ch.script,
                input=str(Path(effective_workspace) / inputs_name),
                output=output_path,
                subject_index=0,
                display_name_zh=ch.display_name_zh,
                args=args,
            )
        )
        return plans

    # per_subject: expand to N PlanCharts, one inputs.json per file.
    multi = len(raw_files) > 1
    for idx, raw_file in enumerate(raw_files):
        suffix = f"_s{idx}" if multi else ""
        inputs_name = f"inputs_{ch.id}{suffix}.json"
        inputs_virtual = _materialise_inputs(inputs_name, [raw_file])
        output_path = str(Path(effective_outputs) / f"plot_{ch.id}{suffix}.png")
        args = ["--inputs", inputs_virtual, "--output", output_path]
        if ch.accepts_paradigm and paradigm:
            args.extend(["--paradigm", paradigm])
        plans.append(
            PlanChart(
                id=ch.id,
                script=ch.script,
                input=str(Path(effective_workspace) / inputs_name),
                output=output_path,
                subject_index=idx,
                display_name_zh=ch.display_name_zh,
                args=args,
            )
        )
    return plans


def _stats_to_plan(
    st: StatisticsEntry, workspace_dir: str, skip_reason: str | None,
    virtual_workspace_dir: str | None = None,
) -> PlanStatistics:
    effective_workspace = virtual_workspace_dir or workspace_dir
    return PlanStatistics(
        id=st.id,
        script=st.script,
        input=str(Path(effective_workspace) / "handoff_code_executor.json"),
        output=str(Path(effective_workspace) / "stats.json"),
        skip_reason=skip_reason,
    )


def _evaluate_when(
    condition: str, *, n_per_group: int | None, n_groups: int | None,
    total_subjects: int | None = None,
    total_duration_seconds: float | None = None,
) -> bool:
    cond = condition.strip()
    if cond == "always":
        return True
    parts = [p.strip() for p in cond.split(" and ")]
    for part in parts:
        if not _evaluate_atomic_when(
            part, n_per_group=n_per_group, n_groups=n_groups, total_subjects=total_subjects,
            total_duration_seconds=total_duration_seconds,
        ):
            return False
    return True


def _evaluate_atomic_when(
    part: str, *, n_per_group: int | None, n_groups: int | None,
    total_subjects: int | None = None,
    total_duration_seconds: float | None = None,
) -> bool:
    tokens = part.split()
    if len(tokens) != 3:
        return False
    var, op, val_str = tokens
    if op not in (">=", ">"):
        return False
    try:
        val = float(val_str)
    except ValueError:
        return False
    if var == "n_per_group":
        return n_per_group is not None and (n_per_group >= val if op == ">=" else n_per_group > val)
    if var == "n_groups":
        return n_groups is not None and (n_groups >= val if op == ">=" else n_groups > val)
    if var == "total_subjects":
        return total_subjects is not None and (total_subjects >= val if op == ">=" else total_subjects > val)
    if var == "total_duration_seconds":
        return total_duration_seconds is not None and (total_duration_seconds >= val if op == ">=" else total_duration_seconds > val)
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
                "id": m.id,
                "script": m.script,
                "input": m.input,
                "output": m.output,
                "required": m.required,
                "reason": m.reason,
                "subject_index": m.subject_index,
                "display_name_zh": m.display_name_zh,
                # W27 (2026-05-27): 透传 catalog 判读 / 展示字段
                "unit_zh": m.unit_zh,
                "one_liner": m.one_liner,
                "output_unit": m.output_unit,
                "direction_for_anxiety": m.direction_for_anxiety,
                "statistical_default": m.statistical_default,
            }
            for m in plan.metrics
        ],
        "statistics": (
            None
            if plan.statistics is None
            else {
                "id": plan.statistics.id,
                "script": plan.statistics.script,
                "input": plan.statistics.input,
                "output": plan.statistics.output,
                "skip_reason": plan.statistics.skip_reason,
            }
        ),
        "charts": [
            {"id": c.id, "script": c.script, "input": c.input, "output": c.output, "subject_index": c.subject_index}
            for c in plan.charts
        ],
        "skipped": [
            {"id": s.id, "reason": s.reason, "detail": s.detail} for s in plan.skipped
        ],
        "notes": plan.notes,
    }


def plan_metrics_to_dict(pm: PlanMetrics) -> dict:
    """PlanMetrics dataclass → JSON-serializable dict."""
    return {
        "schema_version": pm.schema_version,
        "paradigm": pm.paradigm,
        "ev19_template": pm.ev19_template,
        "generated_at": pm.generated_at,
        "inputs": {
            "raw_files": pm.inputs.raw_files,
            "groups_file": pm.inputs.groups_file,
            "columns_file": pm.inputs.columns_file,
        },
        "metrics": [
            {
                "id": m.id,
                "script": m.script,
                "input": m.input,
                "output": m.output,
                "required": m.required,
                "reason": m.reason,
                "subject_index": m.subject_index,
                "display_name_zh": m.display_name_zh,
                # W27 (2026-05-27): 透传 catalog 判读 / 展示字段
                "unit_zh": m.unit_zh,
                "one_liner": m.one_liner,
                "output_unit": m.output_unit,
                "direction_for_anxiety": m.direction_for_anxiety,
                "statistical_default": m.statistical_default,
            }
            for m in pm.metrics
        ],
        "statistics": (
            None
            if pm.statistics is None
            else {
                "id": pm.statistics.id,
                "script": pm.statistics.script,
                "input": pm.statistics.input,
                "output": pm.statistics.output,
                "skip_reason": pm.statistics.skip_reason,
            }
        ),
        "skipped": [
            {"id": s.id, "reason": s.reason, "detail": s.detail} for s in pm.skipped
        ],
        "notes": pm.notes,
    }


def plan_charts_to_dict(pc: PlanCharts) -> dict:
    """PlanCharts dataclass → JSON-serializable dict."""
    return {
        "schema_version": pc.schema_version,
        "paradigm": pc.paradigm,
        "ev19_template": pc.ev19_template,
        "generated_at": pc.generated_at,
        "inputs": {
            "raw_files": pc.inputs.raw_files,
            "groups_file": pc.inputs.groups_file,
            "columns_file": pc.inputs.columns_file,
        },
        "charts": [
            {
                "id": c.id, "script": c.script, "input": c.input, "output": c.output,
                "subject_index": c.subject_index,
                "display_name_zh": c.display_name_zh,
                "args": c.args,
            }
            for c in pc.charts
        ],
        "charts_fallback_available": [
            {
                "id": c.id, "script": c.script, "input": c.input, "output": c.output,
                "subject_index": c.subject_index,
                "display_name_zh": c.display_name_zh,
                "args": c.args,
            }
            for c in pc.charts_fallback_available
        ],
        "skipped": [
            {"id": s.id, "reason": s.reason, "detail": s.detail} for s in pc.skipped
        ],
        "user_intent": pc.user_intent,
        "notes": pc.notes,
    }


if __name__ == "__main__":
    import sys
    from ethoinsight.catalog.cli import main

    sys.exit(main())
