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
import logging
from pathlib import Path
from typing import Any

from ethoinsight.catalog.loader import CatalogError, CommonCatalog, load_catalog, load_common_catalog
from ethoinsight.catalog.schema import (
    Catalog,
    AnonymousZoneOverride,
    ChartEntry,
    MetricEntry,
    ParamSpec,
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

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.1"

# Sentinel for "no alias matched" — distinct from a None alias value (which means ignore).
_UNSET = object()


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
    overrides: dict[str, float | int | str] | None = None,
    common_catalog: CommonCatalog | None = None,
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
        overrides=overrides,
        common_catalog=common_catalog,
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
    overrides: dict[str, float | int | str] | None = None,  # === Sprint 2b ===
    common_catalog: CommonCatalog | None = None,  # === Sprint 2b ===
) -> PlanMetrics:
    """生成 PlanMetrics dataclass（不含 charts）。失败抛 ResolveError。

    workspace_dir: 真实物理路径（用于脚本实际执行时的 input 引用等）。
    virtual_workspace_dir: plan.json output 字段使用的虚拟路径（面向 downstream subagent）。
                           未提供时兜底使用 workspace_dir（兼容旧调用方）。
    column_aliases: {原始列 or normalized: catalog 概念} — Sprint 1 通用列别名表。
                    在 columns 入口单点重映射后喂给所有下游消费入口。
                    含 alias → None/__ignore__ 的列从 columns 移除。

    Sprint 2b 新增:
        overrides: 用户参数覆盖。key=参数名,value=覆盖值。
        common_catalog: 含 shared_parameters,用于解析 metric.parameters_ref。
    """
    overrides = overrides or {}
    shared_params: dict[str, ParamSpec] = (
        common_catalog.shared_parameters.parameters if common_catalog else {}
    )
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

    # #6a 自派生组计数：groups_file 是分组事实的唯一真相源，组计数是它的投影。
    # 调用方漏传计数（prep_metric_plan 的 #6a bug）时从 groups_file 派生，
    # 使"有分组却 statistics={}"结构上不可能。显式入参仍优先（兼容 CLI 兜底/测试）。
    # 派生发生在 plan 期 resolve 内、单一评估点，不引入 runtime 第二评估路径。
    if groups_file and (n_per_group is None or n_groups is None):
        derived_npg, derived_ng = _derive_group_counts(groups_file, workspace_dir)
        if n_per_group is None:
            n_per_group = derived_npg
        if n_groups is None:
            n_groups = derived_ng

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

    # Sprint 1 column-semantics: only remove __ignore__ columns.
    # Physical column names are preserved in the columns list — column_aliases
    # is consulted on-the-fly by _missing_columns and _build_zone_aliases_overrides
    # instead of rewriting columns upfront.
    if column_aliases:
        columns = _apply_aliases(columns, column_aliases)

    # Inject zone param overrides from column_aliases before computing parameters_in_use.
    # This ensures center_zone / open_zones / light_zone get PHYSICAL column names,
    # not catalog concept names (e.g. "center" not "in_zone_center").
    zone_aliases_overrides = _build_zone_aliases_overrides(column_aliases, cat, overrides)
    if zone_aliases_overrides:
        overrides = {**overrides, **zone_aliases_overrides}

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
        missing = _missing_columns(m.requires_columns, columns, column_aliases)
        if missing:
            # Check for anonymous zone: missing pattern like in_zone_* but bare in_zone exists.
            # _detect_anonymous_zone returns:
            #   ResolveError  → zone_unnamed (anonymous zone, no override) — raise immediately
            #   True          → zone pattern resolved by override — proceed (skip column check)
            #   None          → genuine missing column — raise columns_missing below
            zone_check = _detect_anonymous_zone(missing, columns, overrides, cat.anonymous_zone_override)
            if isinstance(zone_check, ResolveError):
                raise zone_check
            if zone_check is None:
                # Genuine missing column (not a zone pattern, or bare in_zone doesn't exist)
                raise ResolveError(
                    code="columns_missing",
                    message=(
                        f"数据缺少指标 '{m.id}' 必需的列：{missing}。"
                        f"这通常意味着实验录制或导出设置不完整。"
                        f"请检查实验设计与导出配置后重新提供数据，不要在缺列情况下勉强分析。"
                    ),
                    details={
                        "metric": m.id,
                        "missing_patterns": missing,
                        "available_columns": columns,
                    },
                )
            # zone_check is True: anonymous zone pattern resolved by override — proceed
        # === Sprint 2b: compute parameters_in_use ===
        params_in_use = _compute_parameters_in_use(
            metric=m,
            shared_params=shared_params,
            paradigm_params=cat.paradigm_parameters.parameters,
            overrides=overrides,
            anonymous_zone_override=cat.anonymous_zone_override,
        )
        plan_metrics.extend(
            _metric_to_plan(
                m, raw_files, workspace_dir, required=True, reason="paradigm.default",
                virtual_workspace_dir=virtual_workspace_dir,
                parameters_in_use=params_in_use,
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
        missing = _missing_columns(m.requires_columns, columns, column_aliases)
        if missing:
            skipped.append(
                PlanSkipped(
                    id=m.id,
                    reason="columns.missing",
                    detail=f"User-included metric {m.id} skipped: missing columns {missing}.",
                )
            )
            continue
        # === Sprint 2b: compute parameters_in_use ===
        params_in_use = _compute_parameters_in_use(
            metric=m,
            shared_params=shared_params,
            paradigm_params=cat.paradigm_parameters.parameters,
            overrides=overrides,
            anonymous_zone_override=cat.anonymous_zone_override,
        )
        plan_metrics.extend(
            _metric_to_plan(
                m, raw_files, workspace_dir, required=False, reason="user.include",
                virtual_workspace_dir=virtual_workspace_dir,
                parameters_in_use=params_in_use,
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
                zone_overrides=zone_aliases_overrides,
            )
        else:
            # #6a 层② 兜底信号：区分两类 skip。
            # - groups_file=None 或计数可得但不满足门槛（如单组 n_groups=1）→ 正常 skip。
            # - groups_file 非空但计数仍为 None（派生/读取失败）→ "组计数不可得"，
            #   这是 bug 信号（哑故障→响亮），区别于"n 真的不足"。不直接 raise
            #   ResolveError（避免一次 groups.json 读取抖动整盘失败），但做成可 grep 的信号。
            if groups_file and (n_per_group is None or n_groups is None):
                skip_reason = (
                    f"groups_file 提供但组计数不可得（n_per_group={n_per_group}, n_groups={n_groups}）"
                    f"——通常是 groups.json 读取/格式问题，非样本量不足；统计被跳过需排查"
                )
                logger.warning("[resolve] %s groups_file=%s", skip_reason, groups_file)
            else:
                skip_reason = f"condition '{cat.statistics_default.when}' not met (n_per_group={n_per_group}, n_groups={n_groups})"
            plan_stats = _stats_to_plan(
                cat.statistics_default,
                workspace_dir,
                skip_reason=skip_reason,
                virtual_workspace_dir=virtual_workspace_dir,
                zone_overrides=zone_aliases_overrides,
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

    # Sprint 1 column-semantics: only remove __ignore__ columns.
    # Physical column names are preserved; column_aliases is consulted on-the-fly
    # by _missing_columns instead of rewriting columns upfront.
    if column_aliases:
        columns = _apply_aliases(columns, column_aliases)

    # === spec 2026-06-18: 列对齐对 plot 脚本生效 ===
    # 与 resolve_metrics 路径（L216）同源——复用同一个 _build_zone_aliases_overrides，
    # 把 column_aliases（HITL 对齐结果）投影成 zone 参数 dict（如 open_arm_zones=['open']），
    # 经 _chart_to_plan 注入每个 chart entry 的 --parameters-json。plot 脚本 parse_parameters
    # 后透传给底层 compute_*（per-subject）或 dispatcher zone_overrides（aggregate）。
    # 单一注入点：plot 脚本只接收+透传，绝不自读 context（避免再造 SSOT 副本）。
    zone_aliases_overrides = _build_zone_aliases_overrides(column_aliases, cat, {})

    charts: list[PlanChart] = []
    skipped: list[PlanSkipped] = []
    for ch in candidate_charts:
        missing = _missing_columns(ch.requires_columns, columns, column_aliases)
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
            # spec 2026-06-22: 按该 chart.requires_columns 声明的 zone 概念裁剪 zone_overrides，
            # 避免给签名严格的 compute 函数（只收 open_arm_zones）塞它不认识的 closed_arm_zones → TypeError。
            chart_zone_overrides = _filter_zone_overrides_for_chart(ch, zone_aliases_overrides, cat)
            charts.extend(_chart_to_plan(
                ch, raw_files, workspace_dir, paradigm=cat.paradigm,
                virtual_workspace_dir=virtual_workspace_dir,
                groups=groups,
                zone_overrides=chart_zone_overrides,
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
            missing = _missing_columns(ch.requires_columns, columns, column_aliases)
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
                # spec 2026-06-22: fallback 路径同主路径走裁剪（守两条路径一致）。
                chart_zone_overrides = _filter_zone_overrides_for_chart(ch, zone_aliases_overrides, cat)
                fallback.extend(_chart_to_plan(
                    ch, raw_files, workspace_dir, paradigm=cat.paradigm,
                    virtual_workspace_dir=virtual_workspace_dir,
                    groups=groups,
                    zone_overrides=chart_zone_overrides,
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


def _flatten_requires_columns(requires_columns) -> list[str]:
    """Flatten CNF requires_columns into a list of str (groups dissolved).

    Consumers that only care about "which patterns were mentioned"
    (zone concept derivation, inspect column collection, zone detection
    has_zone_pattern check) call this to avoid AttributeError/TypeError
    on nested list items.

    None / [] → []; pure str list → shallow copy (order preserved);
    nested items → inlined.
    """
    out: list[str] = []
    for item in requires_columns or []:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out


def _missing_columns(
    patterns: list[str | list[str]],
    available: list[str],
    column_aliases: dict[str, str] | None = None,
) -> list[str | list[str]]:
    """Check each entry in requires_columns against available columns.

    Each entry may be a str (single glob pattern) or a list[str] (OR-group:
    any sub-pattern match satisfies the group).  The returned list preserves
    the original shape: str for unmatched single patterns, list for unmatched
    OR-groups.

    If column_aliases is provided, physical column names not matching a
    requires glob are looked up via aliases (concept name matching).
    """
    missing: list[str | list[str]] = []
    for pat in patterns:
        if isinstance(pat, list):
            # OR-group: any sub-pattern satisfied → group satisfied
            group_ok = False
            for sub in pat:
                if any(fnmatch.fnmatchcase(col, sub) for col in available):
                    group_ok = True
                    break
                if column_aliases and _any_concept_matches_pattern(available, column_aliases, sub):
                    group_ok = True
                    break
            if group_ok:
                continue
            missing.append(pat)  # whole group into missing
            continue
        # Original str path — byte-identical to pre-CNF behaviour
        if any(fnmatch.fnmatchcase(col, pat) for col in available):
            continue
        if column_aliases:
            if _any_concept_matches_pattern(available, column_aliases, pat):
                continue
        missing.append(pat)
    return missing


def _any_concept_matches_pattern(
    available: list[str],
    column_aliases: dict[str, str],
    pat: str,
) -> bool:
    """任一 available 列的 alias 映射到匹配 pattern 的概念名。"""
    from ethoinsight.utils import normalize_column_name

    # Build normalized-key view for robust matching (same as _apply_aliases).
    norm_aliases: dict[str, str] = {}
    for k, v in column_aliases.items():
        norm_aliases.setdefault(normalize_column_name(k), v)

    # 裸 in_zone*（rstrip("*")=="in_zone"，无 concept keyword）是「存在分析 zone 列即可」的弱声明
    # （作者若要精确会写 in_zone_open* 像 metric 那样）。本函数只遍历 column_aliases 的对齐 concept
    # （已排除 None/__ignore__），所以只要有任一 available 列对齐到非忽略 concept，就说明存在真实 zone 列，
    # 弱声明即满足。修前 _concept_matches_pattern 对裸 in_zone* 做 pat[8:] 越界成空串→keyword=""
    # →短路 False，致 EZM/LDB/OFT 的 box/bar 图（requires_columns 用裸 in_zone*）经 alias 对齐后
    # 被误判缺列而 skip。挑对/算对是 chart 脚本层责任（脚本已用 aliases 对齐）。
    # 注意：此放行只作用于本「门」函数；_build_zone_aliases_overrides 的 zone 路由枚举不经过此函数，
    # 裸 in_zone* 在那里仍由 _extract_concept_keyword 判定无 keyword → 不参与具体 concept 路由。
    if pat.startswith("in_zone") and pat.rstrip("*") == "in_zone":
        for col in available:
            concept = column_aliases.get(col, _UNSET)
            if concept is _UNSET:
                concept = norm_aliases.get(normalize_column_name(col), _UNSET)
            if concept is not _UNSET and concept not in (None, "__ignore__"):
                return True
        return False

    for col in available:
        concept = column_aliases.get(col, _UNSET)
        if concept is _UNSET:
            concept = norm_aliases.get(normalize_column_name(col), _UNSET)
        if concept is _UNSET or concept in (None, "__ignore__"):
            continue
        if _concept_matches_pattern(concept, pat):
            return True
    return False


def _concept_matches_pattern(concept: str, pat: str) -> bool:
    """判断 catalog 概念名或概念关键词是否满足 requires_columns glob pattern。

    支持三种匹配层级：
    1. fnmatch 直接命中：concept="in_zone_center_point", pat="in_zone_center_*"
    2. 完整概念名：concept="in_zone_center", pat="in_zone_center_*"（semantic root of pattern）
    3. 概念关键词：concept="center", pat="in_zone_center_*"（LLM writes keywords like "center"）
    """
    if fnmatch.fnmatchcase(concept, pat):
        return True
    # pattern 去掉 wildcard 后缀 → 概念名的前缀
    pattern_base = pat.rstrip("*")
    if concept.startswith(pattern_base):
        return True
    # pat="in_zone_center_*" → base="in_zone_center"
    if concept == pattern_base.rstrip("_"):
        return True
    # 概念关键词匹配：extract keyword from pattern → compare with alias concept
    # "in_zone_center_*" → keyword "center"; "in_zone_open*" → keyword "open"
    if pat.startswith("in_zone") and "*" in pat:
        keyword = pat[len("in_zone_"):].rstrip("*").rstrip("_")
        if keyword and concept == keyword:
            return True
    return False


def _build_zone_aliases_overrides(
    column_aliases: dict[str, str] | None,
    cat: Catalog,
    existing_overrides: dict[str, float | int | str],
) -> dict[str, float | int | str | list[str]]:
    """从 column_aliases 提取 zone param overrides（多 concept 路由）。

    1. 从 zone_concept_params + anonymous_zone_override 收集 (concept → param, wrap_list) 映射
    2. 从 catalog entries 收集 zone_patterns（in_zone* requires_columns）
    3. 对 column_aliases 中匹配 zone_patterns 的物理列 → 按 concept 分组
    4. 每个 concept → 查映射得 (param, wrap_list) → 注入 overrides
    5. existing_overrides 含同名 target_param → 该 param 不覆盖（显式优先）
    """
    if not column_aliases:
        return {}

    # ── Step 1: 从 catalog 统一内部模型构建 (concept_keyword → param, wrap_list) 映射 ──
    concept_param_map: dict[str, tuple[str, bool]] = {}
    for concept_key, rc in cat.resolved_zone_concepts.items():
        if rc.binding is None:
            continue  # 无注入绑定的概念（如 Stage 3 OFT border）不进 param 路由 —— 语义本身
        concept_param_map[concept_key] = (rc.binding.param, rc.binding.wrap_list)

    azo = cat.anonymous_zone_override  # 仍保留：Step 3 物理列路由分支引用 azo

    if not concept_param_map:
        return {}

    # ── Step 2: 收集 zone_patterns ──
    zone_patterns: set[str] = set()
    entries = list(cat.default_metrics) + list(cat.optional_metrics) + list(cat.charts)
    for entry in entries:
        for pat in _flatten_requires_columns(getattr(entry, "requires_columns", [])):
            if pat.startswith("in_zone") and "*" in pat:
                zone_patterns.add(pat)

    if not zone_patterns:
        return {}

    # ── Step 3: 匹配物理列 → concept → 按 concept 分组 ──
    concept_cols: dict[str, list[str]] = {}
    for physical_col, concept in column_aliases.items():
        if concept in (None, "__ignore__"):
            continue
        # 尝试匹配 concept 到 zone pattern
        matched_concept: str | None = None
        # 先检查 concept 是否直接是 concept_param_map 的 key
        if concept in concept_param_map:
            matched_concept = concept
        else:
            # 检查 concept 是否匹配某个 zone_pattern
            for pat in zone_patterns:
                if _concept_matches_pattern(concept, pat):
                    # 从 pattern 提取 keyword 并检查是否在 concept_param_map 中
                    keyword = _extract_concept_keyword(pat)
                    if keyword and keyword in concept_param_map:
                        matched_concept = keyword
                        break
                    # Fallback: 直接用 concept 名作为 keyword
                    if concept in concept_param_map:
                        matched_concept = concept
                        break
            # 额外检查：concept 匹配 zone pattern 但 keyword 不在 map →
            # 用 azo 的 target_param（如果有）
            if matched_concept is None and azo is not None:
                for pat in zone_patterns:
                    if _concept_matches_pattern(concept, pat):
                        matched_concept = _derive_concept_from_zone_patterns(
                            zone_patterns, azo.target_param
                        )
                        if matched_concept and matched_concept in concept_param_map:
                            break
                        matched_concept = None

        if matched_concept is not None:
            concept_cols.setdefault(matched_concept, []).append(physical_col)

    if not concept_cols:
        return {}

    # ── Step 4 & 5: 按概念→参数映射注入 overrides，显式覆盖优先 ──
    overrides: dict[str, float | int | str | list[str]] = {}
    for concept_key, cols in concept_cols.items():
        mapping = concept_param_map.get(concept_key)
        if mapping is None:
            continue
        param_name, wrap_list = mapping
        # 显式覆盖优先：anonymous_zone_is 或同名 target_param 已在 existing_overrides 中
        if param_name in existing_overrides:
            continue
        if azo is not None and azo.target_param == param_name and "anonymous_zone_is" in existing_overrides:
            continue
        # 类型仅由 wrap_list 决定
        overrides[param_name] = cols if wrap_list else cols[0]

    return overrides


def _filter_zone_overrides_for_chart(
    ch: ChartEntry,
    zone_overrides: dict[str, Any],
    cat: Catalog,
) -> dict[str, Any]:
    """按 chart.requires_columns 声明的 zone 概念，把全局 zone_overrides 裁剪到该图子集。

    spec 2026-06-22：`resolve_charts` 拿到的 `zone_aliases_overrides` 是全范式 zone 参数并集
    （如 EPM 同时含 open_arm_zones + closed_arm_zones），无条件注入每张图。底层 compute 函数
    签名严格的图（compute_open_arm_time_ratio 只收 open_arm_zones）被塞了它不认识的
    closed_arm_zones → TypeError。chart 的 requires_columns 里 in_zone_<concept>_* glob 就是
    该图用到的 zone 概念的权威声明（loader 已校验），按它裁剪即可，不需新建 chart↔metric 映射。

    匹配复用 _build_zone_aliases_overrides 的同一套归一化（守 SSOT）：
    对每个可注入 concept_key（resolved_zone_concepts 中 binding 非空），用
    _concept_matches_pattern(concept_key, pat) 检查它是否命中 chart 任一 requires_columns
    pattern（与 _build_zone_aliases_overrides Step 3 同一匹配函数，三层归一化）。
    命中即放行其 binding.param。

    - chart 没声明任何 in_zone_* 概念（如 trajectory/heatmap）→ 返回 {}（不注入 --parameters-json）
    - chart 声明 in_zone_open_arms_* → 命中 concept=open_arms → 留 open_arm_zones
    - chart 声明 in_zone*（OFT/ZM 宽通配，无具体 concept）→ 匹配不到任何具体 concept，
      视为「未明确排除任何 zone」→ 不裁剪，返回原 zone_overrides（避免回归：OFT/ZM chart
      本来就靠全量注入工作，硬裁到空会让 center_zone/closed_zones 注入消失）
    - zone_overrides 为空（无 column_aliases）→ 直接返回 {}
    """
    if not zone_overrides:
        return {}
    # (concept_key → param)，仅 binding 非空的可注入概念（border 等无注入点的跳过）
    concept_to_param: dict[str, str] = {
        key: rc.binding.param
        for key, rc in cat.resolved_zone_concepts.items()
        if rc.binding is not None
    }
    if not concept_to_param:
        return {}
    chart_patterns = _flatten_requires_columns(getattr(ch, "requires_columns", []))
    # 仅 in_zone* glob 才表达 zone 概念依赖（坐标列/Direction 等不算）
    zone_patterns = [p for p in chart_patterns if p.startswith("in_zone") and "*" in p]
    if not zone_patterns:
        return {}
    # 宽通配（in_zone* 无具体 concept keyword）无法据 chart 裁剪 → 不裁剪，守旧行为
    if all(_extract_concept_keyword(p) is None for p in zone_patterns):
        return dict(zone_overrides)
    # 用 _concept_matches_pattern 双向匹配 concept_key ↔ chart pattern（复用注入端同一归一化）
    allowed_params: set[str] = set()
    for concept_key, param in concept_to_param.items():
        if any(_concept_matches_pattern(concept_key, pat) for pat in zone_patterns):
            allowed_params.add(param)
    if not allowed_params:
        return {}
    return {k: v for k, v in zone_overrides.items() if k in allowed_params}


def _extract_concept_keyword(pat: str) -> str | None:
    """从 zone pattern 提取概念关键词。"""
    if not pat.startswith("in_zone") or "*" not in pat:
        return None
    keyword = pat[len("in_zone_"):].rstrip("*").rstrip("_")
    return keyword if keyword else None


def _derive_concept_from_zone_patterns(
    zone_patterns: set[str],
    target_param: str,
) -> str | None:
    """从 zone_patterns 和 target_param 反向推导 concept keyword。

    例如：target_param="center_zone" + pattern "in_zone_center_*" → "center"
    """
    # Remove _zone / _zones suffix from target_param to guess concept keyword
    candidate = target_param.replace("_zones", "").replace("_zone", "")
    for pat in zone_patterns:
        keyword = _extract_concept_keyword(pat)
        if keyword and keyword in candidate:
            return keyword
    return None


def _apply_aliases(columns: list[str], aliases: dict[str, str]) -> list[str]:
    """移除用户标记为忽略的列。不再改写列为概念名（概念名保留在 column_aliases 中供下游查询）。

    Columns whose alias value is None or "__ignore__" are removed
    (user confirmed the column is irrelevant — D4).

    Returns a new list; the input is not mutated.
    """
    from ethoinsight.utils import normalize_column_name

    ignore_values = {None, "__ignore__"}

    # Build a normalized-key view of the alias dict for robust matching.
    norm_aliases: dict[str, str] = {}
    for k, v in aliases.items():
        norm_aliases.setdefault(normalize_column_name(k), v)

    result: list[str] = []
    for col in columns:
        # Match by verbatim key first, then by normalized key.
        if col in aliases:
            target = aliases[col]
        else:
            target = norm_aliases.get(normalize_column_name(col), _UNSET)
        if target is _UNSET:
            result.append(col)
            continue
        if target in ignore_values:
            continue
        result.append(col)
    return result


def _detect_anonymous_zone(
    missing_patterns: list[str | list[str]],
    available_columns: list[str],
    overrides: dict[str, str],
    anonymous_zone_override: AnonymousZoneOverride | None = None,
) -> ResolveError | bool | None:
    """Detect anonymous zone: missing pattern is a named in_zone variant
    (e.g. in_zone_center_*) but bare in_zone exists in data.

    Returns:
        ResolveError — zone_unnamed: anonymous zone detected, no override declared
        True         — zone pattern detected but override resolves it (proceed)
        None         — not a zone pattern, or bare in_zone doesn't exist
    """
    # Flatten to guard against list items in missing (CNF OR-groups)
    flat_missing = _flatten_requires_columns(missing_patterns)
    # Check if any missing pattern is an in_zone variant (has suffix beyond "in_zone")
    has_zone_pattern = any(
        pat.startswith("in_zone") and len(pat) > len("in_zone")
        for pat in flat_missing
    )
    if not has_zone_pattern:
        return None

    # Bare in_zone must exist in available columns
    if "in_zone" not in available_columns:
        return None

    # Paradigm must declare anonymous_zone_override for zone_unnamed to trigger.
    # Paradigms without it (EPM/FST/TST) fall through to columns_missing.
    if anonymous_zone_override is None:
        return None

    # User has provided the unified key → override resolves the zone.
    if "anonymous_zone_is" in overrides:
        return True

    return ResolveError(
        code="zone_unnamed",
        message=(
            "检测到一个未命名的分析区(in_zone)，但指标需要命名区域列"
            f"（缺失: {missing_patterns}）。"
            "请用 ask_clarification 反问用户该区域代表什么。"
        ),
        details={
            "found_column": "in_zone",
            "missing_patterns": missing_patterns,
            "available_columns": available_columns,
        },
    )


def _metric_to_plan(
    m: MetricEntry,
    raw_files: list[str],
    workspace_dir: str,
    *,
    required: bool,
    reason: str,
    virtual_workspace_dir: str | None = None,
    parameters_in_use: dict[str, float | int | str] | None = None,  # === Sprint 2b ===
) -> list[PlanMetric]:
    """Expand one MetricEntry into N PlanMetric (one per raw_file).

    Fix 2026-05-20 (FST E2E): 之前只生成单个 PlanMetric 且 input=raw_files[0],
    用户上传多文件时除第一个外的 subject 全部丢失。现在按 raw_files 展开,
    每个 subject 一个 PlanMetric,output 用 subject_index 后缀避免覆盖。
    单文件(len==1)保持 output 名 m_<id>.json 兼容现有产物。

    W27 (2026-05-27): 透传 catalog 判读 / 展示字段到 PlanMetric,subagent 不再 read catalog YAML。
    详见 docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md

    Sprint 2b: parameters_in_use 填入 PlanMetric; 如非空则追加 --parameters-json 到 args。
    """
    if not raw_files:
        return []
    effective_workspace = virtual_workspace_dir or workspace_dir
    multi = len(raw_files) > 1
    params_in_use = parameters_in_use or {}
    plans: list[PlanMetric] = []
    for idx, raw_file in enumerate(raw_files):
        suffix = f"_s{idx}" if multi else ""
        output_path = str(Path(effective_workspace) / f"m_{m.id}{suffix}.json")
        # === Sprint 2b: build args with --parameters-json ===
        args: list[str] = ["--input", raw_file, "--output", output_path]
        if params_in_use:
            params_json = json.dumps(params_in_use, ensure_ascii=False, sort_keys=True)
            args.extend(["--parameters-json", params_json])
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
                parameters_in_use=params_in_use,  # === Sprint 2b ===
                args=args,  # === Sprint 2b: CLI args with --parameters-json ===
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

    **Open-ended markers ("等" / "之类" / "包括" / "such as" / "etc")**: when the intent names a
    chart type but qualifies it as a non-exhaustive example (e.g. "箱线图/小提琴图**等**"), the
    chart-type keyword is treated as a *hint, not a hard constraint* — all charts pass through.
    Hard-filtering on "box" in that case would silently cut every other chart (dogfood 2026-06-18:
    EPM 6 charts → 1 because the dispatch prompt paraphrased an open intent as "箱线图/小提琴图等").
    """
    if not user_intent:
        return list(charts)
    intent_lower = user_intent.lower()

    # Open-ended markers: the named chart type is an *example*, not the whole ask. Don't hard-filter.
    open_ended_markers = ("等", "之类", "包括", "诸如", "such as", "etc", "e.g", "for example", "including")
    is_open_ended = any(m in user_intent or m in intent_lower for m in open_ended_markers)

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
    if is_open_ended:
        # Keyword present but open-ended → keep all charts (the type is a priority hint only).
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
    zone_overrides: dict[str, Any] | None = None,  # spec 2026-06-18: column_aliases 投影出的 zone 参数
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

        Accepts these upstream shapes:
          - {subject_file: group_name}: the SSOT flat map written by ``prep_metric_plan`` /
            ``prep_chart_plan`` (key = the exact ``raw_files`` virtual path, optionally with a
            ``::sheet`` suffix). Inverted by **exact path identity**, mirroring
            ``scripts._cli.read_groups_json`` so the charts path and the metrics path consume
            the same SSOT identically (P3, 红线二：单一来源 + 同一消费语义).
          - {arena_key: group_name}: legacy short-key form (e.g. ``"Arena 1"``); matched by the
            ``arena_key in filename`` substring heuristic only as a fallback when no exact path
            match succeeds — so a real SSOT map is never silently misbucketed by substring noise.
          - {group_name: [subject_id, ...]}: already-final shape, passed through verbatim.
        Returns None when groups can't be confidently mapped (avoids guessing).
        """
        if not groups:
            return None
        # Already-final shape: values are lists.
        if all(isinstance(v, list) for v in groups.values()):
            return {k: [str(x) for x in v] for k, v in groups.items()}
        # Flat map: values are group-name strings. Two key conventions, exact-path first.
        if all(isinstance(v, str) for v in groups.values()):
            # Index raw_files by full virtual path and by ::sheet-stripped path so SSOT keys
            # (which equal the raw_files paths) match by identity, not by substring.
            by_full = {p: p for p in paths}
            by_clean = {p.split("::", 1)[0]: p for p in paths}

            mapping: dict[str, list[str]] = {}
            unmatched_keys: list[tuple[str, str]] = []
            for key, group_name in groups.items():
                matched = by_full.get(key) or by_clean.get(key.split("::", 1)[0] if "::" in key else key)
                if matched is not None:
                    mapping.setdefault(group_name, []).append(matched)
                else:
                    unmatched_keys.append((key, group_name))

            # Legacy fallback for keys that were NOT exact raw_file paths (e.g. "Arena 1"):
            # substring-match against filenames. Applied per-unmatched-key so an SSOT map that
            # already matched exactly is never reprocessed by the fuzzy heuristic.
            for key, group_name in unmatched_keys:
                for p in paths:
                    if key in Path(p).name:
                        mapping.setdefault(group_name, []).append(p)

            if not any(mapping.values()):
                return None
            # De-dup while preserving order (a path could match both exact + fuzzy in theory).
            return {g: list(dict.fromkeys(v)) for g, v in mapping.items() if v}
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
        if zone_overrides:
            args.extend(["--parameters-json", json.dumps(zone_overrides, ensure_ascii=False, sort_keys=True)])
        plans.append(
            PlanChart(
                id=ch.id,
                script=ch.script,
                input=str(Path(effective_workspace) / inputs_name),
                output=output_path,
                subject_index=0,
                display_name_zh=ch.display_name_zh,
                confidence=ch.confidence,
                args=args,
                output_mode=ch.output_mode,
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
        if zone_overrides:
            args.extend(["--parameters-json", json.dumps(zone_overrides, ensure_ascii=False, sort_keys=True)])
        plans.append(
            PlanChart(
                id=ch.id,
                script=ch.script,
                input=str(Path(effective_workspace) / inputs_name),
                output=output_path,
                subject_index=idx,
                source_filename=Path(raw_file).name,  # basename，保留原文件名（含多空格）；spec 2026-06-29
                display_name_zh=ch.display_name_zh,
                confidence=ch.confidence,
                args=args,
                output_mode=ch.output_mode,
            )
        )
    return plans


def _stats_to_plan(
    st: StatisticsEntry, workspace_dir: str, skip_reason: str | None,
    virtual_workspace_dir: str | None = None,
    zone_overrides: dict[str, list[str] | str] | None = None,
) -> PlanStatistics:
    effective_workspace = virtual_workspace_dir or workspace_dir
    return PlanStatistics(
        id=st.id,
        script=st.script,
        input=str(Path(effective_workspace) / "handoff_code_executor.json"),
        output=str(Path(effective_workspace) / "stats.json"),
        skip_reason=skip_reason,
        # statistics 段复用 metrics 段同一份 zone_aliases_overrides 的投影（SSOT，
        # spec 2026-06-16）。默认空 dict 向后兼容；有 column_aliases 时透传给
        # run_groupwise_stats → dispatcher。
        parameters=dict(zone_overrides) if zone_overrides else {},
    )


def _derive_group_counts(
    groups_file: str | None, workspace_dir: str | None = None
) -> tuple[int | None, int | None]:
    """从 groups_file 派生 (n_per_group=最小组 size, n_groups=组数)。

    groups 是分组事实的唯一真相源；组计数是它的投影。让 resolve 在评估点就近
    自派生计数，使"有分组却漏传组计数"结构上不可能（#6a 根因：prep_metric_plan
    调 resolve_metrics 传了 groups_file 但漏传 n_per_group/n_groups → gate 用 None
    评 False → statistics 静默 skip → handoff statistics={} 毒化 data-analyst）。

    groups.json 两种形状都支持：
      - {subject_path: group_name}（prep_metric_plan 写的正向映射）→ Counter(values)
      - {group_name: [subject_path, ...]}（read_groups_json/charts 的反向）→ {g: len(lst)}
    None/读不到/空/形状不可识别 → (None, None)（单组或无分组场景，gate 正确 skip，
    行为与修复前等价）。fail-safe：读不到不阻断（不引入新失败模式）。

    路径解析（收口到机制 B 的 workspace_base 兜底，不在 resolve 里维护第二套兜底逻辑）：
      调 ``resolve_sandbox_path(groups_file, workspace_base=workspace_dir)``——它先试
      env（沙箱子进程路径），无 env 时用 workspace_base 拼 workspace 前缀虚拟路径的
      真实物理位置。若解析结果不存在（如 groups_file 本就是物理路径或非 workspace 前缀），
      再退回把 groups_file 直接当物理路径读。两个候选覆盖所有调用形态（虚拟+env /
      虚拟+无env+workspace_dir / 物理路径），同一处方收口（2026-06-16 故障族根治改动②）。
    """
    if not groups_file:
        return (None, None)
    data = None
    try:
        from ethoinsight.scripts._cli import resolve_sandbox_path

        candidates: list[Path] = [
            Path(resolve_sandbox_path(groups_file, workspace_base=workspace_dir)),
            Path(groups_file),
        ]
        for cand in candidates:
            if cand.exists():
                data = json.loads(cand.read_text(encoding="utf-8"))
                break
    except Exception:
        return (None, None)  # fail-safe：读不到/坏 JSON 不阻断（与修复前等价）
    if not isinstance(data, dict) or not data:
        return (None, None)
    vals = list(data.values())
    if all(isinstance(v, str) for v in vals):  # {subject: group}
        from collections import Counter

        counts = Counter(vals)
    elif all(isinstance(v, list) for v in vals):  # {group: [subjects]}
        counts = {k: len(v) for k, v in data.items()}
    else:
        return (None, None)  # 形状不可识别（保守不猜）
    if not counts:
        return (None, None)
    return (min(counts.values()), len(counts))


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
# Sprint 2b: parameter resolution helpers
# ============================================================================


def _compute_parameters_in_use(
    metric: MetricEntry,
    shared_params: dict[str, ParamSpec],
    paradigm_params: dict[str, ParamSpec],
    overrides: dict[str, float | int | str],
    anonymous_zone_override: AnonymousZoneOverride | None = None,
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
            continue
        result[ref_name] = spec.default

    # 2. paradigm_params — only inject computation-related params (not dispatcher-only ones)
    if _metric_uses_pendulum(metric):
        for pname, pspec in paradigm_params.items():
            if pname.startswith("pendulum_"):
                result[pname] = pspec.default
        # Also inject pendulum_* from shared_params if not already set
        for pname, pspec in shared_params.items():
            if pname.startswith("pendulum_") and pname not in result:
                result[pname] = pspec.default
    # For velocity-based immobility (used by FST/TST/EPM/OFT/LDB/Zero Maze):
    # inject velocity_threshold / velocity_min_duration if the metric uses immobility
    if _metric_uses_velocity_immobility(metric):
        for pname, pspec in paradigm_params.items():
            if pname.startswith("velocity_"):
                result[pname] = pspec.default
        # Also inject velocity_* from shared_params if not already set
        for pname, pspec in shared_params.items():
            if pname.startswith("velocity_") and pname not in result:
                result[pname] = pspec.default

    # 3. metric.parameters (单 metric 独有)
    for pname, pspec in metric.parameters.items():
        result[pname] = pspec.default

    # Translate unified zone key → paradigm's real param, before replace-only loop.
    # This is where list wrapping happens (zero_maze open_zones requires list[str]).
    effective_overrides = dict(overrides)
    azo = anonymous_zone_override
    if azo is not None and "anonymous_zone_is" in effective_overrides:
        val = effective_overrides.pop("anonymous_zone_is")
        effective_overrides[azo.target_param] = [val] if azo.wrap_list else val

    # 4. overrides (最高优先级)
    for pname, override_val in effective_overrides.items():
        if pname in result:
            result[pname] = override_val

    return result


def _metric_uses_pendulum(metric: MetricEntry) -> bool:
    """启发式:metric.script 含 'fst' 或 'tst' 且 metric.id 含 immobility/struggle/pendulum/activity_intensity。"""
    script_lower = metric.script.lower()
    id_lower = metric.id.lower()
    is_swim_test = ".fst." in script_lower or ".tst." in script_lower
    is_pendulum_metric = any(kw in id_lower for kw in ["immobility", "struggle", "pendulum", "activity_intensity"])
    return is_swim_test and is_pendulum_metric


def _metric_uses_velocity_immobility(metric: MetricEntry) -> bool:
    """启发式:metric.id 含 immobility 时需要 velocity 参数（用于 velocity-based immobility fallback）。"""
    id_lower = metric.id.lower()
    return "immobility" in id_lower


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
                # Sprint 2b: 参数 + CLI args
                "parameters_in_use": m.parameters_in_use,
                "args": m.args,
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
                "parameters": plan.statistics.parameters,
            }
        ),
        "charts": [
            {"id": c.id, "script": c.script, "input": c.input, "output": c.output,
             "subject_index": c.subject_index, "source_filename": c.source_filename}
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
                # Sprint 2b: 参数 + CLI args
                "parameters_in_use": m.parameters_in_use,
                "args": m.args,
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
                "parameters": pm.statistics.parameters,
            }
        ),
        "skipped": [
            {"id": s.id, "reason": s.reason, "detail": s.detail} for s in pm.skipped
        ],
        "notes": pm.notes,
    }


def metric_metadata_to_dict(pm: PlanMetrics) -> dict:
    """去重的展示/判读元数据投影（按 metric id 一条）。

    spec 2026-06-22-metric-metadata-sidecar：plan_metrics.json 的 metrics[] 按 subject
    重复（EPM 28 subject × 5 metric = 140 条，133K），report-writer / data-analyst 只需
    展示+判读元数据（5 条），啃施工文件会撑爆 thinking 撞 turn 超时。本函数生成去重投影，
    落盘为 _metric_metadata.json 供 subagent 单次 read + 按 id 直查。

    守 SSOT：元数据 SSOT 仍是 catalog（MetricEntry 字段），plan_metrics.json 是 catalog 的
    施工投影，本函数是 plan 的去重元数据投影（投影的投影）。去重源是 pm.metrics（resolve 后
    实际包含的 metric），反映 plan 真实集合（plan 会因列缺失 skip metric，旁路同步 skip），
    不是 catalog 全集。
    """
    metrics_meta: dict[str, dict] = {}
    for m in pm.metrics:
        if m.id in metrics_meta:
            continue  # 去重，取首个（所有同 id 行元数据一致）
        metrics_meta[m.id] = {
            "display_name_zh": m.display_name_zh,
            "unit_zh": m.unit_zh,
            "one_liner": m.one_liner,
            "output_unit": m.output_unit,
            "direction_for_anxiety": m.direction_for_anxiety,
            "statistical_default": m.statistical_default,
        }
    return {"paradigm": pm.paradigm, "metrics": metrics_meta}


def select_charts_by_priority(
    charts: list[PlanChart],
    budget: int | None = None,
    group_of: dict[int, str] | None = None,
) -> tuple[list[PlanChart], list[PlanChart]]:
    """按图类型定优先级选图（spec 2026-06-17-chart-budget-by-type）。

    旧预算是"数量上限"，按数组顺序取前 N —— per_subject 个体图（trajectory/heatmap，
    N×类 张）会先吃光名额，挤掉组间对比图（box/bar/rose，aggregate，最有分析价值）。

    新规则（确定性，不靠 LLM 判断"哪张更重要"）：

    1. **aggregate 图全留**（box/bar/rose/zone_distribution 等组间对比，通常 ≤ 5 张，
       是分析核心），不受预算限制。
    2. **per_subject 图受预算限制**：aggregate 画完后用剩余预算画**代表性子集**：
       - ``group_of`` 给定（subject_index→组名）时，**按组轮转**：每组首个 subject
         的全部图先入选，再每组第二个，以此类推——保证子集对每组都有代表
         （dogfood 实证 bug thread 339512dd：前 N 个 subject 同组时，纯 subject_index
         排序会让子集偏向排在前面的组，treatment 一张图都摊不到）。
       - ``group_of`` 为 None（无分组信息）时退回旧行为：纯按 subject_index 升序轮转。
    3. budget=None 表示不限制（全画），remaining=[]。

    返回 ``(selected, remaining)``：
    - ``selected`` 内部顺序：aggregate 在前（chart-maker 按数组顺序执行时 aggregate 先跑），
      per_subject 代表子集在后。
    - ``remaining`` 是被预算截断的 per_subject 图（降级指纹，红线一：挤掉要留痕）。
    - 稳定性：相同输入恒定相同输出（排序键无随机性）。
    """
    if budget is None or budget <= 0:
        return list(charts), []

    aggregate = [c for c in charts if c.output_mode == "aggregate"]
    per_subject = [c for c in charts if c.output_mode != "aggregate"]

    # aggregate 全画优先（核心：组间对比不受预算挤占）。
    selected: list[PlanChart] = list(aggregate)

    # per_subject 预算 = 总预算 - aggregate 占用（floor 0，aggregate 超预算时全画 aggregate）。
    per_subject_budget = max(0, budget - len(aggregate))

    if group_of:
        # 组内排名：每组的 subject_index 升序排位（control 的 idx 0→rank 0, idx 1→rank 1...；
        # treatment 的 idx 7→rank 0, idx 8→rank 1...）。按 (组内排名, 组名, id, script) 排序
        # → 各组首个 subject 各类图先入选，再各组第二个，轮转。组名进键保证确定性。
        rank_within_group: dict[int, int] = {}
        for grp in {group_of.get(c.subject_index) for c in per_subject}:
            members = sorted({c.subject_index for c in per_subject if group_of.get(c.subject_index) == grp})
            for rank, idx in enumerate(members):
                rank_within_group[idx] = rank

        def _key(c: PlanChart) -> tuple:
            return (
                rank_within_group.get(c.subject_index, c.subject_index),
                str(group_of.get(c.subject_index, "")),
                c.id,
                c.script,
            )

        ordered = sorted(per_subject, key=_key)
    else:
        # 无分组信息：纯 subject_index 升序轮转（向后兼容，老 thread 行为不变）。
        ordered = sorted(per_subject, key=lambda c: (c.subject_index, c.id, c.script))

    selected_per_subject = ordered[:per_subject_budget]
    remaining_per_subject = ordered[per_subject_budget:]

    selected.extend(selected_per_subject)
    return selected, remaining_per_subject


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
                "source_filename": c.source_filename,
                "display_name_zh": c.display_name_zh,
                "confidence": c.confidence,
                "output_mode": c.output_mode,
                "args": c.args,
            }
            for c in pc.charts
        ],
        "charts_fallback_available": [
            {
                "id": c.id, "script": c.script, "input": c.input, "output": c.output,
                "subject_index": c.subject_index,
                "source_filename": c.source_filename,
                "display_name_zh": c.display_name_zh,
                "confidence": c.confidence,
                "output_mode": c.output_mode,
                "args": c.args,
            }
            for c in pc.charts_fallback_available
        ],
        "charts_budget_remaining": [
            {
                "id": c.id, "script": c.script, "input": c.input, "output": c.output,
                "subject_index": c.subject_index,
                "source_filename": c.source_filename,
                "display_name_zh": c.display_name_zh,
                "confidence": c.confidence,
                "output_mode": c.output_mode,
                "args": c.args,
            }
            for c in pc.charts_budget_remaining
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
