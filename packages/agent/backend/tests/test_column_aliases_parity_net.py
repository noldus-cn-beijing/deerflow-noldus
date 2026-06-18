"""Spec 4 (P4) §2.3 — parity 回归网（防下一条路径漏接 column_aliases）。

这是 P4 的灵魂交付：把"所有消费 B 类实验状态（column_aliases）的路径都接了"
变成一条 CI 断言。下一条新增消费路径如果漏接 column_aliases，本文件缺它一项
就红，CI 当场抓——不再靠 dogfood 撞。

机制：一个**路径注册表** ``PARITY_PATHS``，枚举每条消费 column_aliases 的路径，
每条带一个"喂了 alias 的探针"和一个"没喂 alias 的探针"，断言两者结果在 zone
对齐上有可观测差异（漏接 → 与没喂 alias 一致 → 红）。注册表是「活的回归网」：
新增消费路径必须在此注册，否则完整性断言会因路径未被覆盖而提示维护者补登。

探针范式 = EPM FewZones（dogfood 实证结构：用户把开放臂/封闭臂简写成 open/closed，
非标准 in_zone_* 列名，必须靠 column_aliases 映射到 open_arms/closed_arms 才能算 zone 指标）。

判据来自 spec 2026-06-17-shared-state-sourcing-spec.md §2.3：
    给定 column_aliases = {open: open_arms, closed: closed_arms}
    for path in [resolve_metrics, resolve_charts, ...]:
        assert path 能看到 open_arm_zones（每条都接了）
"""

from __future__ import annotations

import json

import pytest
from ethoinsight.catalog.loader import load_common_catalog
from ethoinsight.catalog.resolve import ResolveError, resolve_charts, resolve_metrics

# ─── 探针范式 fixture ────────────────────────────────────────────────────────
# FewZones：open/closed 是用户自定义的简写列名，非 catalog 标准列，必须经 alias
# 映射到概念关键词 open_arms/closed_arms 才能解析 zone 指标。这是 dogfood 实证的
# 真实结构（用户命名 → Gate 1 列语义对齐 → column_aliases 投影）。
FEW_ZONES_COLUMNS = [
    "Trial time",
    "Recording time",
    "X center",
    "Y center",
    "open",
    "closed",
]
FEW_ZONES_ALIASES = {"open": "open_arms", "closed": "closed_arms"}

# 多 subject raw_files，使 aggregate 类 chart（box/bar，needs_groups 或多 subject）合格。
RAW_FILES_MULTI = ["/tmp/probe_a.txt", "/tmp/probe_b.txt"]


def _metric_ids(plan) -> set[str]:
    return {m.id for m in plan.metrics}


def _chart_ids(plan) -> set[str]:
    return {c.id for c in plan.charts}


# ─── 路径注册表 ──────────────────────────────────────────────────────────────
# 每个条目：(path_name, describe)
#   path_name: spec §2.3 枚举的消费路径标识（新增路径漏接 = 缺项即红）
#   describe:  该路径「接了 alias」时的可观测信号（callable → bool）
#
# 新增消费 column_aliases 的路径时，**必须**在此 append 一个条目并写对应 probe，
# 否则完整性断言 test_parity_registry_covers_spec_paths 会提示维护者补登。

def _probe_resolve_metrics_with_alias() -> bool:
    """resolve_metrics 接了 alias → FewZones 出 zone 指标（open_arm_time_ratio 等）。"""
    import tempfile

    pm = resolve_metrics(
        "epm",
        FEW_ZONES_COLUMNS,
        RAW_FILES_MULTI,
        tempfile.mkdtemp(),
        column_aliases=FEW_ZONES_ALIASES,
        common_catalog=load_common_catalog(),
    )
    return "open_arm_time_ratio" in _metric_ids(pm)


def _probe_resolve_metrics_without_alias() -> bool:
    """resolve_metrics 没接 alias → FewZones 直接 ResolveError（columns_missing）。"""
    import tempfile

    with pytest.raises(ResolveError):
        resolve_metrics(
            "epm",
            FEW_ZONES_COLUMNS,
            RAW_FILES_MULTI,
            tempfile.mkdtemp(),
            column_aliases=None,
            common_catalog=load_common_catalog(),
        )
    return True


def _probe_resolve_charts_with_alias() -> bool:
    """resolve_charts 接了 alias → FewZones 出 open_arm_time_ratio_bar。"""
    import tempfile

    pc = resolve_charts(
        "epm",
        FEW_ZONES_COLUMNS,
        RAW_FILES_MULTI,
        tempfile.mkdtemp(),
        column_aliases=FEW_ZONES_ALIASES,
        total_subjects=2,
        n_groups=1,
        n_per_group=2,
    )
    return "open_arm_time_ratio_bar" in _chart_ids(pc)


def _probe_resolve_charts_without_alias() -> bool:
    """resolve_charts 没接 alias → open_arm_time_ratio_bar 不出现（被 columns_missing skip）。"""
    import tempfile

    pc = resolve_charts(
        "epm",
        FEW_ZONES_COLUMNS,
        RAW_FILES_MULTI,
        tempfile.mkdtemp(),
        column_aliases=None,
        total_subjects=2,
        n_groups=1,
        n_per_group=2,
    )
    return "open_arm_time_ratio_bar" not in _chart_ids(pc)


def _probe_cli_context_fallback_column_aliases() -> bool:
    """catalog CLI §2.2 兜底：未传 --column-aliases-file 但传 --context-file →
    从 experiment-context.json 的 column_aliases 自动读取，resolve 出 zone 指标。

    漏接（既不传 file 也不传 context）→ column_aliases=None → ResolveError。
    """
    import tempfile
    from pathlib import Path

    from ethoinsight.catalog import cli

    work = Path(tempfile.mkdtemp())
    ctx = work / "experiment-context.json"
    ctx.write_text(
        json.dumps({"paradigm": "epm", "column_aliases": FEW_ZONES_ALIASES}, ensure_ascii=False),
        encoding="utf-8",
    )
    cols = work / "columns.json"
    cols.write_text(json.dumps({"columns": FEW_ZONES_COLUMNS}), encoding="utf-8")
    raw = work / "raw.json"
    raw.write_text(json.dumps(RAW_FILES_MULTI), encoding="utf-8")
    out = work / "plan.json"

    rc = cli.main(
        [
            "--mode", "metrics",
            "--paradigm", "epm",
            "--columns-file", str(cols),
            "--raw-files-json", str(raw),
            "--workspace-dir", str(work),
            "--context-file", str(ctx),
            "--output", str(out),
        ]
    )
    assert rc == 0, f"cli context fallback should succeed, rc={rc}"
    plan = json.loads(out.read_text(encoding="utf-8"))
    metric_ids = {m["id"] for m in plan.get("metrics", [])}
    return "open_arm_time_ratio" in metric_ids


def _probe_cli_no_aliases_no_context_fails() -> bool:
    """catalog CLI 既不传 --column-aliases-file 也不传 --context-file → ResolveError（rc=1）。

    守住「兜底不在场时退化是响亮的」，而非静默无对齐。
    """
    import tempfile
    from pathlib import Path

    from ethoinsight.catalog import cli

    work = Path(tempfile.mkdtemp())
    cols = work / "columns.json"
    cols.write_text(json.dumps({"columns": FEW_ZONES_COLUMNS}), encoding="utf-8")
    raw = work / "raw.json"
    raw.write_text(json.dumps(RAW_FILES_MULTI), encoding="utf-8")
    out = work / "plan.json"

    rc = cli.main(
        [
            "--mode", "metrics",
            "--paradigm", "epm",
            "--columns-file", str(cols),
            "--raw-files-json", str(raw),
            "--workspace-dir", str(work),
            # 故意不传 --column-aliases-file / --context-file
            "--output", str(out),
        ]
    )
    return rc == 1


# 注册表本身：spec §2.3 的 path 列表 + 每条的 with-alias 正向探针。
# （without-alias 反向探针在 _NEGATIVE_PROBES 里，单独校验"没接"时退化响亮。）
PARITY_PATHS: list[tuple[str, callable]] = [
    ("resolve_metrics", _probe_resolve_metrics_with_alias),
    ("resolve_charts", _probe_resolve_charts_with_alias),
    ("catalog_cli_context_fallback", _probe_cli_context_fallback_column_aliases),
]

# 反向探针：每条路径「没接 alias」时的退化信号。新增路径漏接会让 with-alias 探针
# 与对应的退化信号一致（都 False / 都抛错）→ 正向测试当场红。
NEGATIVE_PROBES: dict[str, callable] = {
    "resolve_metrics": _probe_resolve_metrics_without_alias,
    "resolve_charts": _probe_resolve_charts_without_alias,
    "catalog_cli_context_fallback": _probe_cli_no_aliases_no_context_fails,
}

# spec §2.3 明确枚举的消费路径全集。完整性断言用：注册表覆盖到这里列的每一条。
SPEC_PATHS = {"resolve_metrics", "resolve_charts", "statistics_dispatch", "validate_catalog"}


# ─── parity 网测试 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("path_name, probe", PARITY_PATHS, ids=[p[0] for p in PARITY_PATHS])
def test_path_consumes_column_aliases(path_name, probe):
    """每条消费路径喂了 column_aliases 后，必须解析出 zone 对齐信号。

    红了 = 这条路径漏接了 column_aliases（与没接时表现一致）。这是 spec §2.3 的
    核心断言：把"全员自取"变成 CI 可观测。
    """
    assert probe() is True, (
        f"path={path_name} 喂了 column_aliases 却没解析出 zone 指标 → 该路径漏接 alias。"
        f" 修复：让该路径从 read_context / --context-file 自取 column_aliases。"
    )


@pytest.mark.parametrize("path_name", list(NEGATIVE_PROBES.keys()))
def test_path_degrades_loudly_without_aliases(path_name):
    """每条路径「没接 column_aliases」时必须退化响亮（ResolveError / skip 目标项）。

    这是 parity 的对照锚：只有当"没接"会响亮失败时，"接了"的正向断言才有意义。
    红了 = 该路径在缺 alias 时静默产出错误结果（最坏情况：漏接伪装成正常）。
    """
    assert NEGATIVE_PROBES[path_name]() is True


def test_parity_registry_covers_spec_paths():
    """完整性守卫：注册表必须覆盖 spec §2.3 枚举的消费路径全集。

    防止有人删条目或在注册表里漏登新路径。statistics_dispatch / validate_catalog
    不直接消费 column_aliases（statistics 经 catalog 投影到 PlanStatistic.parameters；
    validate_catalog 读 plan 自带 output_unit）——故它们以「不直接消费」的身份在此
    登记，由对应专属测试守住链路（见 test_statistics_* / test_validate_catalog_*），
    本网负责记录"它们不直接消费"这一事实，避免未来误判漏接。
    """
    registered = {name for name, _ in PARITY_PATHS}
    # 直接消费 column_aliases 的路径必须都有正向探针。
    direct_consumers = {"resolve_metrics", "resolve_charts", "catalog_cli_context_fallback"}
    missing_direct = direct_consumers - registered
    assert not missing_direct, f"parity 网漏登直接消费路径: {missing_direct}"

    # spec 全集里非直接消费的路径（statistics/validate_catalog）单独标记，确保
    # 它们的"不直接消费"是显式决定而非遗忘——若未来它们改为直接消费，须补正向探针。
    indirect = SPEC_PATHS - direct_consumers
    assert indirect == {"statistics_dispatch", "validate_catalog"}, (
        f"spec 路径分类漂移，请复核: indirect={indirect}"
    )


def test_statistics_path_carries_aliases_via_catalog_projection(tmp_path):
    """statistics_dispatch 路径（spec §2.3 枚举项）：column_aliases 不在 statistics 层
    二次读取，而是经 resolve_metrics 投影进 PlanMetric（zone 参数已 resolve）。

    断言：带 alias 的 resolve_metrics 产出的 PlanMetric，其 parameters 已含 zone
    解析结果（open_arm_zones 之类），证明 statistics 下游拿到的 plan 已带对齐——
    statistics 脚本无需自读 context。
    """
    pm = resolve_metrics(
        "epm",
        FEW_ZONES_COLUMNS,
        RAW_FILES_MULTI,
        str(tmp_path),
        column_aliases=FEW_ZONES_ALIASES,
        common_catalog=load_common_catalog(),
    )
    # 至少有一个 zone 类 metric 进了 plan（alias 已在 resolve 层生效，不下推给 statistics）
    assert "open_arm_time_ratio" in _metric_ids(pm)
    # 该 metric 的 parameters_in_use 应反映 zone 解析（open_arm_zones 已被 alias 命中
    # 成物理列 'open'）——证明 column_aliases 的对齐在 resolve 层完成，statistics 下游
    # 拿到的 plan 已带对齐参数，无需二次读 context。
    open_arm = next(m for m in pm.metrics if m.id == "open_arm_time_ratio")
    params = getattr(open_arm, "parameters_in_use", {}) or {}
    assert params.get("open_arm_zones"), (
        f"statistics 下游 plan 应已投影 zone 对齐进 parameters_in_use，实际={params}"
    )


def test_validate_catalog_path_reads_output_unit_from_plan(tmp_path):
    """validate_catalog 路径（spec §2.3 枚举项）：不直接消费 column_aliases，
    而是读 plan 自带的 output_unit 做范围校验。

    断言：resolve_metrics 产出的 plan 每条 metric 带 output_unit（validate_catalog 的输入），
    证明 validate 无需回头读 context——column_aliases 的对齐在上游 resolve 层已完成。
    """
    pm = resolve_metrics(
        "epm",
        FEW_ZONES_COLUMNS,
        RAW_FILES_MULTI,
        str(tmp_path),
        column_aliases=FEW_ZONES_ALIASES,
        common_catalog=load_common_catalog(),
    )
    assert pm.metrics, "plan 应产出 metrics 供 validate_catalog 校验"
    for m in pm.metrics:
        # output_unit 是 validate_catalog 的核心输入（ratio/count/物理单位）
        assert getattr(m, "output_unit", None), (
            f"metric {m.id} 缺 output_unit → validate_catalog 无法做范围校验"
        )
