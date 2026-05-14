"""Tests for ethoinsight.catalog module."""

from __future__ import annotations

import json
import subprocess
import sys

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


# ── load_catalog tests ──────────────────────────────────────────────────────


def test_load_catalog_returns_catalog_instance(tmp_path):
    from ethoinsight.catalog.loader import load_catalog
    from ethoinsight.catalog.schema import Catalog

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
    direction_for_anxiety: very_weirdly_anxious
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


# ── Q6 白名单对齐测试 ────────────────────────────────────────────────────────

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


@pytest.mark.parametrize(
    "paradigm,whitelist",
    [
        ("epm", EPM_Q6_WHITELIST),
        ("oft", OFT_Q6_WHITELIST),
        ("fst", FST_Q6_WHITELIST),
    ],
)
def test_catalog_default_metrics_match_q6_whitelist(paradigm, whitelist):
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


# ── resolve tests ────────────────────────────────────────────────────────────


def _epm_columns_full() -> list[str]:
    """Returns column list that satisfies all EPM default metrics' requires_columns."""
    return [
        "time",
        "x_center",
        "y_center",
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
        n_per_group=1,
        n_groups=1,
    )
    assert plan.statistics is not None
    assert plan.statistics.skip_reason is not None
    assert (
        "n_per_group" in plan.statistics.skip_reason
        or "n_groups" in plan.statistics.skip_reason
    )


def test_resolve_plan_to_dict_serializable():
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


# ── CLI tests ──────────────────────────────────────────────────────────────────


def _run_cli(args: list[str]) -> tuple[int, str, str]:
    """Run `python -m ethoinsight.catalog.resolve <args>` and capture."""
    proc = subprocess.run(
        [sys.executable, "-m", "ethoinsight.catalog.resolve", *args],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_happy_path(tmp_path):
    columns_file = tmp_path / "columns.json"
    columns_file.write_text(
        json.dumps(
            {
                "file": "raw.txt",
                "columns": _epm_columns_full(),
                "n_subjects": 1,
                "duration_s": 300.0,
            }
        ),
        encoding="utf-8",
    )

    raw_files_json = tmp_path / "raw_files.json"
    raw_files_json.write_text(json.dumps(["/tmp/raw.txt"]), encoding="utf-8")

    output = tmp_path / "plan.json"

    rc, stdout, stderr = _run_cli(
        [
            "--paradigm",
            "epm",
            "--columns-file",
            str(columns_file),
            "--raw-files-json",
            str(raw_files_json),
            "--workspace-dir",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )
    assert rc == 0, f"stderr: {stderr}"
    plan = json.loads(output.read_text())
    assert plan["paradigm"] == "epm"
    assert plan["schema_version"] == "1.0"
    assert len(plan["metrics"]) == 5


def test_cli_unknown_paradigm_exit1_with_json(tmp_path):
    columns_file = tmp_path / "columns.json"
    columns_file.write_text(json.dumps({"columns": []}), encoding="utf-8")
    raw_files_json = tmp_path / "raw_files.json"
    raw_files_json.write_text(json.dumps(["/tmp/raw.txt"]), encoding="utf-8")

    rc, stdout, stderr = _run_cli(
        [
            "--paradigm",
            "totally_made_up",
            "--columns-file",
            str(columns_file),
            "--raw-files-json",
            str(raw_files_json),
            "--workspace-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "plan.json"),
        ]
    )
    assert rc == 1
    err = json.loads(stderr.strip().splitlines()[-1])
    assert err["code"] == "unknown_paradigm"


def test_cli_user_include_unknown_exit1(tmp_path):
    columns_file = tmp_path / "columns.json"
    columns_file.write_text(
        json.dumps({"columns": _epm_columns_full()}), encoding="utf-8"
    )
    raw_files_json = tmp_path / "raw_files.json"
    raw_files_json.write_text(json.dumps(["/tmp/raw.txt"]), encoding="utf-8")

    rc, _, stderr = _run_cli(
        [
            "--paradigm",
            "epm",
            "--columns-file",
            str(columns_file),
            "--raw-files-json",
            str(raw_files_json),
            "--workspace-dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "plan.json"),
            "--include",
            "nonexistent_metric_xyz",
        ]
    )
    assert rc == 1
    err = json.loads(stderr.strip().splitlines()[-1])
    assert err["code"] == "unknown_metric"
    assert "nonexistent_metric_xyz" in err["details"]["requested"]


# ── Anti-regression: all catalog scripts must be importable ───────────────────


@pytest.mark.parametrize(
    "paradigm", ["epm", "oft", "fst", "tst", "ldb", "zero_maze", "shoaling"]
)
def test_all_catalog_scripts_are_importable(paradigm):
    """catalog 里声明的 script dotted path 必须真的能 import 到一个有 main() 的模块。"""
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
