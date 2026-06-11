"""Stage 2 — 字节等价回归测试（验收核心）。

证明 _build_zone_aliases_overrides 的输出在改动前后逐字节相等。
golden 快照在改动前采集（fixtures/golden_zone_overrides.json）。
"""

from __future__ import annotations

import json
from pathlib import Path

from ethoinsight.catalog.loader import load_catalog
from ethoinsight.catalog.resolve import _build_zone_aliases_overrides


_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_GOLDEN_FILE = _FIXTURE_DIR / "golden_zone_overrides.json"


def _load_golden() -> dict[str, str]:
    """Load golden snapshot dict. Returns {case_name: json_string}. """
    return json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))


def _run_and_serialize(paradigm: str, column_aliases: dict, existing_overrides: dict | None = None) -> str:
    """Run _build_zone_aliases_overrides and serialize result as JSON for comparison."""
    cat = load_catalog(paradigm)
    result = _build_zone_aliases_overrides(column_aliases, cat, existing_overrides or {})
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


# ============================================================================
# Golden snapshot assertions — 中心验收项
# ============================================================================


def test_epm_dual_concept_equivalence():
    """EPM 双 concept 输出应与改动前 golden 逐字节相等。"""
    golden = _load_golden()
    actual = _run_and_serialize("epm", {"OA1": "open_arms", "CA1": "closed_arms"})
    assert actual == golden["epm_dual_concept"], (
        f"EPM dual concept output diverged from golden!\n"
        f"  golden: {golden['epm_dual_concept']}\n"
        f"  actual: {actual}"
    )


def test_oft_scalar_equivalence():
    """OFT scalar 输出应与改动前 golden 逐字节相等。"""
    golden = _load_golden()
    actual = _run_and_serialize("open_field", {"中心区": "center"})
    assert actual == golden["oft_scalar"], (
        f"OFT scalar output diverged from golden!\n"
        f"  golden: {golden['oft_scalar']}\n"
        f"  actual: {actual}"
    )


def test_ldb_scalar_equivalence():
    """LDB scalar 输出应与改动前 golden 逐字节相等。"""
    golden = _load_golden()
    actual = _run_and_serialize("light_dark_box", {"明区": "light"})
    assert actual == golden["ldb_scalar"], (
        f"LDB scalar output diverged from golden!\n"
        f"  golden: {golden['ldb_scalar']}\n"
        f"  actual: {actual}"
    )


def test_zm_list_equivalence():
    """ZM list 输出应与改动前 golden 逐字节相等。"""
    golden = _load_golden()
    actual = _run_and_serialize("zero_maze", {"开放区A": "open", "开放区B": "open"})
    assert actual == golden["zm_list"], (
        f"ZM list output diverged from golden!\n"
        f"  golden: {golden['zm_list']}\n"
        f"  actual: {actual}"
    )


def test_epm_existing_override_equivalence():
    """EPM 显式优先（existing_overrides 含同名 param）输出应与改动前 golden 相等。"""
    golden = _load_golden()
    actual = _run_and_serialize("epm", {"OA1": "open_arms", "CA1": "closed_arms"}, {"open_arm_zones": "already_set"})
    assert actual == golden["epm_existing_override"], (
        f"EPM existing override output diverged from golden!\n"
        f"  golden: {golden['epm_existing_override']}\n"
        f"  actual: {actual}"
    )


# ============================================================================
# 结构护栏：输出 key 的 value 类型由 wrap_list 决定
# ============================================================================


def test_oft_produces_scalar_not_list():
    """OFT wrap_list=False → 输出 value 应为 scalar 不是 list。"""
    cat = load_catalog("open_field")
    result = _build_zone_aliases_overrides({"中心区": "center"}, cat, {})
    assert isinstance(result.get("center_zone"), str), (
        f"OFT center_zone should be scalar (str), got {type(result.get('center_zone')).__name__}"
    )


def test_zm_produces_list_not_scalar():
    """ZM wrap_list=True → 输出 value 应为 list 不是 scalar。"""
    cat = load_catalog("zero_maze")
    result = _build_zone_aliases_overrides({"开放区A": "open", "开放区B": "open"}, cat, {})
    assert isinstance(result.get("open_zones"), list), (
        f"ZM open_zones should be list, got {type(result.get('open_zones')).__name__}"
    )
