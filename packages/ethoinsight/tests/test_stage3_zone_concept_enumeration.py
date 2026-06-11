"""Stage 3 — 补集区概念枚举 (border/dark/closed) TDD 测试。

TDD red 先行：这些测试在实施前应该全红（§5.1 概念缺失 / §5.3 cross-paradigm 失败）。
§5.4 等价性 golden 在改动前采集，实施后验证逐字节相等。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import fields as dc_fields
from pathlib import Path

from ethoinsight.catalog.loader import load_catalog
from ethoinsight.catalog.resolve import _build_zone_aliases_overrides
from ethoinsight.catalog.schema import ParamBinding, ResolvedZoneConcept

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_GOLDEN_FILE = _FIXTURE_DIR / "golden_stage3_zone_overrides.json"


def _load_golden() -> dict[str, str]:
    return json.loads(_GOLDEN_FILE.read_text(encoding="utf-8"))


def _run_and_serialize(paradigm: str, column_aliases: dict, existing_overrides: dict | None = None) -> str:
    cat = load_catalog(paradigm)
    result = _build_zone_aliases_overrides(column_aliases, cat, existing_overrides or {})
    return json.dumps(result, ensure_ascii=False, sort_keys=True)


# ============================================================================
# §5.1 概念枚举存在性（改 yaml/loader 前 red — 概念缺失）
# ============================================================================


def test_ldb_catalog_enumerates_dark_concept():
    """LDB catalog 应枚举 dark 概念（binding=ParamBinding("dark_zone"), source=zone_concept_params）。"""
    cat = load_catalog("light_dark_box")
    rcs = cat.resolved_zone_concepts

    assert "dark" in rcs, f"LDB resolved_zone_concepts missing 'dark'. Keys: {list(rcs.keys())}"
    dark_rc = rcs["dark"]
    assert dark_rc.concept == "dark"
    assert dark_rc.binding is not None, "dark should have a binding (ParamBinding)"
    assert dark_rc.binding.param == "dark_zone"
    assert dark_rc.binding.wrap_list is False
    assert dark_rc.source == "zone_concept_params"


def test_zm_catalog_enumerates_closed_concept():
    """ZM catalog 应枚举 closed 概念（binding=ParamBinding("closed_zones"), source=zone_concept_params）。"""
    cat = load_catalog("zero_maze")
    rcs = cat.resolved_zone_concepts

    assert "closed" in rcs, f"ZM resolved_zone_concepts missing 'closed'. Keys: {list(rcs.keys())}"
    closed_rc = rcs["closed"]
    assert closed_rc.concept == "closed"
    assert closed_rc.binding is not None, "closed should have a binding (ParamBinding)"
    assert closed_rc.binding.param == "closed_zones"
    assert closed_rc.binding.wrap_list is True
    assert closed_rc.source == "zone_concept_params"


def test_oft_catalog_enumerates_border_concept():
    """OFT catalog 应枚举 border 概念（binding=None, source=explicit_concept）。

    border 是可对齐但无注入点的概念——OFT metrics 端无 border_zone 参数，
    thigmotaxis 靠 1−center 反推 + regex 三级降级。
    """
    cat = load_catalog("open_field")
    rcs = cat.resolved_zone_concepts

    assert "border" in rcs, f"OFT resolved_zone_concepts missing 'border'. Keys: {list(rcs.keys())}"
    border_rc = rcs["border"]
    assert border_rc.concept == "border"
    assert border_rc.binding is None, "border should have binding=None (alignable but no injection point)"
    assert border_rc.source == "explicit_concept"


# ============================================================================
# §5.2 param 名对齐 metrics 函数签名（防漂移，不依赖 Stage 2，现在就能写）
# ============================================================================


def test_ldb_dark_param_matches_compute_signature():
    """catalog 声明的 dark_zone param 应对齐 compute_transition_count 函数签名。"""
    from ethoinsight.metrics.ldb import compute_transition_count

    sig = inspect.signature(compute_transition_count)
    assert "dark_zone" in sig.parameters, (
        f"compute_transition_count signature missing 'dark_zone'. "
        f"Params: {list(sig.parameters.keys())}"
    )


def test_zm_closed_param_matches_compute_signature():
    """catalog 声明的 closed_zones param 应对齐 compute_hesitation_count 函数签名。"""
    from ethoinsight.metrics.zero_maze import compute_hesitation_count

    sig = inspect.signature(compute_hesitation_count)
    assert "closed_zones" in sig.parameters, (
        f"compute_hesitation_count signature missing 'closed_zones'. "
        f"Params: {list(sig.parameters.keys())}"
    )


# ============================================================================
# §5.3 不跨范式复用（守边界）
# ============================================================================


def test_concepts_not_cross_paradigm_leaked():
    """dark 只在 LDB、closed 只在 ZM、border 只在 OFT；任一不出现在其他范式的 dict 中。"""
    paradigms = {
        "epm": load_catalog("epm"),
        "open_field": load_catalog("open_field"),
        "light_dark_box": load_catalog("light_dark_box"),
        "zero_maze": load_catalog("zero_maze"),
    }

    for pname, cat in paradigms.items():
        concepts = set(cat.resolved_zone_concepts.keys())

        if pname == "light_dark_box":
            assert "dark" in concepts, f"LDB should have 'dark'"
        else:
            assert "dark" not in concepts, (
                f"{pname} should NOT have 'dark', got {concepts}"
            )

        if pname == "zero_maze":
            assert "closed" in concepts, f"ZM should have 'closed'"
        else:
            assert "closed" not in concepts, (
                f"{pname} should NOT have 'closed', got {concepts}"
            )

        if pname == "open_field":
            assert "border" in concepts, f"OFT should have 'border'"
        else:
            assert "border" not in concepts, (
                f"{pname} should NOT have 'border', got {concepts}"
            )


# ============================================================================
# §5.4 等价性回归（守"不改现有 resolve 输出"——本 stage 的可执行无损证明）
# ============================================================================


def test_stage3_resolve_output_byte_equivalent_bare_columns():
    """裸列集注入：Stage 3 改动前后 _build_zone_aliases_overrides 输出逐字节相同。

    包含 EPM 双臂列、OFT 单 center、LDB light、ZM open 四范式——
    证明新增 dark/closed/border 概念不污染普通注入路径。
    """
    golden = _load_golden()

    # EPM: open_arms + closed_arms（list wrappers）
    epm_actual = _run_and_serialize("epm", {"OA1": "open_arms", "CA1": "closed_arms"})
    assert epm_actual == golden["epm_dual_concept"], (
        f"EPM output diverged!\n"
        f"  golden: {golden['epm_dual_concept']}\n"
        f"  actual: {epm_actual}"
    )

    # OFT: center（scalar）
    oft_actual = _run_and_serialize("open_field", {"中心区": "center"})
    assert oft_actual == golden["oft_center_scalar"], (
        f"OFT center output diverged!\n"
        f"  golden: {golden['oft_center_scalar']}\n"
        f"  actual: {oft_actual}"
    )

    # LDB: light（scalar）
    ldb_actual = _run_and_serialize("light_dark_box", {"明区": "light"})
    assert ldb_actual == golden["ldb_light_scalar"], (
        f"LDB light output diverged!\n"
        f"  golden: {golden['ldb_light_scalar']}\n"
        f"  actual: {ldb_actual}"
    )

    # ZM: open（list）
    zm_actual = _run_and_serialize("zero_maze", {"开放区A": "open", "开放区B": "open"})
    assert zm_actual == golden["zm_open_list"], (
        f"ZM open output diverged!\n"
        f"  golden: {golden['zm_open_list']}\n"
        f"  actual: {zm_actual}"
    )


def test_stage3_border_alias_no_injection():
    """border-alias 路径：OFT column_aliases 含 border 概念时，不产 override、不抛异常。

    这是 §5.4 的核心护栏——border binding=None，resolve 注入循环的
    `if rc.binding is None: continue` 必须命中并跳过 border。
    没有这条 fixture，§5.4 恒绿（假绿——正是 MEMORY 反复警告的哑故障）。
    """
    cat = load_catalog("open_field")
    # before-change behavior: border 概念尚不存在 → concept_param_map 无 border
    # → column_aliases {"in_zone_border": "border"} 中 border 不匹配任何 zone_pattern
    # → 不产 override。after-change: border 有 binding=None → continue 跳过 → 也不产 override。
    result = _build_zone_aliases_overrides({"in_zone_border": "border"}, cat, {})
    assert "center_zone" not in result, (
        f"border alias should NOT produce center_zone override, got {result}"
    )
    assert result == {}, (
        f"border alias should produce empty overrides, got {result}"
    )


def test_stage3_border_alias_with_center_alias_coexists():
    """border + center 同时存在时：center 正常注入，border 被跳过，不抛异常。"""
    cat = load_catalog("open_field")
    result = _build_zone_aliases_overrides(
        {"中心区": "center", "in_zone_border": "border"}, cat, {}
    )
    # center 正常注入
    assert "center_zone" in result, (
        f"center alias should produce center_zone override, got {result}"
    )
    assert result["center_zone"] == "中心区", (
        f"center_zone should be '中心区', got {result.get('center_zone')}"
    )
    # border 不产生额外 override
    assert len(result) == 1, (
        f"Should have exactly 1 override (center only), got {len(result)}: {result}"
    )


# ============================================================================
# §5.5 聚合语义未引入（守 issue#98 边界）
# ============================================================================


def test_no_aggregation_semantics_introduced():
    """ResolvedZoneConcept 仍只是 (concept, binding, source)，无聚合算子。

    防止有人借"补概念"夹带 N:1 聚合语义。ParamBinding 也只有 param + wrap_list。
    """
    rc_field_names = {f.name for f in dc_fields(ResolvedZoneConcept)}
    assert rc_field_names == {"concept", "binding", "source"}, (
        f"ResolvedZoneConcept fields changed: {rc_field_names}. "
        f"Should only be {{concept, binding, source}} — no aggregation operators allowed."
    )

    pb_field_names = {f.name for f in dc_fields(ParamBinding)}
    assert pb_field_names == {"param", "wrap_list"}, (
        f"ParamBinding fields changed: {pb_field_names}. "
        f"Should only be {{param, wrap_list}}."
    )
