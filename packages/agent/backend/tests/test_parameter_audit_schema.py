"""Sprint 3 单元测试 — ParameterAuditFinding schema + DataAnalystHandoff/GateSignals 扩展。

TDD 优先：schema 测试通过后再改 prompt/seal。

测试清单（对应 spec §2.8 + §4）：
1. test_valid_finding_construction — 5 个必填字段 + 1 个可选字段都能正常构造
2. test_mismatch_kind_must_be_enum — mismatch_kind 必须是 5 元枚举之一
3. test_parameter_must_be_nonempty — parameter 字段非空
4. test_observed_distribution_accepts_numeric_only — observed_distribution 值必须是 float/int
5. test_blocks_downstream_default_false — blocks_downstream 默认 False
6. test_severity_critical_does_not_auto_block — severity=critical 不自动让 blocks_downstream=True
7. test_data_analyst_handoff_carries_findings — DataAnalystHandoff 能携带 list[ParameterAuditFinding]
8. test_gate_signals_carries_counts — parameter_audit_findings_count + parameter_audit_critical_count 默认 0

Phase 1.5 新增（对应 spec §7.2）：
9. test_normalize_used_value_none — used_value=None 归一化为 "" 后通过
10. test_normalize_od_note_text — observed_distribution={"note":"文字"} 归一化为 {} 后通过
11. test_normalize_od_mixed — 混合 dict 只保留 numeric 键
12. test_normalize_od_non_dict — 非 dict observed_distribution 归一化为 {}
13. test_normalize_no_effect_on_valid — 正常 finding 不受归一化影响
14. test_normalize_idempotent — 已合规 finding 二次归一化不变
15. test_degenerate_finding_in_handoff — 退化 finding 经归一化后能进 DataAnalystHandoff
"""

import pytest
from pydantic import ValidationError

from deerflow.subagents.handoff_schemas import (
    DataAnalystHandoff,
    GateSignals,
    ParameterAuditFinding,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_finding(**overrides) -> dict:
    """Minimal valid ParameterAuditFinding dict with sensible defaults."""
    base = {
        "parameter": "velocity_threshold",
        "metric": "immobility_time",
        "severity": "warning",
        "used_value": 30.0,
        "observed_distribution": {"median": 5.0, "p90": 12.0, "max": 25.0, "n_subjects": 12},
        "mismatch_kind": "threshold_too_high",
        "suggestion": "当前阈值 30 mm/s 高于本批中位数 5 mm/s 的 6 倍，建议参考 paradigm md 的参数调整指南段",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Valid construction
# ---------------------------------------------------------------------------
class TestValidFindingConstruction:
    def test_valid_finding_construction(self):
        finding = ParameterAuditFinding(**_make_finding())
        assert finding.parameter == "velocity_threshold"
        assert finding.metric == "immobility_time"
        assert finding.severity == "warning"
        assert finding.used_value == 30.0
        assert finding.observed_distribution["median"] == 5.0
        assert finding.mismatch_kind == "threshold_too_high"
        assert finding.suggestion  # non-empty
        assert finding.blocks_downstream is False  # default

    def test_all_severity_values(self):
        for sev in ("critical", "warning", "info"):
            finding = ParameterAuditFinding(**_make_finding(severity=sev))
            assert finding.severity == sev

    def test_all_mismatch_kind_values(self):
        for kind in (
            "threshold_too_high",
            "threshold_too_low",
            "window_too_wide",
            "window_too_narrow",
            "category_mismatch",
        ):
            finding = ParameterAuditFinding(**_make_finding(mismatch_kind=kind))
            assert finding.mismatch_kind == kind

    def test_used_value_accepts_int(self):
        finding = ParameterAuditFinding(**_make_finding(used_value=30))
        assert finding.used_value == 30
        assert isinstance(finding.used_value, int)

    def test_used_value_accepts_str(self):
        finding = ParameterAuditFinding(**_make_finding(used_value="auto"))
        assert finding.used_value == "auto"


# ---------------------------------------------------------------------------
# 2. mismatch_kind must be enum
# ---------------------------------------------------------------------------
class TestMismatchKindEnum:
    def test_invalid_mismatch_kind_rejected(self):
        with pytest.raises(ValidationError, match="mismatch_kind"):
            ParameterAuditFinding(**_make_finding(mismatch_kind="data_outlier"))

    def test_empty_mismatch_kind_rejected(self):
        with pytest.raises(ValidationError, match="mismatch_kind"):
            ParameterAuditFinding(**_make_finding(mismatch_kind=""))


# ---------------------------------------------------------------------------
# 3. parameter must be non-empty
# ---------------------------------------------------------------------------
class TestParameterNonEmpty:
    def test_empty_parameter_rejected(self):
        with pytest.raises(ValidationError, match="non-empty"):
            ParameterAuditFinding(**_make_finding(parameter=""))

    def test_whitespace_only_parameter_rejected(self):
        with pytest.raises(ValidationError, match="non-empty"):
            ParameterAuditFinding(**_make_finding(parameter="   "))

    def test_valid_parameter_with_underscores(self):
        finding = ParameterAuditFinding(**_make_finding(parameter="velocity_threshold"))
        assert finding.parameter == "velocity_threshold"


# ---------------------------------------------------------------------------
# 4. observed_distribution accepts numeric only
# ---------------------------------------------------------------------------
class TestObservedDistributionNumeric:
    def test_valid_numeric_distribution(self):
        finding = ParameterAuditFinding(**_make_finding(observed_distribution={"p90": 12.0, "n": 8}))
        assert finding.observed_distribution == {"p90": 12.0, "n": 8}

    def test_empty_distribution_accepted(self):
        finding = ParameterAuditFinding(**_make_finding(observed_distribution={}))
        assert finding.observed_distribution == {}

    def test_string_value_stripped_by_normalizer(self):
        """Phase 1.5: model_validator strips string values from observed_distribution.

        {"median": "high"} → {} (non-numeric stripped, dict is empty but valid).
        This was previously a ValidationError; now the normalizer handles it.
        """
        finding = ParameterAuditFinding(**_make_finding(observed_distribution={"median": "high"}))
        assert finding.observed_distribution == {}


# ---------------------------------------------------------------------------
# 5. blocks_downstream default false
# ---------------------------------------------------------------------------
class TestBlocksDownstreamDefault:
    def test_default_false(self):
        finding = ParameterAuditFinding(**_make_finding())
        assert finding.blocks_downstream is False

    def test_explicit_true(self):
        finding = ParameterAuditFinding(**_make_finding(blocks_downstream=True))
        assert finding.blocks_downstream is True


# ---------------------------------------------------------------------------
# 6. severity=critical does NOT auto-set blocks_downstream
# ---------------------------------------------------------------------------
class TestSeverityCriticalNoAutoBlock:
    def test_severity_critical_does_not_auto_block(self):
        finding = ParameterAuditFinding(**_make_finding(severity="critical"))
        assert finding.severity == "critical"
        assert finding.blocks_downstream is False  # must be explicitly set

    def test_severity_critical_explicit_block(self):
        finding = ParameterAuditFinding(**_make_finding(severity="critical", blocks_downstream=True))
        assert finding.blocks_downstream is True


# ---------------------------------------------------------------------------
# 7. DataAnalystHandoff carries findings
# ---------------------------------------------------------------------------
class TestDataAnalystHandoffCarriesFindings:
    def test_empty_findings_default(self):
        handoff = DataAnalystHandoff(status="completed", analysis_config_id="abc123", key_findings=["finding"])
        assert handoff.parameter_audit_findings == []

    def test_handoff_with_findings(self):
        findings = [
            ParameterAuditFinding(**_make_finding()),
            ParameterAuditFinding(**_make_finding(parameter="total_entry_threshold", metric="total_entry_count", mismatch_kind="threshold_too_low", used_value=8, observed_distribution={"median": 15.0, "p10": 10.0})),
        ]
        handoff = DataAnalystHandoff(
            status="completed",
            analysis_config_id="abc123",
            key_findings=["finding"],
            parameter_audit_findings=findings,
        )
        assert len(handoff.parameter_audit_findings) == 2
        assert handoff.parameter_audit_findings[0].parameter == "velocity_threshold"
        assert handoff.parameter_audit_findings[1].parameter == "total_entry_threshold"

    def test_handoff_round_trip_json(self):
        """Verify findings survive JSON serialization round-trip."""
        findings = [ParameterAuditFinding(**_make_finding())]
        handoff = DataAnalystHandoff(status="completed", analysis_config_id="abc123", key_findings=["finding"], parameter_audit_findings=findings)
        json_str = handoff.model_dump_json()
        restored = DataAnalystHandoff.model_validate_json(json_str)
        assert len(restored.parameter_audit_findings) == 1
        assert restored.parameter_audit_findings[0].mismatch_kind == "threshold_too_high"


# ---------------------------------------------------------------------------
# 8. GateSignals carries counts
# ---------------------------------------------------------------------------
class TestGateSignalsCounts:
    def test_default_counts_zero(self):
        gs = GateSignals()
        assert gs.parameter_audit_findings_count == 0
        assert gs.parameter_audit_critical_count == 0

    def test_explicit_counts(self):
        gs = GateSignals(parameter_audit_findings_count=3, parameter_audit_critical_count=1)
        assert gs.parameter_audit_findings_count == 3
        assert gs.parameter_audit_critical_count == 1

    def test_counts_in_data_analyst_handoff(self):
        gs = GateSignals(parameter_audit_findings_count=2, parameter_audit_critical_count=0)
        handoff = DataAnalystHandoff(status="completed", analysis_config_id="abc123", key_findings=["finding"], gate_signals=gs)
        assert handoff.gate_signals.parameter_audit_findings_count == 2


# ---------------------------------------------------------------------------
# 9-15. Phase 1.5 — _normalize_audit_finding model_validator tests
#       (mirrors DataQualityWarning._normalize_llm_typeros test pattern)
# ---------------------------------------------------------------------------
class TestNormalizeAuditFinding:
    """Test ParameterAuditFinding._normalize_audit_finding model_validator.

    Phase 1.5 (spec §7.2): conservatively normalise common LLM mistakes
    before strict validation. Mirrors DataQualityWarning._normalize_llm_typeros.
    """

    def test_normalize_used_value_none(self):
        """used_value=None → "" (schema requires float|int|str)."""
        finding = ParameterAuditFinding(**_make_finding(used_value=None))
        assert finding.used_value == ""

    def test_normalize_od_note_text(self):
        """observed_distribution={"note": "文字"} → {} (non-numeric stripped)."""
        finding = ParameterAuditFinding(
            **_make_finding(
                observed_distribution={"note": "per_subject 仅含标量值无法计算百分位判据"},
            )
        )
        assert finding.observed_distribution == {}

    def test_normalize_od_mixed(self):
        """observed_distribution={"p90": 12.0, "note": "x"} → {"p90": 12.0}."""
        finding = ParameterAuditFinding(
            **_make_finding(
                observed_distribution={"p90": 12.0, "note": "some text"},
            )
        )
        assert finding.observed_distribution == {"p90": 12.0}

    def test_normalize_od_non_dict(self):
        """Non-dict observed_distribution (e.g. a string) → {}."""
        finding = ParameterAuditFinding(
            **_make_finding(
                observed_distribution="some text instead of dict",
            )
        )
        assert finding.observed_distribution == {}

    def test_normalize_od_none(self):
        """observed_distribution=None (explicit null) → {}.

        Degenerate skip exit: deepseek may send null when there is no
        distribution. The key is present, so it is normalized to an empty
        numeric dict rather than failing strict validation.
        """
        finding = ParameterAuditFinding(**_make_finding(observed_distribution=None))
        assert finding.observed_distribution == {}

    def test_normalize_no_effect_on_valid(self):
        """Normal finding with valid numeric observed_distribution is unchanged."""
        original = _make_finding()
        finding = ParameterAuditFinding(**original)
        assert finding.used_value == original["used_value"]
        assert finding.observed_distribution == original["observed_distribution"]

    def test_normalize_idempotent(self):
        """Already-compliant finding survives re-validation unchanged."""
        finding1 = ParameterAuditFinding(**_make_finding())
        # Re-construct from dict (simulates second normalization pass)
        finding2 = ParameterAuditFinding(**finding1.model_dump())
        assert finding1.used_value == finding2.used_value
        assert finding1.observed_distribution == finding2.observed_distribution

    def test_degenerate_finding_in_handoff(self):
        """Degenerate finding (the exact pattern from dogfood failure) passes
        through DataAnalystHandoff after normalization.

        Reproduces thread 81051535: used_value=None + observed_distribution
        with note text.
        """
        degenerate = {
            "parameter": "velocity_threshold",
            "metric": "immobility_time",
            "severity": "info",
            "used_value": None,  # ← LLM filled None
            "observed_distribution": {
                "note": "per_subject 仅含标量值，无法计算 median 等百分位判据"
            },  # ← LLM put text instead of numbers
            "mismatch_kind": "threshold_too_high",
            "suggestion": "样本量不足（n=1），无法计算百分位判据，参数审计待上游补逐帧分布后执行",
            "blocks_downstream": False,
        }
        handoff = DataAnalystHandoff(
            status="completed",
            analysis_config_id="abc123",
            key_findings=["finding"],
            parameter_audit_findings=[degenerate],
        )
        assert len(handoff.parameter_audit_findings) == 1
        f = handoff.parameter_audit_findings[0]
        assert f.used_value == ""  # normalized from None
        assert f.observed_distribution == {}  # text stripped
        assert f.parameter == "velocity_threshold"
