"""Contract tests for data-analyst insight depth (per-subject + counterfactual)."""
from deerflow.subagents.handoff_schemas import (
    DataAnalystHandoff,
    OutlierFinding,
)


def test_outlier_finding_schema():
    f = OutlierFinding(
        subject="Subject 3",
        metric="mean_nnd",
        value=70.02,
        deviation="~2x group median",
        counterfactual="NND drops 48.2 → 37.2 mm if excluded",
    )
    assert f.subject == "Subject 3"
    assert f.counterfactual is not None


def test_data_analyst_handoff_accepts_outlier_findings():
    h = DataAnalystHandoff(
        status="completed",
        outlier_findings=[
            OutlierFinding(
                subject="Subject 3",
                metric="mean_nnd",
                value=70.02,
                deviation="2x median",
            ),
        ],
    )
    assert len(h.outlier_findings) == 1


def test_data_analyst_prompt_requires_per_subject_review():
    from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG

    p = DATA_ANALYST_CONFIG.system_prompt
    # Prompt must explicitly direct per-subject outlier inspection
    assert (
        "逐一" in p
        or "按受试者" in p
        or "per_subject" in p
        or "per-subject" in p
    )
    # And counterfactual / leave-one-out analysis
    assert "排除" in p or "leave-one-out" in p or "反事实" in p
