from ethoinsight.templates.tool import get_available_paradigms


def test_available_paradigms_only_contains_implemented():
    available = get_available_paradigms()
    assert "shoaling" in available
    for name in ("epm", "open_field", "novel_object", "y_maze", "forced_swim"):
        assert name not in available, f"{name} listed as available but has no template"


def test_compute_metrics_rejects_unsupported_paradigm():
    from ethoinsight.templates.tool import run_paradigm_analysis_core

    result = run_paradigm_analysis_core(
        paradigm="epm",
        file_pattern="/nonexistent/*.txt",
        groups={"control": ["s1"], "treatment": ["s2"]},
        output_dir="/tmp/test-degradation",
        handoff_path="/tmp/test-degradation/handoff.json",
    )

    assert result["status"] == "failed"
    assert "epm" in result["summary"]
    assert "available_paradigms" in result
    assert "shoaling" in result["available_paradigms"]
