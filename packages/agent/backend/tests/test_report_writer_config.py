"""W14: report-writer SubagentConfig 验收。"""
from __future__ import annotations

from deerflow.subagents.builtins.report_writer import REPORT_WRITER_CONFIG


def test_capability_metadata_set():
    cfg = REPORT_WRITER_CONFIG
    assert cfg.when_to_use and "报告" in cfg.when_to_use
    assert cfg.input_contract and "code-executor" in cfg.input_contract
    assert cfg.output_contract and "report.md" in cfg.output_contract


def test_required_upstream_handoffs_is_code_and_data():
    cfg = REPORT_WRITER_CONFIG
    assert sorted(cfg.required_upstream_handoffs) == ["code_executor", "data_analyst"]


def test_system_prompt_mentions_chart_maker_handoff_optional():
    cfg = REPORT_WRITER_CONFIG
    assert "handoff_chart_maker" in cfg.system_prompt
    assert "可选" in cfg.system_prompt or "optional" in cfg.system_prompt.lower()


def test_image_path_uses_full_virtual_prefix_not_relative():
    """2026-06-04 fix: report.md image paths must use mnt/user-data/outputs/file.png,
    not outputs/file.png. The artifacts API (resolve_thread_virtual_path) requires the
    full virtual prefix — relative paths cause 400 Bad Request in the frontend.
    """
    p = REPORT_WRITER_CONFIG.system_prompt

    # Positive: full virtual path format must be present in the examples
    assert "mnt/user-data/outputs/" in p

    # Negative: the old wrong relative-only form must be gone
    # (the fragment 'outputs/boxplot' was the canonical example of the broken pattern)
    assert "(outputs/boxplot_mean_nnd.png)" not in p
    assert "(outputs/plot_trajectory_plot_s0.png)" not in p
