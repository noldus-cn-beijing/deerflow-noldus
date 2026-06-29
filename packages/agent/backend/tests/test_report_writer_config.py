"""W14: report-writer SubagentConfig 验收。"""
from __future__ import annotations

from deerflow.subagents.builtins.report_writer import REPORT_WRITER_CONFIG


def test_capability_metadata_set():
    cfg = REPORT_WRITER_CONFIG
    assert cfg.when_to_use and "报告" in cfg.when_to_use
    assert cfg.input_contract and "code-executor" in cfg.input_contract
    # 2026-06-29 HTML 报告：输出契约改 report.html
    assert "report.html" in cfg.output_contract


def test_output_targets_html_report():
    """spec 2026-06-29: 产物从 report.md 改 report.html（自包含 + 内联代表性图）。"""
    cfg = REPORT_WRITER_CONFIG
    assert "report.html" in cfg.system_prompt
    assert "report.html" in cfg.output_contract
    # 旧 md 产物路径已退役
    assert "写 /mnt/user-data/outputs/report.md" not in cfg.output_contract


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


def test_system_prompt_no_bash_bundle_dead_guidance():
    """2026-06-18: report-writer 的 disallowed_tools 明确含 ``bash``（report_writer.py），
    它根本没有 bash 工具——``bash cat ... > bundle`` 这条指引是空指令。必须删掉，改逐个
    read_file（spec 2026-06-18-chart-maker-parallel-plotting §3.2）。"""
    p = REPORT_WRITER_CONFIG.system_prompt
    assert "rw_context_bundle" not in p
    assert "bash cat /mnt/user-data/workspace/handoff" not in p


def test_system_prompt_produces_html5_skeleton():
    """spec 2026-06-29: 报告载体改 HTML5——prompt 须指导产合法 HTML5 骨架。"""
    p = REPORT_WRITER_CONFIG.system_prompt
    assert "<!DOCTYPE html>" in p
    # 关键结构标签的指导在
    assert "<table" in p and "<h2" in p


def test_system_prompt_representative_image_only_guidance():
    """spec 2026-06-29: HTML 报告只内联**少量代表性图**（非全量 113 张）。
    prompt 须明示「代表性 + 少量」+ {{img:}} 占位符语法 + seal 转 base64。"""
    p = REPORT_WRITER_CONFIG.system_prompt
    assert "代表性" in p
    assert "{{img:" in p
    assert "base64" in p


def test_system_prompt_preserves_interpretation_constitution():
    """spec 2026-06-29 catastrophic-forgetting: 改 HTML 载体**不松**判读宪法。
    绝对阈值禁令 + 绝对程度（高/低焦虑）禁令必须原样保留（守第 9 条铁律）。"""
    p = REPORT_WRITER_CONFIG.system_prompt
    # 绝对阈值禁令示例仍在
    assert "正常范围" in p
    # <禁止的写法> 段仍在（判读宪法的载体）
    assert "<禁止的写法>" in p or "禁止的写法" in p
