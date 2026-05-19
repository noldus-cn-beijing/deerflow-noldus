"""W12: data-analyst SubagentConfig 验收。"""
from __future__ import annotations

from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG


def test_capability_metadata_set():
    cfg = DATA_ANALYST_CONFIG
    assert cfg.when_to_use and "code-executor 刚完成" in cfg.when_to_use
    assert cfg.input_contract and "用户语言" in cfg.input_contract
    assert cfg.output_contract and "handoff_data_analyst.json" in cfg.output_contract
    assert "gate_signals" in cfg.output_contract


def test_required_upstream_handoffs_is_code_executor():
    cfg = DATA_ANALYST_CONFIG
    assert cfg.required_upstream_handoffs == ["code_executor"]


def test_system_prompt_unchanged_in_substance():
    cfg = DATA_ANALYST_CONFIG
    assert cfg.system_prompt and len(cfg.system_prompt) > 200
