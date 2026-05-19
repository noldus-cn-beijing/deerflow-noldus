"""W15: knowledge-assistant SubagentConfig 验收。"""
from deerflow.subagents.builtins.knowledge_assistant import KNOWLEDGE_ASSISTANT_CONFIG


def test_capability_metadata_set():
    cfg = KNOWLEDGE_ASSISTANT_CONFIG
    assert cfg.when_to_use and "QA_KNOWLEDGE" in cfg.when_to_use
    assert cfg.input_contract
    assert cfg.output_contract


def test_required_upstream_handoffs_empty():
    cfg = KNOWLEDGE_ASSISTANT_CONFIG
    assert cfg.required_upstream_handoffs == []
