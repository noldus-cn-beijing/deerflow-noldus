"""W1: SubagentConfig 4 capability metadata 字段验收。

Spec §3.3:
- 4 新字段都可选,不破坏现有 builtin
- format_subagent_capability 在缺字段时输出 "(未声明)" 而非崩溃
- required_upstream_handoffs 中每个 key 必须在 HANDOFF_FILE_REGISTRY (fail-fast)
"""
from __future__ import annotations

import pytest

from deerflow.subagents.config import SubagentConfig, format_subagent_capability
from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
from deerflow.subagents.handoff_registry import HANDOFF_FILE_REGISTRY


def test_subagent_config_accepts_capability_fields():
    cfg = SubagentConfig(
        name="probe",
        description="d",
        when_to_use="when X",
        input_contract="prompt template",
        output_contract="returns Y",
        required_upstream_handoffs=["code_executor"],
    )
    assert cfg.when_to_use == "when X"
    assert cfg.input_contract == "prompt template"
    assert cfg.output_contract == "returns Y"
    assert cfg.required_upstream_handoffs == ["code_executor"]


def test_capability_fields_default_to_none_or_empty():
    cfg = SubagentConfig(name="bare", description="d")
    assert cfg.when_to_use is None
    assert cfg.input_contract is None
    assert cfg.output_contract is None
    assert cfg.required_upstream_handoffs == []


def test_existing_builtins_still_instantiate():
    for name, cfg in BUILTIN_SUBAGENTS.items():
        assert isinstance(cfg, SubagentConfig)
        assert isinstance(cfg.required_upstream_handoffs, list)


def test_format_subagent_capability_renders_known_fields():
    cfg = SubagentConfig(
        name="x",
        description="desc",
        when_to_use="适合 A",
        input_contract="传 B",
        output_contract="返回 C",
    )
    rendered = format_subagent_capability(cfg)
    assert "x" in rendered
    assert "desc" in rendered
    assert "适合 A" in rendered
    assert "传 B" in rendered
    assert "返回 C" in rendered


def test_format_subagent_capability_renders_placeholder_for_missing():
    cfg = SubagentConfig(name="x", description="desc")
    rendered = format_subagent_capability(cfg)
    assert "(未声明)" in rendered


def test_required_upstream_handoffs_must_reference_known_keys():
    from deerflow.subagents.config import validate_subagent_handoff_refs

    bad = {
        "test-bad": SubagentConfig(
            name="test-bad",
            description="x",
            required_upstream_handoffs=["nonexistent_agent"],
        )
    }
    with pytest.raises(ValueError, match="nonexistent_agent"):
        validate_subagent_handoff_refs(bad, HANDOFF_FILE_REGISTRY)


def test_required_upstream_handoffs_validator_accepts_known():
    from deerflow.subagents.config import validate_subagent_handoff_refs

    good = {
        "test-good": SubagentConfig(
            name="test-good",
            description="x",
            required_upstream_handoffs=["code_executor"],
        )
    }
    validate_subagent_handoff_refs(good, HANDOFF_FILE_REGISTRY)  # should not raise
