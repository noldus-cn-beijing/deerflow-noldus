"""Tests for memory prompt injection formatting."""

import math

from deerflow.agents.memory.prompt import _coerce_confidence, format_memory_for_injection


def test_format_memory_includes_facts_section() -> None:
    memory_data = {
        "user": {},
        "history": {},
        "facts": [
            {"content": "User uses PostgreSQL", "category": "knowledge", "confidence": 0.9},
            {"content": "User prefers SQLAlchemy", "category": "preference", "confidence": 0.8},
        ],
    }

    result = format_memory_for_injection(memory_data, max_tokens=2000)

    assert "Facts:" in result
    assert "User uses PostgreSQL" in result
    assert "User prefers SQLAlchemy" in result


def test_format_memory_sorts_facts_by_confidence_desc() -> None:
    memory_data = {
        "user": {},
        "history": {},
        "facts": [
            {"content": "Low confidence fact", "category": "context", "confidence": 0.4},
            {"content": "High confidence fact", "category": "knowledge", "confidence": 0.95},
        ],
    }

    result = format_memory_for_injection(memory_data, max_tokens=2000)

    assert result.index("High confidence fact") < result.index("Low confidence fact")


def test_format_memory_respects_budget_when_adding_facts(monkeypatch) -> None:
    # Make token counting deterministic for this test by counting characters.
    monkeypatch.setattr("deerflow.agents.memory.prompt._count_tokens", lambda text, encoding_name="cl100k_base": len(text))

    memory_data = {
        "user": {},
        "history": {},
        "facts": [
            {"content": "First fact should fit", "category": "knowledge", "confidence": 0.95},
            {"content": "Second fact should not fit in tiny budget", "category": "knowledge", "confidence": 0.90},
        ],
    }

    first_fact_only_memory_data = {
        "user": {},
        "history": {},
        "facts": [
            {"content": "First fact should fit", "category": "knowledge", "confidence": 0.95},
        ],
    }
    one_fact_result = format_memory_for_injection(first_fact_only_memory_data, max_tokens=2000)
    two_facts_result = format_memory_for_injection(memory_data, max_tokens=2000)
    # Choose a budget that can include exactly one fact section line.
    max_tokens = (len(one_fact_result) + len(two_facts_result)) // 2

    first_only_result = format_memory_for_injection(memory_data, max_tokens=max_tokens)

    assert "First fact should fit" in first_only_result
    assert "Second fact should not fit in tiny budget" not in first_only_result


def test_coerce_confidence_nan_falls_back_to_default() -> None:
    """NaN should not be treated as a valid confidence value."""
    result = _coerce_confidence(math.nan, default=0.5)
    assert result == 0.5


def test_coerce_confidence_inf_falls_back_to_default() -> None:
    """Infinite values should fall back to default rather than clamping to 1.0."""
    assert _coerce_confidence(math.inf, default=0.3) == 0.3
    assert _coerce_confidence(-math.inf, default=0.3) == 0.3


def test_coerce_confidence_valid_values_are_clamped() -> None:
    """Valid floats outside [0, 1] are clamped; values inside are preserved."""
    assert _coerce_confidence(1.5) == 1.0
    assert _coerce_confidence(-0.5) == 0.0
    assert abs(_coerce_confidence(0.75) - 0.75) < 1e-9


def test_format_memory_skips_none_content_facts() -> None:
    """Facts with content=None must not produce a 'None' line in the output."""
    memory_data = {
        "facts": [
            {"content": None, "category": "knowledge", "confidence": 0.9},
            {"content": "Real fact", "category": "knowledge", "confidence": 0.8},
        ],
    }

    result = format_memory_for_injection(memory_data, max_tokens=2000)

    assert "None" not in result
    assert "Real fact" in result


def test_format_memory_skips_non_string_content_facts() -> None:
    """Facts with non-string content (e.g. int/list) must be ignored."""
    memory_data = {
        "facts": [
            {"content": 42, "category": "knowledge", "confidence": 0.9},
            {"content": ["list"], "category": "knowledge", "confidence": 0.85},
            {"content": "Valid fact", "category": "knowledge", "confidence": 0.7},
        ],
    }

    result = format_memory_for_injection(memory_data, max_tokens=2000)

    # The formatted line for an integer content would be "- [knowledge | 0.90] 42".
    assert "| 0.90] 42" not in result
    # The formatted line for a list content would be "- [knowledge | 0.85] ['list']".
    assert "| 0.85]" not in result
    assert "Valid fact" in result


def test_format_memory_renders_correction_source_error() -> None:
    memory_data = {
        "facts": [
            {
                "content": "Use make dev for local development.",
                "category": "correction",
                "confidence": 0.95,
                "sourceError": "The agent previously suggested npm start.",
            }
        ]
    }

    result = format_memory_for_injection(memory_data, max_tokens=2000)

    assert "Use make dev for local development." in result
    assert "avoid: The agent previously suggested npm start." in result


def test_format_memory_renders_correction_without_source_error_normally() -> None:
    memory_data = {
        "facts": [
            {
                "content": "Use make dev for local development.",
                "category": "correction",
                "confidence": 0.95,
            }
        ]
    }

    result = format_memory_for_injection(memory_data, max_tokens=2000)

    assert "Use make dev for local development." in result
    assert "avoid:" not in result


def test_format_memory_does_not_inject_history_or_topofmind() -> None:
    """2026-05-13 隔离更新：history.* 和 user.topOfMind 不再注入到 prompt。

    这些字段容易被 LLM 写入"上传了 X 文件"、"刚跑了 Y 分析"等会话级状态，
    新 thread 读到后会产生"以为当前 thread 也上传过那些文件"的幻觉。本测试
    锁死："即使 memory.json 里有这些字段，注入到 prompt 时也不出现"。

    facts、user.workContext、user.personalContext 仍正常注入。
    """
    memory_data = {
        "user": {
            "workContext": {"summary": "Senior backend engineer"},
            "personalContext": {"summary": "Prefers English markdown output"},
            "topOfMind": {"summary": "已上传 5 个 EPM 文件，等待分组信息"},
        },
        "history": {
            "recentMonths": {"summary": "Recent activity summary"},
            "earlierContext": {"summary": "Earlier context summary"},
            "longTermBackground": {"summary": "Core expertise in distributed systems"},
        },
        "facts": [
            {"content": "User studies zebrafish behavior", "category": "context", "confidence": 0.9},
        ],
    }

    result = format_memory_for_injection(memory_data, max_tokens=2000)

    # 长期画像层 — 注入
    assert "Work: Senior backend engineer" in result
    assert "Personal: Prefers English markdown output" in result
    assert "User studies zebrafish behavior" in result

    # 会话级 / 历史叙述层 — 不注入
    assert "已上传" not in result, "topOfMind 不应注入"
    assert "Current Focus" not in result, "topOfMind label 不应出现"
    assert "Recent activity summary" not in result, "history.recentMonths 不应注入"
    assert "Earlier context summary" not in result, "history.earlierContext 不应注入"
    assert "Core expertise in distributed systems" not in result, "history.longTermBackground 不应注入"
    assert "Background:" not in result
    assert "Recent:" not in result
    assert "Earlier:" not in result
    assert "History:" not in result
