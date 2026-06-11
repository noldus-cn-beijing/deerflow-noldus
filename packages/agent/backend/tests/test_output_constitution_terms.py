"""Tests for output constitution term alignment (Spec B, 2026-06-08).

Validates that qualitative behavioral terms are unbanned while absolute
threshold/degree prohibitions remain intact.
"""

from pathlib import Path


def _constitution_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent.parent
        / "skills" / "custom" / "ethoinsight" / "references" / "output-constitution.md"
    )


def _ban_table_rows(text: str) -> list[str]:
    """Extract lines from the banned-terms markdown table (| 违规模式 | 违规术语 |)."""
    rows = []
    in_table = False
    for line in text.split("\n"):
        if "违规模式" in line and "违规术语" in line:
            in_table = True
            continue
        if in_table:
            if line.startswith("|"):
                rows.append(line)
            else:
                break  # table ended
    return rows


def _row_for(text: str, label: str) -> str:
    """Return the ban-table row whose first column contains `label`."""
    for row in _ban_table_rows(text):
        if label in row:
            return row
    return ""


def test_qualitative_terms_no_longer_banned():
    """焦虑样行为 / 提示焦虑 should be removed from the ban table."""
    text = _constitution_path().read_text(encoding="utf-8")
    anxiety_degree_row = _row_for(text, "绝对焦虑程度")
    assert anxiety_degree_row, "Should still have an '绝对焦虑程度判读' row"
    assert "焦虑样行为" not in anxiety_degree_row, (
        "焦虑样行为 must NOT appear in the ban row — it was unbanned"
    )
    assert "提示焦虑" not in anxiety_degree_row, (
        "提示焦虑 must NOT appear in the ban row — it was unbanned"
    )
    # The row should still ban absolute-degree judgments
    assert "高焦虑" in anxiety_degree_row
    assert "低焦虑" in anxiety_degree_row


def test_absolute_threshold_still_banned():
    """Absolute threshold bans (line 17) remain intact."""
    text = _constitution_path().read_text(encoding="utf-8")
    threshold_row = _row_for(text, "绝对阈值判读")
    assert threshold_row, "Should still have an '绝对阈值判读' row"
    assert "正常范围" in threshold_row
    assert "典型值" in threshold_row
    assert "常模" in threshold_row
    assert "基线水平" in threshold_row
    # Discrimination principles section must also clarify "still banned"
    assert "绝对阈值" in text


def test_absolute_normal_still_banned():
    """Absolute normal bans (line 19) remain intact."""
    text = _constitution_path().read_text(encoding="utf-8")
    normal_row = _row_for(text, "绝对正常判读")
    assert normal_row, "Should still have an '绝对正常判读' row"
    assert "正常活动" in normal_row
    assert "正常行为" in normal_row


def test_constitution_has_qualitative_vs_threshold_distinction():
    """Newly added discrimination principles section exists."""
    text = _constitution_path().read_text(encoding="utf-8")
    assert "定性行为术语" in text, (
        "Should have the new '定性行为术语 vs 绝对阈值/程度' section"
    )
    assert "趋近-回避" in text, (
        "Should mention '趋近-回避冲突' as an example of allowed qualitative term"
    )
    assert "焦虑样行为" in text, (
        "焦虑样行为 should appear somewhere in the text (in allowed section)"
    )
    # The mnemonic口诀 should be present
    assert "口诀" in text


def test_group_comparison_qualitative_example():
    """The correct example combining group comparison + qualitative term is present."""
    text = _constitution_path().read_text(encoding="utf-8")
    assert "组间比较 + 定性术语" in text, (
        "Should have an example marked '组间比较 + 定性术语, ✅'"
    )
    assert "提示焦虑样行为增加" in text, (
        "Should have the concrete example with '提示焦虑样行为增加'"
    )
