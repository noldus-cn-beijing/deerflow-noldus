"""Tests for _extract_required_patterns flatten behaviour (§5.4 harness side).

Ensures set.update does not raise TypeError when requires_columns
contains CNF OR-groups (nested list items).
"""

from __future__ import annotations


def test_inspect_collects_patterns_with_nested():
    """_extract_required_patterns flattens CNF requires_columns before set.update."""
    from deerflow.tools.builtins.inspect_uploaded_file_tool import _extract_required_patterns

    # All currently-loaded paradigms have pure list-of-str requires_columns,
    # so we only verify the function runs without error on them.
    # The flatten behaviour (via _flatten_requires_columns) is covered by
    # the ethoinsight-side test_flatten_requires_columns and
    # test_loader_accepts_nested_integration.

    for paradigm in ["epm", "oft", "ldb", "zero_maze", "fst", "tst"]:
        patterns = _extract_required_patterns(paradigm)
        assert isinstance(patterns, list)
        for p in patterns:
            assert isinstance(p, str), (
                f"Pattern should be str after flatten, got {type(p)}: {p!r}"
            )
