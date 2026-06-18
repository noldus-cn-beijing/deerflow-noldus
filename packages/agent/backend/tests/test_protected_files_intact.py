"""Invariant #6: Noldus protected customizations survive ``scripts/sync-deerflow.sh``.

This is a *liveness* check, not a behavior test: it confirms that the
Noldus-specific anchor strings in protected files still exist after a sync
run. If sync wipes a customization (e.g. an upstream ``paraphrase-merge``
that silently weakens a constraint umbrella sentence — see memory
feedback_sync_protected_file_paraphrase_merge_weakens_constitution), the
anchor grep fails and this test goes red before the regression ships.

The PROTECTED_FILES list is parsed from ``scripts/sync-deerflow.sh`` (the
SSOT — the same list the sync script itself consults), so this test stays
in sync with what the script protects rather than drifting to a hardcoded
copy. See CLAUDE.md "DeerFlow fork 策略" / scripts/sync-deerflow.sh:51.
"""

import re
from pathlib import Path

# tests/test_protected_files_intact.py
#   parents[0]=tests  [1]=backend  [2]=agent  [3]=packages  [4]=repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]
_SYNC_SCRIPT = _REPO_ROOT / "scripts" / "sync-deerflow.sh"
# sync-deerflow.sh resolves PROTECTED_FILES relative to UPSTREAM_HARNESS ==
# packages/harness/deerflow (the vendored harness under the backend workspace).
_HARNESS_ROOT = Path(__file__).resolve().parent.parent / "packages" / "harness" / "deerflow"


def _parse_protected_files() -> list[str]:
    """Parse the ``PROTECTED_FILES=( ... )`` bash array from sync-deerflow.sh.

    Strips ``# comments`` and extracts the double-quoted path on each line.
    Raises AssertionError if the array is not found — guarding the SSOT itself.
    """
    src = _SYNC_SCRIPT.read_text(encoding="utf-8")
    m = re.search(r"PROTECTED_FILES=\(\s*(.*?)\s*\)", src, re.DOTALL)
    assert m, f"PROTECTED_FILES array not found in {_SYNC_SCRIPT}"
    paths: list[str] = []
    for line in m.group(1).splitlines():
        line = re.sub(r"#.*$", "", line).strip()
        mm = re.match(r'"([^"]+)"', line)
        if mm:
            paths.append(mm.group(1))
    return paths


# Noldus-specific anchor strings that MUST survive in each protected file.
# Each entry is a customization that, if wiped by sync, breaks agent behavior
# silently (program runs but loses Chinese scheduling rules / subagent
# registration / shared-path semantics / MCP truncation). Adding a new anchor?
# Ensure the file is also listed in PROTECTED_FILES — the test enforces that
# (registry hygiene) below.
ANCHOR_STRINGS: dict[str, tuple[str, ...]] = {
    # Chinese scheduling rules + ethoinsight subagent ordering + Gate mechanism.
    "agents/lead_agent/prompt.py": (
        "noldus_order",
        "set_experiment_paradigm",
        "from deerflow.subagents.builtins import BUILTIN_SUBAGENTS",
    ),
    # Registration of the 4 ethoinsight subagents (code-executor/data-analyst/
    # chart-maker/report-writer/knowledge-assistant) — upstream has none.
    "subagents/builtins/__init__.py": (
        "BUILTIN_SUBAGENTS = {",
        "CODE_EXECUTOR_CONFIG",
        "DATA_ANALYST_CONFIG",
        "REPORT_WRITER_CONFIG",
        "__all__ = [",
    ),
    # MCP tool-result truncation to keep LLM context bounded — Noldus addition.
    "mcp/tools.py": (
        "MCP_TOOL_RESULT_MAX_CHARS = 4096",
        "def _truncate_result",
    ),
    # Shared workspace path /mnt/shared for cross-thread + subagent comms.
    "sandbox/tools.py": (
        "SHARED_PATH_PREFIX",
        "/mnt/shared",
    ),
    # Where SHARED_PATH_PREFIX is defined (this file is itself protected).
    "config/paths.py": (
        'SHARED_PATH_PREFIX = "/mnt/shared"',
    ),
}


def test_protected_files_list_is_nonempty_and_parseable():
    """Guard the SSOT itself: PROTECTED_FILES must parse to a real list."""
    paths = _parse_protected_files()
    assert len(paths) >= 20, (
        f"Expected >=20 protected files in sync-deerflow.sh, got {len(paths)} "
        "— the SSOT may have been restructured; update _parse_protected_files."
    )


def test_protected_noldus_anchors_survive():
    """Each anchor string in ANCHOR_STRINGS must still exist in its file.

    Failure here means a Noldus protected customization was wiped — the exact
    sync regression (silent weakening / full overwrite of a protected file)
    this invariant exists to catch.
    """
    protected = set(_parse_protected_files())
    missing: list[str] = []

    for rel, anchors in ANCHOR_STRINGS.items():
        # Registry hygiene: a file we anchor-check must itself be protected,
        # otherwise sync can silently overwrite it and these anchors vanish.
        assert rel in protected, (
            f"{rel} has anchors in ANCHOR_STRINGS but is NOT in PROTECTED_FILES "
            "— add it to scripts/sync-deerflow.sh PROTECTED_FILES, or it can be "
            "silently overwritten by an upstream sync."
        )
        target = _HARNESS_ROOT / rel
        assert target.exists(), f"Protected file missing on disk: {target}"
        content = target.read_text(encoding="utf-8")
        for anchor in anchors:
            if anchor not in content:
                missing.append(f"  {rel}: missing anchor {anchor!r}")

    assert not missing, (
        "Noldus protected customization wiped (sync regression detected) — "
        "review the latest scripts/sync-deerflow.sh run:\n"
        + "\n".join(missing)
    )
