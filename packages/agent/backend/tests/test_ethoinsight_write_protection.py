"""Red anchor tests: ethoinsight/ library code is immutable (S4).

These tests verify that agents CANNOT modify ethoinsight/ library code
via write_file, str_replace, or bash operations. All tests expect DENY.

If any test unexpectedly passes (= write is allowed), CI fails —
the protection has been broken by a configuration change.

Protection chain under test:

| Layer  | File                              | Key lines          | Scope                    |
|--------|-----------------------------------|--------------------|--------------------------|
| Sandbox path validation | sandbox/tools.py             | validate_local_tool_path:645-700 | All agents' write_file/str_replace |
| Script invocation guard | guardrails/script_invocation_only_provider.py | _is_path_safe:161 | code-executor/chart-maker bash ops |
"""

import pytest

# Virtual paths that sandbox agents operate in.
# ethoinsight/ is installed inside .venv as a Python package — these
# paths are NOT under /mnt/user-data/, so they must be denied.
_ETHOINSIGHT_VENV_PATHS = [
    "/mnt/.venv/lib/python3.12/site-packages/ethoinsight/validate.py",
    "/mnt/.venv/lib/python3.12/site-packages/ethoinsight/metrics.py",
    "/mnt/.venv/lib/python3.12/site-packages/ethoinsight/statistics.py",
    "/mnt/.venv/lib/python3.12/site-packages/ethoinsight/__init__.py",
]

_SANDBOX_WORKSPACE_BASE = "/mnt/user-data/workspace"


class TestSandboxWriteProtection:
    """Layer 1: validate_local_tool_path denies writes outside /mnt/user-data/."""

    @pytest.mark.parametrize("ethoinsight_path", _ETHOINSIGHT_VENV_PATHS)
    def test_write_file_to_ethoinsight_path_is_denied(self, ethoinsight_path):
        """validate_local_tool_path must raise PermissionError for writes to
        paths outside /mnt/user-data/ (including .venv/site-packages/ethoinsight/)."""
        from deerflow.sandbox.tools import validate_local_tool_path

        # Simulate a write attempt to ethoinsight/ via minimal thread_data mock.
        thread_data = _make_thread_data()

        with pytest.raises(PermissionError):
            validate_local_tool_path(ethoinsight_path, thread_data, read_only=False)

    @pytest.mark.parametrize("ethoinsight_path", _ETHOINSIGHT_VENV_PATHS)
    def test_str_replace_to_ethoinsight_path_is_denied(self, ethoinsight_path):
        """Same guard denies str_replace (write tool) to ethoinsight paths."""
        from deerflow.sandbox.tools import validate_local_tool_path

        thread_data = _make_thread_data()

        with pytest.raises(PermissionError):
            validate_local_tool_path(ethoinsight_path, thread_data, read_only=False)

    def test_write_to_user_data_is_allowed(self):
        """Sanity check: writes to /mnt/user-data/ are permitted."""
        from deerflow.sandbox.tools import validate_local_tool_path

        thread_data = _make_thread_data()
        # Must not raise.
        validate_local_tool_path(
            "/mnt/user-data/workspace/output.txt", thread_data, read_only=False
        )


class TestScriptInvocationWriteProtection:
    """Layer 2: ScriptInvocationOnlyProvider._is_path_safe denies .venv/site-packages."""

    @pytest.mark.parametrize(
        "target_path",
        [
            # Direct .venv writes
            "/mnt/.venv/lib/python3.12/site-packages/ethoinsight/metrics.py",
            "/mnt/.venv/site-packages/ethoinsight/__init__.py",
            "site-packages/ethoinsight/validate.py",
            # Relative path with .venv
            "../.venv/lib/site-packages/ethoinsight/statistics.py",
            # Path containing .venv anywhere
            "/mnt/user-data/workspace/../../../.venv/x.py",
        ],
    )
    def test_bash_path_containing_venv_is_denied(self, target_path):
        """_is_path_safe returns False when path contains .venv or site-packages."""
        from deerflow.guardrails.script_invocation_only_provider import _is_path_safe

        assert _is_path_safe(target_path) is False, (
            f"Expected _is_path_safe to deny path: {target_path}"
        )

    def test_bash_path_in_user_data_is_allowed(self):
        """Sanity check: writes within /mnt/user-data/ are permitted."""
        from deerflow.guardrails.script_invocation_only_provider import _is_path_safe

        assert _is_path_safe("/mnt/user-data/workspace/output.csv") is True

    def test_lookalike_paths_not_in_user_data_are_denied(self):
        """Paths that look safe but are outside /mnt/user-data/ must be denied."""
        from deerflow.guardrails.script_invocation_only_provider import _is_path_safe

        # Not starting with /mnt/user-data/
        assert _is_path_safe("/tmp/output.csv") is False
        assert _is_path_safe("/etc/passwd") is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_thread_data():
    """Create a minimal ThreadDataState mock for path validation."""
    from unittest.mock import MagicMock

    td = MagicMock()
    td.thread_id = "test-thread"
    td.user_id = "test-user"
    return td
