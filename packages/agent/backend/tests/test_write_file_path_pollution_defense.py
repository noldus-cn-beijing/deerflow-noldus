"""Layer A path-pollution defense at write_file_tool / str_replace_tool boundary.

When a subagent (LLM) embeds host-side absolute paths into the *content* of a
write_file call — typically by serialising the result of Path.resolve() into a
handoff JSON — downstream sandbox-aware tools reject the resulting paths. The
mitigation reuses mask_local_paths_in_output() to reverse-map host paths back to
their /mnt/user-data/... virtual equivalents before the bytes hit disk.

These tests exercise the masking integration in isolation; the masking function
itself has dedicated coverage in test_sandbox_tools_security.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from deerflow.sandbox.tools import str_replace_tool, write_file_tool


class _CapturingSandbox:
    """Fake sandbox that records the (path, content, append) tuples it receives."""

    id = "local:test-thread"

    def __init__(self):
        self.writes: list[tuple[str, str, bool]] = []
        self._files: dict[str, str] = {}

    def write_file(self, path: str, content: str, append: bool = False):
        self.writes.append((path, content, append))
        if append and path in self._files:
            self._files[path] += content
        else:
            self._files[path] = content
        return None

    def read_file(self, path: str) -> str:
        return self._files.get(path, "")


class _DummyRuntime:
    """Bare ToolRuntime stand-in (mirrors test_write_file_schema)."""

    state: dict = {}
    context = None
    config: dict = {}
    stream_writer = None
    tool_call_id = None
    store = None


@pytest.fixture
def fake_thread(tmp_path: Path) -> dict:
    """Per-thread filesystem layout matching ThreadDataMiddleware output."""
    workspace = tmp_path / "workspace"
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    for d in (workspace, uploads, outputs):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "workspace_path": str(workspace),
        "uploads_path": str(uploads),
        "outputs_path": str(outputs),
    }


@pytest.fixture
def fake_sandbox(monkeypatch, fake_thread: dict) -> _CapturingSandbox:
    sandbox = _CapturingSandbox()
    monkeypatch.setattr(
        "deerflow.sandbox.tools.ensure_sandbox_initialized", lambda runtime: sandbox
    )
    monkeypatch.setattr(
        "deerflow.sandbox.tools.ensure_thread_directories_exist", lambda runtime: None
    )
    monkeypatch.setattr("deerflow.sandbox.tools.is_local_sandbox", lambda runtime: True)
    monkeypatch.setattr(
        "deerflow.sandbox.tools.get_thread_data", lambda runtime: fake_thread
    )
    monkeypatch.setattr(
        "deerflow.sandbox.tools.validate_local_tool_path",
        lambda path, td, read_only=False: None,
    )
    monkeypatch.setattr(
        "deerflow.sandbox.tools._is_custom_mount_path", lambda path: False
    )
    # Resolve virtual workspace path to a real host-side temp path so
    # sandbox.write_file gets a writeable target. We override the resolver
    # to use the fake thread's workspace_path directly.
    monkeypatch.setattr(
        "deerflow.sandbox.tools._resolve_and_validate_user_data_path",
        lambda path, td: str(Path(td["workspace_path"]) / Path(path).name),
    )
    monkeypatch.setattr(
        "deerflow.sandbox.tools.get_file_operation_lock",
        lambda sb, path: _NoopLock(),
    )
    return sandbox


class _NoopLock:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestWriteFileReverseMasksHostPaths:
    def test_handoff_with_host_uploads_path_is_rewritten(self, fake_sandbox, fake_thread):
        """A polluted handoff_code_executor.json gets its raw_files normalized to virtual paths."""
        uploads_host = fake_thread["uploads_path"]
        polluted_content = (
            '{"inputs": {"raw_files": ['
            f'"{uploads_host}/trial-1.txt", '
            f'"{uploads_host}/trial-2.txt"'
            "]}, \"status\": \"completed\"}"
        )
        result = write_file_tool.func(
            runtime=_DummyRuntime(),
            description="write handoff",
            path="/mnt/user-data/workspace/handoff_code_executor.json",
            content=polluted_content,
            append=False,
        )
        assert result == "OK"
        assert len(fake_sandbox.writes) == 1
        _, written, _ = fake_sandbox.writes[0]
        assert "/mnt/user-data/uploads/trial-1.txt" in written
        assert "/mnt/user-data/uploads/trial-2.txt" in written
        assert uploads_host not in written

    def test_clean_content_passes_through_unchanged(self, fake_sandbox, fake_thread):
        clean_content = '{"raw_files": ["/mnt/user-data/uploads/x.txt"]}'
        write_file_tool.func(
            runtime=_DummyRuntime(),
            description="clean write",
            path="/mnt/user-data/workspace/clean.json",
            content=clean_content,
            append=False,
        )
        _, written, _ = fake_sandbox.writes[0]
        assert written == clean_content

    def test_mixed_paths_only_host_paths_rewritten(self, fake_sandbox, fake_thread):
        uploads_host = fake_thread["uploads_path"]
        outputs_host = fake_thread["outputs_path"]
        content = (
            f"polluted upload: {uploads_host}/a.txt\n"
            f"polluted output: {outputs_host}/chart.png\n"
            "virtual: /mnt/user-data/workspace/x.json\n"
        )
        write_file_tool.func(
            runtime=_DummyRuntime(),
            description="mixed",
            path="/mnt/user-data/workspace/mix.txt",
            content=content,
            append=False,
        )
        _, written, _ = fake_sandbox.writes[0]
        assert "/mnt/user-data/uploads/a.txt" in written
        assert "/mnt/user-data/outputs/chart.png" in written
        assert uploads_host not in written
        assert outputs_host not in written


class TestStrReplaceReverseMasksHostPaths:
    def test_str_replace_new_str_is_masked(self, fake_sandbox, fake_thread):
        """str_replace also leaks host paths via new_str — verify masking applies."""
        uploads_host = fake_thread["uploads_path"]
        # Pre-populate file under the *resolved* host path that the tool will read.
        workspace_host = fake_thread["workspace_path"]
        resolved_target = f"{workspace_host}/handoff.json"
        fake_sandbox.write_file(resolved_target, '{"placeholder": "REPLACE_ME"}')
        fake_sandbox.writes.clear()  # ignore the seed write
        # Replace "REPLACE_ME" with a polluted host path inside JSON value.
        polluted_replacement = f'["{uploads_host}/trial.txt"]'
        result = str_replace_tool.func(
            runtime=_DummyRuntime(),
            description="patch raw_files",
            path="/mnt/user-data/workspace/handoff.json",
            old_str="REPLACE_ME",
            new_str=polluted_replacement,
            replace_all=False,
        )
        assert result == "OK"
        assert len(fake_sandbox.writes) == 1
        _, written, _ = fake_sandbox.writes[-1]
        assert "/mnt/user-data/uploads/trial.txt" in written
        assert uploads_host not in written
