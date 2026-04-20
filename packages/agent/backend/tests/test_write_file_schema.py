"""Tests for write_file tool: explicit Pydantic args_schema + content size cap.

Covers commit 3 of the EthoInsight pipeline redesign: Sonnet occasionally
failed to serialise the `content` field on very large writes (e.g. 10K-char
APA reports) when the tool used parse_docstring=True. The explicit Pydantic
schema plus an 8000-char fail-fast cap give the model unambiguous requirements
and a clear recovery path (split + append).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from deerflow.sandbox.tools import (
    WRITE_FILE_MAX_CONTENT_CHARS,
    _WriteFileArgs,
    write_file_tool,
)


class TestWriteFileArgsSchema:
    def test_accepts_minimal_valid_input(self):
        args = _WriteFileArgs(
            description="save note",
            path="/mnt/user-data/workspace/note.md",
            content="hello",
        )
        assert args.description == "save note"
        assert args.append is False

    def test_rejects_missing_content(self):
        with pytest.raises(ValidationError) as exc:
            _WriteFileArgs(  # type: ignore[call-arg]
                description="save",
                path="/tmp/x.txt",
            )
        assert "content" in str(exc.value).lower()

    def test_rejects_missing_path(self):
        with pytest.raises(ValidationError):
            _WriteFileArgs(  # type: ignore[call-arg]
                description="save",
                content="x",
            )

    def test_rejects_missing_description(self):
        with pytest.raises(ValidationError):
            _WriteFileArgs(  # type: ignore[call-arg]
                path="/tmp/x.txt",
                content="x",
            )

    def test_accepts_empty_content(self):
        args = _WriteFileArgs(
            description="truncate",
            path="/tmp/x.txt",
            content="",
        )
        assert args.content == ""

    def test_append_flag_roundtrip(self):
        args = _WriteFileArgs(
            description="append chunk",
            path="/tmp/x.txt",
            content="chunk",
            append=True,
        )
        assert args.append is True

    def test_schema_is_bound_to_tool(self):
        """Ensure the tool exposes the explicit schema (not a docstring-parsed one)."""
        assert write_file_tool.args_schema is _WriteFileArgs

    def test_json_schema_marks_content_required(self):
        """The LangChain-generated JSON schema must list `content` as required.

        This is the property Sonnet relies on to always emit `content`.
        """
        schema = _WriteFileArgs.model_json_schema()
        required = set(schema.get("required", []))
        assert {"description", "path", "content"}.issubset(required)

    def test_json_schema_content_description_mentions_split_guidance(self):
        """Content description must hint at append-based splitting for long files."""
        schema = _WriteFileArgs.model_json_schema()
        content_desc = schema["properties"]["content"].get("description", "")
        assert "append" in content_desc.lower()
        assert str(WRITE_FILE_MAX_CONTENT_CHARS) in content_desc


class TestWriteFileOversizeFailFast:
    def test_content_at_threshold_does_not_trigger_error(self, monkeypatch):
        """Exactly 8000 chars should be accepted (strict > comparison)."""
        monkeypatch.setattr(
            "deerflow.sandbox.tools.ensure_sandbox_initialized",
            lambda runtime: _FakeSandbox(),
        )
        monkeypatch.setattr(
            "deerflow.sandbox.tools.ensure_thread_directories_exist",
            lambda runtime: None,
        )
        monkeypatch.setattr(
            "deerflow.sandbox.tools.is_local_sandbox", lambda runtime: False
        )
        result = write_file_tool.func(
            runtime=_DummyRuntime(),
            description="at threshold",
            path="/mnt/user-data/workspace/big.md",
            content="x" * WRITE_FILE_MAX_CONTENT_CHARS,
            append=False,
        )
        assert result == "OK"

    def test_content_over_threshold_returns_fail_fast_with_guidance(self):
        oversize = "x" * (WRITE_FILE_MAX_CONTENT_CHARS + 100)
        result = write_file_tool.func(
            runtime=_DummyRuntime(),
            description="too big",
            path="/mnt/user-data/workspace/big.md",
            content=oversize,
            append=False,
        )
        assert result.startswith("Error:")
        # Must tell the model how many chars are over.
        assert str(len(oversize)) in result
        assert str(WRITE_FILE_MAX_CONTENT_CHARS) in result
        # Must tell the model the recovery path.
        assert "append=True" in result
        assert "split" in result.lower() or "multiple" in result.lower()

    def test_oversize_does_not_touch_sandbox(self, monkeypatch):
        """Fail-fast must happen before sandbox lookup / path validation."""
        sandbox_touched = {"count": 0}

        def fake_ensure(runtime):
            sandbox_touched["count"] += 1
            return _FakeSandbox()

        monkeypatch.setattr(
            "deerflow.sandbox.tools.ensure_sandbox_initialized", fake_ensure
        )
        write_file_tool.func(
            runtime=_DummyRuntime(),
            description="skip sandbox",
            path="/mnt/user-data/workspace/x.md",
            content="x" * (WRITE_FILE_MAX_CONTENT_CHARS + 1),
            append=False,
        )
        assert sandbox_touched["count"] == 0


# ------------ minimal fakes (avoid pulling in full sandbox fixtures) -----------


class _DummyRuntime:
    """Bare stand-in — the real ToolRuntime has too many required fields for a unit test."""

    state: dict = {}
    context = None
    config: dict = {}
    stream_writer = None
    tool_call_id = None
    store = None


class _FakeSandbox:
    id = "fake"

    def write_file(self, path, content, append=False):  # noqa: ARG002
        return None
