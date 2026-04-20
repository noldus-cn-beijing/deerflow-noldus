"""Tests for ArchivingSummarizationMiddleware file-backed summary behaviour."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Keep hermetic — archiving_summarization imports deerflow.config.paths.
sys.modules.setdefault("deerflow.subagents.executor", MagicMock())

from deerflow.agents.middlewares.archiving_summarization import (  # noqa: E402
    SUMMARY_FILENAME,
    SUMMARY_VIRTUAL_PATH,
    ArchivingSummarizationMiddleware,
)


def _make_middleware() -> ArchivingSummarizationMiddleware:
    return ArchivingSummarizationMiddleware(model=MagicMock())


def _make_runtime(thread_id: str | None = "test-thread") -> MagicMock:
    runtime = MagicMock()
    runtime.context = {"thread_id": thread_id} if thread_id else {}
    return runtime


class TestBuildFileBackedMessages:
    """_build_file_backed_messages writes the summary file and returns a
    hidden pointer HumanMessage."""

    def test_writes_summary_file_and_returns_hidden_pointer(self, tmp_path, monkeypatch):
        mw = _make_middleware()
        runtime = _make_runtime()

        fake_paths = MagicMock()
        workspace = tmp_path / "threads" / "test-thread" / "user-data" / "workspace"
        fake_paths.sandbox_work_dir.return_value = workspace
        monkeypatch.setattr(
            "deerflow.agents.middlewares.archiving_summarization.get_paths",
            lambda: fake_paths,
        )

        messages = mw._build_file_backed_messages("此前讨论了斑马鱼 shoaling 分析。", runtime)

        # Pointer is a single hidden HumanMessage.
        assert len(messages) == 1
        ptr = messages[0]
        assert ptr.type == "human"
        assert ptr.additional_kwargs.get("hide_from_ui") is True
        assert SUMMARY_VIRTUAL_PATH in ptr.content
        assert ptr.additional_kwargs.get("conversation_summary_path") == SUMMARY_VIRTUAL_PATH

        # Summary file exists with the summary body + a timestamped heading.
        summary_file = workspace / SUMMARY_FILENAME
        assert summary_file.exists()
        content = summary_file.read_text(encoding="utf-8")
        assert "# Conversation Summary" in content
        assert "## 压缩于 " in content
        assert "此前讨论了斑马鱼 shoaling 分析。" in content

    def test_appends_across_multiple_compactions(self, tmp_path, monkeypatch):
        mw = _make_middleware()
        runtime = _make_runtime()

        fake_paths = MagicMock()
        workspace = tmp_path / "workspace"
        fake_paths.sandbox_work_dir.return_value = workspace
        monkeypatch.setattr(
            "deerflow.agents.middlewares.archiving_summarization.get_paths",
            lambda: fake_paths,
        )

        mw._build_file_backed_messages("第一次压缩的内容", runtime)
        mw._build_file_backed_messages("第二次压缩的内容", runtime)

        content = (workspace / SUMMARY_FILENAME).read_text(encoding="utf-8")
        assert content.count("# Conversation Summary") == 1  # header only once
        assert content.count("## 压缩于 ") == 2
        assert "第一次压缩的内容" in content
        assert "第二次压缩的内容" in content
        # Second summary appears after the first.
        assert content.index("第一次压缩的内容") < content.index("第二次压缩的内容")

    def test_falls_back_when_no_thread_id(self, monkeypatch):
        mw = _make_middleware()
        runtime = _make_runtime(thread_id=None)

        # get_paths should NOT even be called since we can't resolve thread_id,
        # but patch defensively in case the implementation changes.
        monkeypatch.setattr(
            "deerflow.agents.middlewares.archiving_summarization.get_paths",
            lambda: (_ for _ in ()).throw(AssertionError("should not fetch paths")),
        )

        messages = mw._build_file_backed_messages("summary text", runtime)

        # Fallback: inline HumanMessage with upstream "Here is a summary..." format,
        # but still hidden from UI.
        assert len(messages) == 1
        assert messages[0].type == "human"
        assert "summary text" in messages[0].content
        assert messages[0].additional_kwargs.get("hide_from_ui") is True

    def test_falls_back_on_file_write_failure(self, tmp_path, monkeypatch):
        mw = _make_middleware()
        runtime = _make_runtime()

        # Point workspace at a location that will fail on mkdir/write.
        fake_paths = MagicMock()
        # Use a path whose parent is a regular file to force OSError on mkdir.
        bad_parent = tmp_path / "blocker"
        bad_parent.write_text("not a directory", encoding="utf-8")
        fake_paths.sandbox_work_dir.return_value = bad_parent / "workspace"
        monkeypatch.setattr(
            "deerflow.agents.middlewares.archiving_summarization.get_paths",
            lambda: fake_paths,
        )

        messages = mw._build_file_backed_messages("summary text", runtime)

        # Falls back to inline HumanMessage on failure.
        assert len(messages) == 1
        assert messages[0].type == "human"
        assert "summary text" in messages[0].content
        assert messages[0].additional_kwargs.get("hide_from_ui") is True


class TestArchiveMessages:
    """The original archive_messages behavior (JSON snapshot for frontend
    history recovery) must still work. This is regression coverage for the
    behavior that existed before B2'."""

    def test_archives_messages_to_json(self, tmp_path, monkeypatch):
        from langchain_core.messages import AIMessage, HumanMessage

        mw = _make_middleware()
        runtime = _make_runtime()

        fake_paths = MagicMock()
        thread_dir = tmp_path / "threads" / "test-thread"
        fake_paths.thread_dir.return_value = thread_dir
        monkeypatch.setattr(
            "deerflow.agents.middlewares.archiving_summarization.get_paths",
            lambda: fake_paths,
        )

        to_archive = [
            HumanMessage(content="user says hi"),
            AIMessage(content="assistant replies"),
        ]
        mw._archive_messages(to_archive, runtime)

        archive_dir = thread_dir / "archived_messages"
        files = list(archive_dir.glob("*.json"))
        assert len(files) == 1
        payload = json.loads(files[0].read_text(encoding="utf-8"))
        assert payload["message_count"] == 2
        assert len(payload["messages"]) == 2

    def test_archive_noop_when_no_thread_id(self, monkeypatch):
        from langchain_core.messages import HumanMessage

        mw = _make_middleware()
        runtime = _make_runtime(thread_id=None)

        monkeypatch.setattr(
            "deerflow.agents.middlewares.archiving_summarization.get_paths",
            lambda: (_ for _ in ()).throw(AssertionError("should not fetch paths")),
        )

        # Should not raise even though there's no thread_id.
        mw._archive_messages([HumanMessage(content="hi")], runtime)

    def test_archive_noop_with_empty_messages(self, tmp_path, monkeypatch):
        mw = _make_middleware()
        runtime = _make_runtime()

        # Even with a valid runtime and paths, an empty list should be a noop.
        fake_paths = MagicMock()
        fake_paths.thread_dir.return_value = tmp_path / "threads" / "test-thread"
        monkeypatch.setattr(
            "deerflow.agents.middlewares.archiving_summarization.get_paths",
            lambda: fake_paths,
        )

        mw._archive_messages([], runtime)

        # Archive dir may or may not be created (implementation choice); we
        # just care no JSON was written.
        archive_dir: Path = tmp_path / "threads" / "test-thread" / "archived_messages"
        if archive_dir.exists():
            assert list(archive_dir.glob("*.json")) == []
