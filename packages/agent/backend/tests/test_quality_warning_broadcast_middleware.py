"""Tests for QualityWarningBroadcastMiddleware.

Verifies that data-analyst handoff quality_warnings get attached to the lead's
broadcast AIMessage `additional_kwargs` so the frontend banner can render.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agents.middlewares.quality_warning_broadcast_middleware import (
    QualityWarningBroadcastMiddleware,
)


def _write_handoff(workspace: Path, warnings: list[dict]) -> None:
    payload = {
        "status": "completed",
        "quality_warnings": warnings,
    }
    (workspace / "handoff_data_analyst.json").write_text(json.dumps(payload), encoding="utf-8")


def _build_state(
    workspace: Path,
    messages: list,
) -> dict:
    return {
        "messages": messages,
        "thread_data": {"workspace_path": str(workspace)},
        "thread_id": "test-thread",
    }


def _critical_warning() -> dict:
    return {
        "severity": "critical",
        "code": "SAMPLE.TOO_SMALL",
        "metric": "all",
        "message": "Group 'ctrl' has n=2 (<3).",
        "evidence": {"n": 2, "threshold": 3, "group": "ctrl"},
        "blocks_downstream": True,
    }


def _data_analyst_dispatch(tool_call_id: str = "call_da_1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {"subagent_type": "data-analyst", "description": "解读结果"},
                "id": tool_call_id,
                "type": "tool_call",
            }
        ],
    )


def _tool_message(tool_call_id: str = "call_da_1") -> ToolMessage:
    return ToolMessage(content="Task Succeeded.\n\n## 最终结果\n...", tool_call_id=tool_call_id, name="task")


def _broadcast_ai_message(content: str = "已收到 data-analyst 结果: 1 条阻断级警告...") -> AIMessage:
    return AIMessage(content=content)


class TestInjectsQualityWarnings:
    """Happy path: handoff exists with warnings → kwargs injected."""

    def test_injects_warnings_into_last_ai_message(self, tmp_path):
        warnings = [_critical_warning()]
        _write_handoff(tmp_path, warnings)

        broadcast = _broadcast_ai_message()
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="分析这批数据"),
                _data_analyst_dispatch(),
                _tool_message(),
                broadcast,
            ],
        )

        mw = QualityWarningBroadcastMiddleware()
        result = mw.after_model(state, runtime=None)

        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1

        updated = result["messages"][0]
        assert isinstance(updated, AIMessage)
        assert updated.id == broadcast.id  # same-id replacement
        injected = updated.additional_kwargs["quality_warnings"]
        assert len(injected) == 1
        assert injected[0]["code"] == "SAMPLE.TOO_SMALL"
        assert injected[0]["blocks_downstream"] is True

    def test_injects_empty_list_when_no_warnings(self, tmp_path):
        _write_handoff(tmp_path, [])

        broadcast = _broadcast_ai_message()
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="分析"),
                _data_analyst_dispatch(),
                _tool_message(),
                broadcast,
            ],
        )

        mw = QualityWarningBroadcastMiddleware()
        result = mw.after_model(state, runtime=None)

        assert result is not None
        updated = result["messages"][0]
        assert updated.additional_kwargs["quality_warnings"] == []

    def test_data_analyst_underscore_alias_also_triggers(self, tmp_path):
        """subagent_type='data_analyst' (underscore) should also match."""
        _write_handoff(tmp_path, [_critical_warning()])

        dispatch = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"subagent_type": "data_analyst"},
                    "id": "call_da_2",
                    "type": "tool_call",
                }
            ],
        )
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="x"),
                dispatch,
                ToolMessage(content="Task Succeeded.", tool_call_id="call_da_2", name="task"),
                _broadcast_ai_message(),
            ],
        )

        result = QualityWarningBroadcastMiddleware().after_model(state, runtime=None)
        assert result is not None


class TestSkipsWhenNotApplicable:
    """Cases where the middleware must NOT modify state."""

    def test_skips_when_no_handoff_file(self, tmp_path):
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="x"),
                _data_analyst_dispatch(),
                _tool_message(),
                _broadcast_ai_message(),
            ],
        )
        result = QualityWarningBroadcastMiddleware().after_model(state, runtime=None)
        assert result is None

    def test_skips_when_no_workspace_path(self, tmp_path):
        state = {
            "messages": [
                HumanMessage(content="x"),
                _data_analyst_dispatch(),
                _tool_message(),
                _broadcast_ai_message(),
            ],
            "thread_data": {},  # no workspace_path
            "thread_id": "x",
        }
        result = QualityWarningBroadcastMiddleware().after_model(state, runtime=None)
        assert result is None

    def test_skips_when_last_message_has_tool_calls(self, tmp_path):
        """Lead is still dispatching tools, not in the broadcast turn yet."""
        _write_handoff(tmp_path, [_critical_warning()])

        # Last AIMessage has another tool call queued
        still_dispatching = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"subagent_type": "chart-maker"},
                    "id": "call_cm_1",
                    "type": "tool_call",
                }
            ],
        )
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="x"),
                _data_analyst_dispatch(),
                _tool_message(),
                still_dispatching,
            ],
        )
        result = QualityWarningBroadcastMiddleware().after_model(state, runtime=None)
        assert result is None

    def test_skips_when_no_prior_data_analyst_call(self, tmp_path):
        """Last AIMessage but no data-analyst task in history → don't inject."""
        _write_handoff(tmp_path, [_critical_warning()])

        # Only a chart-maker task in history, no data-analyst
        chart_dispatch = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"subagent_type": "chart-maker"},
                    "id": "call_cm_1",
                    "type": "tool_call",
                }
            ],
        )
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="x"),
                chart_dispatch,
                ToolMessage(content="Task Succeeded.", tool_call_id="call_cm_1", name="task"),
                _broadcast_ai_message(),
            ],
        )
        result = QualityWarningBroadcastMiddleware().after_model(state, runtime=None)
        assert result is None

    def test_idempotent_skips_when_already_injected(self, tmp_path):
        """If additional_kwargs.quality_warnings already set, do nothing."""
        _write_handoff(tmp_path, [_critical_warning()])

        already_broadcast = AIMessage(
            content="已收到 data-analyst 结果...",
            additional_kwargs={"quality_warnings": [_critical_warning()]},
        )
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="x"),
                _data_analyst_dispatch(),
                _tool_message(),
                already_broadcast,
            ],
        )
        result = QualityWarningBroadcastMiddleware().after_model(state, runtime=None)
        assert result is None

    def test_skips_when_last_is_not_ai_message(self, tmp_path):
        _write_handoff(tmp_path, [_critical_warning()])
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="x"),
                _data_analyst_dispatch(),
                _tool_message(),  # last is ToolMessage, not AIMessage
            ],
        )
        result = QualityWarningBroadcastMiddleware().after_model(state, runtime=None)
        assert result is None

    def test_skips_when_handoff_is_invalid_json(self, tmp_path):
        (tmp_path / "handoff_data_analyst.json").write_text("not json{{{", encoding="utf-8")

        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="x"),
                _data_analyst_dispatch(),
                _tool_message(),
                _broadcast_ai_message(),
            ],
        )
        result = QualityWarningBroadcastMiddleware().after_model(state, runtime=None)
        assert result is None

    def test_skips_when_quality_warnings_field_missing(self, tmp_path):
        (tmp_path / "handoff_data_analyst.json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8"
        )
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="x"),
                _data_analyst_dispatch(),
                _tool_message(),
                _broadcast_ai_message(),
            ],
        )
        result = QualityWarningBroadcastMiddleware().after_model(state, runtime=None)
        assert result is None

    def test_skips_when_quality_warnings_is_not_list(self, tmp_path):
        (tmp_path / "handoff_data_analyst.json").write_text(
            json.dumps({"quality_warnings": "oops"}), encoding="utf-8"
        )
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="x"),
                _data_analyst_dispatch(),
                _tool_message(),
                _broadcast_ai_message(),
            ],
        )
        result = QualityWarningBroadcastMiddleware().after_model(state, runtime=None)
        assert result is None


class TestChartmakerInterleaving:
    """Real-world scenario: lead runs data-analyst, then chart-maker, then broadcasts."""

    def test_injects_after_chartmaker_interleave(self, tmp_path):
        """data-analyst → chart-maker → broadcast → still inject from data-analyst handoff."""
        _write_handoff(tmp_path, [_critical_warning()])

        chart_dispatch = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"subagent_type": "chart-maker"},
                    "id": "call_cm_1",
                    "type": "tool_call",
                }
            ],
        )
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="x"),
                _data_analyst_dispatch("call_da_1"),
                _tool_message("call_da_1"),
                chart_dispatch,
                ToolMessage(content="Task Succeeded.", tool_call_id="call_cm_1", name="task"),
                _broadcast_ai_message(),
            ],
        )

        result = QualityWarningBroadcastMiddleware().after_model(state, runtime=None)
        assert result is not None
        updated = result["messages"][0]
        assert updated.additional_kwargs["quality_warnings"][0]["code"] == "SAMPLE.TOO_SMALL"


class TestAsyncEntrypoint:
    @pytest.mark.asyncio
    async def test_aafter_model_mirrors_sync_path(self, tmp_path):
        _write_handoff(tmp_path, [_critical_warning()])
        state = _build_state(
            tmp_path,
            [
                HumanMessage(content="x"),
                _data_analyst_dispatch(),
                _tool_message(),
                _broadcast_ai_message(),
            ],
        )
        mw = QualityWarningBroadcastMiddleware()
        result = await mw.aafter_model(state, runtime=None)
        assert result is not None
        assert result["messages"][0].additional_kwargs["quality_warnings"]
