"""Tests for TodoPlanningDisciplineProvider."""

from deerflow.guardrails.provider import GuardrailDecision, GuardrailRequest
from deerflow.guardrails.todo_planning_discipline_provider import TodoPlanningDisciplineProvider


def _make_request(args: dict) -> GuardrailRequest:
    return GuardrailRequest(tool_name="write_todos", tool_input=args)


def _make_todos(items: list[dict]) -> list[dict]:
    """Helper to build todo items with defaults."""
    return [
        {"content": item.get("content", ""), "activeForm": item.get("activeForm", ""), "status": item.get("status", "pending")}
        for item in items
    ]


class TestTodoPlanningDisciplineProvider:
    def test_first_two_calls_always_allowed(self):
        provider = TodoPlanningDisciplineProvider(planning_budget=2)
        todos = _make_todos([{"content": "任务1", "activeForm": "执行任务1"}])

        r1 = provider.evaluate(_make_request({"todos": todos}))
        assert r1.allow is True

        r2 = provider.evaluate(_make_request({"todos": todos}))
        assert r2.allow is True

    def test_third_call_same_content_denied(self):
        """3rd call with identical content and status → deny."""
        provider = TodoPlanningDisciplineProvider(planning_budget=2)
        todos = _make_todos([{"content": "任务1", "activeForm": "执行任务1"}])

        provider.evaluate(_make_request({"todos": todos}))  # 1
        provider.evaluate(_make_request({"todos": todos}))  # 2
        r3 = provider.evaluate(_make_request({"todos": todos}))  # 3
        assert r3.allow is False
        assert r3.reasons[0].code == "todo.discipline"

    def test_status_change_allowed(self):
        """Content same, status changed → allow (legitimate status transition)."""
        provider = TodoPlanningDisciplineProvider(planning_budget=2)
        todos_pending = _make_todos([{"content": "任务1", "activeForm": "执行任务1", "status": "pending"}])
        todos_in_progress = _make_todos([{"content": "任务1", "activeForm": "执行任务1", "status": "in_progress"}])

        provider.evaluate(_make_request({"todos": todos_pending}))  # 1
        provider.evaluate(_make_request({"todos": todos_pending}))  # 2
        r3 = provider.evaluate(_make_request({"todos": todos_in_progress}))  # 3: status changed
        assert r3.allow is True

    def test_content_change_allowed(self):
        """New item added → allow."""
        provider = TodoPlanningDisciplineProvider(planning_budget=2)
        todos_1 = _make_todos([{"content": "任务1", "activeForm": "执行任务1"}])
        todos_2 = _make_todos([{"content": "任务1", "activeForm": "执行任务1"}, {"content": "任务2", "activeForm": "执行任务2"}])

        provider.evaluate(_make_request({"todos": todos_1}))  # 1
        provider.evaluate(_make_request({"todos": todos_1}))  # 2
        r3 = provider.evaluate(_make_request({"todos": todos_2}))  # 3: content changed
        assert r3.allow is True

    def test_reason_parameter_always_allowed(self):
        """reason parameter → always allow, bypass signature check."""
        provider = TodoPlanningDisciplineProvider(planning_budget=2)
        todos = _make_todos([{"content": "任务1", "activeForm": "执行任务1"}])

        provider.evaluate(_make_request({"todos": todos}))  # 1
        provider.evaluate(_make_request({"todos": todos}))  # 2
        r3 = provider.evaluate(_make_request({"todos": todos, "reason": "用户要求重新整理计划"}))  # 3
        assert r3.allow is True

    def test_non_write_todos_passes_through(self):
        """Non-write_todos tools should always be allowed."""
        provider = TodoPlanningDisciplineProvider()
        r = provider.evaluate(GuardrailRequest(tool_name="read_file", tool_input={"path": "/foo"}))
        assert r.allow is True
