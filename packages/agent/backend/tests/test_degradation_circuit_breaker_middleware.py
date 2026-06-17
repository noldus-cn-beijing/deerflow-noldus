"""Tests for DegradationCircuitBreakerMiddleware (L1 熔断器, spec 2026-06-17 P7).

P7: lead 的 after_model 读 code-executor handoff 的 gate_signals.statistics_status，
检测到 'crashed'（损害可复现性）→ 自救一次（jump_to=model，提醒重派 code-executor）→
自救超限转 HITL（提醒模型调 ask_clarification 问用户）。只熔断 crashed；
absent_by_design（单组合理 skip）/ ok 只通知不熔断。

复刻 SealGateMiddleware 的测试范式（fake runtime + 构造 state + 写 handoff json）。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares.degradation_circuit_breaker_middleware import (
    _MAX_SELF_HELP,
    DegradationCircuitBreakerMiddleware,
)


def _make_runtime(run_id: str = "test-run") -> MagicMock:
    rt = MagicMock()
    rt.run_id = run_id
    return rt


def _make_state(workspace: Path, messages: list, *, handoff: dict | None = None) -> dict:
    """Build a lead state whose thread_data.workspace_path points at a real workspace.

    If ``handoff`` is given, write handoff_code_executor.json into the workspace.
    """
    if handoff is not None:
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "handoff_code_executor.json").write_text(
            json.dumps(handoff, ensure_ascii=False), encoding="utf-8"
        )
    return {
        "messages": messages,
        "thread_data": {"workspace_path": str(workspace)},
    }


def _ai_no_tools(content: str = "分析完成，准备播报") -> AIMessage:
    return AIMessage(content=content)


def _ai_with_tool_call() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "task", "args": {"subagent_type": "data-analyst"}, "id": "tc1"}],
    )


def _handoff_with_stats_status(stats_status: str, stats_error: str | None = None) -> dict:
    """Minimal valid-ish code-executor handoff carrying gate_signals.statistics_status.

    read_handoff schema-validates via CodeExecutorHandoff; status + gate_signals suffice
    for our signal check (extra fields allowed).
    """
    return {
        "status": "partial",
        "gate_signals": {
            "statistics_status": stats_status,
            "statistics_error": stats_error,
        },
    }


class TestDegradationCircuitBreakerMiddleware:
    """Unit tests for DegradationCircuitBreakerMiddleware."""

    def test_self_help_on_first_crash(self, tmp_path):
        """crashed + last AI 无 tool_calls + count=0 → 注入自救 reminder + jump_to=model。"""
        mw = DegradationCircuitBreakerMiddleware()
        state = _make_state(
            tmp_path,
            messages=[_ai_no_tools()],
            handoff=_handoff_with_stats_status("crashed", "ZeroDivision: float division by zero"),
        )
        result = mw.after_model(state, _make_runtime())
        assert result is not None
        assert result.get("jump_to") == "model"
        msgs = result.get("messages", [])
        assert len(msgs) == 1
        msg = msgs[0]
        assert isinstance(msg, HumanMessage)
        # 自救 reminder 应引导重派 code-executor，并引用崩溃摘要
        assert "code-executor" in msg.content
        assert "ZeroDivision" in msg.content
        assert msg.additional_kwargs.get("hide_from_ui") is True
        # count 变 1
        assert mw._get_count(_make_runtime()) == 1

    def test_no_retrigger_on_same_handoff_mtime(self, tmp_path):
        """同一 handoff 文件（mtime 不变）第二次调 → None（防同条 crashed 反复触发）。"""
        mw = DegradationCircuitBreakerMiddleware()
        runtime = _make_runtime()
        handoff = _handoff_with_stats_status("crashed", "boom")
        state = _make_state(tmp_path, messages=[_ai_no_tools()], handoff=handoff)

        first = mw.after_model(state, runtime)
        assert first is not None  # 第一次自救
        # 第二次：同文件、同 mtime → 不再触发
        second = mw.after_model(state, runtime)
        assert second is None

    def test_hitl_after_self_help_cap_with_new_handoff(self, tmp_path):
        """自救超限（count>=_MAX_SELF_HELP）+ handoff 被重写（新 mtime，仍 crashed）→ HITL reminder。"""
        mw = DegradationCircuitBreakerMiddleware()
        runtime = _make_runtime()
        handoff_path = tmp_path / "handoff_code_executor.json"

        # 第一次 crashed → 自救
        handoff_path.write_text(
            json.dumps(_handoff_with_stats_status("crashed", "boom1")), encoding="utf-8"
        )
        state = _make_state(tmp_path, messages=[_ai_no_tools()])
        first = mw.after_model(state, runtime)
        assert first is not None
        assert "code-executor" in first["messages"][0].content  # 自救文案

        # 模拟 code-executor 重写 handoff（新内容 + 新 mtime）。为确保 mtime 变化，先改时间戳。
        handoff_path.write_text(
            json.dumps(_handoff_with_stats_status("crashed", "boom2")), encoding="utf-8"
        )
        # 强制 mtime 推进（某些文件系统 mtime 精度低，写入同秒可能不变）
        os.utime(handoff_path, ns=(10**9 * 999999, 10**9 * 999999))

        second = mw.after_model(state, runtime)
        assert second is not None
        # HITL 文案：引导调 ask_clarification
        assert "ask_clarification" in second["messages"][0].content

    def test_no_action_when_ok(self, tmp_path):
        mw = DegradationCircuitBreakerMiddleware()
        state = _make_state(
            tmp_path,
            messages=[_ai_no_tools()],
            handoff=_handoff_with_stats_status("ok"),
        )
        assert mw.after_model(state, _make_runtime()) is None

    def test_no_action_when_absent_by_design(self, tmp_path):
        mw = DegradationCircuitBreakerMiddleware()
        state = _make_state(
            tmp_path,
            messages=[_ai_no_tools()],
            handoff=_handoff_with_stats_status("absent_by_design"),
        )
        assert mw.after_model(state, _make_runtime()) is None

    def test_no_action_when_last_ai_has_tool_calls(self, tmp_path):
        """last AIMessage 带 tool_call（lead 还在派遣）→ 不抢断正常工作流。"""
        mw = DegradationCircuitBreakerMiddleware()
        state = _make_state(
            tmp_path,
            messages=[_ai_with_tool_call()],
            handoff=_handoff_with_stats_status("crashed", "boom"),
        )
        assert mw.after_model(state, _make_runtime()) is None

    def test_no_action_when_no_handoff_file(self, tmp_path):
        mw = DegradationCircuitBreakerMiddleware()
        state = _make_state(tmp_path, messages=[_ai_no_tools()], handoff=None)
        assert mw.after_model(state, _make_runtime()) is None

    def test_fail_open_on_exception(self, tmp_path):
        """state 畸形（无 messages / 无 thread_data）→ fail-open 返回 None。"""
        mw = DegradationCircuitBreakerMiddleware()
        # 无 messages 键
        assert mw.after_model({}, _make_runtime()) is None
        # thread_data 无 workspace_path
        assert mw.after_model({"messages": [_ai_no_tools()]}, _make_runtime()) is None

    def test_per_run_count_isolation(self, tmp_path):
        """两个 runtime 不同 run_id → 自救计数互不影响。"""
        mw = DegradationCircuitBreakerMiddleware()
        handoff = _handoff_with_stats_status("crashed", "boom")
        # run A：自救一次
        state_a = _make_state(tmp_path / "a", messages=[_ai_no_tools()], handoff=handoff)
        rt_a = _make_runtime("run-A")
        res_a1 = mw.after_model(state_a, rt_a)
        assert res_a1 is not None
        # run B：独立 workspace，也是第一次 → 自救（不受 run A 计数影响）
        state_b = _make_state(tmp_path / "b", messages=[_ai_no_tools()], handoff=handoff)
        rt_b = _make_runtime("run-B")
        res_b1 = mw.after_model(state_b, rt_b)
        assert res_b1 is not None
        assert "code-executor" in res_b1["messages"][0].content  # run B 仍是自救（非 HITL）

    def test_max_self_help_is_one(self):
        """产品决策：自救上限 = 1（一次重试后转 HITL）。"""
        assert _MAX_SELF_HELP == 1
