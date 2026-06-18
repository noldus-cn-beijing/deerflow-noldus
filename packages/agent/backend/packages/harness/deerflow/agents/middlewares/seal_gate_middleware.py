"""SealGateMiddleware — after_model gate that forces seal_<name>_handoff before exit.

Structural fix for the recurring "terminated without emitting handoff" failure in
data-analyst / chart-maker / report-writer (spec
2026-06-16-seal-gate-middleware-engineering-spec.md).

Root cause: a ReAct agent terminates the moment the model emits an AIMessage with
**no tool_calls**. So "write a pure-text `analysis done` message" naturally ends the
loop — before the seal tool is ever called. Prompt edits ("step 3 must seal") cannot
enforce an ordering ReAct does not have. This middleware changes the **termination
condition itself**: it intercepts in ``after_model`` (before ReAct decides to stop),
and when the model is about to end on pure text but seal has not been called, it
injects a reminder HumanMessage + ``jump_to='model'`` to route control back. With
this gate, the model physically cannot exit a seal-requiring subagent without calling
seal (L1 = structural 100%).

This is a verbatim replication of the ``ParadigmIdentificationGateMiddleware``
pattern (an existing, live precedent in this repo) — same hook shape, same
``@hook_config(can_jump_to=["model"])``, same per-run reminder cap, same fail-open.
No new mechanism.

Behavior (``after_model``), for a subagent that requires seal:
  1. Not a seal-requiring subagent → return None (code-executor / bash / general /
     knowledge pass through; code-executor is already produce-and-deliver merged).
  2. Last AIMessage's tool_calls contain seal_<name>_handoff → return None (it's
     being called right now; allow).
  3. A seal ToolMessage already exists in history → return None (already sealed).
  4. Last AIMessage still carries other tool_calls → return None (still working,
     hasn't decided to end; don't interrupt a normal multi-tool loop).
  5. Reminder count ≥ ``_MAX_REMINDERS`` → return None (cap reached; allow and let
     the existing seal-resume + 5.7 FAILED safety net catch it, no infinite loop).
  6. Otherwise (model wants to end on pure text, seal not yet called) → inject
     reminder + ``jump_to='model'``.

This is L1 (the structural gate), NOT a fallback: a fallback would seal from disk
files *after* the agent ended (admits the miss, cleans up). This gate intercepts
*before* the agent ends and refuses to let it end — the miss cannot occur. See spec
§0 for the L1-vs-fallback distinction.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Subagents whose only physical exit is a seal_<name>_handoff tool call.
# code-executor is intentionally excluded: run_metric_plan already produces-and-
# delivers in one tool (structurally cannot miss seal — verified across many
# dogfoods). bash / general-purpose / knowledge-assistant have no seal contract.
_REQUIRES_SEAL: frozenset[str] = frozenset({"data-analyst", "chart-maker", "report-writer"})

_MAX_REMINDERS = 2


def _seal_tool_name(subagent_name: str) -> str:
    """Derive the seal tool name for a subagent (matches executor.py:1046)."""
    return f"seal_{subagent_name.replace('-', '_')}_handoff"


def _last_ai_message(messages: list) -> AIMessage | None:
    """Return the most recent AIMessage in the conversation, or None."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _seal_in_history(messages: list, seal_tool: str) -> bool:
    """Check if a seal ToolMessage already exists in the conversation history."""
    for msg in messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == seal_tool:
            return True
    return False


class SealGateMiddleware(AgentMiddleware):
    """after_model gate that forces ``seal_<name>_handoff`` before the subagent can exit.

    One middleware *class* covers all three seal-requiring subagents; each subagent
    gets its own instance (constructed with its ``config.name``), which both selects
    the right seal tool name and self-gates via ``_REQUIRES_SEAL``.

    Must be appended after LoopDetectionMiddleware in the subagent middleware chain.
    """

    def __init__(self, subagent_name: str) -> None:
        super().__init__()
        self._subagent_name = subagent_name
        self._seal_tool = _seal_tool_name(subagent_name)
        # Per-instance counter. SUBprocess executor builds a FRESH middleware
        # instance for every subagent run (executor.py:_build_middlewares is
        # called per run — see the "fresh instance each call" precedent for
        # LoopDetectionMiddleware right above SealGate), so a plain int is
        # already per-run isolated. The previous ``{run_id: count}`` dict keyed
        # on ``getattr(runtime, "run_id", None) or id(runtime)`` was the bug:
        # ``runtime`` carries no ``run_id`` ATTRIBUTE (run_id lives in the FLAT
        # ``runtime.context`` dict, not as an attribute — see
        # _thread_id_from_runtime in experiment_context.py), so it always fell
        # back to ``id(runtime)``, which DIFFERS across after_model invocations
        # within one run. The count never accumulated → cap ``>= _MAX_REMINDERS``
        # never tripped → the gate bounced the model indefinitely (capped only by
        # the outer max_turns), producing N wasted re-judgement turns before
        # seal-resume finally caught it. A per-instance int can't drift.
        self._reminder_count: int = 0

    def _get_reminder_count(self, runtime: Runtime) -> int:
        """Get the reminder count for the current run (per-instance isolation)."""
        return self._reminder_count

    def _increment_reminder_count(self, runtime: Runtime) -> None:
        """Increment the reminder count for the current run."""
        self._reminder_count += 1

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Check after model output whether seal should have been called."""
        try:
            return self._check(state, runtime)
        except Exception:
            logger.debug("SealGateMiddleware: check failed, fail-open", exc_info=True)
            return None

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Async version — delegates to sync."""
        return self.after_model(state, runtime)

    def _check(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        # 1. Not a seal-requiring subagent → not our concern
        if self._subagent_name not in _REQUIRES_SEAL:
            return None

        messages = state.get("messages", []) if hasattr(state, "get") else []
        if not messages:
            return None

        last_ai = _last_ai_message(messages)
        if last_ai is None:
            return None

        # 4. (checked early, cheap) Last AIMessage still carries other tool_calls →
        # the agent is still working and hasn't decided to end. Don't interrupt a
        # normal multi-tool loop. Note: a seal tool_call here is also "still working",
        # but criterion 2 below handles it identically — both mean "allow".
        tool_calls = getattr(last_ai, "tool_calls", None) or []
        if tool_calls:
            return None

        # 2/3. seal already being called or already called in history → allow
        if _seal_in_history(messages, self._seal_tool):
            return None

        # 5. Reminder cap reached → allow; existing seal-resume + 5.7 FAILED catch it
        if self._get_reminder_count(runtime) >= _MAX_REMINDERS:
            return None

        # 6. Model wants to end on pure text but seal not called → force back
        self._increment_reminder_count(runtime)
        logger.info(
            "SealGateMiddleware: seal reminder for %s (count=%d, tool=%s)",
            self._subagent_name,
            self._get_reminder_count(runtime),
            self._seal_tool,
        )
        reminder = _build_reminder(self._subagent_name, self._seal_tool)
        return {
            "messages": [
                HumanMessage(
                    content=reminder,
                    name="seal_gate_reminder",
                    additional_kwargs={"hide_from_ui": True},
                )
            ],
            "jump_to": "model",
        }


def _build_reminder(subagent_name: str, seal_tool: str) -> str:
    """Build the positive-instruction reminder (CLAUDE.md §6: no negation framing).

    Names the concrete seal tool and frames the call as "this call is how the
    analysis is produced/delivered" — the same mental model as the L2 prompt merge.
    """
    return (
        "<system_reminder>\n"
        f"你的分析已完成，但尚未发出 {seal_tool} 工具调用。\n"
        "你的分析结论只有通过这次工具调用才会产出落库——这是产出分析的唯一动作。\n"
        f"请现在调用 {seal_tool}，把你的结论（key_findings 等结构化字段）作为工具参数发出。\n"
        "</system_reminder>"
    )
