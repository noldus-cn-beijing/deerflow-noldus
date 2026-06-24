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

ETHO-1 upgrade (spec 2026-06-23): L1 has a release valve — after ``_MAX_REMINDERS``
nudges the gate allows exit (rule 5), so a model that stubbornly ends on pure text
still slips through. For MECHANICALLY RECONSTRUCTABLE producers (report-writer) the
new ``after_agent`` hook closes that valve: at the termination point it runs a
deterministic auto-seal from the output files (reusing
``executor._attempt_auto_seal_from_artifacts``). Side-effect only — no
``can_jump_to`` (after_agent runs post-termination and physically cannot jump).
This eliminates the ``Task failed`` intermediate state + a wasted lead-retry round.
Cognitive producers (data-analyst) and structurally-merged producers (chart-maker,
whose run_chart_plan produces-and-delivers in one tool) are deliberately skipped.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Subagents whose only physical exit is a seal_<name>_handoff tool call.
# code-executor / chart-maker are intentionally excluded: run_metric_plan / run_chart_plan
# already produces-and-delivers in one tool (structurally cannot miss seal — verified across
# many dogfoods; chart-maker aligned 2026-06-24). bash / general-purpose / knowledge-assistant
# have no seal contract.
_REQUIRES_SEAL: frozenset[str] = frozenset({"data-analyst", "report-writer"})

# Subagents whose handoff core fields are MECHANICALLY RECONSTRUCTABLE from
# output files (report.md / plot_*.png). For these, the ``after_agent`` hook can
# run a deterministic auto-seal at the termination point — closing the L1
# reminder-cap release valve and eliminating the ``Task failed`` intermediate
# state the user would otherwise see.
#
# Subset of _REQUIRES_SEAL:
# - report-writer: core fields (report_path) derive deterministically from outputs/ files
#   (executor._attempt_auto_seal_from_artifacts).
# - data-analyst: interpretation is a COGNITIVE product with no file source —
#   fabricating it would be reward-hacking (worse than the miss). Deliberately
#   absent: its seal-miss remains an observable degradation handled by
#   L1 (after_model nudge) + L2 (seal-resume) + L4 (lead retry). See spec §2.2.
# - chart-maker: structurally produce-and-deliver (run_chart_plan), excluded from
#   _REQUIRES_SEAL already; not auto-sealed here.
_RECONSTRUCTABLE: frozenset[str] = frozenset({"report-writer"})

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


def _da_handoff_is_terminal(state: Any) -> bool:
    """data-analyst handoff 是否已封口成终态（spec §二 结构 4）。

    读 workspace/handoff_data_analyst.json 的 status：completed/partial/failed =
    已封口（finalize 已落盘）；in_progress 或读不到 = 未 seal。任何异常 → False
    （fail-open，让 L1 reminder 继续催，绝不误判已 seal）。
    """
    try:
        from deerflow.agents.middlewares.experiment_context import resolve_workspace_from_state

        workspace = resolve_workspace_from_state(state if isinstance(state, dict) else {})
        if not workspace:
            return False
        import json
        from pathlib import Path

        path = Path(workspace) / "handoff_data_analyst.json"
        if not path.exists():
            return False
        data = json.loads(path.read_text(encoding="utf-8"))
        return isinstance(data, dict) and data.get("status") in {"completed", "partial", "failed"}
    except Exception:
        logger.debug("SealGateMiddleware: _da_handoff_is_terminal read failed, fail-open", exc_info=True)
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
        # data-analyst 的唯一封口入口是 finalize_data_analyst_handoff（spec
        # 2026-06-23-data-analyst-seal-stepwise-fill-template §二 结构 4）。它不再
        # 一次性吐 seal args（撞 max_tokens 狭颈），改为 fill_* 逐字段填 → finalize
        # 封口。SealGate 对 data-analyst 认 finalize/终态而非旧 seal。其余 subagent
        # 仍认 seal_<name>_handoff。
        if subagent_name == "data-analyst":
            self._seal_tool = "finalize_data_analyst_handoff"
        else:
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

    # ------------------------------------------------------------------
    # after_agent — termination-point deterministic auto-seal (spec 2026-06-23
    # ETHO-1). Side-effect ONLY: no ``can_jump_to`` (after_agent runs AFTER the
    # agent has decided to terminate and physically cannot jump back to model —
    # see spec §1.4 physical-boundary finding). It only fixes the L1 reminder-
    # cap release valve for RECONSTRUCTABLE producers.
    # ------------------------------------------------------------------
    def after_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Termination-point auto-seal for mechanically-reconstructable producers.

        For report-writer that reaches the termination point without
        having called seal (L1 reminders exhausted via the release valve, or the
        model simply ended on pure text past the cap), reconstruct the handoff
        deterministically from the output files. This eliminates the
        ``Task failed`` intermediate state + a wasted lead-retry round that the
        executor's L3 path (which only runs AFTER L2 seal-resume fails and
        AFTER the FAILED error surfaces to lead) would otherwise impose.

        Cognitive producers (data-analyst) and non-seal subagents are skipped:
        after_agent can neither jump nor fabricate an interpretation.

        ROBUSTNESS: never raises. Any exception (bad workspace, auto-seal
        failure, missing artifacts) → return None and let the executor's L3/L4
        paths be the final arbiter (spec §2.1 "after_agent 是兜底不是主路").
        """
        try:
            return self._after_agent_check(state, runtime)
        except Exception:
            logger.debug(
                "SealGateMiddleware.after_agent: check failed, fail-open",
                exc_info=True,
            )
            return None

    async def aafter_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Async version — delegates to sync (pure side-effect, no awaits)."""
        return self.after_agent(state, runtime)

    def _after_agent_check(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        # 1. Only reconstructable producers get the termination-point auto-seal.
        #    data-analyst (cognitive) and non-seal subagents pass through.
        if self._subagent_name not in _RECONSTRUCTABLE:
            return None

        messages = state.get("messages", []) if hasattr(state, "get") else []
        # 2. Already sealed → nothing to do (don't re-seal / overwrite).
        if _seal_in_history(messages, self._seal_tool):
            return None

        # 3. Resolve the host-side workspace from thread_data (set by
        #    ThreadDataMiddleware; present in subagent state per executor
        #    _build_initial_state). Reuse the experiment_context helper to stay
        #    consistent with how every other middleware resolves workspace.
        from deerflow.agents.middlewares.experiment_context import resolve_workspace_from_state

        workspace = resolve_workspace_from_state(state if isinstance(state, dict) else {})
        if not workspace:
            # No workspace → cannot reconstruct; let executor L3/L4 handle it.
            return None

        # 4. Deterministic auto-seal at the termination point. Lazy import:
        #    seal_gate_middleware must NOT top-level import executor (closes a
        #    harness import cycle → Gateway startup ImportError; CLAUDE.md
        #    import-ring rule). stamp sealed_by so the fallback trigger rate is
        #    observable (HarnessX trace richness; spec §2.3).
        from deerflow.subagents.executor import _attempt_auto_seal_from_artifacts

        sealed = _attempt_auto_seal_from_artifacts(
            self._subagent_name, workspace, sealed_by="after_agent_artifacts",
        )
        if sealed:
            logger.warning(
                "[seal_gate] after_agent auto-sealed %s from artifacts (sealed_by="
                "after_agent_artifacts); L1 reminders did not recover the seal in-run",
                self._subagent_name,
            )
        # Side-effect only — never jump. sealed or not, executor L3/L4 remain the
        # final arbiter.
        return None

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

        # 2b. data-analyst 额外认 workspace 文件终态为已 seal（spec §二 结构 4）。
        # finalize 把 status 从 in_progress 改成 completed/partial/failed 并落盘；
        # 若 ToolMessage 因任何边缘情况没在 history 里（如被截断），workspace 文件
        # 的终态仍是「已封口」的权威证据。in_progress → 未 seal，继续催回（催 finalize）。
        if self._subagent_name == "data-analyst" and _da_handoff_is_terminal(state):
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
