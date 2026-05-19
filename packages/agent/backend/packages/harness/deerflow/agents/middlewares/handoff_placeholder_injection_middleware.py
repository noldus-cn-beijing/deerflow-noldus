"""HandoffPlaceholderInjectionMiddleware.

Auto-injects ``{{handoff://X}}`` placeholders into the ``prompt`` arg of
``task()`` calls BEFORE downstream middlewares (especially
``GuardrailMiddleware(TaskHandoffAuthorizationProvider)``) inspect them.

## Why this exists

The W19 design intended ``task_tool.py`` to auto-inject the placeholders
required by each subagent's ``required_upstream_handoffs``. The W18 guardrail
(``TaskHandoffAuthorizationProvider``) was then meant to be a *safety net* —
denying only the rare case where injection was disabled or skipped.

In practice the ordering was wrong: ``GuardrailMiddleware.wrap_tool_call``
calls ``provider.evaluate(request)`` BEFORE invoking the handler (which is
where ``task_tool`` runs). So the guardrail always saw the raw lead-written
prompt; W19's injection inside ``task_tool`` body could never reach it. The
result was a denial loop in dogfood thread 51b00ac8 (2026-05-19):

    lead 写 task(chart-maker, prompt="出图...")  ← 无 {{handoff://X}}
      → W18 guardrail evaluate → deny: required_handoff_missing
      → ToolMessage(error) → lead re-tries with read_file / re-dispatch
      → LoopDetection fires after 3 calls → ForcedStop

Placing the injection in a dedicated middleware that is appended to the
agent's middleware list BEFORE ``GuardrailMiddleware(TaskHandoffAuthProvider)``
fixes the ordering. ``wrap_tool_call`` middlewares compose like onion layers
(first-appended = outermost), so this middleware rewrites the request and
hands the modified version to the next layer's guardrail.

## What it does

- Skips anything that is not ``tool_call.name == "task"``.
- Skips ``task`` calls whose ``subagent_type`` is unknown or whose subagent
  config declares ``required_upstream_handoffs == []``.
- Otherwise: for each handoff name in the config that is not already present
  as ``{{handoff://name}}`` in the prompt, appends a small auto-injected
  block listing the missing placeholders, then delegates to the handler with
  the rewritten ``ToolCallRequest`` (via the immutable ``override()`` API).

The format of the injected block matches what was previously emitted from
``task_tool._auto_inject_handoff_placeholders`` so that the
``HandoffIsolationProvider`` and the subagent prompt see exactly the same
text whether the lead hand-writes the placeholder or relies on injection.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


_HANDOFF_PLACEHOLDER_RE = re.compile(r"\{\{handoff://([^}]+)\}\}")


def _inject_missing_handoffs(prompt: str, subagent_type: str) -> str:
    """Pure function: returns a possibly-modified prompt with missing
    handoff placeholders auto-injected. Idempotent — no duplicate injection.

    Returns the original ``prompt`` unchanged when:
      - subagent_type is unknown
      - subagent's required_upstream_handoffs is empty
      - all required placeholders are already present
    """
    from deerflow.subagents.builtins import BUILTIN_SUBAGENTS

    config = BUILTIN_SUBAGENTS.get(subagent_type)
    if config is None or not config.required_upstream_handoffs:
        return prompt

    existing = set(_HANDOFF_PLACEHOLDER_RE.findall(prompt))
    additions = [
        f"{{{{handoff://{name}}}}}"
        for name in config.required_upstream_handoffs
        if name not in existing
    ]
    if not additions:
        return prompt

    return (
        f"{prompt}\n\n"
        f"[Upstream handoff (auto-injected by harness)]\n"
        + "\n".join(f"- {p}" for p in additions)
    )


class HandoffPlaceholderInjectionMiddleware(AgentMiddleware[AgentState]):
    """Inject ``{{handoff://X}}`` placeholders into ``task()`` prompts before
    downstream guardrails inspect them.

    Append this middleware BEFORE ``GuardrailMiddleware(TaskHandoffAuthorizationProvider)``
    in the agent middleware list — wrap_tool_call composes first-appended as outermost.
    """

    name = "HandoffPlaceholderInjectionMiddleware"

    def _maybe_override(self, request: ToolCallRequest) -> ToolCallRequest:
        if request.tool_call.get("name") != "task":
            return request
        args = request.tool_call.get("args") or {}
        prompt = args.get("prompt") or ""
        subagent_type = args.get("subagent_type") or ""
        if not prompt or not subagent_type:
            return request

        new_prompt = _inject_missing_handoffs(prompt, subagent_type)
        if new_prompt is prompt or new_prompt == prompt:
            return request

        logger.info(
            "Auto-injected handoff placeholders for task(subagent_type=%s)",
            subagent_type,
        )
        new_args = {**args, "prompt": new_prompt}
        new_tool_call = {**request.tool_call, "args": new_args}
        return request.override(tool_call=new_tool_call)

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        return handler(self._maybe_override(request))

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        return await handler(self._maybe_override(request))
