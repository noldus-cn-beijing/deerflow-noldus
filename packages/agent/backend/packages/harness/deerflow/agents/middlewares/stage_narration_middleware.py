"""StageNarrationMiddleware —— A1 后端事件分轨地基（承重墙）。

Spec: docs/superpowers/specs/2026-06-30-generative-ux-roadmap-and-a1-event-track-foundation-design.md

挂在 lead 中间件链上，职责单一：**在真实流水线意图确定时，确定性地往 custom 轨
写一次 ``stage_plan``**（人话阶段叙事）。

- 触发源 = 真实代码事件（guardrail 强制的 ``[intent]`` 行），不是 LLM 自报。
- 只有「多阶段流水线」意图（E2E_FULL / E2E_FULL_ASKVIZ / E2E_MIN）才发；
  知识问答 / 闲聊 / 单步追问 / 单步出图报告不发（前端据此无 stepper）。
- 幂等：同一 intent 不重复发；intent 变更到新的 pipeline intent 才发新 plan。
- stage_update（每阶段进/出）由 ``task`` 工具在派遣边界同源发射（复用既有观测点），
  不在本中间件——见 stage_narration 模块 + task_tool.py。

与 PR#213（run_chart_plan 确定性登记）/ SealGateMiddleware 同构：真实状态机驱动，
确定性产出，不靠 prompt 打地鼠。人话字段永不携带内脏（工具名/subagent名/gate 关键字），
由 stage_narration 的防泄漏约束 + 本中间件只透传其 payload 共同保证。
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from deerflow.agents.middlewares import stage_narration

logger = logging.getLogger(__name__)

# 复用 guardrails 已有的意图提取 helper（同一 [intent] 正则，SSOT）。
# 惰性导入避免顶层 import 闭环（守 harness import-cycle 铁律）。


def _default_writer(payload: dict) -> None:
    """Default writer: lazily resolve LangGraph stream writer.

    Best-effort: outside a graph context get_stream_writer() raises; callers
    swallow that. Kept as a module-level function so it is trivially injectable.
    """
    from langgraph.config import get_stream_writer

    get_stream_writer()(payload)


class StageNarrationMiddleware(AgentMiddleware[AgentState]):
    """Emit ``stage_plan`` (intent-determined) + 「识别范式」stage_update (tool-call) events.

    Two observation points, both real-event-driven (no LLM self-report):

    1. ``after_model`` — when the lead declares a pipeline ``[intent]``, emit a one-shot
       ``stage_plan`` (人话阶段集) on the custom track.
    2. ``wrap_tool_call`` (缺口 1, spec 2026-06-30-a1-stage-narration-coverage-gap-fix) —
       「识别范式」阶段由 lead 自调工具完成（不派 subagent），故 task 派遣观测点收不到它。
       在 lead 直接调识别类工具的边界同源发射：
         - ``identify_ev19_template`` / ``inspect_uploaded_file`` → 识别范式 active（进入）
         - ``prep_metric_plan`` 返回 ``status=ok`` → 识别范式 completed（识别完成、即将派
           code-executor，grounded：失败/ambiguous/unsupported 不发 completed，叙事不撒谎）。

    Args:
        n_resolver: Optional callable returning the current batch subject count
            (``n``). ``None`` return → n unknown → stage_plan.skipped stays empty
            (we never guess n=1). Defaults to a resolver that reads the lead
            experiment-context if present, else returns None.
        writer: Optional callable ``(payload: dict) -> None`` writing to the custom
            stream track. Defaults to the lazy ``get_stream_writer()`` resolver.
            Injected in tests; in production the default resolves per-call inside
            the graph context.
    """

    def __init__(
        self,
        n_resolver: Callable[[], int | None] | None = None,
        writer: Callable[[dict], None] | None = None,
    ) -> None:
        super().__init__()
        self._n_resolver = n_resolver or _default_n_resolver
        self._writer = writer or _default_writer
        # Last intent we emitted a stage_plan for → idempotency within & across turns.
        self._last_emitted_intent: str | None = None
        # 识别范式 active 是否已发（进入幂等：识别阶段重试不重复发 active，直至 completed 重置）。
        self._identify_active_emitted: bool = False

    # ----- core ------------------------------------------------------------

    def _maybe_emit(self, messages: list | None) -> None:
        from deerflow.guardrails.intent_classification_provider import _latest_declared_intent

        intent = _latest_declared_intent(messages)
        if intent is None or intent == self._last_emitted_intent:
            return
        n = self._safe_resolve_n()
        plan = stage_narration.intent_to_stage_plan(intent, n=n)
        if plan is None:
            # Non-pipeline intent (or unknown): remember it so a later identical
            # declaration doesn't re-trigger, but emit nothing.
            self._last_emitted_intent = intent
            return
        self._safe_write(plan)
        self._last_emitted_intent = intent

    def _safe_resolve_n(self) -> int | None:
        try:
            return self._n_resolver()
        except Exception:  # noqa: BLE001
            logger.debug("stage_narration: n_resolver raised; treating n as unknown", exc_info=True)
            return None

    def _safe_write(self, payload: dict) -> None:
        try:
            self._writer(payload)
        except Exception:  # noqa: BLE001
            logger.debug("stage_narration: writer unavailable or failed; skipping event", exc_info=True)

    # ----- 识别范式 tool-call observation (缺口 1) -------------------------

    def _emit_identify_active(self) -> None:
        """Fire 识别范式 active once per identification phase (idempotent until completed)."""
        if self._identify_active_emitted:
            return
        self._safe_write(stage_narration.stage_update(stage_narration.STAGE_IDENTIFY, "active"))
        self._identify_active_emitted = True

    def _maybe_emit_identify_completed(self, tool_result) -> None:
        """Fire 识别范式 completed only if prep_metric_plan truly succeeded (grounded)."""
        content = getattr(tool_result, "content", None)
        if not stage_narration.identify_done_succeeded(content):
            return  # 失败/ambiguous/unsupported/解析失败 → 不撒谎
        self._safe_write(stage_narration.stage_update(stage_narration.STAGE_IDENTIFY, "completed"))
        # 识别阶段结束：重置 active 标志，下一轮识别（如换数据集重跑）可再发 active。
        self._identify_active_emitted = False

    # ----- hooks -----------------------------------------------------------

    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:  # noqa: ARG002
        messages = state.get("messages") if isinstance(state, dict) else None
        self._maybe_emit(messages)
        return None

    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:  # noqa: ARG002
        messages = state.get("messages") if isinstance(state, dict) else None
        self._maybe_emit(messages)
        return None

    def wrap_tool_call(self, request, handler):  # type: ignore[override]
        """Observe lead's direct identification tools (缺口 1).

        - identify_ev19_template / inspect_uploaded_file → 识别范式 active (before handler).
        - prep_metric_plan → 识别范式 completed iff status=ok (after handler, grounded).
        Handler exceptions propagate (ToolNode's handle_tool_errors handles them).
        """
        name = request.tool_call.get("name") if isinstance(request.tool_call, dict) else None
        if stage_narration.is_identify_enter_tool(name or ""):
            self._emit_identify_active()
        result = handler(request)
        if stage_narration.is_identify_done_tool(name or ""):
            self._maybe_emit_identify_completed(result)
        return result

    async def awrap_tool_call(self, request, handler):  # type: ignore[override]
        name = request.tool_call.get("name") if isinstance(request.tool_call, dict) else None
        if stage_narration.is_identify_enter_tool(name or ""):
            self._emit_identify_active()
        result = await handler(request)
        if stage_narration.is_identify_done_tool(name or ""):
            self._maybe_emit_identify_completed(result)
        return result


def _default_n_resolver() -> int | None:
    """Default subject-count resolver: n unknown at the lead layer.

    ``stage_plan`` fires at intent-determination time, before code-executor has
    counted subjects, so n is generally not yet known here → returns None →
    ``stage_plan.skipped`` stays empty (we never guess n=1). The n=1 skip is
    applied when n is positively known; production callers that have n earlier
    inject their own ``n_resolver``.
    """
    return None
