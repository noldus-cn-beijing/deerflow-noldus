"""AskClarificationOptionsProvider — 强制结构化反问点带快捷选项（ETHO-9）。

根因：ask_clarification 的 options 是可选参数，lead LLM 自由决定传不传。
PATHS 已声明哪些 ask step requires_options=True（viz/report/four_choice 是/否选择题）。
本 provider 拦 ask_clarification，查 next_pending_ask_step 的 requires_options；
若 True 且 options 为空/缺失 → deny，引导 lead 补上快捷选项后重发。

开放澄清（CLARIFY/clarify, requires_options=False）不强制，passthrough。

DeerFlow-native 实现：
- 复用 IntentPostStepAskGateBridge 已设置的 _lead_messages / _lead_workspace contextvar
- 复用 path_registry.next_pending_ask_step（SSOT，不重复逻辑）
- 挂在 GuardrailMiddleware 上，在 IntentPostStepAskGateBridge 之后注册
"""

from __future__ import annotations

from pathlib import Path

from deerflow.guardrails.intent_post_step_ask_gate_provider import (
    _extract_latest_intent,
    _lead_messages,
    _lead_workspace,
)
from deerflow.guardrails.path_registry import (
    ASK_GATE_MAP,
    PATHS,
    next_pending_ask_step,
    to_handoff_name,
)
from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)

logger = __import__("logging").getLogger(__name__)


class AskClarificationOptionsProvider:
    """Deny ask_clarification when the pending ask step requires options but none supplied.

    Fails open (allow) when intent, workspace, or path is unavailable.
    Only intercepts ask_clarification tool calls — all other tools pass through.
    """

    name = "ask_clarification_options"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if request.tool_name != "ask_clarification":
            return GuardrailDecision(allow=True)

        messages = _lead_messages.get()
        intent = _extract_latest_intent(messages)
        if not intent:
            return GuardrailDecision(allow=True)

        steps = PATHS.get(intent)
        if not steps:
            return GuardrailDecision(allow=True)

        workspace = _lead_workspace.get()
        if not workspace:
            return GuardrailDecision(allow=True)

        def _handoff_exists(target: str) -> bool:
            name = to_handoff_name(target)
            return (Path(workspace) / f"handoff_{name}.json").exists()

        # Derive gate_completed from experiment-context.json (fail-open on any error)
        gate_completed: list[str] = []
        try:
            ctx_path = Path(workspace) / "experiment-context.json"
            if ctx_path.exists():
                import json
                with ctx_path.open("r", encoding="utf-8") as f:
                    ctx = json.load(f)
                raw = ctx.get("gate_completed", [])
                if isinstance(raw, list):
                    gate_completed = raw
        except Exception:
            pass

        pending = next_pending_ask_step(steps, gate_completed, _handoff_exists)
        if pending is None or not pending.requires_options:
            return GuardrailDecision(allow=True)

        options = request.tool_input.get("options")
        if not options:
            ask_key = pending.target
            gate_name = ASK_GATE_MAP.get(ask_key, ask_key)
            return GuardrailDecision(
                allow=False,
                reasons=[GuardrailReason(
                    code=f"ethoinsight.ask_{ask_key}_missing_options",
                    message=(
                        f"按 {intent} 路径，ask({ask_key}?) 是结构化选择题，"
                        f"请在 ask_clarification 的 options 参数里给出快捷选项"
                        f"（如 ['A. 是…', 'B. 否…']）后重发。"
                        f"快捷选项让用户单击即答，不必手敲；gate={gate_name}。"
                    ),
                )],
                policy_id="ask_clarification_options",
            )

        return GuardrailDecision(allow=True)

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)
