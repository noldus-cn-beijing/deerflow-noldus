"""TaskHandoffAuthorizationProvider — 校验 lead task() 是否含必需占位符。

Spec §6.2: lead 派遣 task(subagent_type=X) 时, prompt 必须含
BUILTIN_SUBAGENTS[X].required_upstream_handoffs 中每个 name 的
{{handoff://X}} 占位符。

W19 完成后正常 flow 不触发本 provider deny; 本 provider 作为安全网保留。
W19 必须紧跟 W18 落地。
"""
from __future__ import annotations

from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)


class TaskHandoffAuthorizationProvider:
    name = "task_handoff_authorization"

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if request.tool_name != "task":
            return GuardrailDecision(allow=True)

        subagent_type = request.tool_input.get("subagent_type", "")
        prompt = request.tool_input.get("prompt", "") or ""

        from deerflow.subagents.builtins import BUILTIN_SUBAGENTS

        config = BUILTIN_SUBAGENTS.get(subagent_type)
        if config is None or not config.required_upstream_handoffs:
            return GuardrailDecision(allow=True)

        missing = [
            name for name in config.required_upstream_handoffs
            if f"{{{{handoff://{name}}}}}" not in prompt
        ]
        if missing:
            return GuardrailDecision(
                allow=False,
                reasons=[GuardrailReason(
                    code="ethoinsight.required_handoff_missing",
                    message=(
                        f"subagent '{subagent_type}' 需要 upstream handoff "
                        f"{missing}。在 prompt 中加 {{{{handoff://<name>}}}} 占位符,"
                        f"或检查 task_tool 自动注入(W19)是否启用。"
                    ),
                )],
                policy_id="task_handoff_authorization",
            )
        return GuardrailDecision(allow=True)

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)
