"""HandoffIsolationProvider — gates subagent reads of peer handoff_*.json files.

Authorization is supplied by task_tool at dispatch time (parsed from
{{handoff://<name>}} placeholders in the lead's task prompt). The provider
DOES NOT parse the prompt text itself — keeping semantics explicit: no
placeholder = no authorization.

This enforces the 'files are facts' principle in mechanism, not just prompt.
Lead is the sole authorizing party; only paths lead names via placeholder
are accessible to the subagent. A subagent may always read its own outbox
(self-validation case).
"""

from __future__ import annotations

from deerflow.guardrails.provider import (
    GuardrailDecision,
    GuardrailReason,
    GuardrailRequest,
)


class HandoffIsolationProvider:
    """Block subagents from reading peer subagents' handoff files unless lead
    has authorized the path via {{handoff://...}} placeholder in task prompt.
    """

    name = "handoff_isolation"

    def __init__(
        self,
        authorized_paths: set[str],
        self_outbox_subagent_name: str | None = None,
    ):
        self.authorized_paths = authorized_paths
        self.self_outbox_subagent_name = self_outbox_subagent_name

    def _is_own_handoff(self, file_path: str) -> bool:
        """Allow subagent to read its own handoff file (it just wrote it).

        e.g. data-analyst writes handoff_data_analyst.json, then may re-read
        for self-validation. This is not "peeking at peer".
        """
        if not self.self_outbox_subagent_name:
            return False
        # Subagent names use hyphens ('code-executor'); filenames use
        # underscores (handoff_code_executor.json). Normalize for comparison.
        normalized = self.self_outbox_subagent_name.replace("-", "_")
        return f"handoff_{normalized}.json" in file_path

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # No passport = lead call. Lead is never restricted by this provider.
        if not request.agent_id:
            return GuardrailDecision(allow=True)
        # Only gate read_file; other tools pass through.
        if request.tool_name != "read_file":
            return GuardrailDecision(allow=True)
        file_path = request.tool_input.get("file_path", "") or ""
        # Only gate handoff_*.json reads; other files unrestricted.
        if "handoff_" not in file_path or not file_path.endswith(".json"):
            return GuardrailDecision(allow=True)
        # Subagent may always read its own outbox.
        if self._is_own_handoff(file_path):
            return GuardrailDecision(allow=True)
        # Check explicit authorization.
        if file_path in self.authorized_paths:
            return GuardrailDecision(allow=True)
        return GuardrailDecision(
            allow=False,
            reasons=[
                GuardrailReason(
                    code="handoff_isolation.unauthorized",
                    message=(
                        f"Subagent attempted to read {file_path} without "
                        f"lead authorization. Authorized paths: "
                        f"{sorted(self.authorized_paths)}. To authorize, "
                        f"lead must include {{{{handoff://<subagent_name>}}}} "
                        f"placeholder in the task prompt."
                    ),
                )
            ],
            policy_id="handoff_isolation",
        )

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        # Same logic, async signature for GuardrailProvider protocol compliance.
        return self.evaluate(request)
