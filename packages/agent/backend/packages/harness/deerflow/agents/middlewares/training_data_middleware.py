"""Middleware that records every agent turn as a training data sample.

Written as part of the training-data flywheel (docs/plans/2026-04-23-training-data-flywheel.md).
Records Fireworks-compatible JSONL per thread to
`.deer-flow/training-data/auto-collected/<thread_id>.jsonl`.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)


class TrainingDataMiddlewareState(AgentState):
    training_data_path: NotRequired[str | None]


class TrainingDataMiddleware(AgentMiddleware[TrainingDataMiddlewareState]):
    """Record each agent conversation as a training sample JSONL."""

    state_schema = TrainingDataMiddlewareState

    def __init__(self, base_dir: str | None = None):
        super().__init__()
        self._base_dir = Path(base_dir) if base_dir else None

    def _resolve_thread_id(self, runtime: Runtime) -> str | None:
        ctx = runtime.context or {}
        thread_id = ctx.get("thread_id") if isinstance(ctx, dict) else None
        if thread_id:
            return thread_id
        try:
            cfg = get_config()
            return cfg.get("configurable", {}).get("thread_id")
        except Exception:
            return None

    def _resolve_base_dir(self) -> Path:
        if self._base_dir:
            return self._base_dir
        # Default to backend/.deer-flow/
        from deerflow.config.paths import get_paths

        return Path(get_paths().base_dir)

    @override
    def before_agent(self, state, runtime: Runtime) -> dict | None:
        thread_id = self._resolve_thread_id(runtime)
        if not thread_id:
            logger.debug("TrainingDataMiddleware: no thread_id, skipping")
            return None

        out_dir = self._resolve_base_dir() / "training-data" / "auto-collected"
        out_dir.mkdir(parents=True, exist_ok=True)
        return {"training_data_path": str(out_dir / f"{thread_id}.jsonl")}

    def _append_jsonl(self, path: Path, sample: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")

    def _extract_lead_samples(self, messages: list, thread_id: str) -> list[dict]:
        """Pair each HumanMessage with the next AIMessage text reply."""
        samples: list[dict] = []
        pending_human: HumanMessage | None = None
        for msg in messages:
            if isinstance(msg, HumanMessage):
                pending_human = msg
            elif isinstance(msg, AIMessage) and pending_human is not None:
                text = msg.content if isinstance(msg.content, str) else ""
                if text.strip():
                    samples.append({
                        "role": "lead",
                        "thread_id": thread_id,
                        "input": pending_human.content if isinstance(pending_human.content, str) else str(pending_human.content),
                        "output": text,
                        "thinking": (msg.additional_kwargs or {}).get("reasoning_content") or "",
                        "recorded_at": datetime.now(timezone.utc).isoformat(),
                    })
                    pending_human = None
        return samples

    def _extract_subagent_samples(self, messages: list, thread_id: str) -> list[dict]:
        """For each AIMessage.tool_call of task tool, pair with its ToolMessage."""
        tool_results: dict[str, ToolMessage] = {}
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.tool_call_id:
                tool_results[msg.tool_call_id] = msg

        samples: list[dict] = []
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            for call in (msg.tool_calls or []):
                if call.get("name") != "task":
                    continue
                call_id = call.get("id")
                if not call_id or call_id not in tool_results:
                    continue
                args = call.get("args") or {}
                result = tool_results[call_id]
                result_text = result.content if isinstance(result.content, str) else str(result.content)
                samples.append({
                    "role": "subagent",
                    "thread_id": thread_id,
                    "subagent_type": args.get("subagent_type", ""),
                    "input": json.dumps({
                        "description": args.get("description", ""),
                        "prompt": args.get("prompt", ""),
                    }, ensure_ascii=False),
                    "output": result_text,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                })
        return samples

    @override
    def after_agent(self, state, runtime: Runtime) -> dict | None:
        thread_id = self._resolve_thread_id(runtime)
        path_str = state.get("training_data_path") if isinstance(state, dict) else None
        if not thread_id or not path_str:
            return None
        messages = state.get("messages", []) if isinstance(state, dict) else []
        if not messages:
            return None
        samples = self._extract_lead_samples(messages, thread_id)
        samples.extend(self._extract_subagent_samples(messages, thread_id))
        if not samples:
            return None
        path = Path(path_str)
        for s in samples:
            self._append_jsonl(path, s)
        logger.info("TrainingDataMiddleware: wrote %d lead samples to %s", len(samples), path)
        return None
