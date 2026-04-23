"""Middleware that records every agent turn as a training data sample.

Written as part of the training-data flywheel (docs/plans/2026-04-23-training-data-flywheel.md).
Records Fireworks-compatible JSONL per thread to
`.deer-flow/training-data/auto-collected/<thread_id>.jsonl`.
"""
import logging
from pathlib import Path
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
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
