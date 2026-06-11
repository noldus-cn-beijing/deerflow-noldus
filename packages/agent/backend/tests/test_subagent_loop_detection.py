"""Tests for subagent LoopDetectionMiddleware integration (S1).

Verifies:
- LoopDetectionMiddleware is included in _build_middlewares
- Fresh instance per _build_middlewares call (isolation from lead agent)
"""

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Import the real LoopDetectionMiddleware BEFORE mocking anything.
# The isinstance check in tests needs the real class reference.
# ---------------------------------------------------------------------------
from deerflow.agents.middlewares.loop_detection_middleware import (
    LoopDetectionMiddleware,
)

# Now mock the modules that block SubagentExecutor import.
# We must mock deerflow.agents and deerflow.agents.middlewares (which is a
# package), but we keep loop_detection_middleware as the already-imported
# real module so isinstance() works.
_MOCKED_MODULE_NAMES = [
    "deerflow.agents",
    "deerflow.agents.thread_state",
    "deerflow.sandbox",
    "deerflow.sandbox.middleware",
    "deerflow.sandbox.security",
    "deerflow.models",
]


def _ensure_mock_module(name, **attrs):
    """Create a MagicMock module stub in sys.modules."""
    mod = MagicMock()
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


@pytest.fixture(scope="module")
def _executor_module():
    """Load the real SubagentExecutor class with _build_middlewares usable."""
    # Save originals for cleanup
    original_modules = {name: sys.modules.get(name) for name in _MOCKED_MODULE_NAMES}
    original_executor = sys.modules.get("deerflow.subagents.executor")

    # Also track extra modules we set up
    extra_keys = [
        "deerflow.agents.middlewares.tool_error_handling_middleware",
        "deerflow.guardrails.handoff_isolation_provider",
        "deerflow.guardrails.middleware",
        "deerflow.guardrails.script_invocation_only_provider",
    ]
    original_extra = {k: sys.modules.get(k) for k in extra_keys}

    # Remove conftest's executor mock
    if "deerflow.subagents.executor" in sys.modules:
        del sys.modules["deerflow.subagents.executor"]

    # Set up mock modules
    for name in _MOCKED_MODULE_NAMES:
        sys.modules[name] = MagicMock()

    # -- Stub the modules that _build_middlewares imports at call time --

    # tool_error_handling_middleware: provides build_subagent_runtime_middlewares
    _ensure_mock_module(
        "deerflow.agents.middlewares.tool_error_handling_middleware",
        build_subagent_runtime_middlewares=MagicMock(side_effect=lambda **kw: []),
    )

    # handoff_isolation_provider
    from deerflow.guardrails.handoff_isolation_provider import HandoffIsolationProvider

    sys.modules["deerflow.guardrails.handoff_isolation_provider"] = sys.modules[
        HandoffIsolationProvider.__module__
    ]

    # guardrails.middleware
    from deerflow.guardrails.middleware import GuardrailMiddleware

    sys.modules["deerflow.guardrails.middleware"] = sys.modules[
        GuardrailMiddleware.__module__
    ]

    # script_invocation_only_provider
    from deerflow.guardrails.script_invocation_only_provider import (
        ScriptInvocationOnlyProvider,
    )

    sys.modules["deerflow.guardrails.script_invocation_only_provider"] = sys.modules[
        ScriptInvocationOnlyProvider.__module__
    ]

    # loop_detection_middleware — keep the real one we already imported
    sys.modules["deerflow.agents.middlewares.loop_detection_middleware"] = sys.modules[
        LoopDetectionMiddleware.__module__
    ]

    from deerflow.subagents.executor import SubagentExecutor

    yield SubagentExecutor

    # Restore originals
    for name in _MOCKED_MODULE_NAMES:
        if original_modules[name] is not None:
            sys.modules[name] = original_modules[name]
        elif name in sys.modules:
            del sys.modules[name]

    for k in extra_keys:
        if original_extra[k] is not None:
            sys.modules[k] = original_extra[k]
        elif k in sys.modules:
            del sys.modules[k]

    if original_executor is not None:
        sys.modules["deerflow.subagents.executor"] = original_executor
    elif "deerflow.subagents.executor" in sys.modules:
        del sys.modules["deerflow.subagents.executor"]


class TestSubagentLoopDetection:
    """Verify LoopDetectionMiddleware is part of the subagent middleware chain."""

    def test_build_middlewares_includes_loop_detection(self, _executor_module):
        """_build_middlewares should include a LoopDetectionMiddleware instance."""
        SubagentExecutor = _executor_module

        from deerflow.subagents.config import SubagentConfig

        config = SubagentConfig(name="test-agent", description="test")
        executor = SubagentExecutor.__new__(SubagentExecutor)
        executor.config = config
        executor.tools = []
        executor.trace_id = "test-trace"
        executor.authorized_handoff_paths = []
        executor.parent_model = None

        middlewares = executor._build_middlewares()

        loop_detection_instances = [
            m for m in middlewares if isinstance(m, LoopDetectionMiddleware)
        ]
        assert len(loop_detection_instances) == 1, (
            f"Expected 1 LoopDetectionMiddleware, got {len(loop_detection_instances)}"
        )

    def test_loop_detection_fresh_instance_per_call(self, _executor_module):
        """Each _build_middlewares call creates a new LoopDetectionMiddleware.

        This ensures subagent and lead agent loop histories are isolated
        (different instances → different _history dicts).
        """
        SubagentExecutor = _executor_module

        from deerflow.subagents.config import SubagentConfig

        config = SubagentConfig(name="test-agent", description="test")
        executor = SubagentExecutor.__new__(SubagentExecutor)
        executor.config = config
        executor.tools = []
        executor.trace_id = "test-trace"
        executor.authorized_handoff_paths = []
        executor.parent_model = None

        mw1 = executor._build_middlewares()
        mw2 = executor._build_middlewares()

        ld1 = [m for m in mw1 if isinstance(m, LoopDetectionMiddleware)]
        ld2 = [m for m in mw2 if isinstance(m, LoopDetectionMiddleware)]

        assert len(ld1) == 1
        assert len(ld2) == 1
        assert ld1[0] is not ld2[0], (
            "LoopDetectionMiddleware must be a fresh instance each call "
            "to avoid thread_id-based history pollution between lead and subagent"
        )
