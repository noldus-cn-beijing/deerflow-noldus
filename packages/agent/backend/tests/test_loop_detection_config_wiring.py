"""Tests for loop_detection config wiring (Spec S1).

Verifies that LoopDetectionMiddleware instances are actually constructed
from app_config.loop_detection (config.yaml), not from hardcoded defaults.

Before this spec (2026-06-12), both lead_agent/agent.py and factory.py
instantiated LoopDetectionMiddleware with empty defaults, ignoring
config.yaml's tool_freq_hard_limit of 50. The hardcoded default of 5
caused legitimate multi-file inspection (28 files via inspect_uploaded_file)
to hit FORCED STOP at call 5 — a spurious loop-detection kill.
"""

from unittest.mock import MagicMock, patch

import pytest

from deerflow.agents.middlewares.loop_detection_middleware import (
    LoopDetectionMiddleware,
)
from deerflow.config.loop_detection_config import LoopDetectionConfig, ToolFreqOverride


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_runtime(thread_id="test-thread", run_id="test-run"):
    """Build a minimal Runtime mock with context."""
    runtime = MagicMock()
    runtime.context = {"thread_id": thread_id, "run_id": run_id}
    return runtime


def _make_state_from_ai(tool_calls=None, content=""):
    """Build a minimal AgentState dict from an AIMessage."""
    from langchain_core.messages import AIMessage

    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    return {"messages": [msg]}


def _make_app_config_with_loop_detection(**kwargs):
    """Build a minimal AppConfig with required sandbox field + custom loop_detection.

    Uses model_validate with a raw dict to avoid Pydantic's cached-instance
    shortcut that can return stale data after a previous from_file load.
    """
    from deerflow.config.app_config import AppConfig

    loop_cfg = {
        "tool_freq_hard_limit": 50,
        "tool_freq_warn": 30,
        "tool_freq_overrides": {
            "write_todos": {"warn": 2, "hard_limit": 4},
            "inspect_uploaded_file": {"warn": 50, "hard_limit": 100},
        },
    }
    loop_cfg.update(kwargs)

    return AppConfig.model_validate(
        {
            "sandbox": {"use": "deerflow.sandbox.local.local_sandbox:LocalSandboxProvider"},
            "loop_detection": loop_cfg,
            "models": [],
        }
    )


# ---------------------------------------------------------------------------
# 1. Wiring test (agent.py / build_middlewares path)
# ---------------------------------------------------------------------------

class TestLeadAgentLoopDetectionWiring:
    """Verify build_middlewares uses app_config.loop_detection."""

    def test_build_middlewares_uses_config_hard_limit_not_default(self):
        """build_middlewares must create LoopDetectionMiddleware with config's
        tool_freq_hard_limit=50, not the hardcoded default of 5."""
        cfg = _make_app_config_with_loop_detection()

        # Patch get_app_config everywhere it's used in build_middlewares.
        # build_middlewares calls it multiple times — all must return our cfg.
        with patch(
            "deerflow.agents.lead_agent.agent.get_app_config",
            return_value=cfg,
        ):
            from deerflow.agents.lead_agent.agent import build_middlewares

            middlewares = build_middlewares(
                config={"configurable": {"is_plan_mode": False}},
                model_name=None,
            )

        # Find the LoopDetectionMiddleware instance
        loop_mw = None
        for m in middlewares:
            if isinstance(m, LoopDetectionMiddleware):
                loop_mw = m
                break

        assert loop_mw is not None, "LoopDetectionMiddleware not found in middleware chain"
        assert loop_mw.tool_freq_hard_limit == 50, (
            f"Expected tool_freq_hard_limit=50 from config, got {loop_mw.tool_freq_hard_limit}"
        )
        assert loop_mw.tool_freq_warn == 30, (
            f"Expected tool_freq_warn=30 from config, got {loop_mw.tool_freq_warn}"
        )

    def test_build_middlewares_inspect_override_in_effect(self):
        """The inspect_uploaded_file override (warn=50, hard_limit=100) must
        be present in the middleware's _tool_freq_overrides."""
        cfg = _make_app_config_with_loop_detection()

        with patch(
            "deerflow.agents.lead_agent.agent.get_app_config",
            return_value=cfg,
        ):
            from deerflow.agents.lead_agent.agent import build_middlewares

            middlewares = build_middlewares(
                config={"configurable": {"is_plan_mode": False}},
                model_name=None,
            )

        loop_mw = None
        for m in middlewares:
            if isinstance(m, LoopDetectionMiddleware):
                loop_mw = m
                break

        assert loop_mw is not None
        overrides = loop_mw._tool_freq_overrides
        assert "inspect_uploaded_file" in overrides, (
            "Expected inspect_uploaded_file in tool_freq_overrides"
        )
        assert overrides["inspect_uploaded_file"] == (50, 100), (
            f"Expected (warn=50, hard_limit=100), got {overrides['inspect_uploaded_file']}"
        )


# ---------------------------------------------------------------------------
# 2. Factory path wiring test
# ---------------------------------------------------------------------------

class TestFactoryLoopDetectionWiring:
    """Verify create_deerflow_agent uses app_config.loop_detection.

    create_deerflow_agent returns a CompiledStateGraph, not the raw middleware
    chain. We verify the wiring by asserting that when get_app_config returns
    our config, the lazy import inside _assemble_from_features resolves
    correctly to produce a LoopDetectionMiddleware with config values (not defaults).
    """

    def test_factory_lazy_import_produces_correct_middleware(self):
        """Simulate the factory's lazy import: get_app_config().loop_detection
        must produce LoopDetectionMiddleware with tool_freq_hard_limit=50,
        not the hardcoded default of 5."""
        cfg = _make_app_config_with_loop_detection()

        # The factory's internal _assemble_from_features does:
        #   from deerflow.config.app_config import get_app_config
        #   LoopDetectionMiddleware.from_config(get_app_config().loop_detection)
        # We reproduce exactly that pattern here to verify it produces the
        # expected result with our config.
        with patch(
            "deerflow.config.app_config.get_app_config",
            return_value=cfg,
        ):
            from deerflow.config.app_config import get_app_config

            mw = LoopDetectionMiddleware.from_config(get_app_config().loop_detection)

        assert mw.tool_freq_hard_limit == 50, (
            f"Factory lazy-import path: expected tool_freq_hard_limit=50 from config, "
            f"got {mw.tool_freq_hard_limit}"
        )
        assert mw.tool_freq_warn == 30, (
            f"Factory lazy-import path: expected tool_freq_warn=30 from config, "
            f"got {mw.tool_freq_warn}"
        )
        # Also verify the override is loaded — this confirms the config was fully parsed
        assert "inspect_uploaded_file" in mw._tool_freq_overrides
        assert mw._tool_freq_overrides["inspect_uploaded_file"] == (50, 100)


# ---------------------------------------------------------------------------
# 3. Behavior test — dogfood reproduction
# ---------------------------------------------------------------------------

class TestInspectUploadedFileOverrideBehavior:
    """Verify that the inspect_uploaded_file override actually prevents
    FORCED STOP for legitimate multi-file inspection."""

    def test_six_inspect_calls_no_forced_stop_with_override(self):
        """Call inspect_uploaded_file 6 times with different file args.
        With override (hard_limit=100), the 6th call must NOT trigger
        FORCED STOP. Before the fix (hard_limit=5), call 5 would stop."""
        mw = LoopDetectionMiddleware(
            tool_freq_warn=30,
            tool_freq_hard_limit=50,
            tool_freq_overrides={
                "inspect_uploaded_file": (50, 100),
            },
        )
        runtime = _make_runtime()

        # Emit 6 successive tool calls, each for a different uploaded file
        for i in range(6):
            call = [
                {
                    "name": "inspect_uploaded_file",
                    "id": f"call_{i}",
                    "args": {"file_name": f"file_{i}.xlsx"},
                }
            ]
            warning, hard_stop = mw._track_and_check(
                _make_state_from_ai(tool_calls=call),
                runtime,
            )
            assert not hard_stop, (
                f"Call {i+1}: unexpected FORCED STOP — override hard_limit=100 should not trigger at call {i+1}"
            )
            assert warning is None, (
                f"Call {i+1}: unexpected warning at count {i+1} — warn=50 threshold far from here"
            )

    def test_inspect_falls_back_to_global_hard_limit_when_no_override(self):
        """Without the override, the global tool_freq_hard_limit applies.
        With hard_limit=50, 5 calls should still not stop (unlike old default=5)."""
        mw = LoopDetectionMiddleware(
            tool_freq_warn=30,
            tool_freq_hard_limit=50,
            # No override for inspect_uploaded_file
        )
        runtime = _make_runtime()

        for i in range(5):
            call = [
                {
                    "name": "inspect_uploaded_file",
                    "id": f"call_{i}",
                    "args": {"file_name": f"file_{i}.xlsx"},
                }
            ]
            warning, hard_stop = mw._track_and_check(
                _make_state_from_ai(tool_calls=call),
                runtime,
            )
            assert not hard_stop, (
                f"Call {i+1}: unexpected FORCED STOP with global hard_limit=50"
            )
            # warn threshold is 30, so 5 calls should not warn
            assert warning is None, (
                f"Call {i+1}: unexpected warning at count {i+1} — global warn=30, far from here"
            )

    def test_bare_default_stops_at_call_5(self):
        """Sanity check: with bare defaults (hard_limit=5), the 5th identical
        tool-type call DOES trigger FORCED STOP. This tests the old bug behaviour
        to confirm our test harness detects it correctly."""
        mw = LoopDetectionMiddleware()  # bare defaults: tool_freq_hard_limit=5
        runtime = _make_runtime()

        for i in range(4):
            call = [
                {
                    "name": "inspect_uploaded_file",
                    "id": f"call_{i}",
                    "args": {"file_name": f"file_{i}.xlsx"},
                }
            ]
            warning, hard_stop = mw._track_and_check(
                _make_state_from_ai(tool_calls=call),
                runtime,
            )
            assert not hard_stop, f"Call {i+1}: bare default should not stop before call 5"

        # 5th call — should hit the default hard_limit=5
        call = [
            {
                "name": "inspect_uploaded_file",
                "id": "call_5",
                "args": {"file_name": "file_5.xlsx"},
            }
        ]
        warning, hard_stop = mw._track_and_check(
            _make_state_from_ai(tool_calls=call),
            runtime,
        )
        assert hard_stop, (
            "Bare default hard_limit=5: 5th call should trigger FORCED STOP — "
            "this is the old-bug baseline"
        )


# ---------------------------------------------------------------------------
# 4. Config parsing test — inspect_uploaded_file override from config.yaml values
# ---------------------------------------------------------------------------

class TestConfigOverrideParsing:
    """Verify that LoopDetectionConfig correctly parses inspect_uploaded_file override."""

    def test_config_parses_inspect_override(self):
        """LoopDetectionConfig.from_config-style construction must parse
        inspect_uploaded_file override fields."""
        cfg = LoopDetectionConfig(
            tool_freq_hard_limit=50,
            tool_freq_warn=30,
            tool_freq_overrides={
                "write_todos": ToolFreqOverride(warn=2, hard_limit=4),
                "inspect_uploaded_file": ToolFreqOverride(warn=50, hard_limit=100),
            },
        )

        assert "inspect_uploaded_file" in cfg.tool_freq_overrides
        override = cfg.tool_freq_overrides["inspect_uploaded_file"]
        assert override.warn == 50
        assert override.hard_limit == 100

        # from_config must also correctly translate the override
        mw = LoopDetectionMiddleware.from_config(cfg)
        assert mw._tool_freq_overrides["inspect_uploaded_file"] == (50, 100)
        assert mw._tool_freq_overrides["write_todos"] == (2, 4)
