"""Regression tests for SubagentExecutor.calculate_subagent_recursion_limit.

Context: a 2026-05-25 deerflow upstream sync added SafetyFinishReasonMiddleware
to the subagent middleware chain (an extra after_model hook). The old formula
``max_turns * 2 + 1`` assumed 2 graph nodes per turn (model + tools), but each
overridden middleware hook is itself a graph node in langchain.agents.create_agent.
Result: data-analyst max_turns=12 capped recursion_limit at 25 while the actual
chain needed ~32 nodes for the same 12 turns — LangGraph killed the run mid-flight
even though our AI-message counter (the real turn limit) hadn't fired.

The helper now counts hooks dynamically: per-turn = before_model + after_model + 2,
plus one-off before_agent / after_agent at the graph boundary.
"""

import sys
from unittest.mock import MagicMock

import pytest

# Mirror test_subagent_executor.py: undo conftest's executor mock and stub out
# transitive imports so we can load the real helper in isolation.
_MOCKED_MODULE_NAMES = [
    "deerflow.agents",
    "deerflow.agents.thread_state",
    "deerflow.agents.middlewares",
    "deerflow.agents.middlewares.thread_data_middleware",
    "deerflow.sandbox",
    "deerflow.sandbox.middleware",
    "deerflow.sandbox.security",
    "deerflow.models",
]


@pytest.fixture(scope="module")
def helper_and_base():
    """Load the real ``calculate_subagent_recursion_limit`` + AgentMiddleware base."""
    original_modules = {name: sys.modules.get(name) for name in _MOCKED_MODULE_NAMES}
    original_executor = sys.modules.get("deerflow.subagents.executor")

    if "deerflow.subagents.executor" in sys.modules:
        del sys.modules["deerflow.subagents.executor"]
    for name in _MOCKED_MODULE_NAMES:
        sys.modules[name] = MagicMock()

    from langchain.agents.middleware import AgentMiddleware

    from deerflow.subagents.executor import calculate_subagent_recursion_limit

    yield calculate_subagent_recursion_limit, AgentMiddleware

    for name in _MOCKED_MODULE_NAMES:
        if original_modules[name] is not None:
            sys.modules[name] = original_modules[name]
        elif name in sys.modules:
            del sys.modules[name]

    if original_executor is not None:
        sys.modules["deerflow.subagents.executor"] = original_executor
    elif "deerflow.subagents.executor" in sys.modules:
        del sys.modules["deerflow.subagents.executor"]


def _make_middleware(base, *, before_agent=False, before_model=False, after_model=False, after_agent=False, async_only=False):
    """Build an AgentMiddleware subclass overriding the requested hooks.

    Honours the contract from langchain.agents.factory: a hook is "implemented"
    if either its sync OR async variant is overridden on the subclass.
    """
    attrs = {}
    if before_agent:
        if async_only:
            async def _ab(self, state, runtime):  # noqa: ANN001
                return None
            attrs["abefore_agent"] = _ab
        else:
            def _b(self, state, runtime):  # noqa: ANN001
                return None
            attrs["before_agent"] = _b
    if before_model:
        if async_only:
            async def _abm(self, state, runtime):  # noqa: ANN001
                return None
            attrs["abefore_model"] = _abm
        else:
            def _bm(self, state, runtime):  # noqa: ANN001
                return None
            attrs["before_model"] = _bm
    if after_model:
        if async_only:
            async def _aam(self, state, runtime):  # noqa: ANN001
                return None
            attrs["aafter_model"] = _aam
        else:
            def _am(self, state, runtime):  # noqa: ANN001
                return None
            attrs["after_model"] = _am
    if after_agent:
        if async_only:
            async def _aaa(self, state, runtime):  # noqa: ANN001
                return None
            attrs["aafter_agent"] = _aaa
        else:
            def _aa(self, state, runtime):  # noqa: ANN001
                return None
            attrs["after_agent"] = _aa
    cls = type("DynMiddleware", (base,), attrs)
    return cls()


class TestCalculateSubagentRecursionLimit:
    def test_empty_chain_only_baseline(self, helper_and_base):
        calc, _ = helper_and_base
        # 12 turns * (0 + 1 + 0 + 1) + 0 + 3 = 27
        assert calc([], 12) == 27

    def test_only_after_model_hook_matches_safety_middleware_case(self, helper_and_base):
        """Reproduces the 2026-05-25 SafetyFinishReasonMiddleware regression.

        Before fix: max_turns=12 gave recursion_limit=25 — too low to finish.
        After fix: helper must count the after_model hook and yield >= 12 * 3.
        """
        calc, base = helper_and_base
        m = _make_middleware(base, after_model=True)
        # 12 turns * (0 + 1 + 1 + 1) + 0 + 3 = 39
        assert calc([m], 12) == 39

    def test_async_only_hook_counts(self, helper_and_base):
        """Hook is counted when only the async variant is overridden."""
        calc, base = helper_and_base
        m = _make_middleware(base, after_model=True, async_only=True)
        assert calc([m], 12) == 39

    def test_all_four_hooks(self, helper_and_base):
        calc, base = helper_and_base
        m = _make_middleware(base, before_agent=True, before_model=True, after_model=True, after_agent=True)
        # 12 turns * (1 + 1 + 1 + 1) + (1 + 1) + 3 = 53
        assert calc([m], 12) == 53

    def test_before_after_agent_are_one_off(self, helper_and_base):
        """before_agent / after_agent fire exactly once, not per turn."""
        calc, base = helper_and_base
        only_boundary = _make_middleware(base, before_agent=True, after_agent=True)
        # 12 turns * 2 + 2 + 3 = 29  (NOT 12 * 4)
        assert calc([only_boundary], 12) == 29

    def test_mixed_chain_sums_correctly(self, helper_and_base):
        """A realistic chain matching the data-analyst subagent at the time of fix.

        Reproduces the order-of-magnitude of the production chain (~8 middlewares)
        and verifies the formula isn't accidentally double-counting.
        """
        calc, base = helper_and_base
        chain = [
            _make_middleware(base, before_model=True),
            _make_middleware(base, before_model=True, after_model=True),
            _make_middleware(base, after_model=True),
            _make_middleware(base, after_model=True),
            _make_middleware(base, before_agent=True, after_agent=True),
        ]
        # before_model = 2, after_model = 3, before_agent = 1, after_agent = 1
        # 12 * (2 + 1 + 3 + 1) + 2 + 3 = 89
        assert calc(chain, 12) == 89

    def test_data_analyst_pre_fix_recursion_was_insufficient(self, helper_and_base):
        """Sanity check: the old `max_turns * 2 + 1` formula is now provably wrong.

        For any chain with at least one before_model OR after_model hook,
        the dynamic formula must exceed the legacy formula.
        """
        calc, base = helper_and_base
        m = _make_middleware(base, after_model=True)
        legacy = 12 * 2 + 1
        assert calc([m], 12) > legacy

    def test_invalid_max_turns(self, helper_and_base):
        calc, _ = helper_and_base
        with pytest.raises(ValueError):
            calc([], 0)
        with pytest.raises(ValueError):
            calc([], -1)

    def test_scales_linearly_with_max_turns(self, helper_and_base):
        calc, base = helper_and_base
        m = _make_middleware(base, after_model=True)
        # Difference between consecutive max_turns equals per_turn cost
        d10 = calc([m], 10)
        d11 = calc([m], 11)
        d12 = calc([m], 12)
        assert d11 - d10 == d12 - d11  # constant per-turn slope
