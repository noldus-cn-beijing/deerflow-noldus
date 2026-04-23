from deerflow.agents.middlewares.training_data_middleware import TrainingDataMiddleware


def test_lead_agent_includes_training_data_middleware():
    from deerflow.agents.lead_agent.agent import _build_middlewares

    middlewares = _build_middlewares(
        config={"configurable": {"subagent_enabled": False, "is_plan_mode": False}},
        model_name=None,
    )
    types = [type(m).__name__ for m in middlewares]
    assert "TrainingDataMiddleware" in types
    # Must come before ClarificationMiddleware (which must be last)
    assert types.index("TrainingDataMiddleware") < types.index("ClarificationMiddleware")


def test_training_middleware_sits_after_memory():
    """Per plan spec: TrainingDataMiddleware must be registered after MemoryMiddleware.

    Rationale: both are after_agent observers; TrainingDataMiddleware should see
    the final message state alongside MemoryMiddleware, not before it.
    """
    from deerflow.agents.lead_agent.agent import _build_middlewares

    middlewares = _build_middlewares(
        config={"configurable": {"subagent_enabled": False, "is_plan_mode": False}},
        model_name=None,
    )
    types = [type(m).__name__ for m in middlewares]
    assert types.index("MemoryMiddleware") < types.index("TrainingDataMiddleware")
