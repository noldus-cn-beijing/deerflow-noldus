from deerflow.agents.lead_agent.agent import build_middlewares


def _minimal_app_config():
    """A minimal AppConfig so build_middlewares can run without config.yaml on disk.

    build_middlewares gained an ``app_config`` kwarg (sync debt PR#194) precisely so
    callers/tests can inject a config instead of reading the global. These wiring
    tests only assert middleware *presence/ordering*, so an empty-models config is
    sufficient.
    """
    from deerflow.config.app_config import AppConfig
    from deerflow.config.sandbox_config import SandboxConfig

    return AppConfig(
        models=[],
        sandbox=SandboxConfig(use="deerflow.sandbox.local.local_sandbox:LocalSandboxProvider"),
    )


def test_lead_agent_includes_training_data_middleware():
    middlewares = build_middlewares(
        config={"configurable": {"subagent_enabled": False, "is_plan_mode": False}},
        model_name=None,
        app_config=_minimal_app_config(),
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
    middlewares = build_middlewares(
        config={"configurable": {"subagent_enabled": False, "is_plan_mode": False}},
        model_name=None,
        app_config=_minimal_app_config(),
    )
    types = [type(m).__name__ for m in middlewares]
    assert types.index("MemoryMiddleware") < types.index("TrainingDataMiddleware")
