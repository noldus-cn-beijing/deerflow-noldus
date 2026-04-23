import pytest
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.training_data_middleware import TrainingDataMiddleware


class TestTrainingDataMiddlewareInit:
    def test_before_agent_computes_output_dir_from_thread_id(self, tmp_path):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))

        result = middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "thread-abc"}),
        )

        assert result is None or result == {} or "training_data_path" in result
        expected_dir = tmp_path / "training-data" / "auto-collected"
        assert expected_dir.exists()

    def test_before_agent_skips_when_no_thread_id(self, tmp_path, monkeypatch):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        monkeypatch.setattr(
            "deerflow.agents.middlewares.training_data_middleware.get_config",
            lambda: {"configurable": {}},
        )

        result = middleware.before_agent(state={}, runtime=Runtime(context=None))

        assert result is None
        # Should not create the dir when thread_id cannot be resolved
        expected_dir = tmp_path / "training-data" / "auto-collected"
        assert not expected_dir.exists()
