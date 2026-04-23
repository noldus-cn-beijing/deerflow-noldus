import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
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


class TestTrainingDataMiddlewareRecording:
    def test_after_agent_writes_lead_sample(self, tmp_path):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        state_before = middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "thread-xyz"}),
        )
        state = {
            "training_data_path": state_before["training_data_path"],
            "messages": [
                HumanMessage(content="分析这份斑马鱼数据"),
                AIMessage(content="好的，我先解析轨迹文件。"),
            ],
        }

        middleware.after_agent(state=state, runtime=Runtime(context={"thread_id": "thread-xyz"}))

        out = tmp_path / "training-data" / "auto-collected" / "thread-xyz.jsonl"
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        lead_samples = [l for l in lines if l["role"] == "lead"]
        assert len(lead_samples) == 1
        assert lead_samples[0]["input"] == "分析这份斑马鱼数据"
        assert lead_samples[0]["output"] == "好的，我先解析轨迹文件。"
        assert lead_samples[0]["thread_id"] == "thread-xyz"

    def test_after_agent_skips_when_no_messages(self, tmp_path):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "thread-empty"}),
        )
        middleware.after_agent(
            state={"training_data_path": None, "messages": []},
            runtime=Runtime(context={"thread_id": "thread-empty"}),
        )
        out = tmp_path / "training-data" / "auto-collected" / "thread-empty.jsonl"
        assert not out.exists() or out.read_text().strip() == ""


from langchain_core.messages import ToolMessage


class TestSubagentSampleExtraction:
    def test_after_agent_writes_subagent_sample(self, tmp_path):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        sb = middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "thread-sub"}),
        )
        ai_with_task = AIMessage(
            content="我需要 code-executor",
            tool_calls=[{
                "id": "call_1",
                "name": "task",
                "args": {
                    "description": "analyze shoaling",
                    "prompt": "Run ethoinsight-analysis on uploads",
                    "subagent_type": "code-executor",
                },
            }],
        )
        tool_result = ToolMessage(
            content="Analysis complete: 4 metrics computed",
            tool_call_id="call_1",
        )
        state = {
            "training_data_path": sb["training_data_path"],
            "messages": [HumanMessage(content="分析"), ai_with_task, tool_result],
        }
        middleware.after_agent(state=state, runtime=Runtime(context={"thread_id": "thread-sub"}))

        out = tmp_path / "training-data" / "auto-collected" / "thread-sub.jsonl"
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        subagent = [l for l in lines if l["role"] == "subagent"]
        assert len(subagent) == 1
        assert subagent[0]["subagent_type"] == "code-executor"
        assert "analyze shoaling" in subagent[0]["input"]
        assert "Analysis complete" in subagent[0]["output"]


class TestQualityFilter:
    def test_filters_out_error_tool_messages(self, tmp_path):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        sb = middleware.before_agent(state={}, runtime=Runtime(context={"thread_id": "t-err"}))
        ai_with_task = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "task", "args": {"subagent_type": "code-executor", "description": "x", "prompt": "y"}}],
        )
        bad_result = ToolMessage(content='{"error": "subagent timed_out"}', tool_call_id="c1")
        state = {
            "training_data_path": sb["training_data_path"],
            "messages": [HumanMessage(content="hi"), ai_with_task, bad_result],
        }
        middleware.after_agent(state=state, runtime=Runtime(context={"thread_id": "t-err"}))

        out = tmp_path / "training-data" / "auto-collected" / "t-err.jsonl"
        content = out.read_text() if out.exists() else ""
        assert "code-executor" not in content  # the bad subagent sample was filtered


class TestRobustness:
    """Recording must never crash the agent — failures swallowed and logged."""

    def test_after_agent_swallows_exception_and_returns_none(self, tmp_path, monkeypatch, caplog):
        """If extraction raises, agent turn should NOT fail."""
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        sb = middleware.before_agent(state={}, runtime=Runtime(context={"thread_id": "t-crash"}))

        # Force an internal exception
        def boom(*args, **kwargs):
            raise RuntimeError("simulated extractor crash")

        monkeypatch.setattr(middleware, "_extract_lead_samples", boom)

        state = {
            "training_data_path": sb["training_data_path"],
            "messages": [HumanMessage(content="x"), AIMessage(content="y")],
        }

        # Should not raise
        result = middleware.after_agent(state=state, runtime=Runtime(context={"thread_id": "t-crash"}))
        assert result is None

    def test_before_agent_swallows_mkdir_exception(self, tmp_path, monkeypatch):
        """If directory creation fails, before_agent must not crash the agent startup."""
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))

        def boom(*args, **kwargs):
            raise OSError("simulated disk full")

        monkeypatch.setattr(Path, "mkdir", boom)

        result = middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "t-disk-full"}),
        )
        # Should not raise; safest is to return None so downstream skips recording
        assert result is None
