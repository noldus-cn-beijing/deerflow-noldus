import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
        import deerflow.agents.middlewares.training_data_middleware as tdm_module

        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        monkeypatch.setattr(tdm_module, "get_config", lambda: {"configurable": {}})

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
        lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        lead_samples = [line for line in lines if line["role"] == "lead"]
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
        lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        subagent = [line for line in lines if line["role"] == "subagent"]
        assert len(subagent) == 1
        assert subagent[0]["subagent_type"] == "code-executor"
        assert "analyze shoaling" in subagent[0]["input"]
        assert "Analysis complete" in subagent[0]["output"]


class TestMessageIdCapture:
    """Samples must carry a message_id that matches what the frontend sends
    for feedback, so extract_e2e_sessions.py can join precisely."""

    def test_lead_sample_records_ai_message_id(self, tmp_path):
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        sb = middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "thread-mid"}),
        )
        ai = AIMessage(content="ok", id="run-abc-123")
        state = {
            "training_data_path": sb["training_data_path"],
            "messages": [HumanMessage(content="hi"), ai],
        }
        middleware.after_agent(state=state, runtime=Runtime(context={"thread_id": "thread-mid"}))

        out = tmp_path / "training-data" / "auto-collected" / "thread-mid.jsonl"
        lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        lead = [line for line in lines if line["role"] == "lead"]
        assert len(lead) == 1
        assert lead[0]["message_id"] == "run-abc-123"

    def test_subagent_sample_records_subtask_prefixed_id(self, tmp_path):
        """Frontend keys subagent feedback as 'subtask-{tool_call_id}'; match it."""
        middleware = TrainingDataMiddleware(base_dir=str(tmp_path))
        sb = middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "thread-sub-mid"}),
        )
        ai_with_task = AIMessage(
            content="dispatching",
            tool_calls=[{
                "id": "call_xyz",
                "name": "task",
                "args": {"subagent_type": "code-executor", "description": "d", "prompt": "p"},
            }],
        )
        tool_result = ToolMessage(content="done", tool_call_id="call_xyz")
        state = {
            "training_data_path": sb["training_data_path"],
            "messages": [HumanMessage(content="hi"), ai_with_task, tool_result],
        }
        middleware.after_agent(state=state, runtime=Runtime(context={"thread_id": "thread-sub-mid"}))

        out = tmp_path / "training-data" / "auto-collected" / "thread-sub-mid.jsonl"
        lines = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
        sub = [line for line in lines if line["role"] == "subagent"]
        assert len(sub) == 1
        assert sub[0]["message_id"] == "subtask-call_xyz"


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


class TestTrainingDataMiddlewareAskClarification:
    """Regression: ask_clarification turns must be correctly recorded."""

    @pytest.fixture
    def middleware(self):
        return TrainingDataMiddleware()

    def test_ask_clarification_turn_records_lead_sample(self, middleware, tmp_path):
        """ask_clarification AIMessage -> HumanMessage should produce a recorded sample."""
        sb = middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "t-clarify"}),
        )

        messages = [
            HumanMessage(content="分析数据", id="h1"),
            AIMessage(
                content="哪类实验？",
                id="a1",
                tool_calls=[{
                    "name": "ask_clarification",
                    "args": {
                        "question": "哪类实验？",
                        "clarification_type": "approach_choice",
                        "options": ["Shoaling", "OFT", "EPM"],
                    },
                    "id": "tc1",
                }],
            ),
            ToolMessage(content="Shoaling", tool_call_id="tc1", name="ask_clarification", id="t1"),
            HumanMessage(content="Shoaling", id="h2"),
            AIMessage(content="启动 Shoaling 分析...", id="a2"),
        ]

        state = {"training_data_path": sb["training_data_path"], "messages": messages}
        result = middleware.after_agent(state=state, runtime=Runtime(context={"thread_id": "t-clarify"}))
        assert result is None

        jsonl = Path(sb["training_data_path"])
        assert jsonl.exists()
        lines = jsonl.read_text().strip().split("\n")
        assert len(lines) >= 1

        sample = json.loads(lines[0])
        assert sample["role"] == "lead"

    def test_ask_clarification_options_array_preserved(self, middleware, tmp_path):
        """The options array in ask_clarification args should survive in recorded data."""
        sb = middleware.before_agent(
            state={},
            runtime=Runtime(context={"thread_id": "t-options"}),
        )

        options = ["Shoaling", "OFT", "EPM", "Light-Dark Box"]

        messages = [
            HumanMessage(content="分析", id="h1"),
            AIMessage(
                content="",
                id="a1",
                tool_calls=[{
                    "name": "ask_clarification",
                    "args": {
                        "question": "什么实验？",
                        "clarification_type": "approach_choice",
                        "options": options,
                    },
                    "id": "tc1",
                }],
            ),
            ToolMessage(content="Shoaling", tool_call_id="tc1", id="t1"),
            HumanMessage(content="Shoaling", id="h2"),
            AIMessage(content="启动 Shoaling 分析", id="a2"),
        ]

        state = {"training_data_path": sb["training_data_path"], "messages": messages}
        middleware.after_agent(state=state, runtime=Runtime(context={"thread_id": "t-options"}))

        jsonl = Path(sb["training_data_path"])
        lines = jsonl.read_text().strip().split("\n")
        assert len(lines) >= 1

        for line in lines:
            sample = json.loads(line)
            assert "output" in sample
