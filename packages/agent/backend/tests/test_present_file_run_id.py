"""Task 2 (P1) present_file run_id stamping (spec 2026-06-26 §任务2 路 A).

chart artifact 元数据带 ``run_id``，让 merge_artifacts 按 (run_id, path) 去重——
跨 run 同 chart_id 不互覆盖（"113 图只显示 1 张"家族病根之一）。present_file 从
``runtime.context["run_id"]`` 取当前 run（run worker 注入，task_tool 透传）。
"""

import json
from pathlib import Path
from types import SimpleNamespace

present_file_tool_module = __import__(
    "deerflow.tools.builtins.present_file_tool", fromlist=["present_file_tool"]
)


def _make_runtime(outputs_path: str, workspace_path: str, *, run_id: str | None = None) -> SimpleNamespace:
    thread_data = {"outputs_path": outputs_path, "workspace_path": workspace_path}
    ctx = {"thread_id": "thread-1"}
    if run_id is not None:
        ctx["run_id"] = run_id
    return SimpleNamespace(state={"thread_data": thread_data}, context=ctx)


def _patch_get_paths(monkeypatch, outputs_dir: Path):
    monkeypatch.setattr(
        present_file_tool_module,
        "get_paths",
        lambda: SimpleNamespace(resolve_virtual_path=lambda thread_id, path, *, user_id=None: outputs_dir / Path(path).name),
    )


def _write_plan(workspace_dir: Path, charts: list[dict], paradigm: str = "epm") -> None:
    (workspace_dir / "plan_charts.json").write_text(
        json.dumps({"paradigm": paradigm, "charts": charts}), encoding="utf-8"
    )


class TestBuildArtifactMetaRunId:
    """_build_artifact_meta 把 run_id 写进 chart 元数据（无 run_id → 不写字段）。"""

    def test_chart_meta_carries_run_id(self):
        plan = {
            "paradigm": "epm",
            "by_output": {
                "/mnt/user-data/outputs/o.png": {"id": "o", "script": "epm.plot_o_bar", "output_mode": "aggregate"},
            },
        }
        out = present_file_tool_module._build_artifact_meta(
            "/mnt/user-data/outputs/o.png", plan, None, generate_thumb=False, run_id="run-42",
        )
        assert isinstance(out, dict)
        assert out["run_id"] == "run-42"

    def test_chart_meta_no_run_id_omits_field(self):
        """无 run_id（旧调用 / 非 run 上下文）→ 不写 run_id 字段（向后兼容裸 meta）。"""
        plan = {
            "paradigm": "epm",
            "by_output": {
                "/mnt/user-data/outputs/o.png": {"id": "o", "script": "epm.plot_o_bar", "output_mode": "aggregate"},
            },
        }
        out = present_file_tool_module._build_artifact_meta(
            "/mnt/user-data/outputs/o.png", plan, None, generate_thumb=False,
        )
        assert isinstance(out, dict)
        assert "run_id" not in out


class TestPresentFileToolRunIdStamping:
    """present_file_tool.func 端到端：runtime.context.run_id → chart meta.run_id。"""

    def test_present_chart_stamps_context_run_id(self, tmp_path, monkeypatch):
        outputs_dir = tmp_path / "threads/thread-1/user-data/outputs"
        workspace_dir = tmp_path / "threads/thread-1/user-data/workspace"
        outputs_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (outputs_dir / "box_o.png").write_bytes(b"fake-png")
        _patch_get_paths(monkeypatch, outputs_dir)
        _write_plan(
            workspace_dir,
            [{"id": "box_o", "script": "epm.plot_o_bar", "output": "/mnt/user-data/outputs/box_o.png", "output_mode": "aggregate"}],
        )

        result = present_file_tool_module.present_file_tool.func(
            runtime=_make_runtime(str(outputs_dir), str(workspace_dir), run_id="run-7"),
            filepaths=["/mnt/user-data/outputs/box_o.png"],
            tool_call_id="tc",
        )
        artifacts = result.update["artifacts"]
        assert len(artifacts) == 1
        assert artifacts[0]["run_id"] == "run-7"

    def test_present_chart_without_run_id_in_context(self, tmp_path, monkeypatch):
        """runtime.context 无 run_id（如 lead 非 run 上下文）→ meta 不写 run_id 字段。"""
        outputs_dir = tmp_path / "threads/thread-1/user-data/outputs"
        workspace_dir = tmp_path / "threads/thread-1/user-data/workspace"
        outputs_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (outputs_dir / "box_o.png").write_bytes(b"fake-png")
        _patch_get_paths(monkeypatch, outputs_dir)
        _write_plan(
            workspace_dir,
            [{"id": "box_o", "script": "epm.plot_o_bar", "output": "/mnt/user-data/outputs/box_o.png", "output_mode": "aggregate"}],
        )

        result = present_file_tool_module.present_file_tool.func(
            runtime=_make_runtime(str(outputs_dir), str(workspace_dir)),  # 无 run_id
            filepaths=["/mnt/user-data/outputs/box_o.png"],
            tool_call_id="tc",
        )
        artifacts = result.update["artifacts"]
        assert len(artifacts) == 1
        assert "run_id" not in artifacts[0]
