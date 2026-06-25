"""ArtifactMeta contract tests (spec 2026-06-24-frontend-phase0-3-artifact-gallery).

present_file_tool 接出 plan_charts.json 元数据：chart 产物写成 ArtifactMeta dict
（带 chart_id/output_mode/paradigm/chart_type/thumb_path），报告/技能等无 plan
命中的产物仍写裸 string（向后兼容）。failed/remaining 摘要进 charts_status。
"""

import json
from pathlib import Path
from types import SimpleNamespace

present_file_tool_module = __import__(
    "deerflow.tools.builtins.present_file_tool", fromlist=["present_file_tool"]
)


def _make_runtime(outputs_path: str, workspace_path: str | None = None) -> SimpleNamespace:
    thread_data = {"outputs_path": outputs_path}
    if workspace_path:
        thread_data["workspace_path"] = workspace_path
    return SimpleNamespace(
        state={"thread_data": thread_data},
        context={"thread_id": "thread-1"},
    )


def _patch_get_paths(monkeypatch, outputs_dir: Path):
    """present_file 用 get_paths().resolve_virtual_path 把 /mnt/user-data/outputs/<f>
    解析成物理路径。测试里把它指回 tmp_path 的 outputs_dir（镜像现有 core-logic 测试）。"""
    monkeypatch.setattr(
        present_file_tool_module,
        "get_paths",
        lambda: SimpleNamespace(resolve_virtual_path=lambda thread_id, path, *, user_id=None: outputs_dir / Path(path).name),
    )


def _write_plan(workspace_dir: Path, charts: list[dict], paradigm: str = "epm") -> None:
    (workspace_dir / "plan_charts.json").write_text(
        json.dumps({"paradigm": paradigm, "charts": charts}), encoding="utf-8"
    )


class TestBuildArtifactMeta:
    """_build_artifact_meta: plan 命中 → chart dict；未命中 / 无 plan → 裸 string。"""

    def test_plan_miss_returns_bare_string(self):
        plan = {"paradigm": "epm", "by_output": {"/mnt/user-data/outputs/other.png": {}}}
        out = present_file_tool_module._build_artifact_meta(
            "/mnt/user-data/outputs/report.md", plan, None, generate_thumb=False
        )
        assert out == "/mnt/user-data/outputs/report.md"

    def test_no_plan_returns_bare_string(self):
        out = present_file_tool_module._build_artifact_meta("/x.png", None, None, generate_thumb=False)
        assert out == "/x.png"

    def test_plan_hit_returns_chart_meta(self):
        plan = {
            "paradigm": "epm",
            "by_output": {
                "/mnt/user-data/outputs/open_arm_bar.png": {
                    "id": "open_arm_bar",
                    "script": "epm.plot_open_arm_time_ratio_bar",
                    "output_mode": "aggregate",
                    "metric": "open_arm_time_ratio",
                }
            },
        }
        out = present_file_tool_module._build_artifact_meta(
            "/mnt/user-data/outputs/open_arm_bar.png", plan, None, generate_thumb=False
        )
        assert isinstance(out, dict)
        assert out["path"] == "/mnt/user-data/outputs/open_arm_bar.png"
        assert out["kind"] == "chart"
        assert out["chart_id"] == "open_arm_bar"
        assert out["output_mode"] == "aggregate"
        assert out["paradigm"] == "epm"
        assert out["metric"] == "open_arm_time_ratio"
        assert out["chart_type"] == "bar"  # script ..._bar 命中（无 box/trajectory 歧义）

    def test_chart_type_derived_from_trajectory(self):
        out = present_file_tool_module._derive_chart_type("plot_trajectory_subject_1", "epm.trajectory")
        assert out == "trajectory"


class TestPresentFileToolMetadataJoin:
    """present_file_tool.func 端到端：写 plan_charts.json，present chart png → ArtifactMeta。"""

    def test_present_chart_png_emits_artifact_meta(self, tmp_path, monkeypatch):
        outputs_dir = tmp_path / "threads/thread-1/user-data/outputs"
        workspace_dir = tmp_path / "threads/thread-1/user-data/workspace"
        outputs_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        chart_png = outputs_dir / "box_open_arm.png"
        chart_png.write_bytes(b"fake-png")
        _patch_get_paths(monkeypatch, outputs_dir)

        _write_plan(
            workspace_dir,
            [
                {
                    "id": "box_open_arm",
                    "script": "epm.plot_open_arm_time_ratio_bar",
                    "output": "/mnt/user-data/outputs/box_open_arm.png",
                    "output_mode": "aggregate",
                }
            ],
        )

        result = present_file_tool_module.present_file_tool.func(
            runtime=_make_runtime(str(outputs_dir), str(workspace_dir)),
            filepaths=["/mnt/user-data/outputs/box_open_arm.png"],
            tool_call_id="tc-meta",
        )

        artifacts = result.update["artifacts"]
        assert len(artifacts) == 1
        assert isinstance(artifacts[0], dict)
        assert artifacts[0]["path"] == "/mnt/user-data/outputs/box_open_arm.png"
        assert artifacts[0]["chart_id"] == "box_open_arm"
        assert artifacts[0]["output_mode"] == "aggregate"
        assert artifacts[0]["paradigm"] == "epm"

    def test_present_non_chart_file_still_bare_string(self, tmp_path, monkeypatch):
        """报告/技能等无 plan 命中 → 裸 string（向后兼容，现有测试不破）。"""
        outputs_dir = tmp_path / "threads/thread-1/user-data/outputs"
        workspace_dir = tmp_path / "threads/thread-1/user-data/workspace"
        outputs_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (outputs_dir / "report.md").write_text("ok")
        _patch_get_paths(monkeypatch, outputs_dir)

        _write_plan(workspace_dir, [])  # 空 plan

        result = present_file_tool_module.present_file_tool.func(
            runtime=_make_runtime(str(outputs_dir), str(workspace_dir)),
            filepaths=["/mnt/user-data/outputs/report.md"],
            tool_call_id="tc-report",
        )

        assert result.update["artifacts"] == ["/mnt/user-data/outputs/report.md"]

    def test_no_workspace_still_bare_string(self, tmp_path, monkeypatch):
        """无 workspace_path（plan 读不到）→ 退化裸 string，不崩。"""
        outputs_dir = tmp_path / "threads/thread-1/user-data/outputs"
        outputs_dir.mkdir(parents=True)
        (outputs_dir / "x.png").write_bytes(b"png")
        _patch_get_paths(monkeypatch, outputs_dir)

        result = present_file_tool_module.present_file_tool.func(
            runtime=_make_runtime(str(outputs_dir)),  # 无 workspace_path
            filepaths=["/mnt/user-data/outputs/x.png"],
            tool_call_id="tc-no-ws",
        )
        assert result.update["artifacts"] == ["/mnt/user-data/outputs/x.png"]


class TestChartsStatusSurface:
    """present_file 一并带出 failed/remaining 摘要进 charts_status（spec §四 Step 5）。"""

    def test_failed_charts_surfaced_into_state(self, tmp_path, monkeypatch):
        outputs_dir = tmp_path / "threads/thread-1/user-data/outputs"
        workspace_dir = tmp_path / "threads/thread-1/user-data/workspace"
        outputs_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (outputs_dir / "ok.png").write_bytes(b"png")
        _patch_get_paths(monkeypatch, outputs_dir)

        _write_plan(
            workspace_dir,
            [{"id": "ok", "script": "s", "output": "/mnt/user-data/outputs/ok.png", "output_mode": "aggregate"}],
        )
        (workspace_dir / "handoff_chart_maker.json").write_text(
            json.dumps(
                {
                    "chart_files": ["/mnt/user-data/outputs/ok.png"],
                    "failed_charts": [{"chart_id": "bad", "reason": "rc=1"}],
                    "remaining_charts": [{"chart_id": "trunc", "reason": "budget"}],
                }
            ),
            encoding="utf-8",
        )

        result = present_file_tool_module.present_file_tool.func(
            runtime=_make_runtime(str(outputs_dir), str(workspace_dir)),
            filepaths=["/mnt/user-data/outputs/ok.png"],
            tool_call_id="tc-status",
        )

        status = result.update.get("charts_status")
        assert status is not None
        assert status["n_rendered"] == 1
        assert status["failed"] == [{"chart_id": "bad", "reason": "rc=1"}]
        assert status["remaining"] == [{"chart_id": "trunc", "reason": "budget"}]

    def test_no_failures_no_charts_status(self, tmp_path, monkeypatch):
        """全成功 → 不污染 state（charts_status 键不出现）。"""
        outputs_dir = tmp_path / "threads/thread-1/user-data/outputs"
        workspace_dir = tmp_path / "threads/thread-1/user-data/workspace"
        outputs_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (outputs_dir / "ok.png").write_bytes(b"png")
        _patch_get_paths(monkeypatch, outputs_dir)

        _write_plan(workspace_dir, [{"id": "ok", "script": "s", "output": "/mnt/user-data/outputs/ok.png"}])
        (workspace_dir / "handoff_chart_maker.json").write_text(
            json.dumps({"chart_files": ["/mnt/user-data/outputs/ok.png"], "failed_charts": [], "remaining_charts": []}),
            encoding="utf-8",
        )

        result = present_file_tool_module.present_file_tool.func(
            runtime=_make_runtime(str(outputs_dir), str(workspace_dir)),
            filepaths=["/mnt/user-data/outputs/ok.png"],
            tool_call_id="tc-clean",
        )
        assert "charts_status" not in result.update


class TestThumbnailGeneration:
    """Pillow 缩略图（spec §3.1.6）。无 Pillow/读图失败 → 不阻塞，退化原图。"""

    def test_thumbnail_generated_for_chart_png(self, tmp_path, monkeypatch):
        try:
            from PIL import Image
        except ImportError:
            return  # 环境无 Pillow，跳过

        outputs_dir = tmp_path / "threads/thread-1/user-data/outputs"
        workspace_dir = tmp_path / "threads/thread-1/user-data/workspace"
        outputs_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        chart_png = outputs_dir / "traj.png"
        Image.new("RGB", (800, 800), "red").save(chart_png, format="PNG")
        _patch_get_paths(monkeypatch, outputs_dir)

        _write_plan(
            workspace_dir,
            [{"id": "traj", "script": "epm.trajectory", "output": "/mnt/user-data/outputs/traj.png"}],
        )

        result = present_file_tool_module.present_file_tool.func(
            runtime=_make_runtime(str(outputs_dir), str(workspace_dir)),
            filepaths=["/mnt/user-data/outputs/traj.png"],
            tool_call_id="tc-thumb",
        )

        meta = result.update["artifacts"][0]
        assert isinstance(meta, dict)
        assert meta["thumb_path"] == "/mnt/user-data/outputs/traj.thumb.webp"
        assert (outputs_dir / "traj.thumb.webp").exists()

    def test_thumbnail_absent_when_generation_fails(self, tmp_path, monkeypatch):
        """缩略图生成失败 → 无 thumb_path，退化原 path（不崩）。

        直接 mock _generate_thumbnail 返回 None（覆盖无 Pillow/读图失败/写盘失败
        整个「失败即退化」契约，而非仅 import 失败这一条路径）。
        """
        outputs_dir = tmp_path / "threads/thread-1/user-data/outputs"
        workspace_dir = tmp_path / "threads/thread-1/user-data/workspace"
        outputs_dir.mkdir(parents=True)
        workspace_dir.mkdir(parents=True)
        (outputs_dir / "x.png").write_bytes(b"not-a-real-png")
        _patch_get_paths(monkeypatch, outputs_dir)

        _write_plan(workspace_dir, [{"id": "x", "script": "s", "output": "/mnt/user-data/outputs/x.png"}])
        monkeypatch.setattr(present_file_tool_module, "_generate_thumbnail", lambda *a, **k: None)

        result = present_file_tool_module.present_file_tool.func(
            runtime=_make_runtime(str(outputs_dir), str(workspace_dir)),
            filepaths=["/mnt/user-data/outputs/x.png"],
            tool_call_id="tc-nothumb",
        )
        meta = result.update["artifacts"][0]
        assert isinstance(meta, dict)
        assert "thumb_path" not in meta  # 退化：无缩略图
        assert meta["path"] == "/mnt/user-data/outputs/x.png"
