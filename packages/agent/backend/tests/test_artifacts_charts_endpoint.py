"""新端点 GET /api/threads/{thread_id}/artifacts/charts 的单测。

spec 2026-06-26-artifact-bubbling-report-display-gallery-return-fix §1.1：画廊数据源
从 LangGraph state.artifacts（subagent→lead 边界会丢 artifacts，实测 113 张只活 2 张）
换成「磁盘 + plan_charts.json」直取——磁盘是唯一真相，不依赖 state 冒泡。

端点复用 archive_artifacts 的列目录逻辑（resolve_thread_virtual_path + rglob *.png，
排除 .thumb.webp）+ join plan_charts.json 元数据，吐 ArtifactMeta[]。
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

import app.gateway.routers.artifacts as artifacts_router


def _resolve_factory(outputs_dir: Path, plan_path: Path | None):
    """造一个替身 resolve_thread_virtual_path：
    /mnt/user-data/outputs → outputs_dir；/mnt/user-data/workspace/plan_charts.json → plan_path。
    """

    def _resolve(_thread_id: str, virtual_path: str) -> Path:
        vp = virtual_path.rstrip("/")
        if vp.endswith("/outputs"):
            return outputs_dir
        if vp.endswith("plan_charts.json"):
            return plan_path if plan_path is not None else outputs_dir / "_missing_plan.json"
        return outputs_dir / "_other"

    return _resolve


def test_list_chart_artifacts_returns_disk_pngs_with_plan_metadata(tmp_path, monkeypatch) -> None:
    """磁盘 2 张 png + plan_charts.json 命中 → 返回 2 个 ArtifactMeta，带 chart 元数据。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "plot_box_open_arm.png").write_bytes(b"\x89PNG fake")
    (outputs_dir / "trajectory_subject_0.png").write_bytes(b"\x89PNG fake")

    plan = {
        "paradigm": "epm",
        "charts": [
            {
                "id": "box_open_arm",
                "output": "/mnt/user-data/outputs/plot_box_open_arm.png",
                "output_mode": "aggregate",
                "script": "ethoinsight.scripts.epm.plot_box_open_arm",
                "metric": "open_arm_time_ratio",
            },
            {
                "id": "trajectory",
                "output": "/mnt/user-data/outputs/trajectory_subject_0.png",
                "output_mode": "per_subject",
                "subject": "subject_0",
                "script": "ethoinsight.scripts.epm.plot_trajectory",
            },
        ],
    }
    plan_path = tmp_path / "plan_charts.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir, plan_path))

    result = artifacts_router.list_chart_artifacts("thread-1", None)

    assert isinstance(result, list)
    assert len(result) == 2
    by_path = {m["path"]: m for m in result}

    agg = by_path["/mnt/user-data/outputs/plot_box_open_arm.png"]
    assert agg["kind"] == "chart"
    assert agg["chart_id"] == "box_open_arm"
    assert agg["output_mode"] == "aggregate"
    assert agg["paradigm"] == "epm"
    assert agg["metric"] == "open_arm_time_ratio"
    assert agg["chart_type"] == "box"

    per_subj = by_path["/mnt/user-data/outputs/trajectory_subject_0.png"]
    assert per_subj["output_mode"] == "per_subject"
    assert per_subj["chart_type"] == "trajectory"


def test_list_chart_artifacts_excludes_thumb_files(tmp_path, monkeypatch) -> None:
    """缩略图 .thumb.webp 不算画廊图（它是渲染衍生物）。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "plot_box_open_arm.png").write_bytes(b"\x89PNG")
    # 缩略图：同 stem 的 .thumb.webp，必须被排除
    (outputs_dir / "plot_box_open_arm.thumb.webp").write_bytes(b"WEBP")
    # 命中缩略图时 thumb_path 带出
    plan_path = tmp_path / "plan_charts.json"
    plan_path.write_text(
        json.dumps(
            {
                "paradigm": "epm",
                "charts": [
                    {"id": "box_open_arm", "output": "/mnt/user-data/outputs/plot_box_open_arm.png", "output_mode": "aggregate"},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir, plan_path))

    result = artifacts_router.list_chart_artifacts("thread-1", None)

    assert len(result) == 1
    meta = result[0]
    assert meta["path"] == "/mnt/user-data/outputs/plot_box_open_arm.png"
    # 缩略图存在 → thumb_path 带出
    assert meta.get("thumb_path") == "/mnt/user-data/outputs/plot_box_open_arm.thumb.webp"


def test_list_chart_artifacts_bare_meta_when_path_not_in_plan(tmp_path, monkeypatch) -> None:
    """磁盘有图但 plan 没命中（极少）→ 退裸 {path}，不丢图。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "orphan.png").write_bytes(b"\x89PNG")
    plan_path = tmp_path / "plan_charts.json"
    plan_path.write_text(json.dumps({"paradigm": "epm", "charts": []}), encoding="utf-8")

    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir, plan_path))

    result = artifacts_router.list_chart_artifacts("thread-1", None)

    assert len(result) == 1
    assert result[0]["path"] == "/mnt/user-data/outputs/orphan.png"
    assert "kind" not in result[0] or result[0].get("kind") is None


def test_list_chart_artifacts_returns_empty_when_no_outputs_dir(tmp_path, monkeypatch) -> None:
    """0 图 thread：端点返回空列表（200），不报错（spec §五 回归项）。"""
    outputs_dir = tmp_path / "outputs"  # 不 mkdir
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir, None))

    result = artifacts_router.list_chart_artifacts("thread-1", None)

    assert result == []


def test_list_chart_artifacts_missing_plan_still_lists_disk_pngs(tmp_path, monkeypatch) -> None:
    """plan_charts.json 缺失/坏掉 → 仍按磁盘列图（裸 {path}），plan 是增强不是前提。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "plot_a.png").write_bytes(b"\x89PNG")
    # plan_path 不存在
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir, None))

    result = artifacts_router.list_chart_artifacts("thread-1", None)

    assert len(result) == 1
    assert result[0]["path"] == "/mnt/user-data/outputs/plot_a.png"


class TestDeriveChartType:
    """_derive_chart_type: token 漏掉的图名靠显式映射表补正（spec 2026-06-29-assets-gallery-fixes 问题2）。

    根因坐实：``zone_entry_distribution`` 的 chart_id/script 不含任何 token
    （trajectory/timeseries/box/bar/...）→ 返回 None → 前端 box/bar filter 用精确匹配
    把它永久吞掉。修法=显式 chart_id → type 映射表优先，token 启发式作 fallback。

    两处推导（artifacts.py 网关画廊端点 + present_file_tool 同构）必须同型——这里同时
    import 两份做 parity 断言，锁住「SSOT 一处推导」不漂移（见 CLAUDE.md 三大病理自检 #2）。
    """

    # 显式映射应覆盖的已知图（chart_id/script → type），证据来自 ethoinsight/charts.py 实际调用：
    #   zone_entry_distribution / center_entry_summary → ax.bar → "bar"（非 box）
    #   activity_intensity / time_progress → ax.plot 时间序列 → "line"
    #   struggle_distribution → broken_barh 随时间 → "timeseries"
    #   rose → 极坐标 bar → "rose"
    CASES = [
        # (chart_id, script, expected_type)
        ("zone_entry_distribution", "ethoinsight.scripts.epm.plot_zone_entry_distribution", "bar"),
        ("zone_entry_distribution", "ethoinsight.scripts.ldb.plot_zone_entry_distribution", "bar"),
        ("center_entry_summary", "ethoinsight.scripts.oft.plot_center_entry_summary", "bar"),
        ("activity_intensity", "ethoinsight.scripts.fst.plot_activity_intensity", "line"),
        ("time_progress", "ethoinsight.scripts.oft.plot_time_progress", "line"),
        ("struggle_distribution", "ethoinsight.scripts.fst.plot_struggle_distribution", "timeseries"),
        ("rose", "ethoinsight.scripts.epm.plot_rose", "rose"),
        # token 命中的仍走启发式 fallback，行为不变：
        ("box_open_arm", "ethoinsight.scripts.epm.plot_box_open_arm", "box"),
        ("open_arm_time_ratio_bar", "ethoinsight.scripts.epm.plot_open_arm_time_ratio_bar", "bar"),
        ("trajectory", "ethoinsight.scripts._common.plot_trajectory", "trajectory"),
        ("heatmap", "ethoinsight.scripts._common.plot_heatmap", "heatmap"),
        ("timeseries_plot", "ethoinsight.scripts._common.plot_timeseries", "timeseries"),
        # 未知（无显式映射、无 token 命中）仍 None——不强行猜：
        ("something_unknown", "ethoinsight.scripts.epm.compute_mystery", None),
    ]

    def test_router_derive_covers_explicit_and_token(self) -> None:
        for chart_id, script, expected in self.CASES:
            got = artifacts_router._derive_chart_type(chart_id, script)
            assert got == expected, f"router: {chart_id}/{script} → {got!r}, expected {expected!r}"

    def test_present_file_tool_derive_matches_router(self) -> None:
        """present_file_tool 与网关画廊端点两处 _derive_chart_type 必须同型（SSOT 不漂移）。"""
        present_file_tool_module = __import__(
            "deerflow.tools.builtins.present_file_tool", fromlist=["present_file_tool"]
        )
        for chart_id, script, expected in self.CASES:
            got_router = artifacts_router._derive_chart_type(chart_id, script)
            got_present = present_file_tool_module._derive_chart_type(chart_id, script)
            assert got_router == got_present == expected, (
                f"parity drift at {chart_id}/{script}: "
                f"router={got_router!r} present={got_present!r} expected={expected!r}"
            )


def test_list_chart_artifacts_route_registered_and_authenticated(tmp_path, monkeypatch) -> None:
    """端点经 TestClient 真路径可达（路由已注册 + 鉴权放行）。"""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    (outputs_dir / "plot_box_open_arm.png").write_bytes(b"\x89PNG")
    monkeypatch.setattr(artifacts_router, "resolve_thread_virtual_path", _resolve_factory(outputs_dir, None))

    from _router_auth_helpers import make_authed_test_app

    app = make_authed_test_app()
    app.include_router(artifacts_router.router)
    with TestClient(app) as client:
        # /charts 必须注册在 catch-all /{path:path} 之前；这里测它不被吞。
        resp = client.get("/api/threads/thread-1/artifacts/charts")

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["path"] == "/mnt/user-data/outputs/plot_box_open_arm.png"
