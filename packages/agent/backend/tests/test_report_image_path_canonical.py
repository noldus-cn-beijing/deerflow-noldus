"""report 图片路径 SSOT 统一 —— 规范形态验收测试。

spec: docs/superpowers/specs/2026-06-18-report-image-path-ssot-spec.md

锁定三件事：
1. 规范形态 = 带前导斜杠的虚拟绝对路径 ``/mnt/user-data/outputs/<name>.png``。
2. seal 的两个产出点（placeholder 解析 + path normalize）对同一文件名产出
   **字节相同**的路径 —— SSOT 一致性（同一约定绝不多处定义）。
3. ``resolve_virtual_path`` 接受规范形态对应的 API path 段（``mnt/user-data/...``），
   而**放宽**校验（接受丢前缀的 ``outputs/...``）不是修复手段 —— 必须靠前端送对前缀。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from deerflow.config.paths import Paths
from deerflow.tools.builtins.seal_handoff_tools import (
    _to_canonical_artifact_path,
    _load_chart_files_map,
    _normalize_report_image_paths,
    _resolve_report_image_placeholders,
)


# ---------------------------------------------------------------------------
# §4.1 规范形态定义 + helper 行为
# ---------------------------------------------------------------------------


class TestCanonicalArtifactPath:
    """_to_canonical_artifact_path 是 report 图片路径规范形态的 SSOT 唯一定义点。"""

    def test_canonical_form_has_leading_slash(self):
        # 规范形态：带前导斜杠的虚拟绝对路径
        assert _to_canonical_artifact_path("plot.png") == "/mnt/user-data/outputs/plot.png"

    def test_idempotent_on_canonical_input(self):
        # 对已是规范形态的输入幂等（不会二次加前缀）
        assert _to_canonical_artifact_path("/mnt/user-data/outputs/plot.png") == "/mnt/user-data/outputs/plot.png"

    def test_strips_extra_leading_slashes(self):
        # chart_files 可能含多前导斜杠的脏数据；helper 归一化
        assert _to_canonical_artifact_path("///mnt/user-data/outputs/plot.png") == "/mnt/user-data/outputs/plot.png"

    def test_handles_subdirs_in_name(self):
        # 文件名内含子路径段（理论上的 outputs 子目录）也只规范前缀
        assert _to_canonical_artifact_path("sub/plot.png") == "/mnt/user-data/outputs/sub/plot.png"


# ---------------------------------------------------------------------------
# §4.1 SSOT 一致性：两个产出点对同一文件名字节相同
# ---------------------------------------------------------------------------


class TestSealProducesCanonicalForm:
    """seal 两个图片路径产出点必须产出同一规范形态（SSOT 一致性）。"""

    def _make_chart_handoff(self, workspace: Path, chart_files: list[str]) -> None:
        data = {
            "status": "completed",
            "paradigm": "epm",
            "summary": "charts done",
            "chart_files": chart_files,
        }
        (workspace / "handoff_chart_maker.json").write_text(json.dumps(data), encoding="utf-8")

    def test_load_chart_files_map_matches_canonical(self, tmp_path):
        """_load_chart_files_map 的值 == _to_canonical_artifact_path(basename)。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        self._make_chart_handoff(ws, ["/mnt/user-data/outputs/plot_box_open_arm.png"])

        chart_map = _load_chart_files_map(ws)

        assert chart_map == {"plot_box_open_arm.png": _to_canonical_artifact_path("plot_box_open_arm.png")}

    def test_placeholder_resolution_matches_canonical(self, tmp_path):
        """placeholder 解析产出的路径 == _to_canonical_artifact_path(basename)。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        out = tmp_path / "outputs"
        out.mkdir()
        self._make_chart_handoff(ws, ["/mnt/user-data/outputs/plot_box_open_arm.png"])
        report = out / "report.md"
        report.write_text("![x]({{img:plot_box_open_arm.png}})", encoding="utf-8")

        _resolve_report_image_placeholders(report, ws)

        expected = "![x](" + _to_canonical_artifact_path("plot_box_open_arm.png") + ")"
        assert report.read_text(encoding="utf-8") == expected

    def test_two_production_points_byte_identical(self, tmp_path):
        """SSOT 铁律：load_chart_files_map 与 normalize 对同一文件名产出字节相同。"""
        ws = tmp_path / "workspace"
        ws.mkdir()
        basename = "plot_canonical_check.png"
        self._make_chart_handoff(ws, [f"/mnt/user-data/outputs/{basename}"])

        # 产出点 A: placeholder 解析（经 load_chart_files_map）
        out_a = tmp_path / "out_a"
        out_a.mkdir()
        report_a = out_a / "report.md"
        report_a.write_text("![x]({{img:" + basename + "}})", encoding="utf-8")
        _resolve_report_image_placeholders(report_a, ws)
        produced_a = report_a.read_text(encoding="utf-8")

        # 产出点 B: normalize（裸路径归一化）
        with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False, encoding="utf-8") as f:
            f.write("![x](outputs/" + basename + ")")
            report_b = Path(f.name)
        _normalize_report_image_paths(report_b)
        produced_b = report_b.read_text(encoding="utf-8")
        report_b.unlink(missing_ok=True)

        # 两处产出的图片路径段必须字节一致
        expected = "![x](/mnt/user-data/outputs/" + basename + ")"
        assert produced_a == produced_b == expected


# ---------------------------------------------------------------------------
# §4.2 resolve_virtual_path 接受规范形态对应的 API path
# ---------------------------------------------------------------------------


class TestResolveAcceptsCanonicalArtifactPath:
    """后端 resolve_virtual_path 接受规范形态对应的 path 段，但拒绝丢前缀的兜底形态。

    证明修复手段是「前端送对前缀」而非「后端放宽校验」。
    """

    @pytest.mark.parametrize("user_id", [None, "real-uid"])
    def test_accepts_canonical_api_path(self, tmp_path, user_id):
        """前端送 mnt/user-data/outputs/X.png（规范 src 去掉前导斜杠）→ resolve 成功。"""
        paths = Paths(base_dir=tmp_path)
        outputs_dir = paths.sandbox_user_data_dir("thread-1", user_id=user_id) / "outputs"
        outputs_dir.mkdir(parents=True)
        (outputs_dir / "plot.png").write_bytes(b"png-bytes")

        resolved = paths.resolve_virtual_path(
            "thread-1",
            "mnt/user-data/outputs/plot.png",
            user_id=user_id,
        )
        assert resolved.read_bytes() == b"png-bytes"

    @pytest.mark.parametrize("user_id", [None, "real-uid"])
    def test_rejects_prefix_dropped_form(self, tmp_path, user_id):
        """丢掉 mnt/user-data 前缀的旧兜底形态 outputs/X.png 仍抛 ValueError。

        锁定「不靠放宽 resolve_virtual_path 校验来修」——正解是前端送对前缀。
        """
        paths = Paths(base_dir=tmp_path)
        outputs_dir = paths.sandbox_user_data_dir("thread-1", user_id=user_id) / "outputs"
        outputs_dir.mkdir(parents=True)
        (outputs_dir / "plot.png").write_bytes(b"png-bytes")

        with pytest.raises(ValueError):
            paths.resolve_virtual_path(
                "thread-1",
                "outputs/plot.png",
                user_id=user_id,
            )


# ---------------------------------------------------------------------------
# §4.4 dev/deploy user_id 一致性：seal 写路径与 artifact 读路径同源
# ---------------------------------------------------------------------------


class TestArtifactReadPathMatchesSealWritePath:
    """seal 写 outputs 与 artifact 读 outputs 必须用同一 user_id 来源，否则
    dev（无 user）与 deploy（有 user）resolve 到不同物理目录。"""

    @pytest.mark.parametrize("user_id", [None, "real-uid"])
    def test_read_hits_seal_written_file(self, tmp_path, user_id):
        # seal 把图片写到 outputs（与 workspace 同一 user-data 根）
        paths = Paths(base_dir=tmp_path)
        outputs_dir = paths.sandbox_user_data_dir("thread-1", user_id=user_id) / "outputs"
        outputs_dir.mkdir(parents=True)
        (outputs_dir / "plot.png").write_bytes(b"sealed-by-seal")

        # artifact 读路径用同一 user_id
        resolved = paths.resolve_virtual_path(
            "thread-1",
            "mnt/user-data/outputs/plot.png",
            user_id=user_id,
        )
        assert resolved.read_bytes() == b"sealed-by-seal"
        # 物理路径就是 seal 写的那一个
        assert resolved == (outputs_dir / "plot.png").resolve()
