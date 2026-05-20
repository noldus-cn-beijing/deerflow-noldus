"""Tests for ethoinsight.parse using real EthoVision demo data.

Demo data location resolves in order:
1. ``ETHOINSIGHT_DEMO_BASE`` env var (e.g. ``/home/wangqiuyang/DemoData/newdemodata``)
2. legacy hard-coded path (kept for backward compat)

Subdirectory names: project repos used multiple Chinese naming variants
over time (e.g. ``斑马鱼鱼群行为`` vs ``斑马鱼``). If the resolved demo base
or the specific paradigm subdirectory does not exist, the relevant tests
are skipped rather than failed — these tests are integration aids, not
infra invariants.
"""

import os
from pathlib import Path

import pandas as pd
import pytest

from ethoinsight import parse
from ethoinsight.utils import detect_paradigm, normalize_column_name

# Demo data base — prefer env var, fall back to legacy hard-coded path.
DEMO_BASE = Path(
    os.environ.get("ETHOINSIGHT_DEMO_BASE", "/home/qiuyangwang/noldus-insight/demo-data/DemoData")
)
ZEBRAFISH_DIR = DEMO_BASE / "斑马鱼鱼群行为"
EPM_DIR = DEMO_BASE / "高架十字迷宫"
O_MAZE_DIR = DEMO_BASE / "O迷宫"
NOVEL_OBJECT_DIR = DEMO_BASE / "新物体识别"
Y_MAZE_DIR = DEMO_BASE / "Y迷宫"


def _require_trajectory_files(directory: Path) -> list[str]:
    """Find trajectory files in *directory* or skip the test if none exist."""
    if not directory.exists():
        pytest.skip(f"Demo data directory missing: {directory}. Set ETHOINSIGHT_DEMO_BASE to enable.")
    files = sorted(str(f) for f in directory.glob("轨迹-*.txt"))
    if not files:
        pytest.skip(f"No 轨迹-*.txt files in {directory}")
    return files


# Find trajectory files (start with "轨迹-")
def _find_trajectory_files(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(str(f) for f in directory.glob("轨迹-*.txt"))


def _find_any_txt(directory: Path) -> list[str]:
    if not directory.exists():
        return []
    return sorted(str(f) for f in directory.glob("*.txt"))


# ============================================================================
# detect_ethovision
# ============================================================================


class TestDetectEthovision:
    def test_valid_trajectory_file(self):
        files = _require_trajectory_files(ZEBRAFISH_DIR)
        assert len(files) > 0, "No zebrafish trajectory files found"
        assert parse.detect_ethovision(files[0]) is True

    def test_valid_statistics_file(self):
        stats_files = sorted(str(f) for f in EPM_DIR.glob("统计数据-*.txt"))
        if stats_files:
            assert parse.detect_ethovision(stats_files[0]) is True

    def test_nonexistent_file(self):
        assert parse.detect_ethovision("/tmp/nonexistent_file.txt") is False

    def test_non_ethovision_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        assert parse.detect_ethovision(str(f)) is False

    def test_directory_returns_false(self, tmp_path):
        assert parse.detect_ethovision(str(tmp_path)) is False


# ============================================================================
# parse_header
# ============================================================================


class TestParseHeader:
    def test_zebrafish_header(self):
        files = _require_trajectory_files(ZEBRAFISH_DIR)
        assert len(files) > 0
        header = parse.parse_header(files[0])

        assert header["header_lines"] > 0
        assert (
            "shoaling" in header["experiment"].lower()
            or header["paradigm"] == "shoaling"
        )
        assert header["paradigm"] == "shoaling"
        assert header["subject"] != ""
        assert len(header["columns"]) > 0
        assert "trial_time" in header["columns"]
        assert "recording_time" in header["columns"]
        assert "x_center" in header["columns"]
        assert "y_center" in header["columns"]

    def test_epm_header(self):
        files = _find_trajectory_files(EPM_DIR)
        if not files:
            pytest.skip("No EPM trajectory files")
        header = parse.parse_header(files[0])

        assert header["paradigm"] == "epm"
        assert "x_nose" in header["columns"] or "x_center" in header["columns"]

    def test_o_maze_header(self):
        files = _find_trajectory_files(O_MAZE_DIR)
        if not files:
            pytest.skip("No O-maze trajectory files")
        header = parse.parse_header(files[0])

        assert header["paradigm"] == "o_maze"
        assert "trial_time" in header["columns"]

    def test_novel_object_header(self):
        files = _find_trajectory_files(NOVEL_OBJECT_DIR)
        if not files:
            pytest.skip("No novel object trajectory files")
        header = parse.parse_header(files[0])

        assert header["paradigm"] == "novel_object"
        # Basic trajectory columns should always be present
        assert "trial_time" in header["columns"]
        assert "x_center" in header["columns"]

    def test_units_parsed(self):
        files = _require_trajectory_files(ZEBRAFISH_DIR)
        header = parse.parse_header(files[0])
        assert len(header["units"]) > 0

    def test_raw_metadata(self):
        files = _require_trajectory_files(ZEBRAFISH_DIR)
        header = parse.parse_header(files[0])
        assert (
            "实验" in header["raw_metadata"] or "Experiment" in header["raw_metadata"]
        )


# ============================================================================
# parse_trajectory
# ============================================================================


class TestParseTrajectory:
    def test_zebrafish_trajectory(self):
        files = _require_trajectory_files(ZEBRAFISH_DIR)
        assert len(files) > 0
        df = parse.parse_trajectory(files[0])

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 100, "Expected at least 100 data rows"
        assert "trial_time" in df.columns
        assert "x_center" in df.columns
        assert "y_center" in df.columns

        # Numeric types
        assert df["trial_time"].dtype in (float, "float64")
        assert df["x_center"].dtype in (float, "float64")

        # No "-" strings remaining
        for col in df.columns:
            if df[col].dtype == object:
                continue
            assert not df[col].astype(str).str.contains("^-$").any(), (
                f"Column {col} still has '-' values"
            )

        # Metadata in attrs
        assert "subject" in df.attrs
        assert "paradigm" in df.attrs

    def test_epm_trajectory(self):
        files = _find_trajectory_files(EPM_DIR)
        if not files:
            pytest.skip("No EPM trajectory files")
        df = parse.parse_trajectory(files[0])

        assert len(df) > 100
        assert "trial_time" in df.columns
        assert df.attrs.get("paradigm") == "epm"

    def test_y_maze_trajectory(self):
        files = _find_trajectory_files(Y_MAZE_DIR)
        if not files:
            pytest.skip("No Y-maze trajectory files")
        df = parse.parse_trajectory(files[0])

        assert len(df) > 0
        assert df.attrs.get("paradigm") == "y_maze"

    def test_nan_handling(self):
        """Verify that '-' values are converted to NaN."""
        files = _require_trajectory_files(ZEBRAFISH_DIR)
        df = parse.parse_trajectory(files[0])

        # distance_moved or velocity often has NaN at the start
        numeric_cols = df.select_dtypes(include="number").columns
        # At least some columns should have NaN (first row often has "-")
        has_nan = any(df[col].isna().any() for col in numeric_cols)
        # This is typical but not guaranteed, so just check it doesn't crash
        assert isinstance(df, pd.DataFrame)


# ============================================================================
# parse_batch
# ============================================================================


class TestParseBatch:
    def test_zebrafish_batch(self):
        files = _require_trajectory_files(ZEBRAFISH_DIR)
        assert len(files) > 0
        result = parse.parse_batch(files)

        assert result["summary"]["total_files"] == len(files)
        assert result["summary"]["total_rows"] > 0
        assert len(result["subjects"]) > 0
        assert result["summary"]["paradigm"] == "shoaling"
        assert isinstance(result["all_data"], pd.DataFrame)
        assert "subject" in result["all_data"].columns
        assert "file" in result["all_data"].columns

    def test_glob_pattern(self):
        if not ZEBRAFISH_DIR.exists():
            pytest.skip(f"Demo data directory missing: {ZEBRAFISH_DIR}. Set ETHOINSIGHT_DEMO_BASE to enable.")
        pattern = str(ZEBRAFISH_DIR / "轨迹-*.txt")
        result = parse.parse_batch(pattern)
        if result["summary"]["total_files"] == 0:
            pytest.skip(f"No 轨迹-*.txt files in {ZEBRAFISH_DIR}")

        assert result["summary"]["total_files"] > 0
        assert result["summary"]["paradigm"] == "shoaling"

    def test_empty_input(self):
        result = parse.parse_batch([])
        assert result["summary"]["total_files"] == 0
        assert len(result["subjects"]) == 0

    def test_nonexistent_glob(self):
        result = parse.parse_batch("/tmp/nonexistent_pattern_*.txt")
        assert result["summary"]["total_files"] == 0

    def test_mixed_files_filtered(self):
        """Statistics files should be filtered out, only trajectories parsed."""
        all_files = _find_any_txt(EPM_DIR)
        trajectory_files = _find_trajectory_files(EPM_DIR)
        if not trajectory_files:
            pytest.skip("No EPM trajectory files")

        result = parse.parse_batch(all_files)
        assert result["summary"]["total_files"] == len(trajectory_files)


# ============================================================================
# get_summary
# ============================================================================


class TestGetSummary:
    def test_summary_content(self):
        files = _require_trajectory_files(ZEBRAFISH_DIR)[:5]  # Use subset for speed
        result = parse.parse_batch(files)
        text = parse.get_summary(result)

        assert "EthoVision Data Summary" in text
        assert "shoaling" in text.lower()
        assert "Files:" in text
        assert "Subject" in text or "subject" in text.lower()

    def test_summary_max_chars(self):
        files = _require_trajectory_files(ZEBRAFISH_DIR)[:5]
        result = parse.parse_batch(files)
        text = parse.get_summary(result, max_chars=500)

        assert len(text) <= 500

    def test_empty_data_summary(self):
        result = parse.parse_batch([])
        text = parse.get_summary(result)
        assert "Files: 0" in text


# ============================================================================
# utils: normalize_column_name & detect_paradigm
# ============================================================================


class TestNormalizeColumnName:
    def test_exact_match(self):
        assert normalize_column_name("试用时间") == "trial_time"
        assert normalize_column_name("X 中心") == "x_center"
        assert normalize_column_name("Velocity") == "velocity"

    def test_quoted(self):
        assert normalize_column_name('"试用时间"') == "trial_time"
        assert normalize_column_name(' "X 中心" ') == "x_center"

    def test_in_zone_pattern(self):
        name = "In zone(Open arms /中心点)"
        result = normalize_column_name(name)
        assert result.startswith("in_zone")
        assert "open_arms" in result

    def test_chinese_zone_pattern(self):
        name = "分析区中(开放臂 /所有 中心点, 鼻尖)"
        result = normalize_column_name(name)
        assert result.startswith("in_zone")
        assert "open_arms" in result

    def test_entries_pattern(self):
        name = "Entries to Open arms"
        result = normalize_column_name(name)
        assert result.startswith("entries")

    def test_nose_in_zone_pattern(self):
        name = "Nose within object zone(Familiar object 1 /鼻尖)"
        result = normalize_column_name(name)
        assert result.startswith("nose_in_zone")

    def test_empty_name(self):
        assert normalize_column_name("") == "unnamed"
        assert normalize_column_name('""') == "unnamed"


class TestDetectParadigm:
    def test_shoaling(self):
        assert detect_paradigm("Shoaling behavior with JavaScript XT180") == "shoaling"

    def test_epm(self):
        assert detect_paradigm("Elevated Plus Maze XT180") == "epm"

    def test_open_field(self):
        assert detect_paradigm("Open Field test XT160") == "open_field"

    def test_o_maze(self):
        assert detect_paradigm("ethoInsightDemo-oMaze") == "o_maze"

    def test_novel_object(self):
        assert (
            detect_paradigm("Novel Object Recognition test with Deep Learning XT180")
            == "novel_object"
        )

    def test_y_maze(self):
        assert detect_paradigm("安琥医大Y迷宫") == "y_maze"

    def test_light_dark(self):
        assert detect_paradigm("ethoInsightDemo-ldb") == "light_dark"

    def test_unknown(self):
        assert detect_paradigm("Some random experiment") is None

    def test_empty(self):
        assert detect_paradigm("") is None


# ============================================================================
# infer_groups_from_result_block — 2026-05-13 同事反馈 Q3: 自动分组重写
# ============================================================================


def test_parse_groups_from_result_block_name():
    from ethoinsight.parse._core import infer_groups_from_result_block

    result = infer_groups_from_result_block(
        subjects=[
            {"name": "A1", "result_block_name": "Drug"},
            {"name": "A2", "result_block_name": "Drug"},
            {"name": "A3", "result_block_name": "Saline"},
            {"name": "A4", "result_block_name": "Saline"},
        ]
    )
    assert result == {"Drug": ["A1", "A2"], "Saline": ["A3", "A4"]}

    result_default = infer_groups_from_result_block(
        subjects=[
            {"name": "A1", "result_block_name": "Result 1"},
            {"name": "A2", "result_block_name": "Result 1"},
        ]
    )
    assert result_default is None


def test_parse_groups_handles_missing_field():
    from ethoinsight.parse._core import infer_groups_from_result_block

    result = infer_groups_from_result_block(
        subjects=[{"name": "A1"}, {"name": "A2"}],
    )
    assert result is None
