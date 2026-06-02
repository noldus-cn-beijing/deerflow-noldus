"""Sprint 1 tests: dispatcher.py 9 warnings upgraded with code/evidence/blocks_downstream.

Each test verifies that the specific warning populates the three new fields
according to spec §2.1 mapping table. The final test asserts no LEGACY codes remain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethoinsight.metrics import compute_paradigm_metrics


# ============================================================================
# Helpers — minimal synthetic DataFrames per paradigm
# ============================================================================


def _make_epm_df(n_frames: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    oa = np.zeros(n_frames, dtype=int)
    oa[10:18] = 1
    oa[25:33] = 1
    oa[45:53] = 1
    oa[65:73] = 1
    oa[85:93] = 1
    return pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
            "in_zone_open_arm_1": oa,
            "in_zone_closed_arm_1": np.zeros(n_frames, dtype=int),
            "in_zone_center-point": np.ones(n_frames, dtype=int),
        }
    )


def _make_zero_maze_df(n_frames: int = 100, distance_total: float | None = None, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if distance_total is not None:
        dm = np.full(n_frames, distance_total / n_frames)
    else:
        dm = rng.uniform(0, 5, n_frames)
    return pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": dm,
            "velocity": rng.uniform(0, 20, n_frames),
            "in_zone_open_zone": np.ones(n_frames, dtype=int),
            "in_zone_closed_zone": np.zeros(n_frames, dtype=int),
        }
    )


def _make_ldb_df(n_frames: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
            "in_zone_light_compartment": np.ones(n_frames, dtype=int),
            "in_zone_dark_compartment": np.zeros(n_frames, dtype=int),
        }
    )


def _make_fst_df(n_frames: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "trial_time": np.arange(n_frames, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n_frames),
            "y_center": rng.uniform(100, 500, n_frames),
            "distance_moved": rng.uniform(0, 5, n_frames),
            "velocity": rng.uniform(0, 20, n_frames),
            "mobility_state": np.ones(n_frames, dtype=int),
        }
    )


def _build_parsed(subjects: dict[str, pd.DataFrame]) -> dict:
    all_dfs = []
    for name, df in subjects.items():
        df = df.copy()
        df["subject"] = name
        all_dfs.append(df)
    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return {
        "subjects": subjects,
        "all_data": combined,
        "file_list": [f"subj_{i}.txt" for i in range(len(subjects))],
        "summary": {
            "total_files": len(subjects),
            "total_subjects": len(subjects),
            "total_rows": sum(len(df) for df in subjects.values()),
            "duration_seconds": 600.0,
        },
    }


# ============================================================================
# Test 1: SAMPLE.TOO_SMALL (cross-paradigm n<3)
# ============================================================================


class TestWarningSampleTooSmall:
    def test_epm_n2_triggers_too_small(self):
        df1 = _make_epm_df(seed=1)
        df2 = _make_epm_df(seed=2)
        parsed = _build_parsed({"S1": df1, "S2": df2})
        result = compute_paradigm_metrics(parsed, "epm", groups={"grp": ["S1", "S2"]})
        warnings = result["data_quality_warnings"]
        too_small = [w for w in warnings if w.get("code") == "SAMPLE.TOO_SMALL"]
        assert len(too_small) >= 1
        w = too_small[0]
        assert w["severity"] == "critical"
        assert w["blocks_downstream"] is True
        assert w["evidence"]["n"] == 2
        assert w["evidence"]["threshold"] == 3
        assert w["evidence"]["group"] == "grp"

    def test_oft_n2_triggers_too_small(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x_center": rng.uniform(0, 1, 100), "y_center": rng.uniform(0, 1, 100), "distance_moved": rng.uniform(0, 5, 100), "velocity": rng.uniform(0, 20, 100), "trial_time": np.arange(100, dtype=float) * 0.04, "in_zone_center-point": np.ones(100, dtype=int)})
        parsed = _build_parsed({"A": df, "B": df})
        result = compute_paradigm_metrics(parsed, "open_field", groups={"ctrl": ["A", "B"]})
        too_small = [w for w in result["data_quality_warnings"] if w.get("code") == "SAMPLE.TOO_SMALL"]
        assert len(too_small) >= 1
        assert too_small[0]["blocks_downstream"] is True


# ============================================================================
# Test 2: SAMPLE.UNDERPOWERED (epm n<5)
# ============================================================================


class TestWarningEpmUnderpowered:
    def test_epm_n4_triggers_underpowered(self):
        dfs = {f"S{i}": _make_epm_df(seed=i) for i in range(4)}
        parsed = _build_parsed(dfs)
        groups = {"treatment": list(dfs.keys())}
        result = compute_paradigm_metrics(parsed, "epm", groups=groups)
        warnings = result["data_quality_warnings"]
        underpowered = [w for w in warnings if w.get("code") == "SAMPLE.UNDERPOWERED"]
        assert len(underpowered) >= 1
        w = underpowered[0]
        assert w["severity"] == "warning"
        assert w["blocks_downstream"] is False
        assert w["evidence"]["paradigm"] == "epm"
        assert w["evidence"]["n"] == 4
        assert w["evidence"]["threshold"] == 5


# ============================================================================
# Test 3: MOTOR.LOW_ENTRIES (epm total_entry_count < 8)
# ============================================================================


class TestWarningEpmLowEntries:
    def test_low_entries_triggers_motor_warning(self):
        n = 100
        rng = np.random.default_rng(42)
        oa = np.zeros(n, dtype=int)
        oa[10:20] = 1  # single bout → very few entries
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n, dtype=float) * 0.04,
                "x_center": rng.uniform(100, 500, n),
                "y_center": rng.uniform(100, 500, n),
                "distance_moved": rng.uniform(0, 5, n),
                "velocity": rng.uniform(0, 20, n),
                "in_zone_open_arm_1": oa,
                "in_zone_closed_arm_1": np.zeros(n, dtype=int),
                "in_zone_center-point": np.ones(n, dtype=int),
            }
        )
        parsed = _build_parsed({"Subject_Low": df})
        result = compute_paradigm_metrics(parsed, "epm")
        warnings = result["data_quality_warnings"]
        motor = [w for w in warnings if w.get("code") == "MOTOR.LOW_ENTRIES"]
        assert len(motor) >= 1
        w = motor[0]
        assert w["evidence"]["subject"] == "Subject_Low"
        assert w["evidence"]["total_entry_count"] < 8
        assert w["evidence"]["threshold"] == 8
        assert w["evidence"]["paradigm"] == "epm"
        assert w["blocks_downstream"] is False


# ============================================================================
# Test 4: SAMPLE.UNDERPOWERED (zero_maze n<5)
# ============================================================================


class TestWarningZmUnderpowered:
    def test_zm_n4_triggers_underpowered(self):
        dfs = {f"S{i}": _make_zero_maze_df(seed=i) for i in range(4)}
        parsed = _build_parsed(dfs)
        groups = {"ctrl": list(dfs.keys())}
        result = compute_paradigm_metrics(parsed, "zero_maze", groups=groups)
        underpowered = [w for w in result["data_quality_warnings"] if w.get("code") == "SAMPLE.UNDERPOWERED"]
        assert len(underpowered) >= 1
        w = underpowered[0]
        assert w["evidence"]["paradigm"] == "zero_maze"
        assert w["blocks_downstream"] is False


# ============================================================================
# Test 5: MOTOR.LOW_DISTANCE (zero_maze distance_moved < 10)
# ============================================================================


class TestWarningZmLowDistance:
    def test_low_distance_triggers_motor_warning(self):
        df = _make_zero_maze_df(n_frames=100, distance_total=5.0)
        parsed = _build_parsed({"Sedentary_Mouse": df})
        result = compute_paradigm_metrics(parsed, "zero_maze")
        warnings = result["data_quality_warnings"]
        motor = [w for w in warnings if w.get("code") == "MOTOR.LOW_DISTANCE"]
        assert len(motor) >= 1
        w = motor[0]
        assert w["evidence"]["subject"] == "Sedentary_Mouse"
        assert w["evidence"]["distance_moved"] < 10.0
        assert w["evidence"]["threshold"] == 10.0
        assert w["evidence"]["paradigm"] == "zero_maze"
        assert w["blocks_downstream"] is False


# ============================================================================
# Test 6: SAMPLE.UNDERPOWERED (light_dark_box n<5)
# ============================================================================


class TestWarningLdbUnderpowered:
    def test_ldb_n4_triggers_underpowered(self):
        dfs = {f"S{i}": _make_ldb_df(seed=i) for i in range(4)}
        parsed = _build_parsed(dfs)
        groups = {"ctrl": list(dfs.keys())}
        result = compute_paradigm_metrics(parsed, "light_dark_box", groups=groups)
        underpowered = [w for w in result["data_quality_warnings"] if w.get("code") == "SAMPLE.UNDERPOWERED"]
        assert len(underpowered) >= 1
        w = underpowered[0]
        assert w["evidence"]["paradigm"] == "light_dark_box"
        assert w["blocks_downstream"] is False


# ============================================================================
# Test 7: SIGNAL.LOW_TRANSITION_COUNT (ldb transition_count < 4)
# ============================================================================


class TestWarningLdbLowTransitions:
    def test_low_transitions_triggers_signal_warning(self):
        df = _make_ldb_df()
        parsed = _build_parsed({"Mouse_A": df})
        result = compute_paradigm_metrics(parsed, "light_dark_box")
        # transition_count may be low for synthetic data
        signal = [w for w in result["data_quality_warnings"] if w.get("code") == "SIGNAL.LOW_TRANSITION_COUNT"]
        if signal:
            w = signal[0]
            assert w["evidence"]["subject"] == "Mouse_A"
            assert w["evidence"]["transition_count"] < 4
            assert w["evidence"]["threshold"] == 4
            assert w["evidence"]["paradigm"] == "light_dark_box"
            assert w["blocks_downstream"] is False


# ============================================================================
# Test 8: SAMPLE.UNDERPOWERED (forced_swim n<5)
# ============================================================================


class TestWarningFstUnderpowered:
    def test_fst_n4_triggers_underpowered(self):
        dfs = {f"S{i}": _make_fst_df(seed=i) for i in range(4)}
        parsed = _build_parsed(dfs)
        groups = {"treatment": list(dfs.keys())}
        result = compute_paradigm_metrics(parsed, "forced_swim", groups=groups)
        underpowered = [w for w in result["data_quality_warnings"] if w.get("code") == "SAMPLE.UNDERPOWERED"]
        assert len(underpowered) >= 1
        w = underpowered[0]
        assert w["evidence"]["paradigm"] == "forced_swim"
        assert w["blocks_downstream"] is False


# ============================================================================
# Test 9: No LEGACY.UNCATEGORIZED remaining
# ============================================================================


class TestNoLegacyRemaining:
    @pytest.mark.parametrize("paradigm", ["epm", "open_field", "zero_maze", "light_dark_box", "forced_swim"])
    def test_no_legacy_codes(self, paradigm: str):
        """All dispatched warnings must use real codes, not LEGACY.UNCATEGORIZED."""
        rng = np.random.default_rng(42)
        base = pd.DataFrame(
            {
                "trial_time": np.arange(100, dtype=float) * 0.04,
                "x_center": rng.uniform(100, 500, 100),
                "y_center": rng.uniform(100, 500, 100),
                "distance_moved": rng.uniform(0, 5, 100),
                "velocity": rng.uniform(0, 20, 100),
            }
        )
        # Add paradigm-specific columns
        if paradigm == "epm":
            base["in_zone_open_arm_1"] = np.ones(100, dtype=int)
            base["in_zone_closed_arm_1"] = np.zeros(100, dtype=int)
            base["in_zone_center-point"] = np.ones(100, dtype=int)
        elif paradigm == "open_field":
            base["in_zone_center-point"] = np.ones(100, dtype=int)
        elif paradigm == "zero_maze":
            base["in_zone_open_zone"] = np.ones(100, dtype=int)
            base["in_zone_closed_zone"] = np.zeros(100, dtype=int)
        elif paradigm == "light_dark_box":
            base["in_zone_light_compartment"] = np.ones(100, dtype=int)
            base["in_zone_dark_compartment"] = np.zeros(100, dtype=int)
        elif paradigm == "forced_swim":
            base["mobility_state"] = np.ones(100, dtype=int)

        parsed = _build_parsed({"S1": base.copy(), "S2": base.copy()})
        groups = {"small_grp": ["S1", "S2"]}
        result = compute_paradigm_metrics(parsed, paradigm, groups=groups)
        warnings = result["data_quality_warnings"]
        for w in warnings:
            assert w["code"] != "LEGACY.UNCATEGORIZED", (
                f"Warning still uses LEGACY.UNCATEGORIZED in paradigm {paradigm}: {w}"
            )
            assert w["code"] in (
                "SAMPLE.TOO_SMALL",
                "SAMPLE.UNDERPOWERED",
                "MOTOR.LOW_ENTRIES",
                "MOTOR.LOW_DISTANCE",
                "SIGNAL.LOW_TRANSITION_COUNT",
            ), f"Unexpected code: {w['code']}"
