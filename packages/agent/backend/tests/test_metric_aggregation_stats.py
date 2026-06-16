"""Spec 2026-06-16 缺陷 2 — metric_aggregation 聚合累加器正确性。

red 锚点（修复前）：``aggregate_metrics_to_handoff`` 的累加器是半成品——只 n+=1，
mean 恒等于该组首个被处理 subject 的值、std 恒 null。data-analyst 读到与 per_subject
真实 mean 矛盾的 metrics_summary，陷入"验证/重算/怀疑"螺旋、漏调 seal → terminated
without emitting handoff。

修复：循环内只收集 values 到临时 ``_values`` 列表，循环结束后用 ``_compute_stat`` 统一
算 mean/std/n（忽略 None；std 样本标准差 n−1；n<2 时 std=None）。

importlib 加载 worktree 源：worktree 共享主仓 venv，editable deerflow 指主仓，直接
``from deerflow.subagents.metric_aggregation import aggregate_metrics_to_handoff`` 测主仓
代码（worktree 改动不生效=假绿）。守 feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

# ---------------------------------------------------------------------------
# Load the REAL metric_aggregation.py source from this worktree
# ---------------------------------------------------------------------------
_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages" / "harness" / "deerflow" / "subagents" / "metric_aggregation.py"
)


def _load() -> ModuleType:
    assert _FILE.exists(), f"metric_aggregation.py not found at {_FILE}"
    spec = importlib.util.spec_from_file_location("metric_agg_worktree_ioboundary", _FILE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_MOD = _load()
aggregate_metrics_to_handoff = _MOD.aggregate_metrics_to_handoff
_compute_stat = _MOD._compute_stat


# ---------------------------------------------------------------------------
# Helpers: build plan + m_*.json artifacts (real host paths, no /mnt)
# ---------------------------------------------------------------------------


def _m_file(ws: Path, mid: str, metric: str, value, subject_index: int, params=None) -> None:
    """Write one m_<id>.json compute artifact to workspace."""
    payload = {"metric": metric, "value": value, "parameters_used": params or {}}
    (ws / f"m_{mid}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _plan(ws: Path, metric_ids: list[str], metric_name: str, n: int, *, groups=None) -> dict:
    """Build plan_metrics.json with n subjects of one metric, all in one group."""
    metrics = []
    for i in range(n):
        mid = metric_ids[i]
        metrics.append({
            "id": mid,
            "output": str(ws / f"m_{mid}.json"),
            "subject_index": i,
            "script": "fake",
        })
    raw_files = [f"/mnt/user-data/uploads/Trial {i + 1}.xlsx" for i in range(n)]
    return {
        "schema_version": "1.1",
        "paradigm": "epm",
        "ev19_template": "epm",
        "inputs": {"raw_files": raw_files, "groups_file": str(ws / "groups.json")},
        "metrics": metrics,
        "statistics": {},
        "skipped": [],
        "notes": [],
    }


def _write_groups(ws: Path, groups: dict[str, str]) -> None:
    """Write SSOT groups.json {subject_file: group_name}."""
    (ws / "groups.json").write_text(json.dumps(groups, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# _compute_stat pure function (direct unit test — red anchor: old impl had no such fn)
# ---------------------------------------------------------------------------


class TestComputeStat:
    def test_basic_mean_std(self):
        # [2,4,6] → mean 4.0, 样本标准差 2.0 (ddof=1), n 3
        s = _compute_stat([2.0, 4.0, 6.0])
        assert s["mean"] == pytest.approx(4.0)
        assert s["std"] == pytest.approx(2.0)
        assert s["n"] == 3

    def test_single_value_std_none(self):
        s = _compute_stat([5.0])
        assert s["mean"] == pytest.approx(5.0)
        assert s["std"] is None
        assert s["n"] == 1

    def test_skips_none(self):
        # None 不计入 mean/std/n（compute 脚本报"不适用"）
        s = _compute_stat([1.0, None, 3.0, None])
        assert s["mean"] == pytest.approx(2.0)
        assert s["std"] == pytest.approx(math_stdev([1.0, 3.0]))
        assert s["n"] == 2

    def test_all_none(self):
        s = _compute_stat([None, None])
        assert s["mean"] is None
        assert s["std"] is None
        assert s["n"] == 0


def math_stdev(vals):
    import statistics

    return statistics.stdev(vals)


# ---------------------------------------------------------------------------
# aggregate_metrics_to_handoff — full path (red anchor: old mean==first value, std==None)
# ---------------------------------------------------------------------------


class TestAggregationStats:
    def test_mean_std_n_correct(self, tmp_path):
        """核心 red→green：3 个同组同 metric subject [2,4,6] → mean 4, std 2, n 3。

        修复前：mean==2.0（首个值）、std is None、n==3（n 对、mean/std 错）。
        """
        ws = tmp_path
        ids = ["open_arm_time_ratio_s0", "open_arm_time_ratio_s1", "open_arm_time_ratio_s2"]
        for i, v in enumerate([2.0, 4.0, 6.0]):
            _m_file(ws, ids[i], "open_arm_time_ratio", v, i)
        plan = _plan(ws, ids, "open_arm_time_ratio", 3)
        _write_groups(ws, {
            "/mnt/user-data/uploads/Trial 1.xlsx": "control",
            "/mnt/user-data/uploads/Trial 2.xlsx": "control",
            "/mnt/user-data/uploads/Trial 3.xlsx": "control",
        })
        (ws / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")

        agg = aggregate_metrics_to_handoff(plan, ws, run_validation=False)
        stat = agg["metrics_summary"]["control"]["open_arm_time_ratio"]
        assert stat["mean"] == pytest.approx(4.0)
        assert stat["std"] == pytest.approx(2.0)
        assert stat["n"] == 3
        # 临时 _values 字段必须清除（不泄漏进 handoff）
        assert "_values" not in stat

    def test_multi_group_no_pollution(self, tmp_path):
        """control [1,2,3]、treatment [4,5,6] 各自 mean/std/n 正确、互不污染。"""
        ws = tmp_path
        ids = [f"m_s{i}" for i in range(6)]
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        groups = {}
        for i in range(6):
            _m_file(ws, ids[i], "ratio", vals[i], i)
            g = "control" if i < 3 else "treatment"
            groups[f"/mnt/user-data/uploads/Trial {i + 1}.xlsx"] = g
        plan = _plan(ws, ids, "ratio", 6)
        _write_groups(ws, groups)
        (ws / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")

        agg = aggregate_metrics_to_handoff(plan, ws, run_validation=False)
        c = agg["metrics_summary"]["control"]["ratio"]
        t = agg["metrics_summary"]["treatment"]["ratio"]
        assert c["mean"] == pytest.approx(2.0)
        assert c["std"] == pytest.approx(1.0)
        assert c["n"] == 3
        assert t["mean"] == pytest.approx(5.0)
        assert t["std"] == pytest.approx(1.0)
        assert t["n"] == 3

    def test_skips_none_values(self, tmp_path):
        """某 subject metric 值 None → 不计入 mean/std/n（与 MetricStat.applicable 对齐）。"""
        ws = tmp_path
        ids = ["m_s0", "m_s1", "m_s2"]
        _m_file(ws, ids[0], "ratio", 1.0, 0)
        _m_file(ws, ids[1], "ratio", None, 1)
        _m_file(ws, ids[2], "ratio", 3.0, 2)
        plan = _plan(ws, ids, "ratio", 3)
        _write_groups(ws, {
            "/mnt/user-data/uploads/Trial 1.xlsx": "control",
            "/mnt/user-data/uploads/Trial 2.xlsx": "control",
            "/mnt/user-data/uploads/Trial 3.xlsx": "control",
        })
        (ws / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")

        agg = aggregate_metrics_to_handoff(plan, ws, run_validation=False)
        stat = agg["metrics_summary"]["control"]["ratio"]
        assert stat["mean"] == pytest.approx(2.0)
        assert stat["std"] == pytest.approx(math_stdev([1.0, 3.0]))
        assert stat["n"] == 2

    def test_single_subject_std_none(self, tmp_path):
        """组内仅 1 个值 → std None、mean==该值、n==1。"""
        ws = tmp_path
        ids = ["m_s0"]
        _m_file(ws, ids[0], "ratio", 0.42, 0)
        plan = _plan(ws, ids, "ratio", 1)
        _write_groups(ws, {"/mnt/user-data/uploads/Trial 1.xlsx": "control"})
        (ws / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")

        agg = aggregate_metrics_to_handoff(plan, ws, run_validation=False)
        stat = agg["metrics_summary"]["control"]["ratio"]
        assert stat["mean"] == pytest.approx(0.42)
        assert stat["std"] is None
        assert stat["n"] == 1

    def test_per_subject_unchanged(self, tmp_path):
        """聚合不破坏明细：per_subject 仍逐 subject 记原值（data-analyst 仍能读）。"""
        ws = tmp_path
        ids = ["m_s0", "m_s1", "m_s2"]
        for i, v in enumerate([2.0, 4.0, 6.0]):
            _m_file(ws, ids[i], "ratio", v, i)
        plan = _plan(ws, ids, "ratio", 3)
        _write_groups(ws, {
            "/mnt/user-data/uploads/Trial 1.xlsx": "control",
            "/mnt/user-data/uploads/Trial 2.xlsx": "control",
            "/mnt/user-data/uploads/Trial 3.xlsx": "control",
        })
        (ws / "plan_metrics.json").write_text(json.dumps(plan), encoding="utf-8")

        agg = aggregate_metrics_to_handoff(plan, ws, run_validation=False)
        # subject_name = Path(raw_file).stem = "Trial 1" 等
        assert agg["per_subject"]["Trial 1"]["ratio"] == 2.0
        assert agg["per_subject"]["Trial 2"]["ratio"] == 4.0
        assert agg["per_subject"]["Trial 3"]["ratio"] == 6.0
