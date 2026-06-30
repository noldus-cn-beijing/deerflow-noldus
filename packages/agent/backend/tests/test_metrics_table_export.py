"""Spec 2026-06-30 C1 (module 1) — Tests for the pure metrics-table exporter.

Strategy: the exporter is stdlib-only with no deerflow imports, so import it
fresh via importlib (bypassing any conftest sys.modules manipulation). The
function is pure (filesystem write only); tests feed hand-built metrics_summary
+ per_subject + subject_groups dicts and assert the on-disk CSV/JSON.

IQR assertions are hand-computed (Tukey median-of-halves) so a drift in the
method fails loudly.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

_EXPORT_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "harness"
    / "deerflow"
    / "subagents"
    / "metrics_table_export.py"
)


def _load_export_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "deerflow.subagents.metrics_table_export_real",
        _EXPORT_FILE,
        submodule_search_locations=[],
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_EXPORT = _load_export_module()

# Keys that must NEVER appear in the clean JSON (handoff viscera).
_VISCERA_KEYS = {
    "gate_signals",
    "handoff",
    "assessment",
    "statistics",
    "confidence",
    "inputs",
    "sealed_by",
}


def _stat(mean, std, n, params=None) -> dict:
    return {"mean": mean, "std": std, "n": n, "parameters_used": params or {}}


# ---------------------------------------------------------------------------
# SSOT: CSV + JSON same numbers
# ---------------------------------------------------------------------------


def test_csv_and_json_values_match_input(tmp_path: Path) -> None:
    """CSV cells + JSON per_subject values equal the input (SSOT, no double-compute drift)."""
    metrics_summary = {
        "Control": {"ratio": _stat(0.30, None, 1)},
        "Treatment": {"ratio": _stat(0.55, None, 1)},
    }
    per_subject = {
        "ctrl": {"ratio": 0.30},
        "drug": {"ratio": 0.55},
    }
    subject_groups = {"ctrl": "Control", "drug": "Treatment"}

    csv_path, json_path = _EXPORT.export_metrics_table(
        metrics_summary=metrics_summary,
        per_subject=per_subject,
        subject_groups=subject_groups,
        outputs_dir=tmp_path,
        paradigm="epm",
    )
    assert csv_path.name == "metrics_table.csv"
    assert json_path.name == "metrics_table.json"

    j = json.loads(json_path.read_text(encoding="utf-8"))
    by_subj = {row["subject"]: row for row in j["per_subject"]}
    assert by_subj["ctrl"]["values"]["ratio"] == 0.30
    assert by_subj["drug"]["values"]["ratio"] == 0.55
    assert by_subj["ctrl"]["group"] == "Control"
    assert by_subj["drug"]["group"] == "Treatment"

    # CSV carries the same values in the right columns.
    csv_text = csv_path.read_text(encoding="utf-8")
    header = csv_text.splitlines()[0]
    assert header == "subject,group,ratio"
    assert "ctrl,Control,0.3" in csv_text
    assert "drug,Treatment,0.55" in csv_text


# ---------------------------------------------------------------------------
# Strip viscera by construction (vacuous guard)
# ---------------------------------------------------------------------------


def test_json_has_no_viscera_keys(tmp_path: Path) -> None:
    """JSON must not contain any handoff viscera key (the export never sees them)."""
    csv_path, json_path = _EXPORT.export_metrics_table(
        metrics_summary={"Control": {"ratio": _stat(0.3, None, 1)}},
        per_subject={"ctrl": {"ratio": 0.3}},
        subject_groups={"ctrl": "Control"},
        outputs_dir=tmp_path,
        paradigm="epm",
    )
    assert csv_path.is_file() and json_path.is_file()
    # Recursively scan all string keys in the JSON for viscera leakage.
    j = json.loads(json_path.read_text(encoding="utf-8"))

    def _walk_keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _walk_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                yield from _walk_keys(item)

    present_viscera = set(_walk_keys(j)) & _VISCERA_KEYS
    assert present_viscera == set(), f"viscera leaked into JSON: {present_viscera}"


# ---------------------------------------------------------------------------
# IQR outlier flags — hand-computed
# ---------------------------------------------------------------------------


def test_outlier_flags_match_handcomputed_iqr(tmp_path: Path) -> None:
    """Fixed dataset; hand-compute Q1/Q3/IQR → assert flags match (Tukey)."""
    # Group "G" values for metric "m": [0, 10, 11, 12, 13, 100].
    # Sorted = [0, 10, 11, 12, 13, 100], n=6 (even). mid=3.
    # lower = first 3 = [0, 10, 11] → Q1 = median = 10.
    # upper = last 3 = [12, 13, 100] → Q3 = median = 13.
    # IQR = 3. lo = 10 - 4.5 = 5.5. hi = 13 + 4.5 = 17.5.
    # → 0 (< 5.5) outlier; 100 (> 17.5) outlier; 10/11/12/13 in-range.
    values = {"s1": 0, "s2": 10, "s3": 11, "s4": 12, "s5": 13, "s6": 100}
    per_subject = {s: {"m": v} for s, v in values.items()}
    subject_groups = {s: "G" for s in values}
    metrics_summary = {"G": {"m": _stat(24.33, None, 6)}}

    _csv, json_path = _EXPORT.export_metrics_table(
        metrics_summary=metrics_summary,
        per_subject=per_subject,
        subject_groups=subject_groups,
        outputs_dir=tmp_path,
    )
    j = json.loads(json_path.read_text(encoding="utf-8"))
    flags = {row["subject"]: row["outlier_flags"]["m"] for row in j["per_subject"]}

    assert flags["s1"] is True   # 0 < 5.5
    assert flags["s6"] is True   # 100 > 17.5
    assert flags["s2"] is False  # 10 in [5.5, 17.5]
    assert flags["s3"] is False
    assert flags["s4"] is False
    assert flags["s5"] is False


def test_outlier_flags_all_false_when_group_n_under_2(tmp_path: Path) -> None:
    """Single-subject group → no outliers detectable (all flags False)."""
    per_subject = {"only": {"m": 999.0}}
    subject_groups = {"only": "Solo"}
    metrics_summary = {"Solo": {"m": _stat(999.0, None, 1)}}

    _csv, json_path = _EXPORT.export_metrics_table(
        metrics_summary=metrics_summary,
        per_subject=per_subject,
        subject_groups=subject_groups,
        outputs_dir=tmp_path,
    )
    j = json.loads(json_path.read_text(encoding="utf-8"))
    assert j["per_subject"][0]["outlier_flags"]["m"] is False


def test_none_values_excluded_from_iqr_and_not_flagged(tmp_path: Path) -> None:
    """None value: excluded from the IQR reference set AND flagged False.

    Non-None set = [1, 2, 3, 4, 5, 6, 100] (subject b=None excluded), n=7 odd, mid=3.
    lower=[:3]=[1,2,3] → Q1=2; upper=[4:]=[5,6,100] → Q3=6; IQR=4.
    lo=2-6=-4; hi=6+6=12 → 100 outlier (>12); 1..6 all in [-4,12].
    """
    per_subject = {
        "a": {"m": 1.0}, "b": {"m": None}, "c": {"m": 2.0}, "d": {"m": 3.0},
        "e": {"m": 4.0}, "f": {"m": 5.0}, "g": {"m": 6.0}, "h": {"m": 100.0},
    }
    subject_groups = {s: "G" for s in per_subject}
    metrics_summary = {"G": {"m": _stat(None, None, 7)}}

    _csv, json_path = _EXPORT.export_metrics_table(
        metrics_summary=metrics_summary,
        per_subject=per_subject,
        subject_groups=subject_groups,
        outputs_dir=tmp_path,
    )
    j = json.loads(json_path.read_text(encoding="utf-8"))
    flags = {row["subject"]: row["outlier_flags"]["m"] for row in j["per_subject"]}
    # None subject is not flagged (excluded from IQR AND not flagged).
    assert flags["b"] is False
    # 100 is an outlier; the rest are not.
    assert flags["h"] is True
    assert all(flags[s] is False for s in ("a", "c", "d", "e", "f", "g"))


# ---------------------------------------------------------------------------
# CSV formatting
# ---------------------------------------------------------------------------


def test_csv_none_value_renders_empty_cell(tmp_path: Path) -> None:
    """None → empty CSV cell (not 'None' / 'null')."""
    per_subject = {"s": {"m1": 0.5, "m2": None}}
    subject_groups = {"s": "G"}
    metrics_summary = {"G": {"m1": _stat(0.5, None, 1), "m2": _stat(None, None, 0)}}

    csv_path, _json = _EXPORT.export_metrics_table(
        metrics_summary=metrics_summary,
        per_subject=per_subject,
        subject_groups=subject_groups,
        outputs_dir=tmp_path,
    )
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "subject,group,m1,m2"
    # m1=0.5, m2 empty.
    row = lines[1]
    assert row.startswith("s,G,0.5,")
    assert row.endswith(",")  # trailing empty cell for None
    assert "None" not in row and "null" not in row


def test_csv_column_order_is_sorted_union(tmp_path: Path) -> None:
    """Metric columns = sorted union across groups + subjects."""
    per_subject = {"s1": {"zebra": 1.0}, "s2": {"alpha": 2.0}}
    subject_groups = {"s1": "G1", "s2": "G2"}
    metrics_summary = {"G1": {"zebra": _stat(1.0, None, 1)}, "G2": {"alpha": _stat(2.0, None, 1)}}

    csv_path, _json = _EXPORT.export_metrics_table(
        metrics_summary=metrics_summary,
        per_subject=per_subject,
        subject_groups=subject_groups,
        outputs_dir=tmp_path,
    )
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header == "subject,group,alpha,zebra"


def test_csv_header_is_subject_group_then_metrics(tmp_path: Path) -> None:
    per_subject = {"s": {"m": 1.0}}
    subject_groups = {"s": "G"}
    metrics_summary = {"G": {"m": _stat(1.0, None, 1)}}
    csv_path, _ = _EXPORT.export_metrics_table(
        metrics_summary=metrics_summary,
        per_subject=per_subject,
        subject_groups=subject_groups,
        outputs_dir=tmp_path,
    )
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[:2] == ["subject", "group"]


# ---------------------------------------------------------------------------
# Determinism + directory creation
# ---------------------------------------------------------------------------


def test_export_is_deterministic(tmp_path: Path) -> None:
    """Two calls into two dirs → byte-identical files."""
    per_subject = {"s": {"m": 1.0}}
    subject_groups = {"s": "G"}
    metrics_summary = {"G": {"m": _stat(1.0, None, 1)}}

    d1 = tmp_path / "a"
    d2 = tmp_path / "b"
    args = dict(
        metrics_summary=metrics_summary,
        per_subject=per_subject,
        subject_groups=subject_groups,
        paradigm="epm",
    )
    _EXPORT.export_metrics_table(outputs_dir=d1, **args)
    _EXPORT.export_metrics_table(outputs_dir=d2, **args)

    assert (d1 / "metrics_table.csv").read_bytes() == (d2 / "metrics_table.csv").read_bytes()
    assert (d1 / "metrics_table.json").read_bytes() == (d2 / "metrics_table.json").read_bytes()


def test_export_creates_outputs_dir_if_missing(tmp_path: Path) -> None:
    """Pass a non-existent outputs_dir → it's created."""
    outputs = tmp_path / "nested" / "outputs"  # does not exist
    per_subject = {"s": {"m": 1.0}}
    subject_groups = {"s": "G"}
    metrics_summary = {"G": {"m": _stat(1.0, None, 1)}}

    csv_path, json_path = _EXPORT.export_metrics_table(
        metrics_summary=metrics_summary,
        per_subject=per_subject,
        subject_groups=subject_groups,
        outputs_dir=outputs,
    )
    assert csv_path.is_file() and json_path.is_file()


# ---------------------------------------------------------------------------
# JSON groups: mean/std/n read straight from metrics_summary (not recomputed)
# ---------------------------------------------------------------------------


def test_json_groups_match_metrics_summary_mean_std_n(tmp_path: Path) -> None:
    """groups[].metrics mean/std/n come straight from metrics_summary (no recompute)."""
    metrics_summary = {
        "G": {"m": _stat(0.42, 0.08, 24, params={"zones": ["open"]})},
    }
    per_subject = {"s": {"m": 0.42}}
    subject_groups = {"s": "G"}

    _csv, json_path = _EXPORT.export_metrics_table(
        metrics_summary=metrics_summary,
        per_subject=per_subject,
        subject_groups=subject_groups,
        outputs_dir=tmp_path,
        paradigm="epm",
    )
    j = json.loads(json_path.read_text(encoding="utf-8"))
    assert j["paradigm"] == "epm"
    grp = j["groups"][0]
    assert grp["group"] == "G"
    assert grp["n"] == 24
    assert grp["metrics"]["m"]["mean"] == 0.42
    assert grp["metrics"]["m"]["std"] == 0.08
    # parameters_used is intentionally NOT leaked into the clean JSON.
    assert "parameters_used" not in grp["metrics"]["m"]
