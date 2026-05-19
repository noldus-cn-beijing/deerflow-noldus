"""W5: _evaluate_when 扩展 total_subjects 变量。"""
from __future__ import annotations

from ethoinsight.catalog.resolve import _evaluate_when


def test_total_subjects_passes_when_threshold_met():
    assert _evaluate_when("total_subjects >= 1", n_per_group=None, n_groups=None, total_subjects=1) is True
    assert _evaluate_when("total_subjects >= 1", n_per_group=None, n_groups=None, total_subjects=10) is True


def test_total_subjects_fails_when_threshold_not_met():
    assert _evaluate_when("total_subjects >= 3", n_per_group=None, n_groups=None, total_subjects=1) is False


def test_total_subjects_none_evaluates_false():
    assert _evaluate_when("total_subjects >= 1", n_per_group=None, n_groups=None, total_subjects=None) is False


def test_compound_with_total_subjects():
    assert _evaluate_when("n_per_group >= 1 and total_subjects >= 1",
                          n_per_group=1, n_groups=None, total_subjects=1) is True
    assert _evaluate_when("n_per_group >= 1 and total_subjects >= 3",
                          n_per_group=1, n_groups=None, total_subjects=1) is False


def test_backward_compat_n_per_group_still_works():
    assert _evaluate_when("n_per_group >= 3", n_per_group=3, n_groups=None, total_subjects=None) is True
    assert _evaluate_when("n_per_group >= 3", n_per_group=2, n_groups=None, total_subjects=None) is False
    assert _evaluate_when("n_groups >= 2", n_per_group=None, n_groups=2, total_subjects=None) is True
    assert _evaluate_when("always", n_per_group=None, n_groups=None, total_subjects=None) is True


def test_n_per_group_call_without_total_subjects_kwarg_still_works():
    """既有 resolve() 调用未传 total_subjects 时不应崩。"""
    assert _evaluate_when("n_per_group >= 1", n_per_group=1, n_groups=None) is True
