"""2026-05-20: GateSignals.statistical_validity enum 扩展 (handoff #3).

之前枚举只有 "ok" / "warning" / "failed",单样本场景下 code-executor 正确跳过
统计后写 "ok",语义不准确(没做 != 统计 OK)。扩 "skipped" 值表达"未运行统计检验"。
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from deerflow.subagents.handoff_schemas import GateSignals


class TestStatisticalValiditySkipped:
    def test_accepts_skipped(self):
        """skipped 是合法值。"""
        signals = GateSignals(statistical_validity="skipped")
        assert signals.statistical_validity == "skipped"

    def test_accepts_existing_values(self):
        """ok / warning / failed 仍合法。"""
        for v in ("ok", "warning", "failed"):
            signals = GateSignals(statistical_validity=v)
            assert signals.statistical_validity == v

    def test_rejects_unknown_value(self):
        """非枚举值仍被拒绝。"""
        with pytest.raises(ValidationError):
            GateSignals(statistical_validity="bogus")

    def test_default_is_ok(self):
        """缺省值不变,仍是 ok。"""
        assert GateSignals().statistical_validity == "ok"
