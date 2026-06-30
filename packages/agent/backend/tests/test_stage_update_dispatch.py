"""TDD tests for stage_update emission at the task-tool dispatch boundary.

对应 spec 2026-06-30-... 模块 2（复用 task 工具派遣观测点发 stage_update）+
验收标准 4（subagent 失败 → stage_update 不报 completed，叙事不撒谎）。

被测：``deerflow.agents.middlewares.stage_narration`` 的派遣边界发射助手。
发射逻辑抽成纯函数（不依赖 SubagentExecutor），由 ``task`` 工具在既有
task_started / task_completed / task_failed 旁同源调用——复用观测点，不重复造。
"""

from __future__ import annotations

import pytest

from deerflow.agents.middlewares import stage_narration


def _collected(emitted):
    return [e for e in emitted if e.get("kind") == "stage_update"]


class TestDispatchBoundaryStageUpdate:
    """task 工具派遣边界 → stage_update active/completed（grounded 于真实 status）。"""

    def test_dispatch_enter_emits_active(self):
        """subagent 进入（task_started）→ 发 stage_update(active)。"""
        emitted: list[dict] = []
        stage_narration.emit_dispatch_enter("chart-maker", writer=emitted.append)
        updates = _collected(emitted)
        assert len(updates) == 1
        assert updates[0]["stage"] == "生成图表"
        assert updates[0]["status"] == "active"

    def test_dispatch_enter_unknown_subagent_emits_nothing(self):
        """未登记的 subagent 不发 stage_update（不猜阶段名）。"""
        emitted: list[dict] = []
        stage_narration.emit_dispatch_enter("mystery-agent", writer=emitted.append)
        assert emitted == []

    def test_dispatch_exit_completed_emits_completed(self):
        """subagent 成功退出（task_completed）→ 发 stage_update(completed)。"""
        emitted: list[dict] = []
        stage_narration.emit_dispatch_exit("code-executor", succeeded=True, writer=emitted.append)
        updates = _collected(emitted)
        assert len(updates) == 1
        assert updates[0]["stage"] == "计算指标"
        assert updates[0]["status"] == "completed"

    @pytest.mark.parametrize(
        "terminal_status",
        ["failed", "timed_out", "cancelled", "disappeared"],
    )
    def test_dispatch_exit_failure_emits_no_completed(self, terminal_status):
        """spec 验收 4：subagent 失败/超时/取消/消失 → **不**发 completed。

        叙事不撒谎（grounded）：阶段未成功结束，绝不报 completed。
        本测试是 neuter 探针——若实现错了（失败也发 completed），这里会红。
        """
        emitted: list[dict] = []
        stage_narration.emit_dispatch_exit(
            "data-analyst",
            succeeded=False,
            terminal_status=terminal_status,
            writer=emitted.append,
        )
        completed = [e for e in _collected(emitted) if e["status"] == "completed"]
        assert completed == [], f"失败状态 {terminal_status} 不应发 completed，却发了 {completed}"

    def test_dispatch_exit_failure_emits_nothing(self):
        """失败时干脆不发 stage_update（active 已发过，completed 不发，留空比撒谎好）。"""
        emitted: list[dict] = []
        stage_narration.emit_dispatch_exit("report-writer", succeeded=False, terminal_status="failed", writer=emitted.append)
        assert _collected(emitted) == []

    def test_dispatch_exit_unknown_subagent_emits_nothing(self):
        emitted: list[dict] = []
        stage_narration.emit_dispatch_exit("mystery", succeeded=True, writer=emitted.append)
        assert emitted == []

    def test_writer_failure_swallowed(self):
        """writer 抛错不应传播（task 工具的 turn 不该被叙事崩掉）。"""
        def boom(_):
            raise RuntimeError("writer down")

        # 不应抛
        stage_narration.emit_dispatch_enter("code-executor", writer=boom)
        stage_narration.emit_dispatch_exit("code-executor", succeeded=True, writer=boom)


class TestStageUpdateNoVisceraInNarration:
    """dispatch 边界产出的 narration 也不含内脏（防泄漏一致性）。"""

    FORBIDDEN = ["code-executor", "code_executor", "data-analyst", "chart-maker", "report-writer", "identify_ev19_template", "gate_signals"]

    def test_active_narration_clean(self):
        emitted: list[dict] = []
        stage_narration.emit_dispatch_enter("code-executor", writer=emitted.append)
        narr = _collected(emitted)[0]["narration"]
        for bad in self.FORBIDDEN:
            assert bad not in narr

    def test_completed_narration_clean(self):
        emitted: list[dict] = []
        stage_narration.emit_dispatch_exit("code-executor", succeeded=True, writer=emitted.append)
        narr = _collected(emitted)[0]["narration"]
        for bad in self.FORBIDDEN:
            assert bad not in narr
