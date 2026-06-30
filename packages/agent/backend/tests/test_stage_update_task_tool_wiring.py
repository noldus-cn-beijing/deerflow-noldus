"""集成测：task_tool.py 派遣边界接线坐实（spec 验收 4 grounded 的 neuter 探针）。

unit 测（test_stage_update_dispatch.py）已证 emit_dispatch_enter/exit 的语义；
本文件证 ``task_tool`` 在真实代码路径的每个终态分支都接了线——
- task_started 边界 → emit_dispatch_enter
- COMPLETED → emit_dispatch_exit(succeeded=True)
- FAILED / CANCELLED / TIMED_OUT / disappeared / poll-timeout → emit_dispatch_exit(succeeded=False)

这是 neuter 探针：若有人误删某分支的发射调用，本测试红（防「内脏泄漏」与
「失败报 completed」两类回归静默通过）。
"""

from __future__ import annotations

import importlib
from pathlib import Path

# Import the task_tool *module* (not the StructuredTool re-exported under same name).
_task_tool_module = importlib.import_module("deerflow.tools.builtins.task_tool")
SOURCE = Path(_task_tool_module.__file__).read_text(encoding="utf-8")


def _count(needle: str) -> int:
    return SOURCE.count(needle)


class TestTaskToolWiring:
    def test_imports_emitters_lazily(self):
        """惰性 import 在函数体内（守 harness import-cycle 铁律）。"""
        assert "from deerflow.agents.middlewares.stage_narration import emit_dispatch_enter, emit_dispatch_exit" in SOURCE

    def test_started_boundary_emits_active(self):
        assert _count("emit_dispatch_enter(subagent_type, writer=writer)") >= 1

    def test_completed_branch_emits_succeeded_true(self):
        """COMPLETED 分支必须发 succeeded=True（叙事说完成 = 真完成）。"""
        assert 'emit_dispatch_exit(subagent_type, succeeded=True, terminal_status="completed", writer=writer)' in SOURCE

    def test_every_failure_branch_emits_succeeded_false(self):
        """所有失败终态分支都接 emit_dispatch_exit(succeeded=False)。

        neuter：去掉任何一个 → 这里计数下降 → 红。
        失败分支：disappeared / failed / cancelled / timed_out(x2: executor + poll-fallback)。
        """
        false_calls = SOURCE.count("emit_dispatch_exit(subagent_type, succeeded=False")
        # disappeared + failed + cancelled + timed_out(executor) + timed_out(poll) = 5
        assert false_calls >= 5, f"期望 ≥5 个 succeeded=False 发射点，实际 {false_calls}"

    def test_no_succeeded_true_outside_completed(self):
        """succeeded=True 只该出现在 completed 分支（一处）。"""
        true_calls = SOURCE.count("emit_dispatch_exit(subagent_type, succeeded=True")
        assert true_calls == 1, f"succeeded=True 应只在 COMPLETED 出现一次，实际 {true_calls}"
