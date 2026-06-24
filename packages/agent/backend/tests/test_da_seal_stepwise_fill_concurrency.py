"""Tests for fill/finalize/preset 读-改-写并发竞态修复 —— spec
2026-06-24-fill-handoff-concurrent-write-race.

根因：data-analyst 同一条 AIMessage 并行发多个 ``fill_data_analyst_*`` tool_call，
它们都走「``_load_da_payload`` 读 → 改 payload → ``_write_da_payload`` 写固定
``handoff_data_analyst.json.tmp`` → ``os.rename``」无锁且 tmp 路径固定共享 →
竞态：丢字段 / ``FileNotFoundError`` rename / 读到 0 字节 ``Expecting value char 0``。

修法：按 ``(workspace, filename)`` 的模块级 ``threading.Lock`` 串行化所有
data-analyst handoff 的读-改-写临界区 + tmp 路径加 pid+tid 唯一后缀（纵深）。

⚠️ worktree 借主仓 venv（editable 指主仓），运行时必须 ``PYTHONPATH=packages/harness``
使 ``deerflow.*`` 解析到被测 worktree 源（守 memory
feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib）。
与 test_da_seal_stepwise_fill.py 同运行环境。
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "experiment-context.json").write_text(
        json.dumps({"paradigm": "epm", "analysis_config_id": "cfg-123"}),
        encoding="utf-8",
    )
    return ws


def _make_runtime(workspace_dir: str) -> MagicMock:
    runtime = MagicMock()
    runtime.state = {"thread_data": {"workspace_path": workspace_dir}}
    return runtime


# ---------------------------------------------------------------------------
# T1 — 锁注册表：同 workspace 同一 Lock；不同 workspace 不同 Lock
# ---------------------------------------------------------------------------


class TestLockRegistry:
    def test_same_workspace_returns_same_lock(self, tmp_path):
        from deerflow.tools.builtins.seal_handoff_tools import _get_da_handoff_lock

        ws_a = _make_workspace(tmp_path)
        lock1 = _get_da_handoff_lock(ws_a)
        lock2 = _get_da_handoff_lock(ws_a)
        assert lock1 is lock2, "同一 workspace 必须返回同一 Lock 对象（串行化前提）"

    def test_different_workspaces_return_different_locks(self, tmp_path):
        from deerflow.tools.builtins.seal_handoff_tools import _get_da_handoff_lock

        ws_a = tmp_path / "ws_a"
        ws_a.mkdir()
        ws_b = tmp_path / "ws_b"
        ws_b.mkdir()
        lock_a = _get_da_handoff_lock(ws_a)
        lock_b = _get_da_handoff_lock(ws_b)
        assert lock_a is not lock_b, "不同 workspace 必须不同 Lock（不互相阻塞）"


# ---------------------------------------------------------------------------
# T2 — 并发 fill 无损（核心红→绿）：多线程并发 set 不同 field → 全成功、无丢字段、无崩
# ---------------------------------------------------------------------------


class TestConcurrentFillNoLoss:
    def test_concurrent_set_different_fields_all_persist(self, tmp_path):
        """≥8 线程并发各 fill 不同 field → 最终 handoff 含所有 field 的值。

        改前（无锁）：竞态丢字段 / FileNotFoundError rename / 读到 0 字节。
        改后（加锁）：全部成功、字段完整。
        """
        from deerflow.tools.builtins.seal_handoff_tools import (
            fill_data_analyst_text_list,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))

        # 5 个 text-list 字段 × 多线程/重复 → 提高竞态概率
        fields = ["key_findings", "excluded_metrics", "method_warnings", "recommendations", "errors"]
        n_threads_per_field = 4  # 共 20 个并发 fill（每线程一次 set）

        barrier = threading.Barrier(20, timeout=30)

        def fill_one(field: str, idx: int) -> str:
            barrier.wait()  # 对齐起跑，最大化竞态窗口
            return fill_data_analyst_text_list.func(
                field=field, mode="set", value=[f"{field}-t{idx}"], runtime=runtime,
            )

        errors: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = []
            for field in fields:
                for idx in range(n_threads_per_field):
                    futures.append(pool.submit(fill_one, field, idx))
            for fut in as_completed(futures):
                try:
                    fut.result()
                except BaseException as e:  # noqa: BLE001 — 收集所有线程异常
                    errors.append(e)

        # 断言 1：无任何线程抛 FileNotFoundError / unreadable JSON
        assert not errors, f"并发 fill 抛错（竞态未消除）: {errors[:3]}"

        # 断言 2：最终 handoff 是合法 JSON 且每个 field 有且仅有最后一次 set 的值（无丢字段）
        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        for field in fields:
            vals = data[field]
            assert len(vals) == 1, (
                f"field {field} 应只剩最后一次 set 的 1 条，实际 {vals}（并发覆盖下"
                f"丢字段=竞态；多于一=append 语义错）"
            )


# ---------------------------------------------------------------------------
# T3 — 并发 append 累加：多线程并发 append 同一 field → 总条数 = append 之和
# ---------------------------------------------------------------------------


class TestConcurrentAppendAccumulates:
    def test_concurrent_append_same_field_no_loss(self, tmp_path):
        """多线程并发 append 同一 field → 最终条数 = 所有 append 之和（无覆盖丢失）。"""
        from deerflow.tools.builtins.seal_handoff_tools import (
            fill_data_analyst_text_list,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))

        n_threads = 12
        barrier = threading.Barrier(n_threads, timeout=30)

        def append_one(idx: int) -> None:
            barrier.wait()
            fill_data_analyst_text_list.func(
                field="recommendations", mode="append", value=[f"rec-{idx}"], runtime=runtime,
            )

        errors: list[BaseException] = []
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futs = [pool.submit(append_one, i) for i in range(n_threads)]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)

        assert not errors, f"并发 append 抛错: {errors[:3]}"
        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        # 12 次 append × 1 条 = 12 条，无一丢失
        assert len(data["recommendations"]) == n_threads, (
            f"并发 append 丢条：期望 {n_threads} 条，实际 {len(data['recommendations'])} 条 → 竞态覆盖"
        )


# ---------------------------------------------------------------------------
# T4 — preset + fill 不竞态：preset 写后立即多线程 fill → 模板不被读到半截
# ---------------------------------------------------------------------------


class TestPresetFillNoRace:
    def test_preset_then_concurrent_fill_template_intact(self, tmp_path):
        """preset 写模板后立即多线程 fill → 每个 fill 读到的都是合法完整模板。"""
        from deerflow.tools.builtins.seal_handoff_tools import (
            fill_data_analyst_text_list,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))

        n = 10
        barrier = threading.Barrier(n, timeout=30)
        errors: list[BaseException] = []

        def fill(idx: int) -> None:
            barrier.wait()
            fill_data_analyst_text_list.func(
                field="key_findings", mode="append", value=[f"k{idx}"], runtime=runtime,
            )

        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = [pool.submit(fill, i) for i in range(n)]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)

        assert not errors, f"preset+fill 竞态抛错: {errors[:3]}"
        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        assert len(data["key_findings"]) == n


# ---------------------------------------------------------------------------
# T5 — tmp 路径唯一后缀（纵深）：tmp 名含 pid/tid（断言 _write_da_payload 用唯一 tmp）
# ---------------------------------------------------------------------------


class TestUniqueTmpSuffix:
    def test_tmp_path_uses_pid_tid_suffix(self, tmp_path, monkeypatch):
        """_write_da_payload 写出的 tmp 文件名含 os.getpid()/threading.get_ident() 唯一后缀。

        断言实现用 ``.{pid}.{tid}.tmp`` 形态，保证万一锁失效也不抢同一 tmp。
        守 spec §七「确定性可测：tmp 唯一后缀用 pid/tid，不用 Date.now()/随机数」。
        """
        import deerflow.tools.builtins.seal_handoff_tools as mod

        src = Path(mod.__file__).read_text(encoding="utf-8")
        # 实现须用 os.getpid() + threading.get_ident() 构造 tmp（不用随机数/时间戳）
        assert "os.getpid()" in src, "_write_da_payload tmp 须用 os.getpid() 做唯一后缀"
        assert "threading.get_ident()" in src, "_write_da_payload tmp 须用 threading.get_ident() 做唯一后缀"

    def test_concurrent_writes_distinct_tmp_files(self, tmp_path):
        """并发写产生不同的 tmp 文件名（间接验证唯一后缀）。"""
        from deerflow.tools.builtins.seal_handoff_tools import (
            fill_data_analyst_text_list,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))

        # 捕获 _write_da_payload 用的 tmp 路径（monkeypatch os.replace 看从哪 replace）。
        # _write_da_payload 用 os.replace 做 POSIX atomic rename。
        import deerflow.tools.builtins.seal_handoff_tools as mod

        seen_tmp: list[str] = []
        orig_replace = mod.os.replace

        def spy_replace(src_path, dst_path):
            seen_tmp.append(str(src_path))
            return orig_replace(src_path, dst_path)

        mod.os.replace = spy_replace
        try:
            n = 8
            barrier = threading.Barrier(n, timeout=30)

            def fill(idx: int) -> None:
                barrier.wait()
                fill_data_analyst_text_list.func(
                    field="key_findings", mode="append", value=[f"k{idx}"], runtime=runtime,
                )

            with ThreadPoolExecutor(max_workers=n) as pool:
                futs = [pool.submit(fill, i) for i in range(n)]
                for fut in as_completed(futs):
                    fut.result()
        finally:
            mod.os.replace = orig_replace

        # 每个 replace 的 src 是一个 tmp 路径；断言 tmp 名带唯一后缀且不全相同
        assert len(seen_tmp) == n
        # 所有 tmp 都不是固定的 bare .tmp（应带 pid.tid 后缀）
        for t in seen_tmp:
            assert t.endswith(".tmp")
            assert "handoff_data_analyst.json." in t
            # bare tmp（无 pid.tid）= 抢同一文件 = 竞态
            assert t != str(ws / "handoff_data_analyst.json.tmp"), (
                "tmp 仍是固定 bare 路径 → 并发抢同一文件（无唯一后缀）"
            )
        # 不同线程的 tmp 名应不同（pid 同、tid 不同）
        assert len(set(seen_tmp)) > 1, f"所有并发 tmp 名相同 → 无唯一后缀: {seen_tmp[:3]}"


# ---------------------------------------------------------------------------
# T6 — 单线程不回归：串行 fill→finalize 仍绿（守既有 stepwise fill 流程）
# ---------------------------------------------------------------------------


class TestSingleThreadNoRegression:
    def test_serial_fill_then_finalize(self, tmp_path):
        from deerflow.tools.builtins.seal_handoff_tools import (
            fill_data_analyst_gate_signals,
            fill_data_analyst_text_list,
            finalize_data_analyst_handoff,
            preset_data_analyst_template_to_workspace,
        )

        ws = _make_workspace(tmp_path)
        preset_data_analyst_template_to_workspace(ws)
        runtime = _make_runtime(str(ws))

        fill_data_analyst_text_list.func(
            field="key_findings", mode="set", value=["finding"], runtime=runtime,
        )
        fill_data_analyst_gate_signals.func(
            value={"statistical_validity": "ok", "data_quality": {"critical_count": 0}},
            runtime=runtime,
        )
        result = finalize_data_analyst_handoff.func(final_status="completed", runtime=runtime)
        assert "status=completed" in result
        data = json.loads((ws / "handoff_data_analyst.json").read_text(encoding="utf-8"))
        assert data["status"] == "completed"
        assert data["key_findings"] == ["finding"]


# ---------------------------------------------------------------------------
# T7 — import 环：改 seal_handoff_tools.py 后裸导入两入口 0 退出
# （由 tests/test_gateway_import_no_cycle.py 在 clean subprocess 覆盖；
#  本处只断言本模块本身可被干净导入，不触发 conftest mock 之外的环。）
# ---------------------------------------------------------------------------


class TestModuleImportClean:
    def test_seal_handoff_tools_imports_threading_weakref(self):
        """seal_handoff_tools 模块顶层 import 了 threading + weakref（锁注册表依赖）。"""
        import deerflow.tools.builtins.seal_handoff_tools as mod

        assert hasattr(mod, "threading")
        assert hasattr(mod, "weakref")
        assert hasattr(mod, "_get_da_handoff_lock")
        assert hasattr(mod, "_DA_HANDOFF_LOCKS")
