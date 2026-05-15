"""Validate DeerFlowAsyncSqliteSaver.adelete_for_runs.

Issue #10 from thread 5046a6e6: langgraph.log warns every run about
"Custom checkpointer missing adelete_for_runs".  This test simulates
a cancelled-run state cleanup.
"""

import tempfile
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_adelete_for_runs_removes_cancelled_run_state():
    """adelete_for_runs removes checkpoints that match the given run_ids."""
    from deerflow.runtime.checkpointer.deerflow_saver import (
        DeerFlowAsyncSqliteSaver,
    )

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        async with DeerFlowAsyncSqliteSaver.from_conn_string(db_path) as saver:
            await saver.setup()

            await saver.aput(
                {"configurable": {"thread_id": "test-thread", "checkpoint_ns": ""}},
                {"channel_values": {"messages": ["hello"]}, "channel_versions": {}, "versions_seen": {}, "id": "cp-1"},
                {"run_id": "run-1", "source": "loop", "step": 0},
                {},
            )

            result = await saver.aget_tuple({"configurable": {"thread_id": "test-thread"}})
            assert result is not None, "checkpoint should exist after aput"
            assert result.metadata.get("run_id") == "run-1"

            await saver.adelete_for_runs(["run-1"])

            result = await saver.aget_tuple({"configurable": {"thread_id": "test-thread"}})
            assert result is None, "adelete_for_runs should remove the checkpoint"
    finally:
        Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_adelete_for_runs_preserves_other_runs():
    """adelete_for_runs only removes matching run_ids, leaving others intact."""
    from deerflow.runtime.checkpointer.deerflow_saver import (
        DeerFlowAsyncSqliteSaver,
    )

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        async with DeerFlowAsyncSqliteSaver.from_conn_string(db_path) as saver:
            await saver.setup()

            for run_id in ["run-1", "run-2"]:
                await saver.aput(
                    {"configurable": {"thread_id": "test-thread", "checkpoint_ns": ""}},
                    {
                        "channel_values": {"messages": [f"msg-{run_id}"]},
                        "channel_versions": {},
                        "versions_seen": {},
                        "id": f"cp-{run_id}",
                    },
                    {"run_id": run_id, "source": "loop", "step": 0},
                    {},
                )

            await saver.adelete_for_runs(["run-1"])

            result = await saver.aget_tuple({"configurable": {"thread_id": "test-thread"}})
            assert result is not None, "run-2 checkpoint should still exist"
            assert result.metadata.get("run_id") == "run-2"

            configs = []
            async for cp in saver.alist({"configurable": {"thread_id": "test-thread"}}):
                configs.append(cp)
            assert len(configs) == 1, f"expected 1 checkpoint left, got {len(configs)}"
            assert configs[0].metadata.get("run_id") == "run-2"
    finally:
        Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_adelete_for_runs_empty_ids_is_noop():
    """adelete_for_runs with empty iterable is a no-op."""
    from deerflow.runtime.checkpointer.deerflow_saver import (
        DeerFlowAsyncSqliteSaver,
    )

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        async with DeerFlowAsyncSqliteSaver.from_conn_string(db_path) as saver:
            await saver.setup()

            await saver.aput(
                {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
                {"channel_values": {}, "channel_versions": {}, "versions_seen": {}, "id": "cp-x"},
                {"run_id": "r-x", "source": "loop", "step": 0},
                {},
            )

            await saver.adelete_for_runs([])

            result = await saver.aget_tuple({"configurable": {"thread_id": "t"}})
            assert result is not None, "empty run_ids should not delete anything"
    finally:
        Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_adelete_for_runs_non_existent_run_ids_is_noop():
    """adelete_for_runs with non-matching run_ids is a no-op."""
    from deerflow.runtime.checkpointer.deerflow_saver import (
        DeerFlowAsyncSqliteSaver,
    )

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        async with DeerFlowAsyncSqliteSaver.from_conn_string(db_path) as saver:
            await saver.setup()

            await saver.aput(
                {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
                {"channel_values": {}, "channel_versions": {}, "versions_seen": {}, "id": "cp-x"},
                {"run_id": "real-run", "source": "loop", "step": 0},
                {},
            )

            await saver.adelete_for_runs(["non-existent-run"])

            result = await saver.aget_tuple({"configurable": {"thread_id": "t"}})
            assert result is not None, "non-matching run_ids should not delete anything"
            assert result.metadata.get("run_id") == "real-run"
    finally:
        Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_adelete_for_runs_also_cleans_writes():
    """adelete_for_runs deletes associated writes, not just checkpoints."""
    from deerflow.runtime.checkpointer.deerflow_saver import (
        DeerFlowAsyncSqliteSaver,
    )

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        async with DeerFlowAsyncSqliteSaver.from_conn_string(db_path) as saver:
            await saver.setup()

            config = await saver.aput(
                {"configurable": {"thread_id": "t", "checkpoint_ns": ""}},
                {"channel_values": {}, "channel_versions": {}, "versions_seen": {}, "id": "cp-w"},
                {"run_id": "run-w", "source": "loop", "step": 0},
                {},
            )

            await saver.aput_writes(
                config,
                [("messages", "write-1"), ("messages", "write-2")],
                task_id="task-1",
            )

            # Verify writes exist before deletion
            async with saver.conn.execute(
                "SELECT COUNT(*) FROM writes WHERE thread_id = ?", ("t",)
            ) as cur:
                count = (await cur.fetchone())[0]
            assert count == 2, f"expected 2 writes, got {count}"

            await saver.adelete_for_runs(["run-w"])

            # Verify writes are gone
            async with saver.conn.execute(
                "SELECT COUNT(*) FROM writes WHERE thread_id = ?", ("t",)
            ) as cur:
                count = (await cur.fetchone())[0]
            assert count == 0, f"expected 0 writes after adelete_for_runs, got {count}"

            # Verify checkpoint is gone
            result = await saver.aget_tuple({"configurable": {"thread_id": "t"}})
            assert result is None, "checkpoint should be deleted"
    finally:
        Path(db_path).unlink(missing_ok=True)
