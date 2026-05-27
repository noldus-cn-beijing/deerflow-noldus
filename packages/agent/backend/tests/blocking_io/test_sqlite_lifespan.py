"""Regression test: sqlite path setup must run off the event loop.

Anchors the production offload from
`runtime/checkpointer/async_provider.py:_async_checkpointer`, where SQLite
path resolution and `ensure_sqlite_parent_dir` are dispatched via
`await asyncio.to_thread(...)`.

This test invokes the production `_async_checkpointer()` path under the
strict Blockbuster context. The target path's parent does not yet exist, so
the underlying path resolution and `os.mkdir` both execute. If either step is
regressed to run directly on the event loop, Blockbuster raises
`BlockingError` and this test fails.

Noldus adaptation: patches `DeerFlowAsyncSqliteSaver.from_conn_string`
instead of the upstream `sys.modules["langgraph.checkpoint.sqlite.aio"]`
approach, because our production code uses the custom saver subclass.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


async def test_async_checkpointer_sqlite_setup_does_not_block_event_loop(tmp_path: Path) -> None:
    from deerflow.config.checkpointer_config import CheckpointerConfig
    from deerflow.runtime.checkpointer.async_provider import _async_checkpointer

    db_file = tmp_path / "subdir" / "store.db"

    mock_saver = AsyncMock()
    mock_context_manager = AsyncMock()
    mock_context_manager.__aenter__.return_value = mock_saver
    mock_context_manager.__aexit__.return_value = False

    mock_saver_cls = MagicMock()
    mock_saver_cls.from_conn_string.return_value = mock_context_manager

    with patch(
        "deerflow.runtime.checkpointer.deerflow_saver.DeerFlowAsyncSqliteSaver.from_conn_string",
        mock_saver_cls.from_conn_string,
    ):
        async with _async_checkpointer(CheckpointerConfig(type="sqlite", connection_string=str(db_file))) as saver:
            assert saver is mock_saver

    assert db_file.parent.exists()
    mock_saver_cls.from_conn_string.assert_called_once_with(str(db_file.resolve()))
    mock_saver.setup.assert_awaited_once()
