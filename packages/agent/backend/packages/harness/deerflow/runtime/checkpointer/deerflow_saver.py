"""DeerFlow extended SQLite saver.

Provides a thin subclass of langgraph's AsyncSqliteSaver that adds
``adelete_for_runs``, which is required by the langgraph_api for
multitask_strategy='rollback' cleanup of cancelled-run checkpoints.
"""

from __future__ import annotations

from collections.abc import Iterable

import aiosqlite
from langgraph.checkpoint.base import SerializerProtocol
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deerflow.runtime.store._sqlite_utils import ensure_sqlite_parent_dir


class DeerFlowAsyncSqliteSaver(AsyncSqliteSaver):
    """AsyncSqliteSaver extended with adelete_for_runs support.

    langgraph_api wraps custom checkpointers in a capability-detecting
    adapter. When ``adelete_for_runs`` is missing, it logs a warning on
    every server startup::

       Custom checkpointer missing adelete_for_runs:
       multitask_strategy='rollback' will not clean up
       checkpoints from cancelled runs.

    This subclass fills the gap by querying the ``metadata`` column
    (JSON blob) for matching ``run_id`` values and deleting the
    associated checkpoints + writes.
    """

    def __init__(
        self,
        conn: aiosqlite.Connection,
        *,
        serde: SerializerProtocol | None = None,
    ):
        super().__init__(conn, serde=serde)

    async def adelete_for_runs(self, run_ids: Iterable[str]) -> None:
        """Delete checkpoints for specific cancelled runs.

        Implements the langgraph ``FullCheckpointerProtocol`` contract.
        Called by the runtime when ``multitask_strategy='rollback'``
        cancels a run, so that subsequent runs don't see stale state.

        Args:
            run_ids: Iterable of run IDs whose checkpoints should be
                     removed from this saver's storage.
        """
        run_id_list = list(run_ids)
        if not run_id_list:
            return

        placeholders = ",".join("?" * len(run_id_list))
        await self.setup()

        async with self.lock, self.conn.cursor() as cur:
            # 1. Find matching checkpoints by run_id in metadata JSON
            await cur.execute(
                f"SELECT thread_id, checkpoint_ns, checkpoint_id "
                f"FROM checkpoints "
                f"WHERE json_extract(CAST(metadata AS TEXT), '$.run_id') "
                f"IN ({placeholders})",
                run_id_list,
            )
            matches = await cur.fetchall()

            if not matches:
                return

            # 2. Delete associated writes for each matched checkpoint
            for thread_id, checkpoint_ns, checkpoint_id in matches:
                await cur.execute(
                    "DELETE FROM writes "
                    "WHERE thread_id = ? AND checkpoint_ns = ? "
                    "AND checkpoint_id = ?",
                    (thread_id, checkpoint_ns, checkpoint_id),
                )

            # 3. Delete the matched checkpoints
            await cur.execute(
                f"DELETE FROM checkpoints "
                f"WHERE json_extract(CAST(metadata AS TEXT), '$.run_id') "
                f"IN ({placeholders})",
                run_id_list,
            )

            await self.conn.commit()
