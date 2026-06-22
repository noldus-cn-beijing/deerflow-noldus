"""Dump a thread's full conversation as JSON, straight from the checkpointer.

Given a thread_id, opens the configured LangGraph checkpointer (same path the
Gateway uses at runtime — legacy ``checkpointer:`` section in config.yaml wins,
so checkpoints live in ``backend/checkpoints.db``), reads the latest checkpoint,
and serializes ``channel_values`` (notably ``messages``) the same way the
``POST /api/runs/wait`` endpoint does — via
``deerflow.runtime.serialize_channel_values``. This is the ONLY way to correctly
recover LangChain message objects (incl. tool calls / thinking blocks); do not
hand-decode the msgpack blob in ``checkpoint_blobs``.

Also joins in lightweight metadata from the app DB (``backend/.deer-flow/data/
deerflow.db``): thread_meta + runs (status, token usage, first/last message).
``run_events`` is NOT consulted — it defaults to ``backend: memory`` and is not
persisted.

Usage (run inside the gateway container, or anywhere with the backend venv +
config.yaml on the config path)::

    # one thread
    .venv/bin/python scripts/dump_thread.py <thread_id>

    # write to a file instead of stdout
    .venv/bin/python scripts/dump_thread.py <thread_id> -o thread.json

    # pretty vs compact
    .venv/bin/python scripts/dump_thread.py <thread_id> --indent 2

From the host with docker::

    docker exec deer-flow-gateway sh -c \
      'cd backend && .venv/bin/python scripts/dump_thread.py <thread_id>'

Exit codes: 0 ok; 2 thread not found in checkpointer; 3 thread found but has
no messages channel.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

# Resolve config before any deerflow import that might cache a stale config.
# Scripts run from backend/, so config.yaml is on the default search path.
from deerflow.config.app_config import get_app_config
from deerflow.runtime import serialize_channel_values
from deerflow.runtime.checkpointer.async_provider import make_checkpointer


def _read_app_db_metadata(thread_id: str) -> dict[str, Any]:
    """Pull thread_meta + latest run row from the application SQLite DB.

    Returns {} if the app DB or tables are absent (e.g. backend=memory). Never
    raises — metadata is a best-effort enrichment on top of the checkpointer.
    """
    cfg = get_app_config()
    db_path = cfg.database.app_sqlite_path if cfg.database.backend != "memory" else None
    if not db_path or not Path(db_path).exists():
        return {}

    meta: dict[str, Any] = {"app_db": db_path}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        try:
            row = conn.execute(
                "SELECT thread_id, display_name, status, user_id, created_at, updated_at "
                "FROM threads_meta WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            if row is not None:
                meta["thread_meta"] = dict(row)
        except sqlite3.Error:
            # threads_meta may not exist on a fresh DB — not fatal.
            pass

        try:
            runs = conn.execute(
                "SELECT run_id, status, model_name, message_count, "
                "first_human_message, last_ai_message, total_tokens, "
                "llm_call_count, error, created_at, updated_at "
                "FROM runs WHERE thread_id = ? ORDER BY created_at DESC",
                (thread_id,),
            ).fetchall()
            if runs:
                meta["runs"] = [dict(r) for r in runs]
        except sqlite3.Error:
            pass
    finally:
        conn.close()
    return meta


async def _dump(thread_id: str) -> dict[str, Any]:
    app_config = get_app_config()
    async with make_checkpointer(app_config) as checkpointer:
        config = {"configurable": {"thread_id": thread_id}}
        tup = await checkpointer.aget_tuple(config)
        if tup is None:
            # No checkpoint — nothing durable for this thread.
            sys.stderr.write(
                f"[dump_thread] thread {thread_id!r} not found in checkpointer "
                f"(type={app_config.checkpointer.type if app_config.checkpointer else 'memory'}). "
                f"Confirm the thread_id and that the gateway has written a checkpoint for it.\n"
            )
            sys.exit(2)

        checkpoint = getattr(tup, "checkpoint", {}) or {}
        channel_values = checkpoint.get("channel_values", {}) or {}
        values = serialize_channel_values(channel_values)

        messages = values.get("messages")
        if not messages:
            sys.stderr.write(
                f"[dump_thread] thread {thread_id!r} has a checkpoint but no 'messages' channel "
                f"(channels present: {sorted(channel_values.keys())}).\n"
            )
            # Not fatal — still emit metadata + other channel values.

        result: dict[str, Any] = {
            "thread_id": thread_id,
            "checkpoint": {
                "id": checkpoint.get("id"),
                "parent_checkpoint_id": checkpoint.get("parent_checkpoint_id"),
            },
            "metadata": getattr(tup, "metadata", {}) or {},
            "values": values,
        }
        if messages:
            result["messages"] = messages

        result.update(_read_app_db_metadata(thread_id))
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump a thread's conversation as JSON.")
    parser.add_argument("thread_id", help="LangGraph thread_id (UUID string)")
    parser.add_argument("-o", "--output", help="Write JSON to this file (default: stdout)")
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (default: 2). Use 0 for compact one-line output.",
    )
    args = parser.parse_args()

    data = asyncio.run(_dump(args.thread_id))

    indent = args.indent if args.indent > 0 else None
    payload = json.dumps(data, ensure_ascii=False, indent=indent, default=str)

    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        msg_count = len(data.get("messages", []))
        sys.stderr.write(
            f"[dump_thread] wrote {args.output} — {msg_count} messages, "
            f"thread={args.thread_id}\n"
        )
    else:
        sys.stdout.write(payload)
        if indent is not None:
            sys.stdout.write("\n")


if __name__ == "__main__":
    main()
