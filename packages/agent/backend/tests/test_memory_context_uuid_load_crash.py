"""Regression tests for the memory-context UUID crash (spec 2026-06-26).

Root cause (gateway.log ERROR ``Failed to load memory context: expected
string or bytes-like object, got 'UUID'``):

``lead_agent/prompt.py`` resolved ``user_id = current_user.id`` where
``current_user.id`` is a **UUID object** at runtime (the
``CurrentUser`` protocol declares ``id: str`` but the concrete
``app.gateway.auth.models.User.id`` is a UUID). That UUID flowed into
``FileMemoryStorage`` → ``_validate_user_id`` → ``re.match`` and raised
``TypeError``. The exception was swallowed by the broad ``except`` in
``_get_memory_context`` and turned into ``return ""``, so the lead agent
silently ran with **no memory context at all** — a loud failure
degraded into a silent one.

These tests lock in:
  * The end-to-end fix — ``_get_memory_context`` with a UUID-valued
    ``current_user.id`` returns a non-empty ``<memory>`` block instead
    of the swallowed ``""``.
  * The storage-layer defense — ``FileMemoryStorage`` normalizes a UUID
    ``user_id`` to ``str`` at the boundary so any future caller that
    forgets ``str()`` cannot reintroduce the crash.
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deerflow.agents.lead_agent.prompt import _get_memory_context
from deerflow.agents.memory.storage import FileMemoryStorage
from deerflow.config.memory_config import MemoryConfig
from deerflow.runtime.user_context import reset_current_user, set_current_user


def _seed_user_memory(tmp_path, user_id, fact_content="User studies EPM anxiety behavior"):
    """Write a memory.json for ``user_id`` under a tmp ``Paths.base_dir``.

    Returns a ``Paths`` instance whose ``base_dir`` is ``tmp_path`` so the
    real ``FileMemoryStorage`` reads the seeded file when ``get_paths``
    is patched to return it.
    """
    from deerflow.config.paths import Paths

    paths = Paths(base_dir=tmp_path)
    memory_file = paths.user_memory_file(str(user_id))
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_data = {
        "user": {"workContext": {"summary": "Behavioral researcher"}},
        "history": {},
        "facts": [
            {"content": fact_content, "category": "context", "confidence": 0.9},
        ],
    }
    memory_file.write_text(json.dumps(memory_data), encoding="utf-8")
    return paths


@pytest.mark.no_auto_user
def test_get_memory_context_with_uuid_user_id_returns_non_empty(tmp_path):
    """A UUID ``current_user.id`` must not crash; memory must be injected.

    Before the fix this raised ``TypeError: expected string or bytes-like
    object, got 'UUID'`` inside ``_validate_user_id``; the broad
    ``except`` in ``_get_memory_context`` swallowed it and returned ``""``,
    silently running the lead agent with no memory.
    """
    user_uuid = uuid.uuid4()
    paths = _seed_user_memory(tmp_path, user_uuid)

    user = SimpleNamespace(id=user_uuid)
    token = set_current_user(user)
    try:
        storage = FileMemoryStorage()
        with (
            patch("deerflow.agents.memory.storage.get_paths", return_value=paths),
            patch("deerflow.agents.memory.updater.get_memory_storage", return_value=storage),
        ):
            # Default MemoryConfig has enabled=True, injection_enabled=True.
            app_config = SimpleNamespace(memory=MemoryConfig())
            result = _get_memory_context(agent_name=None, app_config=app_config)
    finally:
        reset_current_user(token)

    assert result != "", "memory context was swallowed to '' — UUID user_id still crashes"
    assert "<memory>" in result
    assert "User studies EPM anxiety behavior" in result


def test_file_memory_storage_normalizes_uuid_user_id(tmp_path):
    """Defense-in-depth: the storage boundary coerces UUID → str.

    Even if a future caller bypasses ``str()`` and passes a raw UUID,
    ``FileMemoryStorage`` must not crash on path validation. This guards
    every call path (load/reload/save) with one boundary normalization.
    """
    user_uuid = uuid.uuid4()
    paths = _seed_user_memory(tmp_path, user_uuid)

    storage = FileMemoryStorage()
    with patch("deerflow.agents.memory.storage.get_paths", return_value=paths):
        # A raw UUID object passed as user_id must not raise.
        memory_data = storage.load(agent_name=None, user_id=user_uuid)

    assert memory_data is not None
    assert memory_data["user"]["workContext"]["summary"] == "Behavioral researcher"
    assert any(f["content"] == "User studies EPM anxiety behavior" for f in memory_data["facts"])
