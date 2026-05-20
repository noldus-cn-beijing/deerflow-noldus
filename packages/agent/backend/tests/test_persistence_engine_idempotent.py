"""Regression tests for the persistence engine's idempotency guard.

Why this matters:
    `app.gateway.deps.get_local_provider()` calls `init_engine_from_config`
    on every authenticated request (langgraph_auth.authenticate uses it from
    inside an ASGI handler). Without an idempotency guard, init_engine() would
    re-run os.makedirs(sqlite_dir) on every request — which langgraph's
    blockbuster middleware (langgraph_api >= 0.7) flags as a blocking call,
    surfacing as HTTP 500 on every /threads/* endpoint.

    The guard at the top of init_engine() returns early if the engine is
    already initialized, eliminating both the syscall and the wasted engine
    rebuild.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from deerflow.persistence import engine as engine_module


@pytest.fixture(autouse=True)
def _reset_engine_state():
    """Ensure each test starts with a clean global engine state."""
    engine_module._engine = None
    engine_module._session_factory = None
    yield
    engine_module._engine = None
    engine_module._session_factory = None


@pytest.mark.anyio
async def test_init_engine_returns_early_when_already_initialized():
    """Once _engine is set, init_engine must not call os.makedirs again."""
    # Simulate an already-initialized engine.
    engine_module._engine = "SENTINEL_ENGINE"  # type: ignore[assignment]

    call_log: list[str] = []

    def trap_makedirs(*args, **kwargs):
        call_log.append("os.makedirs")

    with patch("os.makedirs", trap_makedirs):
        await engine_module.init_engine(
            backend="sqlite",
            url="sqlite+aiosqlite:////tmp/nope.db",
            sqlite_dir="/tmp/should-not-be-created",
        )

    assert call_log == [], (
        "init_engine must be idempotent: when _engine is already set, "
        "no syscalls (especially os.makedirs) should run. blockbuster "
        f"would 500 on this. Got: {call_log}"
    )


@pytest.mark.anyio
async def test_init_engine_from_config_is_idempotent():
    """The convenience wrapper also short-circuits when engine exists."""
    engine_module._engine = "SENTINEL"  # type: ignore[assignment]

    call_log: list[str] = []

    def trap_makedirs(*args, **kwargs):
        call_log.append("os.makedirs")

    class _FakeConfig:
        backend = "sqlite"
        sqlite_dir = "/tmp/whatever"
        app_sqlalchemy_url = "sqlite+aiosqlite:////tmp/whatever/x.db"
        echo_sql = False
        pool_size = 5

    with patch("os.makedirs", trap_makedirs):
        await engine_module.init_engine_from_config(_FakeConfig())

    assert call_log == []


@pytest.mark.anyio
async def test_first_init_still_runs_setup(tmp_path):
    """The guard must not prevent the first real initialization."""
    db_dir = tmp_path / "data"

    await engine_module.init_engine(
        backend="sqlite",
        url=f"sqlite+aiosqlite:///{db_dir}/deerflow.db",
        sqlite_dir=str(db_dir),
    )

    assert engine_module._engine is not None
    assert db_dir.is_dir()  # makedirs ran on first call
    assert engine_module._session_factory is not None

    # Cleanup the engine before fixture teardown.
    await engine_module.close_engine()
