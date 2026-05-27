"""Regression tests for DatabaseConfig.

The single most important invariant exercised here is that property access on
a constructed DatabaseConfig MUST NOT touch the filesystem. langgraph's
blockbuster middleware raises BlockingError if `os.getcwd` (called transitively
by `Path.resolve` or `os.path.abspath`) runs inside the ASGI event loop, which
manifests as 500s on every /threads/* request under multi-user load.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from deerflow.config.database_config import DatabaseConfig, _resolve_sqlite_dir


def test_no_path_resolve_or_getcwd_at_any_step(tmp_path: Path, monkeypatch) -> None:
    """Neither construction nor property access must call Path.resolve / os.getcwd / os.path.abspath."""
    # Clear the lru_cache so this test sees actual call attempts.
    _resolve_sqlite_dir.cache_clear()

    monkeypatch.setenv("PWD", str(tmp_path))

    call_log: list[str] = []

    def trap_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        call_log.append("Path.resolve")
        return self  # type: ignore[return-value]

    def trap_getcwd() -> str:
        call_log.append("os.getcwd")
        return str(tmp_path)

    def trap_abspath(p: str) -> str:
        call_log.append("os.path.abspath")
        return p

    with patch.object(Path, "resolve", trap_resolve), \
         patch("os.getcwd", trap_getcwd), \
         patch("os.path.abspath", trap_abspath):
        cfg = DatabaseConfig(backend="sqlite", sqlite_dir=".deer-flow/data")
        _ = cfg.sqlite_path
        _ = cfg.sqlite_path
        _ = cfg.app_sqlite_path
        _ = cfg.checkpointer_sqlite_path
        _ = cfg.app_sqlalchemy_url

    assert call_log == [], (
        f"DatabaseConfig must avoid filesystem syscalls (blockbuster forbids them "
        f"on the ASGI event loop). Calls observed: {call_log}"
    )


def test_relative_sqlite_dir_joined_to_pwd(tmp_path: Path, monkeypatch) -> None:
    _resolve_sqlite_dir.cache_clear()
    monkeypatch.setenv("PWD", str(tmp_path))
    cfg = DatabaseConfig(backend="sqlite", sqlite_dir=".deer-flow/data")
    assert cfg._resolved_sqlite_dir == str(tmp_path / ".deer-flow/data")
    assert cfg.sqlite_path == str(tmp_path / ".deer-flow/data" / "deerflow.db")


def test_absolute_sqlite_dir_used_as_is(monkeypatch) -> None:
    _resolve_sqlite_dir.cache_clear()
    cfg = DatabaseConfig(backend="sqlite", sqlite_dir="/var/lib/deerflow")
    assert cfg._resolved_sqlite_dir == "/var/lib/deerflow"
    assert cfg.sqlite_path == "/var/lib/deerflow/deerflow.db"


def test_pwd_missing_falls_back_to_root(monkeypatch) -> None:
    _resolve_sqlite_dir.cache_clear()
    monkeypatch.delenv("PWD", raising=False)
    cfg = DatabaseConfig(backend="sqlite", sqlite_dir="rel/path")
    assert cfg._resolved_sqlite_dir == "/rel/path"


def test_sqlalchemy_url_sqlite_backend(monkeypatch) -> None:
    _resolve_sqlite_dir.cache_clear()
    cfg = DatabaseConfig(backend="sqlite", sqlite_dir="/tmp/deerflow-test")
    assert cfg.app_sqlalchemy_url == "sqlite+aiosqlite:////tmp/deerflow-test/deerflow.db"


def test_sqlalchemy_url_postgres_backend_rewrites_driver() -> None:
    cfg = DatabaseConfig(
        backend="postgres",
        postgres_url="postgresql://user:pass@host:5432/deerflow",
    )
    assert cfg.app_sqlalchemy_url == "postgresql+asyncpg://user:pass@host:5432/deerflow"


def test_memory_backend_has_no_sqlalchemy_url() -> None:
    cfg = DatabaseConfig(backend="memory")
    import pytest

    with pytest.raises(ValueError, match="No SQLAlchemy URL for backend"):
        _ = cfg.app_sqlalchemy_url


def test_lru_cache_keys_on_raw_string(monkeypatch, tmp_path: Path) -> None:
    """Repeated resolutions of the same sqlite_dir share a cache slot."""
    _resolve_sqlite_dir.cache_clear()
    monkeypatch.setenv("PWD", str(tmp_path))

    # Construction itself does NOT call _resolve_sqlite_dir (that's the whole
    # point of avoiding blockbuster). Only property access does.
    configs = [DatabaseConfig(backend="sqlite", sqlite_dir=".deer-flow/data") for _ in range(5)]
    assert _resolve_sqlite_dir.cache_info().misses == 0

    # Now exercise the property; first hit misses, the rest are cached hits.
    for cfg in configs:
        _ = cfg.sqlite_path

    info = _resolve_sqlite_dir.cache_info()
    assert info.misses == 1
    assert info.hits >= 4
