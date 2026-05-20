"""Unified database backend configuration.

Controls BOTH the LangGraph checkpointer and the DeerFlow application
persistence layer (runs, threads metadata, users, etc.). The user
configures one backend; the system handles physical separation details.

SQLite mode: checkpointer and app share a single .db file
({sqlite_dir}/deerflow.db) with WAL journal mode enabled on every
connection. WAL allows concurrent readers and a single writer without
blocking, making a unified file safe for both workloads.  Writers
that contend for the lock wait via the default 5-second sqlite3
busy timeout rather than failing immediately.

Postgres mode: both use the same database URL but maintain independent
connection pools with different lifecycles.

Memory mode: checkpointer uses MemorySaver, app uses in-memory stores.
No database is initialized.

Sensitive values (postgres_url) should use $VAR syntax in config.yaml
to reference environment variables from .env:

    database:
      backend: postgres
      postgres_url: $DATABASE_URL

The $VAR resolution is handled by AppConfig.resolve_env_variables()
before this config is instantiated -- DatabaseConfig itself does not
need to do any environment variable processing.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field


@lru_cache(maxsize=128)
def _resolve_sqlite_dir(raw: str) -> str:
    """Module-level cached resolver for sqlite_dir paths.

    Why module-level (vs. an instance attribute):
        AppConfig.from_file() is called on every request that needs the local
        provider (e.g. langgraph_auth.authenticate). Each call constructs a
        fresh DatabaseConfig, so an instance-level cache would still recompute
        on every request. lru_cache here keys on the raw config string, so the
        *process* only ever resolves each distinct sqlite_dir value once.

    Why we don't call Path.resolve() / os.path.abspath() / os.getcwd():
        All three call os.getcwd under the hood, which langgraph's blockbuster
        middleware (langgraph_api >= 0.7) flags as a blocking call when invoked
        from the ASGI event loop. AppConfig.from_file() runs inside the
        authenticate handler, so we cannot rely on the cache being warm before
        the first ASGI hit — meaning the first miss would still trip
        blockbuster.

        Workaround: read PWD from the environment (set by every POSIX shell and
        propagated through Docker's exec). This is a pure-Python dict lookup,
        not a syscall, so blockbuster does not see it. If PWD is missing or
        does not exist as a directory we fall back to "/" — the only scenarios
        where that matters are pathological (e.g. someone unset PWD before
        starting the process), and they would surface as a clear sqlite file
        path error rather than as silent corruption.
    """
    if os.path.isabs(raw):
        return raw
    cwd = os.environ.get("PWD") or "/"
    return os.path.normpath(os.path.join(cwd, raw))


class DatabaseConfig(BaseModel):
    backend: Literal["memory", "sqlite", "postgres"] = Field(
        default="memory",
        description=("Storage backend for both checkpointer and application data. 'memory' for development (no persistence across restarts), 'sqlite' for single-node deployment, 'postgres' for production multi-node deployment."),
    )
    sqlite_dir: str = Field(
        default=".deer-flow/data",
        description=("Directory for the SQLite database file. Both checkpointer and application data share {sqlite_dir}/deerflow.db."),
    )
    postgres_url: str = Field(
        default="",
        description=(
            "PostgreSQL connection URL, shared by checkpointer and app. "
            "Use $DATABASE_URL in config.yaml to reference .env. "
            "Example: postgresql://user:pass@host:5432/deerflow "
            "(the +asyncpg driver suffix is added automatically where needed)."
        ),
    )
    echo_sql: bool = Field(
        default=False,
        description="Echo all SQL statements to log (debug only).",
    )
    pool_size: int = Field(
        default=5,
        description="Connection pool size for the app ORM engine (postgres only).",
    )

    # -- Derived helpers (not user-configured) --

    @property
    def _resolved_sqlite_dir(self) -> str:
        """Absolute path for sqlite_dir.

        Resolution is cached at module level via _resolve_sqlite_dir, so this
        property is event-loop-safe: at most one Path.resolve() call per
        distinct sqlite_dir value across the process lifetime, and zero calls
        if the configured value is already absolute.
        """
        return _resolve_sqlite_dir(self.sqlite_dir)

    @property
    def sqlite_path(self) -> str:
        """Unified SQLite file path shared by checkpointer and app."""
        return os.path.join(self._resolved_sqlite_dir, "deerflow.db")

    # Backward-compatible aliases
    @property
    def checkpointer_sqlite_path(self) -> str:
        """SQLite file path for the LangGraph checkpointer (alias for sqlite_path)."""
        return self.sqlite_path

    @property
    def app_sqlite_path(self) -> str:
        """SQLite file path for application ORM data (alias for sqlite_path)."""
        return self.sqlite_path

    @property
    def app_sqlalchemy_url(self) -> str:
        """SQLAlchemy async URL for the application ORM engine."""
        if self.backend == "sqlite":
            return f"sqlite+aiosqlite:///{self.sqlite_path}"
        if self.backend == "postgres":
            url = self.postgres_url
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url
        raise ValueError(f"No SQLAlchemy URL for backend={self.backend!r}")
