#!/usr/bin/env bash
#
# run-db-migrations.sh — Apply DeerFlow application-table Alembic migrations to a
# SQLite database, safely, idempotently, and including the legacy-DB auto-stamp
# path. Designed to run BOTH:
#
#   (a) inside the gateway container on the prod host (via `docker compose run`),
#       invoked automatically by deploy-via-tar.sh before `docker compose up`; and
#   (b) standalone on a local backend checkout (pass --local) for dev DBs.
#
# WHY THIS EXISTS
# ---------------
# This repo's deploy chain does NOT run Alembic, and the gateway boots with
# `Base.metadata.create_all` — which creates MISSING TABLES but never ALTERs an
# existing table to add a column (see persistence/engine.py:153 "Production
# should use Alembic"). So every upstream sync that adds an ORM column leaves the
# already-existing prod/dev `runs`/`feedback`/... tables missing that column, and
# the next request explodes with `sqlite3.OperationalError: no such column`
# (e.g. runs.token_usage_by_model from sync e418d729 → HTTP 500 on /chats/new).
#
# Historically these DBs were built by create_all and have NO `alembic_version`
# table at all, so a plain `alembic upgrade head` would replay from the first
# revision and collide with columns create_all already made. This script handles
# that by STAMPING a legacy DB to the correct baseline (detected by which
# migration-added columns already exist) before upgrading.
#
# TWO ALEMBIC GOTCHAS THIS SCRIPT WORKS AROUND (both bit us in practice):
#   1. migrations/env.py reads ONLY config.get_main_option("sqlalchemy.url")
#      from alembic.ini — it IGNORES `alembic -x sqlalchemy.url=...`. So we do NOT
#      try to pass the URL on the command line; instead we make the ini's RELATIVE
#      default url resolve to the right file (gotcha #2).
#   2. alembic.ini's default url is RELATIVE: sqlite+aiosqlite:///./data/deerflow.db
#      We therefore `cd` to the directory whose ./data/deerflow.db is the target
#      DB before invoking alembic, and also export PWD (database_config.py resolves
#      relative dirs from $PWD, not getcwd).
#
# USAGE
#   In-container (what deploy-via-tar.sh calls):
#       /app/scripts/run-db-migrations.sh
#     Resolves the DB from config.yaml's database.sqlite_dir under the backend dir.
#
#   Local dev checkout:
#       ./scripts/run-db-migrations.sh --local
#     Uses packages/agent/backend as the backend root.
#
# Env overrides:
#   BACKEND_DIR   — backend root (default: auto-detect; /app/backend in container)
#   DB_PATH       — explicit path to deerflow.db (default: derived from sqlite_dir)
#   NO_BACKUP=1   — skip the timestamped .bak copy (NOT recommended for prod)
#   DRY_RUN=1     — print what would happen, change nothing
#
set -euo pipefail

GREEN='\033[0;32m'; BLUE='\033[0;34m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
err()  { echo -e "${RED}✗ $*${NC}" >&2; exit 1; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
info() { echo -e "${BLUE}→ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }

# ── Resolve backend dir ────────────────────────────────────────────────────────
MODE_LOCAL=0
[ "${1:-}" = "--local" ] && MODE_LOCAL=1

if [ -n "${BACKEND_DIR:-}" ]; then
    :
elif [ "$MODE_LOCAL" = "1" ]; then
    BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/backend"
elif [ -d /app/backend ]; then
    BACKEND_DIR=/app/backend            # inside the gateway container
else
    BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/backend"
fi
[ -d "$BACKEND_DIR" ] || err "backend dir not found: $BACKEND_DIR"

HARNESS_DIR="$BACKEND_DIR/packages/harness"
INI="$HARNESS_DIR/deerflow/persistence/migrations/alembic.ini"
[ -f "$INI" ] || err "alembic.ini not found: $INI"

# ── Resolve the DB path ────────────────────────────────────────────────────────
# config.yaml database.sqlite_dir (default .deer-flow/data) is relative to the
# backend dir; the unified DB file is <sqlite_dir>/deerflow.db. config.yaml lives
# at <backend>/config.yaml in the container (compose mount) but at the project
# root (<backend>/../config.yaml) in a local checkout — check both.
if [ -z "${DB_PATH:-}" ]; then
    CONFIG_YAML=""
    for cand in "$BACKEND_DIR/config.yaml" "$BACKEND_DIR/../config.yaml"; do
        [ -f "$cand" ] && { CONFIG_YAML="$cand"; break; }
    done
    SQLITE_DIR=""
    if [ -n "$CONFIG_YAML" ]; then
        SQLITE_DIR="$(grep -A6 '^database:' "$CONFIG_YAML" 2>/dev/null | grep -E '^[[:space:]]*sqlite_dir:' | head -1 | sed -E 's/.*sqlite_dir:[[:space:]]*//; s/[[:space:]]*(#.*)?$//' || true)"
    fi
    SQLITE_DIR="${SQLITE_DIR:-.deer-flow/data}"
    case "$SQLITE_DIR" in
        /*) : ;;                                   # absolute, leave as-is
        *)  SQLITE_DIR="$BACKEND_DIR/$SQLITE_DIR" ;;
    esac
    DB_PATH="$SQLITE_DIR/deerflow.db"
fi

info "Backend dir : $BACKEND_DIR"
info "Database    : $DB_PATH"

# A truly fresh host may not have the DB yet — boot's create_all will build it at
# the latest schema. Nothing to migrate; stamping happens on the next deploy once
# the file exists. (We do NOT create it here to avoid racing the app's WAL setup.)
if [ ! -f "$DB_PATH" ]; then
    warn "DB file does not exist yet — skipping migration (boot create_all will build latest schema)."
    warn "Re-run this script (or the next deploy) after first boot to stamp the version table."
    exit 0
fi

# ── alembic runner: cd into the dir whose ./data/deerflow.db is our target ──────
# alembic.ini default url is sqlite+aiosqlite:///./data/deerflow.db (relative).
# Running from <db_parent_parent> makes ./data/deerflow.db resolve to DB_PATH.
DB_DIR="$(cd "$(dirname "$DB_PATH")/.." && pwd)"   # the dir that contains data/
DB_BASENAME_DIR="$(basename "$(dirname "$DB_PATH")")"
[ "$DB_BASENAME_DIR" = "data" ] || warn "DB parent dir is '$DB_BASENAME_DIR' (expected 'data'); relative-url trick assumes ./data/deerflow.db"

# Prefer the project venv's alembic binary (deterministic, no dependency-resolve
# step — important offline / in-container), then `uv run`, then a PATH alembic.
alembic_cmd() {
    if [ -x "$BACKEND_DIR/.venv/bin/alembic" ]; then
        ( cd "$DB_DIR" && PYTHONPATH="$HARNESS_DIR" PWD="$DB_DIR" "$BACKEND_DIR/.venv/bin/alembic" -c "$INI" "$@" )
    elif command -v uv >/dev/null 2>&1 && [ -f "$BACKEND_DIR/pyproject.toml" ]; then
        ( cd "$DB_DIR" && PYTHONPATH="$HARNESS_DIR" PWD="$DB_DIR" uv run --project "$BACKEND_DIR" alembic -c "$INI" "$@" )
    else
        ( cd "$DB_DIR" && PYTHONPATH="$HARNESS_DIR" PWD="$DB_DIR" alembic -c "$INI" "$@" )
    fi
}

# ── Does alembic_version exist? Which migration-added columns are present? ──────
# We use sqlite3 if available (container has it); else a tiny python via uv.
sql() {
    if command -v sqlite3 >/dev/null 2>&1; then
        sqlite3 "$DB_PATH" "$1"
    else
        ( cd "$BACKEND_DIR" && python3 - "$DB_PATH" "$1" <<'PY'
import sqlite3, sys
con = sqlite3.connect(sys.argv[1]); cur = con.cursor()
for row in cur.execute(sys.argv[2]).fetchall():
    print("|".join("" if c is None else str(c) for c in row))
con.close()
PY
        )
    fi
}

has_table() { [ -n "$(sql "SELECT name FROM sqlite_master WHERE type='table' AND name='$1';")" ]; }
has_col()   { sql "PRAGMA table_info($1);" | grep -qiE "(^|\|)$2(\||$)"; }

# Alembic head revision (newest migration). Bump this every time a migration is
# added under persistence/migrations/versions/ — the final verify and the
# "legacy DB already at head column" branch compare against it.
HEAD_REV="20260626_1700"   # 20260626_1700_thread_cascade_fk (runs/run_events → threads_meta ON DELETE CASCADE)

# Map of (column → revision that introduced it), newest first. The correct
# stamp baseline for a legacy (no alembic_version) DB is the NEWEST revision
# whose column is ALREADY present — i.e. the DB is "at" that revision already.
# NOTE: 20260626_1700 (thread_cascade_fk) adds NO column — it only adds CASCADE
# foreign keys (batch_alter recreates tables). A legacy DB therefore detects at
# most 20260622_1700 (newest column-adding rev) and `upgrade head` then applies
# the FK migration on top. So this map intentionally stops at 20260622_1700.
detect_baseline() {
    # newest → oldest
    if has_table runs     && has_col runs     token_usage_by_model; then echo "20260622_1700"; return; fi
    if has_table feedback && has_col feedback paradigm;             then echo "20260601_1500"; return; fi
    if has_table feedback && has_col feedback verdict;              then echo "20260512_1200"; return; fi
    echo ""   # none of the known columns present → stamp from base (None)
}

if [ "${DRY_RUN:-0}" = "1" ]; then
    info "[DRY_RUN] would back up $DB_PATH, then:"
fi

# ── Backup (prod safety) ────────────────────────────────────────────────────────
if [ "${NO_BACKUP:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    TS="$(date +%Y%m%d-%H%M%S)"
    BAK="${DB_PATH}.bak-${TS}"
    cp "$DB_PATH" "$BAK"
    ok "Backed up DB → $BAK"
fi

# ── Decide path: normal upgrade vs legacy auto-stamp ────────────────────────────
if has_table alembic_version; then
    CUR="$(sql "SELECT version_num FROM alembic_version;" | head -1)"
    info "alembic_version present (current: ${CUR:-<empty>}) — running upgrade head"
    if [ "${DRY_RUN:-0}" = "1" ]; then info "[DRY_RUN] alembic upgrade head"; exit 0; fi
    alembic_cmd upgrade head
else
    BASELINE="$(detect_baseline)"
    if [ -z "$BASELINE" ]; then
        warn "No alembic_version table and none of the known migration columns are present."
        warn "Treating as a pre-scaffold DB: stamping from base and upgrading head."
        if [ "${DRY_RUN:-0}" = "1" ]; then info "[DRY_RUN] alembic stamp base; alembic upgrade head"; exit 0; fi
        alembic_cmd stamp base
        alembic_cmd upgrade head
    elif [ "$BASELINE" = "$HEAD_REV" ]; then
        info "Legacy DB already has the head column set — stamping head (no schema change needed)."
        if [ "${DRY_RUN:-0}" = "1" ]; then info "[DRY_RUN] alembic stamp head"; exit 0; fi
        alembic_cmd stamp "$HEAD_REV"
    else
        info "Legacy DB detected (no alembic_version). Schema is at revision $BASELINE."
        info "Stamping $BASELINE then upgrading to head ($HEAD_REV)."
        if [ "${DRY_RUN:-0}" = "1" ]; then info "[DRY_RUN] alembic stamp $BASELINE; alembic upgrade head"; exit 0; fi
        alembic_cmd stamp "$BASELINE"
        alembic_cmd upgrade head
    fi
fi

# ── Verify ──────────────────────────────────────────────────────────────────────
FINAL="$(sql "SELECT version_num FROM alembic_version;" | head -1)"
if [ "$FINAL" = "$HEAD_REV" ]; then
    ok "Migration complete. alembic_version = $FINAL"
else
    err "Post-migration version is '$FINAL', expected '$HEAD_REV'. Investigate before serving traffic."
fi
