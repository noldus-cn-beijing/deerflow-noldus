#!/usr/bin/env bash
#
# run-db-migrations.sh — Apply DeerFlow application-table Alembic migrations
# (`alembic upgrade head`) to the SQLite DB, idempotently.
#
# Runs BOTH:
#   (a) inside the gateway container on the prod host (via `docker compose run`),
#       invoked automatically by deploy-via-tar.sh BEFORE `docker compose up`; and
#   (b) standalone on a local backend checkout (pass --local) for dev DBs.
#
# WHY THIS EXISTS
# ---------------
# The gateway boots with `bootstrap_schema` (persistence/bootstrap.py), which on
# an already-versioned DB runs `alembic upgrade head`. That is sufficient on its
# own — so why also run migrations at deploy time? Because the deploy-time run
# FAILS FAST: a migration error aborts the deploy and leaves the old (working)
# containers serving traffic, instead of surfacing as a Gateway startup crash
# after the new code is already live. This is the "migrate before serving" rail
# (memory feedback_deploy_alembic_migration_for_added_columns).
#
# WHY THIS IS NOW A THIN `alembic upgrade head` (vs the old column-detecting
# legacy auto-stamper)
# ------------------------------------------------------------
# We collapsed onto the upstream single-root alembic chain
# (`0001_baseline → 0002_runs_token_usage → 0003_noldus_feedback_ext`), which
# ships a real `0001_baseline` revision. Three DB states are now handled
# uniformly by `bootstrap_schema` at gateway startup:
#   - empty DB        -> create_all + stamp head
#   - legacy DB       -> create_all (baseline backfill) + stamp 0001_baseline + upgrade head
#   - versioned DB    -> upgrade head
# The old script hard-coded `HEAD_REV="20260626_1700"` (our deleted chain head)
# and detected a legacy baseline by probing which migration-added columns were
# present. Neither is needed anymore: the chain has a proper root, and the
# legacy/empty cases are owned by bootstrap at first boot.
#
# This script therefore does ONE thing for a versioned DB: `alembic upgrade
# head`. If the DB has no `alembic_version` table yet (fresh host, never booted,
# or a wiped dev DB), it SKIPS — the first gateway boot's bootstrap owns
# provisioning. Stamping/autogenerating baselines here would race bootstrap and
# risk replaying `0001_baseline.upgrade()` (create_table) against tables
# create_all already made.
#
# TWO ALEMBIC GOTCHAS (unchanged from before):
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

# A truly fresh host has no DB file yet — the first gateway boot's bootstrap
# provisions it (empty branch: create_all + stamp head). Nothing to migrate here;
# running alembic against a non-existent file would just create an empty one and
# race the app's WAL setup.
if [ ! -f "$DB_PATH" ]; then
    warn "DB file does not exist yet — skipping migration (first boot's bootstrap will provision + stamp it)."
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

# ── Is this DB already under alembic management? ────────────────────────────────
# sqlite3 if available (container has it); else a tiny python via uv.
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

if ! has_table alembic_version; then
    # No alembic_version table. The first gateway boot's bootstrap owns this:
    #   - empty DB  -> create_all + stamp head
    #   - legacy DB -> create_all (baseline backfill) + stamp 0001_baseline + upgrade head
    # Running `alembic upgrade head` here would replay 0001_baseline.upgrade()
    # (create_table) and collide with existing tables on a legacy DB. Defer to
    # bootstrap, then the next deploy catches up via the versioned path below.
    warn "No alembic_version table — deferring to first-boot bootstrap (create_all + stamp + upgrade)."
    warn "Re-run this script (or the next deploy) after the first boot to confirm head."
    exit 0
fi

CUR="$(sql "SELECT version_num FROM alembic_version;" | head -1)"
info "alembic_version present (current: ${CUR:-<empty>}) — running upgrade head"

if [ "${DRY_RUN:-0}" = "1" ]; then
    info "[DRY_RUN] alembic upgrade head"
    exit 0
fi

# ── Backup (prod safety) ────────────────────────────────────────────────────────
if [ "${NO_BACKUP:-0}" != "1" ]; then
    TS="$(date +%Y%m%d-%H%M%S)"
    BAK="${DB_PATH}.bak-${TS}"
    cp "$DB_PATH" "$BAK"
    ok "Backed up DB → $BAK"
fi

alembic_cmd upgrade head

# ── Verify we reached head ─────────────────────────────────────────────────────
FINAL="$(alembic_cmd heads | head -1 | awk '{print $1}')"
CUR_AFTER="$(sql "SELECT version_num FROM alembic_version;" | head -1)"
if [ "$CUR_AFTER" = "$FINAL" ]; then
    ok "Migration complete. alembic_version = $CUR_AFTER (head)"
else
    err "Post-migration version is '$CUR_AFTER', expected head '$FINAL'. Investigate before serving traffic."
fi
