#!/usr/bin/env bash
#
# deploy-via-tar.sh - Build images locally, ship them to a remote host as a tar,
# load + restart on the remote. No image registry required.
#
# Use case: you have a remote host (ECS / VPS) running 1Panel or plain Docker,
# don't want to pay for an image registry (ACR / GHCR), and don't want to rsync
# the whole repo every deploy. This script ships only:
#   - frontend + backend image tar (compressed, ~300-500 MB)
#   - runtime configs: docker-compose.yaml, nginx/, skills/, config.yaml, extensions_config.json
#
# Prerequisites:
#   - Local: docker (with buildx), ssh, rsync, gzip
#   - Remote: docker, sshd, writable target directory
#   - SSH key auth set up to the remote host
#
# Configuration (env vars, all required unless noted):
#   DEPLOY_HOST       — ssh target, e.g. "deploy@1.2.3.4" or a Host alias in ~/.ssh/config
#   DEPLOY_PATH       — remote directory, e.g. "/opt/ethoinsight"
#   DEPLOY_CONFIG     — path to your production config.yaml (LOCAL path)
#   DEPLOY_EXTENSIONS — path to your production extensions_config.json (LOCAL path)
#
# Optional:
#   DEPLOY_TAG        — image tag, defaults to git short hash
#   SKIP_BUILD=1      — skip docker build (reuse last build)
#   SKIP_SHIP=1       — skip rsync (build only, useful for testing)
#   KEEP_TAR=1        — keep the .tar.gz on local /tmp after deploy
#   SKIP_PRUNE=1      — skip local docker image + build cache prune after deploy
#
# Examples:
#   DEPLOY_HOST=deploy@my-ecs DEPLOY_PATH=/opt/ethoinsight \
#     DEPLOY_CONFIG=$HOME/prod/config.yaml \
#     DEPLOY_EXTENSIONS=$HOME/prod/extensions_config.json \
#     ./scripts/deploy-via-tar.sh
#
#   # Skip rebuild (reuse last local images)
#   SKIP_BUILD=1 ./scripts/deploy-via-tar.sh
#
# Must be run from the repo root (packages/agent/).

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

err() { echo -e "${RED}✗ $*${NC}" >&2; exit 1; }
ok()  { echo -e "${GREEN}✓ $*${NC}"; }
info() { echo -e "${BLUE}→ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }

# ── Validate required env ─────────────────────────────────────────────────────
[ -n "${DEPLOY_HOST:-}" ]       || err "DEPLOY_HOST not set (e.g. deploy@1.2.3.4)"
[ -n "${DEPLOY_PATH:-}" ]       || err "DEPLOY_PATH not set (e.g. /opt/ethoinsight)"
[ -n "${DEPLOY_CONFIG:-}" ]     || err "DEPLOY_CONFIG not set (path to production config.yaml)"
[ -n "${DEPLOY_EXTENSIONS:-}" ] || err "DEPLOY_EXTENSIONS not set (path to production extensions_config.json)"
[ -n "${DEPLOY_AGENT_ENV:-}" ]  || err "DEPLOY_AGENT_ENV not set (path to production .env for gateway/langgraph)"
[ -f "$DEPLOY_CONFIG" ]         || err "DEPLOY_CONFIG file not found: $DEPLOY_CONFIG"
[ -f "$DEPLOY_EXTENSIONS" ]     || err "DEPLOY_EXTENSIONS file not found: $DEPLOY_EXTENSIONS"
[ -f "$DEPLOY_AGENT_ENV" ]      || err "DEPLOY_AGENT_ENV file not found: $DEPLOY_AGENT_ENV"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEPLOY_TAG="${DEPLOY_TAG:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d-%H%M%S)}"
TAR_PATH="/tmp/ethoinsight-${DEPLOY_TAG}.tar.gz"

# Compose builds these image names (project=deer-flow, service names from docker-compose.yaml).
# The deployment runs Gateway-embedded mode: the agent runtime lives inside the gateway
# service (uvicorn app.gateway.app:app) — there is NO standalone LangGraph container in
# docker-compose.yaml (跟随上游 Gateway 化). So only two app images are built/shipped:
# deer-flow-frontend and deer-flow-gateway. (Historical note: an earlier topology had a
# separate deer-flow-langgraph service; it was removed when the runtime moved to
# Gateway-embedded, so this script no longer retags/ships a langgraph image.)
FRONTEND_IMAGE="deer-flow-frontend"
BACKEND_IMAGE="deer-flow-gateway"

# ── Step 1: build images ─────────────────────────────────────────────────────
if [ "${SKIP_BUILD:-0}" != "1" ]; then
    info "[1/5] Building images locally (linux/amd64)"
    DOCKER_DEFAULT_PLATFORM=linux/amd64 \
        ./scripts/deploy.sh build
    ok "Build complete"
else
    warn "[1/5] SKIP_BUILD=1, reusing existing local images"
    # Sanity check: images must exist
    docker image inspect "${FRONTEND_IMAGE}:latest" >/dev/null 2>&1 \
        || err "${FRONTEND_IMAGE}:latest not found; run without SKIP_BUILD first"
    docker image inspect "${BACKEND_IMAGE}:latest" >/dev/null 2>&1 \
        || err "${BACKEND_IMAGE}:latest not found; run without SKIP_BUILD first"
fi

# ── Step 2: tag and save ─────────────────────────────────────────────────────
info "[2/5] Tagging and exporting images (tag: ${DEPLOY_TAG})"
docker tag "${FRONTEND_IMAGE}:latest" "${FRONTEND_IMAGE}:${DEPLOY_TAG}"
docker tag "${BACKEND_IMAGE}:latest"  "${BACKEND_IMAGE}:${DEPLOY_TAG}"

# Save both :latest and :tag — the compose file references implicit ":latest" by
# default, but pinning the tag lets the remote roll back.
docker save \
    "${FRONTEND_IMAGE}:latest" "${FRONTEND_IMAGE}:${DEPLOY_TAG}" \
    "${BACKEND_IMAGE}:latest"  "${BACKEND_IMAGE}:${DEPLOY_TAG}" \
    | gzip -c > "$TAR_PATH"

TAR_SIZE_MB=$(du -m "$TAR_PATH" | cut -f1)
ok "Image tar written: $TAR_PATH (${TAR_SIZE_MB} MB)"

# ── Step 3: ship image tar + runtime configs ─────────────────────────────────
if [ "${SKIP_SHIP:-0}" = "1" ]; then
    warn "[3/5] SKIP_SHIP=1, stopping after build+save (tar at $TAR_PATH)"
    exit 0
fi

info "[3/5] Ensuring remote target directories exist"
ssh "$DEPLOY_HOST" "mkdir -p '$DEPLOY_PATH/images' '$DEPLOY_PATH/docker' '$DEPLOY_PATH/skills' '$DEPLOY_PATH/runtime' '$DEPLOY_PATH/scripts'"

info "[4/5] Shipping image tar and runtime files to ${DEPLOY_HOST}:${DEPLOY_PATH}"

# 4a. Image tar — into images/ subdir, named with tag so multiple versions coexist
rsync -avz --progress "$TAR_PATH" "$DEPLOY_HOST:$DEPLOY_PATH/images/"

# 4b. Compose file + nginx config (under docker/)
rsync -avz --delete docker/docker-compose.yaml "$DEPLOY_HOST:$DEPLOY_PATH/docker/"
# nginx.conf must be a file (not a directory) at docker/nginx/nginx.conf on the remote
# Use explicit file transfer to avoid rsync creating a directory named nginx.conf
ssh "$DEPLOY_HOST" "mkdir -p '$DEPLOY_PATH/docker/nginx'"
rsync -avz docker/nginx/nginx.conf "$DEPLOY_HOST:$DEPLOY_PATH/docker/nginx/nginx.conf"

# 4c. Skills directory (read-only mount inside containers)
rsync -avz --delete --exclude='__pycache__' --exclude='*.pyc' \
    skills/ "$DEPLOY_HOST:$DEPLOY_PATH/skills/"

# 4c-bis. DB migration runner — shipped so the remote can run Alembic inside the
# gateway image before bringing services up (the image bakes in alembic + the
# migration files, but not scripts/). Without this, an existing prod DB stays
# missing any column a sync added → 500s on the hot path (see run-db-migrations.sh).
rsync -avz scripts/run-db-migrations.sh "$DEPLOY_HOST:$DEPLOY_PATH/scripts/run-db-migrations.sh"

# 4d. Runtime configs (config.yaml + extensions_config.json + .env) — never commit secrets to repo
rsync -avz "$DEPLOY_CONFIG"     "$DEPLOY_HOST:$DEPLOY_PATH/runtime/config.yaml"
rsync -avz "$DEPLOY_EXTENSIONS" "$DEPLOY_HOST:$DEPLOY_PATH/runtime/extensions_config.json"
rsync -avz "$DEPLOY_AGENT_ENV"  "$DEPLOY_HOST:$DEPLOY_PATH/.env"

# 4e. frontend/.env — no secrets, just URL config; docker compose requires this file to exist
ssh "$DEPLOY_HOST" "mkdir -p '$DEPLOY_PATH/frontend'"
rsync -avz frontend/.env "$DEPLOY_HOST:$DEPLOY_PATH/frontend/.env"

ok "Files shipped"

# ── Step 5: remote — load images + restart services ──────────────────────────
info "[5/5] Loading images and restarting services on remote"

ssh "$DEPLOY_HOST" bash -s -- "$DEPLOY_PATH" "$DEPLOY_TAG" <<'REMOTE_SCRIPT'
set -euo pipefail
DEPLOY_PATH="$1"
DEPLOY_TAG="$2"

echo "→ Loading images from images/ethoinsight-${DEPLOY_TAG}.tar.gz"
gunzip -c "$DEPLOY_PATH/images/ethoinsight-${DEPLOY_TAG}.tar.gz" | docker load

echo "→ Resolving DEER_FLOW_HOME for compose"
# Persist runtime data outside the image so upgrades don't wipe it
export DEER_FLOW_HOME="$DEPLOY_PATH/deer-flow-home"
mkdir -p "$DEER_FLOW_HOME"

# Compose needs these env vars to resolve volume specs
export DEER_FLOW_CONFIG_PATH="$DEPLOY_PATH/runtime/config.yaml"
export DEER_FLOW_EXTENSIONS_CONFIG_PATH="$DEPLOY_PATH/runtime/extensions_config.json"
export DEER_FLOW_REPO_ROOT="$DEPLOY_PATH"
export DEER_FLOW_DOCKER_SOCKET="${DEER_FLOW_DOCKER_SOCKET:-/var/run/docker.sock}"

# BETTER_AUTH_SECRET: generate once, persist
SECRET_FILE="$DEER_FLOW_HOME/.better-auth-secret"
if [ -f "$SECRET_FILE" ]; then
    export BETTER_AUTH_SECRET="$(cat "$SECRET_FILE")"
else
    BETTER_AUTH_SECRET="$(openssl rand -hex 32)"
    echo "$BETTER_AUTH_SECRET" > "$SECRET_FILE"
    chmod 600 "$SECRET_FILE"
    export BETTER_AUTH_SECRET
    echo "→ Generated BETTER_AUTH_SECRET → $SECRET_FILE"
fi

# HOME for the SSH session may not be set the way compose expects;
# the auth bind mounts default to $HOME/.claude and $HOME/.codex but on a
# headless deployment those aren't relevant — point them at the deploy path
# so compose can resolve them (mount becomes empty but non-fatal).
export HOME="${HOME:-$DEPLOY_PATH}"
mkdir -p "$HOME/.claude" "$HOME/.codex"

echo "→ Restarting services via docker compose"
cd "$DEPLOY_PATH/docker"

# ── DB migrations (BEFORE bringing services up) ──────────────────────────────
# The gateway boots with `bootstrap_schema` (persistence/bootstrap.py), which
# also runs `alembic upgrade head` on versioned DBs — so why migrate at deploy
# time? To FAIL FAST: a migration error aborts the deploy and leaves the old
# (working) containers serving, instead of surfacing as a Gateway startup crash
# after the new code is already live ("migrate before serving" rail).
# We run Alembic inside the freshly-loaded gateway image (it bakes in alembic +
# the migration files) against the host-persisted DB
# (${DEER_FLOW_HOME} → /app/backend/.deer-flow). run-db-migrations.sh is now a
# thin `alembic upgrade head` for versioned DBs; empty/legacy (no
# alembic_version) DBs defer to first-boot bootstrap, which owns provisioning
# (create_all + stamp). No hardcoded HEAD_REV — head is read from the chain.
# Failure here ABORTS the deploy — better to keep the old (working) containers
# running than to bring up new code against an un-migrated schema.
echo "→ Running DB migrations (alembic) inside gateway image before service start"
# The script is bind-mounted :ro (the container must not mutate the shipped
# script), so it is NOT chmod'd +x — instead it is invoked explicitly via `bash`
# (present in python:3.12-slim), which honors the script's `#!/usr/bin/env bash`
# shebang and Bash semantics (BASH_SOURCE, `set -euo pipefail`). Using `sh`
# instead would ignore the shebang and could misinterpret Bash-isms.
docker compose -p deer-flow -f docker-compose.yaml run --rm --no-deps \
    -v "$DEPLOY_PATH/scripts/run-db-migrations.sh:/app/scripts/run-db-migrations.sh:ro" \
    --entrypoint bash \
    gateway /app/scripts/run-db-migrations.sh \
    || { echo "✗ DB migration FAILED — aborting deploy, leaving existing containers running"; exit 1; }
echo "✓ DB migrations applied (or already up to date)"

# Explicitly list services to start — skips 'provisioner' (Kubernetes-only, not needed here).
# --force-recreate is mandatory: compose normally skips rebuilding a container when its
# service spec is unchanged, but that misses two real change classes we hit in practice:
#   (1) env_file (../.env) contents change but the inline `environment:` keys do not
#   (2) the image tag points at a new image ID (e.g. deer-flow-gateway:latest rebuilt
#       to a fresh image) but compose's spec-hash diff doesn't always notice
# Both cases caused silent "deploy succeeded but container still runs old code" incidents
# (see docs/handoffs/2026-05/2026-05-27-channel-todos-bug-resolved-handoff.md).
# A few seconds of unconditional recreate is cheaper than another sleuthing session.
docker compose -p deer-flow -f docker-compose.yaml up -d --remove-orphans --force-recreate \
    frontend gateway nginx

echo "→ Pruning unused images older than 7 days"
docker image prune -af --filter "until=168h" >/dev/null 2>&1 || true

echo "✓ Remote deploy complete (tag: $DEPLOY_TAG)"
REMOTE_SCRIPT

ok "Deploy finished. Tag: $DEPLOY_TAG"
info "Image tar kept at: $TAR_PATH"

# ── Cleanup ───────────────────────────────────────────────────────────────────
if [ "${KEEP_TAR:-0}" != "1" ]; then
    rm -f "$TAR_PATH"
    info "Local tar removed (set KEEP_TAR=1 to retain)"
fi

if [ "${SKIP_PRUNE:-0}" != "1" ]; then
    info "Pruning unused local images and build cache"
    docker image prune -af >/dev/null 2>&1 || true
    docker builder prune -f >/dev/null 2>&1 || true
fi

echo ""
echo "=========================================="
echo "  Next steps on the remote:"
echo "    1Panel → 容器 → 编排  to see status"
echo "    docker logs deer-flow-gateway  to debug"
echo "=========================================="
