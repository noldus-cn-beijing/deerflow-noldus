#!/usr/bin/env bash
#
# serve.sh — Unified DeerFlow service launcher
#
# Usage:
#   ./scripts/serve.sh [--dev|--prod] [--daemon] [--stop|--restart]
#
# Modes:
#   --dev       Development mode with hot-reload (default)
#   --prod      Production mode, pre-built frontend, no hot-reload
#   --daemon    Run all services in background (nohup), exit after startup
#
# Actions:
#   --skip-install  Skip dependency installation (faster restart)
#   --stop      Stop all running services and exit
#   --restart   Stop all services, then start with the given mode flags
#
# Runtime: Gateway-embedded (3 processes — Gateway+runtime / Frontend / Nginx).
# The agent runtime runs inside the Gateway API; there is no standalone
# LangGraph server. Nginx rewrites the public /api/langgraph/* prefix onto the
# Gateway's /api/* LangGraph-compatible endpoints.
#
# Examples:
#   ./scripts/serve.sh --dev                 # Dev, hot-reload
#   ./scripts/serve.sh --prod                # Prod, pre-built frontend
#   ./scripts/serve.sh --dev --daemon        # Dev, background
#   ./scripts/serve.sh --stop                # Stop all services
#   ./scripts/serve.sh --restart --dev       # Restart in dev mode
#
# Must be run from the repo root directory.

set -e

REPO_ROOT="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd -P)"
cd "$REPO_ROOT"

# ── Load .env ────────────────────────────────────────────────────────────────

if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

# ── Argument parsing ─────────────────────────────────────────────────────────

DEV_MODE=true
DAEMON_MODE=false
SKIP_INSTALL=false
ACTION="start"   # start | stop | restart

for arg in "$@"; do
    case "$arg" in
        --dev)     DEV_MODE=true ;;
        --prod)    DEV_MODE=false ;;
        --daemon)  DAEMON_MODE=true ;;
        --skip-install) SKIP_INSTALL=true ;;
        --stop)    ACTION="stop" ;;
        --restart) ACTION="restart" ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--dev|--prod] [--daemon] [--skip-install] [--stop|--restart]"
            exit 1
            ;;
    esac
done

# ── Stop helper ──────────────────────────────────────────────────────────────

stop_all() {
    echo "Stopping all services..."
    # Only kill processes owned by current user (避免误杀同机器其他同事的服务)
    pkill -u "$USER" -f "uvicorn app.gateway.app:app" 2>/dev/null || true
    pkill -u "$USER" -f "next dev" 2>/dev/null || true
    pkill -u "$USER" -f "next start" 2>/dev/null || true
    pkill -u "$USER" -f "next-server" 2>/dev/null || true
    # Try graceful nginx quit via the rendered conf if it exists, fall back to default
    if [ -f "$REPO_ROOT/temp/nginx.local.rendered.conf" ]; then
        nginx -c "$REPO_ROOT/temp/nginx.local.rendered.conf" -p "$REPO_ROOT" -s quit 2>/dev/null || true
    else
        nginx -c "$REPO_ROOT/docker/nginx/nginx.local.conf" -p "$REPO_ROOT" -s quit 2>/dev/null || true
    fi
    sleep 1
    pkill -u "$USER" -9 nginx 2>/dev/null || true
    ./scripts/cleanup-containers.sh deer-flow-sandbox 2>/dev/null || true
    echo "✓ All services stopped"
}

# ── Action routing ───────────────────────────────────────────────────────────

if [ "$ACTION" = "stop" ]; then
    stop_all
    exit 0
fi

ALREADY_STOPPED=false
if [ "$ACTION" = "restart" ]; then
    stop_all
    sleep 1
    ALREADY_STOPPED=true
fi

# ── Derive runtime flags ────────────────────────────────────────────────────

# Mode label for banner
if $DEV_MODE; then
    MODE_LABEL="DEV (Gateway runtime, hot-reload enabled)"
else
    MODE_LABEL="PROD (Gateway runtime, optimized)"
fi

if $DAEMON_MODE; then
    MODE_LABEL="$MODE_LABEL [daemon]"
fi

# ── Port configuration (overridable for multi-user shared host) ──────────────
# 多人共享同一台机器开发时，给每人一份偏移避免端口冲突。例如 `PORT_OFFSET=10000 make dev`
# 让所有端口 +10000。也可以单独覆盖某个端口。
PORT_OFFSET="${PORT_OFFSET:-0}"
NGINX_PORT="${NGINX_PORT:-$((2026 + PORT_OFFSET))}"
GATEWAY_PORT="${GATEWAY_PORT:-$((8001 + PORT_OFFSET))}"
FRONTEND_PORT="${FRONTEND_PORT:-$((3000 + PORT_OFFSET))}"

# Frontend command — Next.js reads PORT env var
if $DEV_MODE; then
    FRONTEND_CMD="PORT=$FRONTEND_PORT pnpm run dev"
else
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "Python is required to generate BETTER_AUTH_SECRET."
        exit 1
    fi
    FRONTEND_CMD="env BETTER_AUTH_SECRET=$($PYTHON_BIN -c 'import secrets; print(secrets.token_hex(16))') PORT=$FRONTEND_PORT pnpm run preview"
fi

# Extra flags for uvicorn (Gateway)
if $DEV_MODE && ! $DAEMON_MODE; then
    GATEWAY_EXTRA_FLAGS="--reload --reload-dir=app --reload-dir=packages/harness --reload-include=\*.yaml --reload-include=.env --reload-exclude=\*.pyc --reload-exclude=__pycache__ --reload-exclude=sandbox/\*\* --reload-exclude=.deer-flow/\*\*"
else
    GATEWAY_EXTRA_FLAGS=""
fi

# ── Stop existing services (skip if restart already did it) ──────────────────

if ! $ALREADY_STOPPED; then
    stop_all
    sleep 1
fi

# ── Config check ─────────────────────────────────────────────────────────────

if ! { \
        [ -n "$DEER_FLOW_CONFIG_PATH" ] && [ -f "$DEER_FLOW_CONFIG_PATH" ] || \
        [ -f backend/config.yaml ] || \
        [ -f config.yaml ]; \
    }; then
    echo "✗ No DeerFlow config file found."
    echo "  Run 'make config' to generate config.yaml."
    exit 1
fi

"$REPO_ROOT/scripts/config-upgrade.sh"

# ── Install dependencies ────────────────────────────────────────────────────

if ! $SKIP_INSTALL; then
    echo "Syncing dependencies..."
    (cd backend && uv sync --quiet) || { echo "✗ Backend dependency install failed"; exit 1; }
    (cd frontend && pnpm install --silent) || { echo "✗ Frontend dependency install failed"; exit 1; }
    echo "✓ Dependencies synced"
else
    echo "⏩ Skipping dependency install (--skip-install)"
fi

# ── Sync frontend .env.local ─────────────────────────────────────────────────
# Next.js .env.local takes precedence over process env vars. In Gateway-runtime
# mode the frontend keeps calling the standard /api/langgraph/* prefix and nginx
# rewrites it onto the Gateway's /api/* endpoints, so no override is needed.
# Strip any stale NEXT_PUBLIC_LANGGRAPH_BASE_URL line (e.g. a legacy
# /api/langgraph-compat value from the old transition mode) so the default wins.

FRONTEND_ENV_LOCAL="$REPO_ROOT/frontend/.env.local"
ENV_KEY="NEXT_PUBLIC_LANGGRAPH_BASE_URL"

sync_frontend_env() {
    if [ -f "$FRONTEND_ENV_LOCAL" ] && grep -q "^${ENV_KEY}=" "$FRONTEND_ENV_LOCAL"; then
        sed -i.bak "/^${ENV_KEY}=/d" "$FRONTEND_ENV_LOCAL" && rm -f "${FRONTEND_ENV_LOCAL}.bak"
    fi
}

sync_frontend_env

# ── Banner ───────────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "  Starting DeerFlow"
echo "=========================================="
echo ""
echo "  Mode: $MODE_LABEL"
echo ""
echo "  Services:"
echo "    Gateway     → localhost:$GATEWAY_PORT  (REST API + agent runtime)"
echo "    Frontend    → localhost:$FRONTEND_PORT  (Next.js)"
echo "    Nginx       → localhost:$NGINX_PORT  (reverse proxy)"
echo ""

# ── Cleanup handler ──────────────────────────────────────────────────────────

cleanup() {
    trap - INT TERM
    echo ""
    stop_all
    exit 0
}

trap cleanup INT TERM

# ── Helper: start a service ──────────────────────────────────────────────────

# run_service NAME COMMAND PORT TIMEOUT
# In daemon mode, wraps with nohup. Waits for port to be ready.
run_service() {
    local name="$1" cmd="$2" port="$3" timeout="$4"

    echo "Starting $name..."
    if $DAEMON_MODE; then
        nohup sh -c "$cmd" > /dev/null 2>&1 &
    else
        sh -c "$cmd" &
    fi

    ./scripts/wait-for-port.sh "$port" "$timeout" "$name" || {
        local logfile="logs/$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '-').log"
        echo "✗ $name failed to start."
        [ -f "$logfile" ] && tail -20 "$logfile"
        cleanup
    }
    echo "✓ $name started on localhost:$port"
}

# ── Start services ───────────────────────────────────────────────────────────

mkdir -p logs
mkdir -p temp/client_body_temp temp/proxy_temp temp/fastcgi_temp temp/uwsgi_temp temp/scgi_temp

# 1. Gateway API (embeds the agent runtime — no standalone LangGraph server)
run_service "Gateway" \
    "cd backend && PYTHONPATH=. uv run uvicorn app.gateway.app:app --host 0.0.0.0 --port $GATEWAY_PORT $GATEWAY_EXTRA_FLAGS > ../logs/gateway.log 2>&1" \
    $GATEWAY_PORT 180

# 2. Frontend
run_service "Frontend" \
    "cd frontend && $FRONTEND_CMD > ../logs/frontend.log 2>&1" \
    $FRONTEND_PORT 120

# 3. Nginx — render local conf with current ports (env overrides default 2026/8001/3000)
NGINX_CONF_RENDERED="$REPO_ROOT/temp/nginx.local.rendered.conf"
sed -e "s/listen 2026;/listen $NGINX_PORT;/g" \
    -e "s/listen \[::\]:2026;/listen [::]:$NGINX_PORT;/g" \
    -e "s/127.0.0.1:8001/127.0.0.1:$GATEWAY_PORT/g" \
    -e "s/127.0.0.1:3000/127.0.0.1:$FRONTEND_PORT/g" \
    "$REPO_ROOT/docker/nginx/nginx.local.conf" > "$NGINX_CONF_RENDERED"

run_service "Nginx" \
    "nginx -g 'daemon off;' -c '$NGINX_CONF_RENDERED' -p '$REPO_ROOT' > logs/nginx.log 2>&1" \
    $NGINX_PORT 10

# ── Ready ────────────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "  ✓ DeerFlow is running!  [$MODE_LABEL]"
echo "=========================================="
echo ""
echo "  🌐 http://localhost:$NGINX_PORT"
echo ""
echo "  Routing: Frontend → Nginx → Gateway (embedded runtime)"
echo "  API:     /api/langgraph/*  →  Gateway agent runtime ($GATEWAY_PORT)"
echo "           /api/*              →  Gateway REST API ($GATEWAY_PORT)"
echo ""
echo "  📋 Logs: logs/{gateway,frontend,nginx}.log"
echo ""

if $DAEMON_MODE; then
    echo "  🛑 Stop: make stop"
    # Detach — trap is no longer needed
    trap - INT TERM
else
    echo "  Press Ctrl+C to stop all services"
    wait
fi
