#!/usr/bin/env bash
#
# serve.sh — Unified DeerFlow service launcher
#
# Usage:
#   ./scripts/serve.sh [--dev|--prod] [--gateway] [--daemon] [--stop|--restart]
#
# Modes:
#   --dev       Development mode with hot-reload (default)
#   --prod      Production mode, pre-built frontend, no hot-reload
#   --gateway   Gateway mode (experimental): skip LangGraph server,
#               agent runtime embedded in Gateway API
#   --daemon    Run all services in background (nohup), exit after startup
#
# Actions:
#   --skip-install  Skip dependency installation (faster restart)
#   --stop      Stop all running services and exit
#   --restart   Stop all services, then start with the given mode flags
#
# Examples:
#   ./scripts/serve.sh --dev                 # Standard dev (4 processes)
#   ./scripts/serve.sh --dev --gateway       # Gateway dev  (3 processes)
#   ./scripts/serve.sh --prod --gateway      # Gateway prod (3 processes)
#   ./scripts/serve.sh --dev --daemon        # Standard dev, background
#   ./scripts/serve.sh --dev --gateway --daemon  # Gateway dev, background
#   ./scripts/serve.sh --stop                # Stop all services
#   ./scripts/serve.sh --restart --dev --gateway # Restart in gateway mode
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
GATEWAY_MODE=false
DAEMON_MODE=false
SKIP_INSTALL=false
ACTION="start"   # start | stop | restart

for arg in "$@"; do
    case "$arg" in
        --dev)     DEV_MODE=true ;;
        --prod)    DEV_MODE=false ;;
        --gateway) GATEWAY_MODE=true ;;
        --daemon)  DAEMON_MODE=true ;;
        --skip-install) SKIP_INSTALL=true ;;
        --stop)    ACTION="stop" ;;
        --restart) ACTION="restart" ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [--dev|--prod] [--gateway] [--daemon] [--skip-install] [--stop|--restart]"
            exit 1
            ;;
    esac
done

# ── Stop helper ──────────────────────────────────────────────────────────────

stop_all() {
    echo "Stopping all services..."
    # Only kill processes owned by current user (避免误杀同机器其他同事的服务)
    pkill -u "$USER" -f "langgraph dev" 2>/dev/null || true
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

if $GATEWAY_MODE; then
    export SKIP_LANGGRAPH_SERVER=1
fi

# Mode label for banner
if $DEV_MODE && $GATEWAY_MODE; then
    MODE_LABEL="DEV + GATEWAY (experimental)"
elif $DEV_MODE; then
    MODE_LABEL="DEV (hot-reload enabled)"
elif $GATEWAY_MODE; then
    MODE_LABEL="PROD + GATEWAY (experimental)"
else
    MODE_LABEL="PROD (optimized)"
fi

if $DAEMON_MODE; then
    MODE_LABEL="$MODE_LABEL [daemon]"
fi

# ── Port configuration (overridable for multi-user shared host) ──────────────
# 多人共享同一台机器开发时，给每人一份偏移避免端口冲突。例如 `PORT_OFFSET=10000 make dev`
# 让所有端口 +10000。也可以单独覆盖某个端口。
PORT_OFFSET="${PORT_OFFSET:-0}"
NGINX_PORT="${NGINX_PORT:-$((2026 + PORT_OFFSET))}"
LANGGRAPH_PORT="${LANGGRAPH_PORT:-$((2024 + PORT_OFFSET))}"
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

# Extra flags for uvicorn/langgraph
LANGGRAPH_EXTRA_FLAGS="--no-reload"
if $DEV_MODE && ! $DAEMON_MODE; then
    GATEWAY_EXTRA_FLAGS="--reload --reload-dir=app --reload-dir=packages/harness --reload-include=\*.py --reload-include=\*.yaml --reload-include=.env --reload-exclude=\*.pyc --reload-exclude=__pycache__ --reload-exclude=sandbox/\*\* --reload-exclude=.deer-flow/\*\* --reload-exclude=.venv/\*\* --reload-exclude=node_modules/\*\* --reload-exclude=tests/\*\* --reload-exclude=.git/\*\*"
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
# Next.js .env.local takes precedence over process env vars.
# The script manages the NEXT_PUBLIC_LANGGRAPH_BASE_URL line to ensure
# the frontend routes match the active backend mode.

FRONTEND_ENV_LOCAL="$REPO_ROOT/frontend/.env.local"
ENV_KEY="NEXT_PUBLIC_LANGGRAPH_BASE_URL"

sync_frontend_env() {
    if $GATEWAY_MODE; then
        # Point frontend to Gateway's compat API
        if [ -f "$FRONTEND_ENV_LOCAL" ] && grep -q "^${ENV_KEY}=" "$FRONTEND_ENV_LOCAL"; then
            sed -i.bak "s|^${ENV_KEY}=.*|${ENV_KEY}=/api/langgraph-compat|" "$FRONTEND_ENV_LOCAL" && rm -f "${FRONTEND_ENV_LOCAL}.bak"
        else
            echo "${ENV_KEY}=/api/langgraph-compat" >> "$FRONTEND_ENV_LOCAL"
        fi
    else
        # Remove override — frontend falls back to /api/langgraph (standard)
        if [ -f "$FRONTEND_ENV_LOCAL" ] && grep -q "^${ENV_KEY}=" "$FRONTEND_ENV_LOCAL"; then
            sed -i.bak "/^${ENV_KEY}=/d" "$FRONTEND_ENV_LOCAL" && rm -f "${FRONTEND_ENV_LOCAL}.bak"
        fi
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
if ! $GATEWAY_MODE; then
    echo "    LangGraph   → localhost:$LANGGRAPH_PORT  (agent runtime)"
fi
echo "    Gateway     → localhost:$GATEWAY_PORT  (REST API$(if $GATEWAY_MODE; then echo " + agent runtime"; fi))"
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

# 1. LangGraph (skip in gateway mode)
if ! $GATEWAY_MODE; then
    CONFIG_LOG_LEVEL=$(grep -m1 '^log_level:' config.yaml 2>/dev/null | awk '{print $2}' | tr -d ' ')
    LANGGRAPH_LOG_LEVEL="${LANGGRAPH_LOG_LEVEL:-${CONFIG_LOG_LEVEL:-info}}"
    LANGGRAPH_JOBS_PER_WORKER="${LANGGRAPH_JOBS_PER_WORKER:-10}"
    LANGGRAPH_ALLOW_BLOCKING="${LANGGRAPH_ALLOW_BLOCKING:-0}"
    LANGGRAPH_ALLOW_BLOCKING_FLAG=""
    if [ "$LANGGRAPH_ALLOW_BLOCKING" = "1" ]; then
        LANGGRAPH_ALLOW_BLOCKING_FLAG="--allow-blocking"
    fi
    run_service "LangGraph" \
        "cd backend && NO_COLOR=1 BG_JOB_ISOLATED_LOOPS=true uv run langgraph dev --no-browser --host 127.0.0.1 --port $LANGGRAPH_PORT $LANGGRAPH_ALLOW_BLOCKING_FLAG --n-jobs-per-worker $LANGGRAPH_JOBS_PER_WORKER --server-log-level $LANGGRAPH_LOG_LEVEL $LANGGRAPH_EXTRA_FLAGS > ../logs/langgraph.log 2>&1" \
        $LANGGRAPH_PORT 120
else
    echo "⏩ Skipping LangGraph (Gateway mode — runtime embedded in Gateway)"
fi

# 2. Gateway API
run_service "Gateway" \
    "cd backend && PYTHONPATH=. uv run uvicorn app.gateway.app:app --host 0.0.0.0 --port $GATEWAY_PORT $GATEWAY_EXTRA_FLAGS > ../logs/gateway.log 2>&1" \
    $GATEWAY_PORT 180

# 3. Frontend
run_service "Frontend" \
    "cd frontend && $FRONTEND_CMD > ../logs/frontend.log 2>&1" \
    $FRONTEND_PORT 120

# 4. Nginx — render local conf with current ports (env overrides default 2026/2024/8001/3000)
NGINX_CONF_RENDERED="$REPO_ROOT/temp/nginx.local.rendered.conf"
sed -e "s/listen 2026;/listen $NGINX_PORT;/g" \
    -e "s/listen \[::\]:2026;/listen [::]:$NGINX_PORT;/g" \
    -e "s/127.0.0.1:2024/127.0.0.1:$LANGGRAPH_PORT/g" \
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
if $GATEWAY_MODE; then
    echo "  Routing: Frontend → Nginx → Gateway (embedded runtime)"
    echo "  API:     /api/langgraph-compat/*  →  Gateway agent runtime"
else
    echo "  Routing: Frontend → Nginx → LangGraph + Gateway"
    echo "  API:     /api/langgraph/*  →  LangGraph server ($LANGGRAPH_PORT)"
fi
echo "           /api/*              →  Gateway REST API ($GATEWAY_PORT)"
echo ""
echo "  📋 Logs: logs/{langgraph,gateway,frontend,nginx}.log"
echo ""

if $DAEMON_MODE; then
    echo "  🛑 Stop: make stop"
    # Detach — trap is no longer needed
    trap - INT TERM
else
    echo "  Press Ctrl+C to stop all services"
    wait
fi
