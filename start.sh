#!/usr/bin/env bash
# Start Heliograph on Linux or macOS (the equivalent of start.ps1 on Windows).
#
#   ./start.sh            # backend on :8000, built UI (vite preview) on :5173
#   ./start.sh --dev      # UI with hot reload (vite dev server)
#   ./start.sh --single   # one process: FastAPI serves the API and the built UI on :8000
#
# Environment: HELIOGRAPH_DEVICE=auto|cpu|cuda, API_PORT (default 8000), UI_PORT (default 5173).
# Press Ctrl+C to stop everything. Nothing trains automatically.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE=preview
for arg in "$@"; do
  case "$arg" in
    --dev) MODE=dev ;;
    --single) MODE=single ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-5173}"
VPY="$ROOT/backend/.venv/bin/python"
fail() { printf 'START FAILED: %s\n' "$1" >&2; exit 1; }

[ -x "$VPY" ] || fail "backend/.venv not found. Run ./setup.sh first."
if [ "$MODE" != dev ] && [ ! -f frontend/dist/index.html ]; then fail "frontend/dist not built. Run ./setup.sh first (or use --dev)."; fi
if [ "$MODE" != single ] && [ ! -f frontend/node_modules/vite/bin/vite.js ]; then fail "Frontend dependencies missing. Run ./setup.sh first."; fi

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
port_busy "$API_PORT" && fail "Port $API_PORT is already in use (set API_PORT=... to use another port)."
if [ "$MODE" != single ]; then
  port_busy "$UI_PORT" && fail "Port $UI_PORT is already in use (set UI_PORT=... to use another port)."
fi

mkdir -p logs
export PYTHONUNBUFFERED=1 HELIOGRAPH_DEVICE="${HELIOGRAPH_DEVICE:-auto}"
PIDS=()
cleanup() {
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
  echo "Heliograph stopped."
}
trap cleanup EXIT
trap 'exit 0' INT TERM

wait_url() {
  local url="$1" name="$2" pid="$3" log="$4"
  for _ in $(seq 1 120); do
    kill -0 "$pid" 2>/dev/null || { echo "$name exited early. Last log lines:"; tail -n 25 "$log"; exit 1; }
    if curl -fsS -o /dev/null "$url" 2>/dev/null; then return 0; fi
    sleep 0.75
  done
  fail "$name did not respond at $url. See $log"
}

if [ "$MODE" = single ]; then
  export HELIOGRAPH_SERVE_FRONTEND=1
fi
echo "Starting backend (FastAPI on :$API_PORT)..."
(cd backend && exec "$VPY" -m uvicorn app.main:app --host 127.0.0.1 --port "$API_PORT") \
  >logs/backend.out.log 2>logs/backend.err.log &
PIDS+=($!)
wait_url "http://127.0.0.1:$API_PORT/api/health" "Backend" "${PIDS[0]}" logs/backend.err.log

if [ "$MODE" = single ]; then
  echo "  Heliograph ready   http://localhost:$API_PORT   (API docs: http://localhost:$API_PORT/api/docs)"
else
  echo "  backend ready      http://localhost:$API_PORT   (API docs: http://localhost:$API_PORT/docs)"
  echo "Starting web UI (vite $MODE on :$UI_PORT)..."
  (cd frontend && HELIOGRAPH_API_PORT="$API_PORT" exec node node_modules/vite/bin/vite.js "$MODE" \
      --port "$UI_PORT" --strictPort --host 127.0.0.1) >logs/frontend.out.log 2>logs/frontend.err.log &
  PIDS+=($!)
  wait_url "http://127.0.0.1:$UI_PORT/" "Web UI" "${PIDS[1]}" logs/frontend.err.log
  echo "  web UI ready       http://localhost:$UI_PORT"
fi
echo
echo "Heliograph is running. Logs: logs/   Press Ctrl+C to stop."
wait
