#!/usr/bin/env bash
# One-time setup of Heliograph on Linux or macOS (the equivalent of setup.ps1 on Windows):
# Python virtual environment, backend dependencies, frontend build, NASA dataset and database.
#
#   ./setup.sh                 # typical first run (CPU PyTorch, downloads the dataset)
#   ./setup.sh --cuda          # Linux with an NVIDIA GPU (default PyTorch wheels include CUDA)
#   ./setup.sh --no-data       # skip the dataset download (the reference results still load)
#   PYTHON=python3.12 ./setup.sh
#
# Requires Python 3.11-3.13 and Node.js 18+.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CUDA=0
DATA=1
for arg in "$@"; do
  case "$arg" in
    --cuda) CUDA=1 ;;
    --no-data) DATA=0 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$1"; }
fail() { printf '\nSETUP FAILED: %s\n' "$1" >&2; exit 1; }

step "Locating Python 3.11 - 3.13"
PY=""
for c in "${PYTHON:-}" python3.12 python3.11 python3.13 python3; do
  [ -n "$c" ] || continue
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
[ -n "$PY" ] || fail "Python 3.11-3.13 not found. Install it (https://www.python.org/downloads/) or set PYTHON=/path/to/python."
echo "    using $("$PY" --version) ($(command -v "$PY"))"

step "Checking Node.js 18+"
command -v node >/dev/null 2>&1 || fail "Node.js not found. Install the LTS version from https://nodejs.org/"
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
[ "$NODE_MAJOR" -ge 18 ] || fail "Node.js $NODE_MAJOR found; version 18 or newer is required."
echo "    using node $(node --version)"

step "Creating backend/.venv and installing backend dependencies"
[ -x backend/.venv/bin/python ] || "$PY" -m venv backend/.venv
VPY="$ROOT/backend/.venv/bin/python"
"$VPY" -m pip install --upgrade pip --quiet
if [ "$CUDA" = 1 ] || [ "$(uname -s)" = "Darwin" ]; then
  "$VPY" -m pip install torch
else
  # CPU-only wheels are much smaller than the default CUDA build on Linux.
  "$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
"$VPY" -m pip install -r backend/requirements-dev.txt

step "Installing and building the web UI"
(cd frontend && if [ -f package-lock.json ]; then npm ci --no-audit --no-fund; else npm install --no-audit --no-fund; fi && npm run build)

if [ "$DATA" = 1 ]; then
  step "Downloading and validating the NASA SMAP/MSL benchmark"
  "$VPY" scripts/acquire_data.py || fail "dataset acquisition (see the message above for the manual alternative)"
fi

step "Initialising the database (imports the shipped reference results)"
"$VPY" scripts/init_db.py

printf '\nSetup complete. Start Heliograph with:  ./start.sh\n'
