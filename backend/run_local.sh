#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Quick local run of the FastAPI backend (demo mode). No GPU, no model needed.
#
#   ./run_local.sh
#
# Then open frontend/index.html with BACKEND_URL = "http://localhost:8000"
# and BACKEND_TYPE = "fastapi".
# ---------------------------------------------------------------------------
set -e
cd "$(dirname "$0")"
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Starting FastAPI backend on http://localhost:8000  (Ctrl+C to stop)"
uvicorn rest_api:app --host 0.0.0.0 --port 8000
