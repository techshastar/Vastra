#!/usr/bin/env bash
# Host the frontend on a tiny local static server (helps with camera
# permissions and fetch requests). Open http://localhost:8080
set -e
cd "$(dirname "$0")"
python3 -m http.server 8080
