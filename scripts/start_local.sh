#!/usr/bin/env bash
set -euo pipefail
uvicorn api.app:app --host 127.0.0.1 --port 8000
