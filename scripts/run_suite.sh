#!/usr/bin/env bash
set -euo pipefail
BASE_URL=${1:-http://127.0.0.1:8000}
python bench/run_bench.py --base-url $BASE_URL --endpoint /judge --out artifacts --n 50 --warmup 8 --discard 5
