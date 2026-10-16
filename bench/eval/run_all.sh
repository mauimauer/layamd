#!/usr/bin/env bash
# Benchmark every checkpoint (quality + speed) against the current laya:gfx1151 image.
#   bench/eval/run_all.sh [laya-en laya-multi laya-typed]
# results/<variant>.json keeps any earlier "speed_shipped" run (the pre-NODELAY server) for comparison.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); EVAL="$ROOT/bench/eval"; cd "$ROOT"
for v in ${@:-laya-en laya-multi laya-typed}; do
  echo "=== $v"
  LAYA_VARIANT=$v docker compose up -d --force-recreate >/dev/null 2>&1
  for i in $(seq 1 90); do curl -sf -m 2 localhost:8080/health >/dev/null && break; sleep 2; done
  "$EVAL/.venv/bin/python" -I "$EVAL/run_bench.py" --variant "$v"
done
