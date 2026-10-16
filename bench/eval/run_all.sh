#!/usr/bin/env bash
# Benchmark every checkpoint: stock server (speed only) then TCP_NODELAY overlay (quality + speed).
#   bench/eval/run_all.sh [laya-en laya-multi laya-typed]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd); EVAL="$ROOT/bench/eval"; cd "$ROOT"
up() {  # variant, extra compose args...
  local v=$1; shift
  LAYA_VARIANT=$v docker compose -f docker-compose.yml "$@" up -d --force-recreate >/dev/null 2>&1
  for i in $(seq 1 90); do curl -sf -m 2 localhost:8080/health >/dev/null && return; sleep 2; done
  echo "server did not come up" >&2; exit 1
}
for v in "${@:-laya-en laya-multi laya-typed}"; do
  for vv in $v; do
    echo "=== $vv (stock server)";  up "$vv";                               "$EVAL/.venv/bin/python" -I "$EVAL/run_bench.py" --variant "$vv" --only speed --speed-key speed_shipped
    echo "=== $vv (TCP_NODELAY)";   up "$vv" -f bench/eval/compose.nodelay.yml; "$EVAL/.venv/bin/python" -I "$EVAL/run_bench.py" --variant "$vv"
  done
done
