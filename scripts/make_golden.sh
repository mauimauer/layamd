#!/usr/bin/env bash
# Regenerate golden reference tensors with upstream's PyTorch code (CPU, fp32).
#   scripts/make_golden.sh <variant|model dir under weights/> [out dir]   e.g. scripts/make_golden.sh laya-en golden
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
v=${1:?variant, e.g. laya-en}; out=${2:-golden-$v}
docker image inspect laya-ref:cpu >/dev/null 2>&1 || docker build -f docker/Dockerfile.reference -t laya-ref:cpu .
mkdir -p "$out"
docker run --rm -v "$ROOT/weights:/weights:ro" -v "$(realpath "$out"):/golden" -e LAYA_MODEL_DIR="/weights/$v" laya-ref:cpu
