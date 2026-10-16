#!/usr/bin/env bash
# Check the HIP engine against golden tensors (layer by layer + final probabilities).
#   scripts/parity.sh <variant> <golden dir>      e.g. scripts/parity.sh laya-en golden
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
v=${1:?variant}; g=${2:?golden dir}
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined -v "$ROOT/weights:/weights:ro" -v "$(realpath "$g"):/golden:ro" \
  laya:gfx1151 laya-parity "/weights/$v" /golden
