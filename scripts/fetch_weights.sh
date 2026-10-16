#!/usr/bin/env bash
# Download Laya checkpoints from Hugging Face into weights/<variant>/ (the layout the engine loads).
#   scripts/fetch_weights.sh [laya-en] [laya-multi] [laya-typed]     (default: all three, ~2.3 GB)
# Checkpoints are Apache-2.0, published by ConvAI Innovations at huggingface.co/convaiinnovations/laya.
set -euo pipefail
REPO=${LAYA_HF_REPO:-convaiinnovations/laya}
REV=${LAYA_HF_REVISION:-7b928d828b7b0e022f929d9bd2e44165aa270148}   # pinned: golden tensors and benchmarks were made with this revision
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FILES="model.safetensors rl_agent_config.json encoder/config.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json"

subfolder() { case $1 in
  laya-en) echo "" ;; laya-multi) echo "multilingual/" ;; laya-typed) echo "typed-decisions/" ;;
  *) echo "unknown variant: $1 (laya-en | laya-multi | laya-typed)" >&2; exit 2 ;; esac; }

for v in ${@:-laya-en laya-multi laya-typed}; do
  sub=$(subfolder "$v"); dst="$ROOT/weights/$v"
  echo "== $v  <- $REPO/${sub:-<root>} @ $REV"
  for f in $FILES; do
    out="$dst/$f"; mkdir -p "$(dirname "$out")"
    url="https://huggingface.co/$REPO/resolve/$REV/$sub$f"
    if [[ -s "$out" ]]; then
      # skip files whose size already matches the hub's
      remote=$(curl -sfIL "$url" | tr -d '\r' | awk 'tolower($1)=="x-linked-size:"{s=$2} tolower($1)=="content-length:"{c=$2} END{print s?s:c}')
      [[ -n "$remote" && "$(stat -c %s "$out")" == "$remote" ]] && { echo "   ok   $f"; continue; }
    fi
    echo "   get  $f"
    curl -fsSL --retry 5 --retry-delay 5 -o "$out.part" "$url" && mv "$out.part" "$out"
  done
done
