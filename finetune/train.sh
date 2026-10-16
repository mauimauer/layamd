#!/usr/bin/env bash
# Fine-tune a Laya checkpoint on gfx1151 with upstream's trainer, then export it for the engine.
#
#   finetune/train.sh --name my-ft --data path/train.jsonl [--eval path/test.jsonl] [--base laya-en]
#                     [-- <extra laya-train args, e.g. --epochs 2 --micro-batch 16>]
#
# Paths are relative to the repo root (it is mounted at /work). The result lands in weights/<name>/
# and serves with:  LAYA_VARIANT=<name> docker compose up -d
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
NAME= DATA= EVAL= BASE=laya-en
while [[ $# -gt 0 ]]; do case $1 in
  --name) NAME=$2; shift 2 ;; --data) DATA=$2; shift 2 ;; --eval) EVAL=$2; shift 2 ;;
  --base) BASE=$2; shift 2 ;; --) shift; break ;; *) echo "unknown option $1" >&2; exit 2 ;; esac; done
[[ -n $NAME && -n $DATA ]] || { sed -n '4,6p' "$0"; exit 2; }
[[ -e weights/$NAME ]] && { echo "weights/$NAME already exists" >&2; exit 1; }
docker image inspect laya-train:gfx1151 >/dev/null 2>&1 || docker build -f finetune/Dockerfile -t laya-train:gfx1151 .
mkdir -p weights/"$NAME"
EVAL_ARG=; [[ -n $EVAL ]] && EVAL_ARG="--eval /work/$EVAL"

# Base weights are mounted read-only and copied: laya.train rewrites tokenizer_config.json in place.
docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add "$(getent group video | cut -d: -f3)" --group-add "$(getent group render | cut -d: -f3)" \
  --security-opt seccomp=unconfined --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/tmp/hf \
  -e TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-0}" \
  -v "$ROOT:/work" -v "$ROOT/weights/$BASE:/base:ro" -v "$ROOT/weights/$NAME:/out" \
  laya-train:gfx1151 bash -c '
    set -e
    cp -r /base /tmp/base
    laya-train --data "/work/'"$DATA"'" '"$EVAL_ARG"'  --base /tmp/base --out /out \
               --device cuda '"$*"'
    python /work/finetune/export.py /base /out' 2>&1 | tee "weights/$NAME.train.log"
echo "done: weights/$NAME  (log: weights/$NAME.train.log)"
