# CLAUDE.md

layamd is a HIP inference engine and HTTP server for Laya decision models (ModernBERT/mmBERT
encoder plus a 2-layer decision head), compiled for one GPU: gfx1151 (Strix Halo). The HTTP API
is a drop-in for upstream `rl_agent_api.RLAgent.system_one`. Read `docs/architecture.md`
before changing the engine, and `docs/optimizations.md` before proposing a speed-up; most
obvious ideas have been measured, and several were rejected.

## Build, run, check

```
docker compose build                              # engine image laya:gfx1151 (~30 s; compiles src/)
LAYA_VARIANT=laya-en docker compose up -d         # laya-en | laya-multi | laya-typed | <fine-tune>
scripts/parity.sh laya-en golden                  # golden check; expect "18/18 rows pass"
scripts/make_golden.sh <variant> <dir>            # (re)generate golden tensors with PyTorch fp32
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add <video gid> --group-add <render gid> \
  -v $PWD/weights:/weights:ro laya:gfx1151 laya-bench /weights/laya-en     # engine latency
bench/eval/.venv/bin/python bench/eval/check_many_options.py diff <ref.json>  # >64-option check
```

Only the GPU path exists, so nothing builds or runs without gfx1151 and `/dev/kfd`. Golden
tensors live in `golden/`, `golden-multi/` and `golden-typed/` (generated, gitignored except
`manifest.json`). Weights come from `scripts/fetch_weights.sh`.

## Rules

- **Correctness first.** After any change under `src/` or to `server/laya_server.py`'s
  sequence building, run parity on all three checkpoints and compare the worst deltas with
  the previous run. A change that shouldn't touch numerics must leave them identical. The
  current values are 0.001923 (en), 0.000824 (multi) and 0.000760 (typed).
- **Measure speed claims** with `laya-bench` (engine) or `bench/eval/run_bench.py --only
  speed` (HTTP). Compare against a baseline taken in the same session, because absolute
  numbers drift by about ±10% between days.
- Keep the fp32 residual stream. ModernBERT's residual has outliers near 3e4, and fp16 there
  breaks accuracy (docs/optimizations.md#numerics).
- Don't allocate on the request path. Buffers are sized for `LAYA_MAX_TOKENS` at load, and
  HIP graphs replay fixed addresses. New per-pass inputs need a pinned host buffer, a device
  buffer, an `h2d` copy in `run()`, and the graph key must cover any shape they change.
- `ref/rl_common.py` and `ref/rl_agent_api.py` are upstream files and stay unmodified (see
  NOTICE). `server/laya_server.py` ports their behaviour; keep it in step with them.
- The engine implements the `sequential` option layout only. `finetune/export.py` rejects
  anything else.
- The server must keep writing each response in one `send` with `TCP_NODELAY`; splitting the
  write brings back a ~40 ms stall on keep-alive connections.

## Map

- `src/engine.hip`: `Engine` (load, buffers, GEMM plans and autotuning, `run()` for one pass,
  `forward_batch()` packing, HIP graphs) and the C API. `src/kernels.hip`: attention (WMMA
  flash), fused residual+LayerNorm, RoPE, GeGLU, gather, act head. `src/wgemm.hip`: opt-in
  custom GEMM. `src/laya.cpp`: config and safetensors mmap.
- `server/app.py`: HTTP. `server/laya_server.py`: tokenizer, `build_sequence`, `Batcher`
  (continuous batching), temperature and answer post-processing.
- `bench/eval/`: benchmark harness and report (docs/benchmarks.md). `bench/probes/`:
  microbenchmarks.
- `finetune/`: training image, `train.sh`, `export.py` (docs/finetuning.md).

## Repo conventions

- Code style is dense and comment-light: comments explain *why* (measurements, hardware
  facts), not what. Match the surrounding file.
- Docs state measured numbers with their conditions. When a number changes, update
  docs/benchmarks.md and the README table together.
- Commits are small and topical, and the message says what was measured.
