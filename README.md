# Laya on Strix Halo (gfx1151)

Custom HIP inference engine for [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
(ModernBERT/mmBERT encoder + 2-layer decision head), packaged as a gfx1151-only ROCm container.
HTTP API is a drop-in for upstream `rl_agent_api.RLAgent.system_one`.

## Run
    docker compose up -d                          # English checkpoint
    LAYA_VARIANT=laya-multi docker compose up -d  # or laya-typed

    curl -s localhost:8080/v1/decide -H 'content-type: application/json' -d '{
      "state": "User says: my package never arrived",
      "questions": {"route": {"type": "choice", "instructions": "Which queue?",
                              "criteria": {"billing": null, "shipping": null, "tech": null}}}}'

`GET /health`, `GET /info` (running request/latency stats).

## Layout
    src/        HIP engine: laya.cpp (config + mmap safetensors), kernels.hip, engine.hip, wgemm.hip, C API (laya_api.h)
                test_parity.cpp (golden check), bench_engine.cpp
    server/     laya_server.py (tokenizer + build_sequence port + answer post-processing), app.py (HTTP)
    docker/     Dockerfile (engine), Dockerfile.reference (CPU PyTorch golden generator), probe/skeleton benches
    ref/        upstream rl_common.py / rl_agent_api.py + gen_golden.py
    weights/    laya-en, laya-typed, laya-multi (HF layout)
    golden*/    reference tensors per checkpoint

## Verification
    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 991 \
      -v $PWD/weights:/weights:ro -v $PWD/golden:/golden:ro laya:gfx1151 laya-parity /weights/laya-en /golden
All three checkpoints: 9/9 cases pass (incl. full 512/1024-token context), worst probability
delta vs fp32 PyTorch <= 0.0019. Regenerate golden with `Dockerfile.reference` (see gen_golden.py).

## Performance (p50, in container)
Engine, single question:

| tokens | en / typed | multilingual |
|-------:|-----------:|-------------:|
| 64     | 5.1 ms     | 2.5 ms       |
| 128    | 5.9 ms     | 3.0 ms       |
| 256    | 8.0 ms     | 4.2 ms       |
| 512    | 13.3 ms    | 5.5 ms       |
| 1024   | 29.3 ms (typed) | 12.6 ms |

Engine, batched (256 tokens each): en/typed ~155 questions/s (6.4 ms/question), multilingual
~380 questions/s (2.6 ms/question), flat from 2 to 32 questions per call.

HTTP under concurrent load (`bench/load.py`, a new connection per request; `--mixed` = states of
~60-512 tokens):

| | 1 client | 16-32 clients |
|---|---:|---:|
| en, uniform (225 tok) | 105 req/s, p50 9.3 ms | 173 req/s |
| en, mixed | 70 req/s | 86 req/s |
| multilingual, uniform | 182 req/s, p50 5.3 ms | 417 req/s |
| multilingual, mixed | 98 req/s | 119 req/s |

## Accuracy
All checkpoints: 9/9 golden cases pass eager and graph-replayed (incl. full 512/1024 context).
API answers vs the fp32 PyTorch reference, compared with upstream's own GPU path
(`rl_agent_api` runs bf16 autocast on CUDA) on the same inputs:

| case | engine abs prob error | upstream bf16 |
|---|---:|---:|
| choice, 12 options (T=0.10) | 0.0057 | 0.0602 |
| score, 5 levels, long state | 0.0002 | 0.0051 |
| everything else | <= 0.0011 | 0.0003-0.0052 |

`ref/bf16_drift.py` reproduces the bf16 column. The `choice:11+` bucket's fitted temperature
of 0.10 amplifies any numeric noise ~10x; that is the model's calibration, not the engine.

## Design notes
- Compute-bound prefill, not bandwidth-bound decode: no KV cache, one forward pass.
- **FP32 residual stream**, FP16 GEMM inputs, FP32 accumulation. ModernBERT's residual has outlier
  features up to ~3e4 (fp16 step there is 16); keeping it fp16 gave 1.2% RMS drift by the last layer.
  GEMM outputs stay fp16 and are added into the fp32 stream by one reduce kernel (which also
  fuses the bias and split-K). No quantization.
- **Continuous batching** (server): request threads tokenize in parallel and enqueue; one batcher
  thread drains the queue and makes one engine call. No added wait at low load.
- **Packing** (engine): questions are laid out back to back, each padded only to a multiple of 16
  rows; per-row position / sequence-id arrays drive RoPE and the type embedding, a tile map drives
  attention. Packs are split into **~1024-row GPU passes**: larger packs spill the 32 MB LLC (GeGLU,
  LayerNorm and attention K/V reads go to DRAM) -- 32 x 256 tokens went 9.2 -> 6.5 ms/question.
- **Attention** is a fused flash kernel on the RDNA3 WMMA instructions (`k_flash_wmma`): one wave
  per 16 queries, QK^T and PV as 16x16x16 WMMAs with fp32 accumulation, online softmax in fp32
  registers, Q/K/V read straight from the fused qkv buffer. No S x S matrix in memory; sliding-window
  layers only visit key blocks in their band. vs. GEMM-based attention: 1024 tokens 44 -> 29 ms.
- **Decision head pruning**: after the last head layer's attention, only option-marker and CLS rows
  are read, so its out-projection and FFN run on those rows only: every question's markers packed
  back to back (padded to 64 per pass) plus one CLS row per question. A question takes up to 255
  options (whatever fits `max_len`; upstream's act head encodes the count as k/255).
  `bench/eval/check_many_options.py` checks 77- and 120-option questions against PyTorch.
- **Fused residual + LayerNorm**: each residual update (x += GEMM output + bias, fp32) and the
  LayerNorm that follows it are one row-wise kernel (~2%).
- **GEMM autotuning**: every candidate (hipBLAS with split-K 1/2/4/8 using fp16 partials, and the
  top hipBLASLt heuristic algorithms) is timed per (M bucket, N, K); the fastest is kept. All buckets
  are tuned at startup (~4 s), so dynamic batch sizes never tune on the request path. For M <= 128
  (DRAM-bound in the real pass) candidates are timed streaming through all 28 layers' weights
  instead of one cache-hot matrix -- a cache-hot tuner picked plans ~3% slower in the model.
- **HIP graphs**: each (rows, sequences) shape is captured once and replayed. Measured gain ~0 --
  launch overhead was never the floor -- but it costs nothing.
- Weights are transposed once at load so every GEMM is NN (TN was 1.3-3.3x slower).
- The act/escalate head is saturated in all three checkpoints (logits ~+/-1000-4700), so
  `act_probability` is always 1.0. That is the model, not the engine.

## Knobs (env)
`LAYA_MODEL_DIR`, `LAYA_PASS_TOKENS` (rows per GPU pass, default 1024), `LAYA_MAX_TOKENS` (buffer
capacity, default 8192), `LAYA_BATCH_TOKENS` (server: tokens gathered per batching round, 4096),
`LAYA_GRAPHS=0`, `LAYA_HIPBLASLT=0`, `LAYA_SPLITK=<max splits>`, `LAYA_WGEMM=1`, `LAYA_TUNE_COLD_MAXM` (128), `LAYA_PROFILE=<len>` (+
`LAYA_PROFILE_B=<batch>`; per-stage timings in laya-bench), `LAYA_VERBOSE=1` (autotune choices),
`LAYA_DUMP_DIR` (per-layer residual dumps for debugging).

## Tried and not adopted (measured)
- **LLC weight prefetch** (side stream pulling the next GEMM's weights into the 32 MB cache during
  attention/LayerNorm): 5-40% *slower*. Small-M GEMMs turned out efficiency-bound, not DRAM-bound
  (a 64x5248x1024 GEMM is 64.7 us cold vs 57.9 us with its weights fully cached), so the prefetch
  only added contention.
- **Custom WMMA GEMM** (`src/wgemm.hip`, opt-in `LAYA_WGEMM=1`): correct, 8/4-wave tiles, split-K,
  double-buffered LDS. Peaks ~25-36 TFLOPS vs ~42 for hipBLASLt on the wide projections. Beats the
  libraries only on small-M N=1024 shapes (e.g. 64x1024x2624: 17 vs 26 us in isolation), which nets
  to ~0 in the model.
- **GeGLU fused into the MLP-in GEMM epilogue** (the same kernel with `GEGLU=true`; the [M, 2I]
  intermediate never hits memory): correct, but the fused GEMM is slower than hipBLASLt + a separate
  GeGLU kernel at every M (1024 rows: 400 vs ~305 us; 64 rows: 55 vs ~40 us).

## Where the time goes now (4 x 256 tokens)
GEMMs ~73% (28-43 TFLOPS against a ~45 TFLOPS measured peak for these shapes), attention ~9%,
elementwise ~10%, head ~7%. Moving this further needs a GEMM that beats hipBLASLt on gfx1151
(a Tensile-grade tuning effort), or a smaller/lower-precision model.
