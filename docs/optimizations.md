# Optimizations

Every choice here was measured on gfx1151. The table at the end lists what was tried and
rejected, with numbers, so nobody has to repeat it. The microbenchmarks behind several of
these choices are in `bench/probes/`.

## The shape of the problem

Laya does **one prefill and no decoding**: one forward pass over at most 512 or 1024 tokens,
with no KV cache. That makes it compute-bound GEMM work, unlike LLM decoding, which is limited
by memory bandwidth. In a typical pass (4 questions × 256 tokens) the time splits like this:

| Stage | Share |
|---|---:|
| GEMMs | ~73% (28–43 TFLOPS against ~45 measured peak for these shapes) |
| Elementwise (LayerNorm, GeGLU, RoPE, reductions) | ~10% |
| Attention | ~9% |
| Decision head | ~7% |

The engine already runs near the GEMM libraries' limit. Going much further would need a GEMM
faster than hipBLASLt on gfx1151, a smaller model, or lower precision.

## Numerics

**FP32 residual stream, FP16 GEMMs with FP32 accumulation, no quantization.** ModernBERT's
residual stream has outlier features up to about 3e4, where the fp16 step size is 16. Keeping
the residual in fp16 produced 1.2% RMS drift by the last layer. The residual therefore stays
fp32, while GEMM inputs and outputs stay fp16, so the GEMMs still run at fp16 speed. The result
is more accurate than upstream's own GPU path, which uses bf16 autocast
(see [correctness.md](correctness.md)).

## Batching and packing

- **Continuous batching in the server.** Request threads tokenize in parallel; one batcher
  thread drains the queue into one engine call. Nothing waits for a batch to fill, so a lone
  request pays no extra latency.
- **Packing instead of padding.** Questions are laid out back to back and each is padded only
  to 16 rows. A 40-token question next to a 500-token one wastes 8 rows, not 460. Per-row
  position and sequence arrays and a 16-row tile map keep the sequences separate.
- **About 1024-row passes.** Packs larger than this spill the 32 MB last-level cache: GeGLU,
  LayerNorm and attention K/V reads go to DRAM. Splitting 32 × 256 tokens into 1024-row passes
  took it from 9.2 to 6.5 ms per question.
- **Decision-head pruning with packed markers and gathered attention.** In the last head layer,
  only option-marker rows and CLS rows are read after attention. `k_flash_wmma_gather` computes
  attention queries only for those gathered rows (attending over each sequence's full K/V) and
  writes directly into the gathered buffer, skipping non-marker/CLS queries and the post-attention
  gather pass (~10% faster head stage at 1024 tokens). The out-projection, FFN and scorer then
  run on just those rows. Markers are packed per pass (padded to 64), so a 3-option question
  costs 4 rows, not a fixed 65. Before this change the engine reserved 64 slots per question and
  rejected questions with more options. Packing removed that limit (now 255) and raised batched
  throughput by 1–4%.

## Kernels

- **Fused flash attention on WMMA** (`k_flash_wmma`, `k_flash_wmma_gather`). One wave handles
  16 queries. QKᵀ and PV are 16×16×16 WMMA instructions with fp32 accumulation, the softmax runs
  online in fp32 registers, and Q/K/V are read straight from the fused QKV buffer. The S×S score
  matrix never touches memory, and sliding-window layers only visit key blocks inside their band.
  Compared with GEMM-based attention, 1024 tokens went from 44 to 29 ms. The encoder uses a
  10-argument specialization without gather parameters to avoid extra SGPR pressure.
- **Fused residual + LayerNorm** (`k_res_ln`) and **dual-output embedding LayerNorm** (`k_ln`).
  Each residual update (fp16 GEMM output, plus bias and split-K partials, added into fp32) is
  one row-wise kernel together with the LayerNorm that follows it. The final norm also adds the
  type embedding there. At layer 0 (which has no `attn_norm`), `k_ln` writes both the fp32
  residual stream and the fp16 GEMM input in one launch, cutting `enc:layernorm` by ~25%.
- **RoPE in place** on the QKV buffer, so there are no split or transpose copies.
- **`half2`-vectorized bias + activation** (`k_bias_act`) for head QKV bias, head FFN ReLU, and
  scorer GELU, halving thread count and memory transactions.

## GEMMs

- **Autotuned plans.** No single library wins every shape on gfx1151. For each (M bucket, N,
  K) up to `LAYA_PASS_TOKENS` the engine times hipBLAS with split-K 1/2/4/8 and hipBLASLt's top
  heuristic algorithms, and keeps the fastest. All reachable buckets are tuned at startup (~1.6 s
  on `laya-en`/`laya-typed`, ~0.7 s on `laya-multi`), never on the request path.
- **Cold-weight timing for small M.** With M ≤ 128 the real pass streams 28 layers of distinct
  weights from DRAM. Timing candidates on one cache-hot matrix picked plans about 3% slower in
  the model, so small-M candidates are timed streaming through every layer's weights.
- **Split-K with fp16 partials.** Small-M GEMMs make too few output tiles to fill 40 CUs, so
  they run as K-slices. The partials are fp16 because hipBLAS's fp32-output path is 2–4×
  slower on gfx1151; they're summed in fp32.
- **NN layout.** Weights are transposed once at load. TN GEMMs were 1.3–3.3× slower.

## Launch overhead

**HIP graphs.** Each pass shape is captured once and replayed. The measured gain is about
zero, because launch overhead was never the bottleneck, but it costs nothing and keeps CPU
jitter out of the latency.

## Server

- **One write per response, with `TCP_NODELAY`.** Python's `BaseHTTPRequestHandler` writes
  headers and body separately with Nagle's algorithm on. On a keep-alive connection the body
  then waits for the client's delayed ACK, about 40 ms. Every HTTP client that pools connections
  hit this on every request: one decision took 48–51 ms instead of 6–10 ms. Benchmarks that
  opened a new connection per request never saw it, because TCP acknowledges immediately at the
  start of a connection.
- **Shared state tokenization, option LRU cache, and `array('i')` C marshaling.** A multi-question
  request shares one `state` payload across all its questions; `Laya.prepare` serializes and
  tokenizes `state` once per request, caches tokenized option strings (`lru_cache`), and marshals
  flat integer arrays via `array('i')` buffer sharing instead of Python tuple unpacking (~7×
  faster tokenization on multi-question requests, ~2.3× faster ctypes marshaling).

## Tried and not adopted

| Idea | Result |
|---|---|
| **LLC weight prefetch**: a side stream pulls the next GEMM's weights into the cache during attention and LayerNorm | 5–40% *slower*. Small-M GEMMs are limited by efficiency, not DRAM: a 64×5248×1024 GEMM takes 64.7 µs cold vs 57.9 µs with its weights fully cached, so prefetch only added contention. |
| **Custom WMMA GEMM** (`wgemm.hip`, opt-in `LAYA_WGEMM=1`): 8/4-wave tiles, split-K, double-buffered LDS | Correct. Peaks at 25–36 TFLOPS vs ~42 for hipBLASLt on the wide projections. Wins only on small-M N=1024 shapes (64×1024×2624: 17 vs 26 µs in isolation), which nets to about zero in the model. |
| **GeGLU fused into the MLP-in GEMM epilogue** (the [M, 2I] intermediate never reaches memory) | Correct, but slower than hipBLASLt plus a separate GeGLU kernel at every M: 1024 rows 400 vs ~305 µs, 64 rows 55 vs ~40 µs. |
| **Raising the fixed option slots from 64 to 256** | Rejected without building it. Every question would pay for 257 gathered rows; packing does the same job at the actual option count. |
| **First-Fit Decreasing (FFD) bin-packing across passes** | Sorts short sequences into the final pass (`B = 14–18`), pushing `KT` from 64 to 128 and increasing HIP graph cache misses under mixed-length load. Greedy in-order packing is faster. |
| **Rayon `encode_batch` in `Laya.prepare`** | Slightly faster for 1 client (`0.24 -> 0.18 ms` on 20 questions), but under concurrent HTTP load (`c = 4–64`) Rayon wakes all 32 CPU threads across both CCDs, contending with the batcher thread and stealing shared APU package power from the iGPU (~10% lower concurrent `req/s`). |

## Knobs (environment)

| Variable | Default | Effect |
|---|---|---|
| `LAYA_MODEL_DIR` | `/weights/laya-en` | Checkpoint directory |
| `LAYA_PASS_TOKENS` | 1024 | Rows per GPU pass |
| `LAYA_MAX_TOKENS` | 8192 | Buffer capacity in rows |
| `LAYA_BATCH_TOKENS` | 4096 | Server: tokens gathered per batching round |
| `LAYA_GRAPHS=0` | on | Disable HIP graphs |
| `LAYA_HIPBLASLT=0` | on | hipBLAS only |
| `LAYA_SPLITK` | 8 | Maximum split-K factor |
| `LAYA_WGEMM=1` | off | Add the custom WMMA GEMM to the tuning candidates |
| `LAYA_TUNE_COLD_MAXM` | 128 | Largest M bucket timed with cold weights |
| `LAYA_PROFILE=<len>` (+ `LAYA_PROFILE_B`) | off | Per-stage timings in `laya-bench` |
| `LAYA_VERBOSE=1` | off | Print autotuning choices |
| `LAYA_DUMP_DIR` | unset | Dump the residual stream after each layer, for debugging |
