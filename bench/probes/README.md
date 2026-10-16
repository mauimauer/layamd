# Probes

One-off microbenchmarks written while designing the engine. They are not part of the build;
each one answered a question that led to a choice in `docs/optimizations.md`.

| File | Question it answered |
|---|---|
| `gemm.cpp` | Kernel launch overhead and achievable DRAM bandwidth on gfx1151 |
| `skel.cpp` | Achievable latency of the full network shape with dummy weights, and where the time goes |
| `sk.cpp` | Does split-K (fp16 or fp32 partials) help the small-M GEMMs? |
| `lt.cpp` | hipBLAS vs the best hipBLASLt heuristic algorithm, per GEMM shape |
| `pf.cpp` | Cost of a cold (LLC-flushed) GEMM vs prefetched vs cache-hot weights |
| `wgemm_bench.cpp` | Custom WMMA GEMM (`src/wgemm.hip`) vs hipBLAS, and fused GeGLU |

`skel` and `gemm` build in a lean image (run from the repo root):

    docker build -f bench/probes/Dockerfile -t laya-probe:gfx1151 .
    docker run --rm --device=/dev/kfd --device=/dev/dri laya-probe:gfx1151 skel 512 1

The others build the same way inside the engine's builder stage, e.g.
`hipcc -O3 --offload-arch=gfx1151 -std=c++17 lt.cpp -o lt -lhipblas -lhipblaslt`.
