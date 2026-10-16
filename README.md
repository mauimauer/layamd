# layamd

A fast, exact inference server for [Laya](https://huggingface.co/convaiinnovations/laya)
decision models on AMD Strix Halo (Ryzen AI Max, Radeon 8060S, gfx1151).

Laya reads a state (a ticket, an email, a JSON record) and answers typed questions in one
forward pass: pick one of several options, score on a scale, or estimate yes/no, each with
calibrated probabilities. layamd runs it with a custom HIP engine and serves the same HTTP
request and answer format as upstream's `RLAgent.system_one`:

- **Fast.** 4–10 ms per decision over HTTP, up to ~450 questions/s on one chip. Upstream
  PyTorch on a T4 takes 33–40 ms.
- **Exact.** Every published checkpoint matches upstream's fp32 PyTorch layer by layer, and
  matches upstream's benchmark results to within one answer per suite.
- **Trainable on the same box.** There's a recipe for fine-tuning Laya on your own decisions
  on Strix Halo, then serving the result.

> layamd is an independent project. It is not affiliated with or endorsed by ConvAI
> Innovations (the authors of Laya) or AMD.

## Requirements

- A gfx1151 GPU: Strix Halo / Ryzen AI Max 300 series. The engine is compiled for this chip
  only.
- Linux with the `amdgpu` kernel driver (`/dev/kfd` and `/dev/dri` present) and Docker. The
  ROCm user space ships inside the image.
- About 2.5 GB for the three checkpoints.

## Quickstart

```
scripts/fetch_weights.sh                      # laya-en, laya-multi, laya-typed (pinned revision)
echo "VIDEO_GID=$(getent group video | cut -d: -f3)"  >  .env
echo "RENDER_GID=$(getent group render | cut -d: -f3)" >> .env
docker compose up -d --build                  # English checkpoint on :8080

curl -s localhost:8080/v1/decide -H 'content-type: application/json' -d '{
  "state": "Hi, we were billed twice for March. Please refund the duplicate.",
  "questions": {
    "dept":    {"type": "choice", "instructions": "Which team handles this?",
                "criteria": {"billing": "payments, refunds", "tech": "bugs, outages", "other": null}},
    "urgency": {"type": "score", "instructions": "How urgent is it?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "refund":  {"type": "noul", "instructions": "Does the customer ask for a refund?"}}}'
```

Choose the checkpoint with `LAYA_VARIANT=laya-multi` (100+ languages, about 2× faster) or
`laya-typed` (fine-tuned on typed business decisions), or the name of your own fine-tune in
`weights/`. Other endpoints: `GET /health`, and `GET /info` for model and running stats.

## Checkpoints

| Variant | Encoder | Context | Use for |
|---|---|---|---|
| `laya-en` | ModernBERT-large, 421M | 512 | English classification, NLI, guardrails |
| `laya-multi` | mmBERT-base, 322M | 1024 | Anything not in English |
| `laya-typed` | ModernBERT-large, 421M | 1024 | Typed business decisions (the typed-decisions workflows) |

On business decisions the base checkpoints are near chance. Fine-tune them on your own data
([docs/finetuning.md](docs/finetuning.md)).

## Results

| | laya-en | laya-multi | laya-typed |
|---|---:|---:|---:|
| typed-decisions accuracy | 0.362 | 0.352 | 0.766 |
| XNLI accuracy, English / 14 other languages | 0.863 / 0.521 | 0.843 / 0.731 | 0.860 / 0.536 |
| One decision, p50 over HTTP | 10.2 ms | 4.4 ms | 10.1 ms |
| Peak throughput | ~176 q/s | ~450 q/s | ~168 q/s |

Full results, calibration and speed curves are in the interactive report
(`bench/eval/report.html`) and [docs/benchmarks.md](docs/benchmarks.md).

## Documentation

- [Architecture](docs/architecture.md): request path, model, engine, kernels.
- [Optimizations](docs/optimizations.md): what was done, why, what was measured and what was
  rejected.
- [Correctness](docs/correctness.md): golden tensors, parity checks, comparison with upstream.
- [Benchmarks](docs/benchmarks.md): public test sets, results, how to reproduce them.
- [Fine-tuning](docs/finetuning.md): train on your own decisions, check, evaluate and serve.

## Repository

```
src/          HIP engine (C API in laya_api.h), laya-parity, laya-bench
server/       HTTP server, tokenizer and sequence building, batching, answers
ref/          upstream reference code (unmodified) and golden-tensor generator
docker/       engine image and CPU reference image
scripts/      fetch_weights.sh, make_golden.sh, parity.sh
bench/eval/   benchmark harness and report;  bench/probes/: microbenchmarks
finetune/     fine-tuning recipe
docs/         documentation
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). layamd includes and adapts Apache-2.0 code from Laya and
vendors JSON for Modern C++ (MIT); see [NOTICE](NOTICE). The checkpoints and benchmark datasets
are downloaded separately and carry their own licenses.
