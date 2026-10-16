# Benchmarks

`bench/eval/` measures decision quality and speed on public test sets and builds an
interactive HTML report (`bench/eval/report.html`). Measured on a Ryzen AI Max+ 395 (Radeon
8060S, gfx1151) in October 2026.

## Test sets

The suites reproduce upstream's published benchmark
(`research/scripts/laya_benchmark_colab.ipynb`): same rows, prompts, seed (13), option
sampling and metric code. Each suite therefore lines up case for case with upstream's Tesla T4
results (`bench/eval/upstream/t4_colab_benchmark.json`).

| Family | Suites | Question shape |
|---|---|---|
| typed-decisions (LocalLLaMA) | 1 × 2,000 questions | Business decisions in 4 workflows, mixed choice/score/yes-no against JSON state, soft teacher labels |
| MASSIVE intent / scenario | 14 languages × 300 each | 20-option choice (gold label plus 19 sampled) |
| XNLI | 15 languages × 300 | 3-way choice |
| English | SST-5 (5-level score), DAIR Emotion (6-way), deepset prompt-injections (yes/no), Banking77 (77 options), AG News, BoolQ | |

AG News and BoolQ were in Laya's training mix, so they measure retention, not generalisation.
None of the published checkpoints was trained on MASSIVE, XNLI or typed-decisions. The
typed-decisions checkpoint was trained on typed-decisions' own *training* split.

## Results

| | laya (English) | laya-multilingual | laya-typed-decisions |
|---|---:|---:|---:|
| typed-decisions | 0.362 | 0.352 | **0.766** |
| XNLI, English | 0.863 | 0.843 | 0.860 |
| XNLI, 14 other languages (mean) | 0.521 | **0.731** | 0.536 |
| MASSIVE intent, English | 0.783 | 0.657 | 0.810 |
| AG News / BoolQ (trained on) | 0.948 / 0.830 | 0.937 / 0.788 | 0.947 / 0.822 |
| SST-5 / Emotion / prompt-injections | 0.372 / 0.572 / 0.698 | 0.282 / 0.513 / 0.578 | 0.482 / 0.583 / 0.672 |
| Banking77, 77 options | 0.460 | 0.496 | 0.528 |

What the numbers say:

- **The engine matches upstream.** On the checkpoints upstream benchmarked, every suite is
  within one correct answer of the T4 run ([correctness.md](correctness.md)).
- **Laya out of the box is a strong NLU model and a weak business decider.** Base checkpoints
  score near chance on typed-decisions; the majority-class baseline is 0.461. Fine-tuning on
  in-domain decisions doubles that. Plan to fine-tune ([finetuning.md](finetuning.md)).
- **Route non-English text to the multilingual checkpoint.** The English model drops close to
  chance in Hindi, Korean, Tamil and Swahili while staying confident.
- **Many-label questions are weak.** With 77 options each label gets about 4 tokens. Shortlist
  the labels first, or run a tournament over groups of labels.

The report also covers calibration (reliability diagrams, accuracy versus coverage), option-
order robustness, and per-workflow and per-type typed-decisions results.

## Speed

Measured over keep-alive HTTP from the same host, so tokenization and JSON are included.

| | laya (English) | laya-multilingual | laya-typed-decisions |
|---|---:|---:|---:|
| One decision, p50 | 10.2 ms | 4.4 ms | 10.1 ms |
| 50 questions in one request | 356 ms | 131 ms | 357 ms |
| Full context, one question | 17.5 ms (512 tok) | 16.6 ms (1024 tok) | 38.2 ms (1024 tok) |
| Peak throughput (16–64 clients) | ~176 questions/s | ~449–486 questions/s | ~168 questions/s |

Upstream on a T4, in-process with no HTTP: 39.5 ms for one decision and 771 ms for 50
(English); 32.8 and 337 ms (multilingual).

Engine only (`laya-bench`, no HTTP), p50:

| Tokens | 64 | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|
| laya (English) | 5.7 ms | 6.2 ms | 9.2 ms | 15.4 ms | – |
| laya-multilingual | 2.8 ms | 3.2 ms | 4.9 ms | 6.5 ms | 15.1 ms |

Batched at 256 tokens per question: about 133 questions/s (English) and 324 questions/s
(multilingual), flat from 2 to 32 questions per call.

## Running it

```
cd bench/eval
python3 -m venv .venv && .venv/bin/pip install numpy pyarrow
.venv/bin/python build_suites.py            # downloads datasets, writes suites/ (~5 min)
cd ../.. && docker compose build
bench/eval/run_all.sh                       # all three checkpoints, ~10 min
bench/eval/.venv/bin/python bench/eval/metrics.py
bench/eval/.venv/bin/python bench/eval/build_report.py   # -> bench/eval/report.html
```

To score a checkpoint on **your own data**, see `score_suite.py` in
[finetuning.md](finetuning.md#evaluate).
