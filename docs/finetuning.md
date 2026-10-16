# Fine-tuning

Base Laya scores near chance on business decisions (0.36 on typed-decisions, where always
picking the most common answer scores 0.46). Fine-tuned on in-domain decisions, it reaches 0.75
(this recipe) to 0.77 (the published checkpoint). If you have a few thousand labelled decisions
from your own workload, fine-tuning is the biggest improvement available, and on Strix Halo it
runs on the same machine that serves.

The recipe uses **upstream's trainer**: `laya-train` from the `laya` package (Apache-2.0),
pinned to 0.4.8. It trains with RLCD, a reinforcement-learning objective built on proper
scoring rules, and fits calibration temperatures afterwards. layamd adds four things: a
training image for gfx1151, a guard for your base weights, an export check against the
engine's requirements, and evaluation through the server you'll actually run.

## Requirements

- The base checkpoint in `weights/` (`scripts/fetch_weights.sh`).
- About 9 GB of disk for the training image. It ships PyTorch built for gfx1151 by AMD, and
  those wheels bundle the ROCm runtime.
- Memory isn't a concern on a 128 GB Strix Halo. Training the 421M model uses a small
  fraction of it.

## 1. Prepare data

One JSON object per line, in the same request shape the server takes, plus the right answer:

```json
{"state": {"ticket": "Charged twice for March, refund please"},
 "questions": {"dept":    {"type": "choice", "instructions": "Which team handles this?",
                           "criteria": {"billing": "payments, refunds", "tech": "bugs", "other": null}},
               "urgency": {"type": "score",  "instructions": "How urgent?",
                           "criteria": ["not urgent", "soon", "blocking"]},
               "refund":  {"type": "noul",   "instructions": "Does the user ask for a refund?"}},
 "expected": {"dept": "billing", "urgency": 1, "refund": true}}
```

Instead of `expected` you can give `gold` with soft labels, for example from several
annotators or a teacher model: `{"dept": {"label": "billing", "probabilities": {"billing":
0.8, "other": 0.2}}}`. For a plain classification CSV, `laya-train` also accepts
`--text-column` and `--label-column`.

**Keep a held-out file.** The trainer fits calibration on a slice of the training data, so
only a separate file tells you whether the model got better.

The worked example below uses typed-decisions:

```
docker build -f finetune/Dockerfile -t laya-train:gfx1151 .
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -v $PWD:/work laya-train:gfx1151 \
  python /work/finetune/prepare_typed_decisions.py /work/finetune/data/td
# train: 1200 rows, 6000 questions / test: 400 rows, 2000 questions
```

## 2. Train

```
finetune/train.sh --name my-ft --base laya-en \
  --data finetune/data/td/train.jsonl --eval finetune/data/td/test.jsonl \
  -- --epochs 4 --max-len 1024 --head-max-len 256 --seed 42
```

What `train.sh` does:

1. Mounts the base checkpoint **read-only** and trains from a copy. Upstream's loader rewrites
   `tokenizer_config.json` in place, and this keeps your `weights/` untouched.
2. Runs `laya-train` on the GPU (`--device cuda`; ROCm builds of PyTorch use that name).
   Everything after `--` goes to `laya-train`. Useful options: `--epochs`, `--micro-batch` and
   `--grad-accum` (the effective batch is their product, 64 by default), `--freeze-encoder` for
   a fast head-only run, and `--shuffle-options` when option order shouldn't matter.
3. Runs `finetune/export.py`. It restores the base checkpoint's exact tokenizer and encoder
   config (fine-tuning doesn't change them), removes the trainer's resume copy, and refuses the
   result if the engine can't serve it. Reasons include a different tensor layout, an
   unsupported dtype, the `parallel` option layout (the engine implements the `sequential`
   layout every published checkpoint uses), or a context longer than the encoder supports.

The result is `weights/my-ft/`, with a log in `weights/my-ft.train.log`.

`--max-len 1024 --head-max-len 256` matches the published typed-decisions checkpoint. The
English base defaults to 512/192. Longer budgets let more of each state and more options in,
but single-question latency roughly doubles at full length (see
[benchmarks.md](benchmarks.md#speed)).

## 3. Check it computes correctly

```
scripts/make_golden.sh my-ft golden-my-ft
scripts/parity.sh my-ft golden-my-ft        # expect 18/18 rows pass
```

## 4. Evaluate

Score the base and the fine-tune on your held-out file **through the server**:

```
LAYA_VARIANT=laya-en docker compose up -d
bench/eval/.venv/bin/python bench/eval/score_suite.py finetune/data/td/test.jsonl --label base
LAYA_VARIANT=my-ft docker compose up -d --force-recreate
bench/eval/.venv/bin/python bench/eval/score_suite.py finetune/data/td/test.jsonl --label my-ft
```

`score_suite.py` prints accuracy, macro F1, calibration error (ECE), Brier score, accuracy on
the most confident half, and a breakdown by question type, with every earlier run in the
same table. Trust these numbers over the trainer's own before/after table. The trainer
evaluates its in-memory fp32 weights, before they're saved as fp16, and its figures can differ
by a few answers. `score_suite.py` measures exactly what you serve, and in our checks it agrees
with PyTorch on the saved weights.

## 5. Serve

```
LAYA_VARIANT=my-ft docker compose up -d
```

## Results on this machine

typed-decisions, laya-en base, 4 epochs, 1024/256 budgets, seed 42:

| Served by layamd, 2,000 held-out questions | Accuracy | Choice | Score | Yes/no | ECE | Acc. on confident half |
|---|---:|---:|---:|---:|---:|---:|
| `laya-en` (base) | 0.362 | 0.290 | 0.323 | 0.487 | 0.174 | 0.427 |
| This recipe's fine-tune | **0.752** | 0.732 | 0.714 | 0.825 | **0.135** | 0.907 |
| Published `laya-typed` | 0.766 | 0.733 | 0.724 | 0.857 | 0.214 | 0.903 |

- 5,600 training questions (400 more held back for calibration), 352 optimizer updates. About
  75 minutes wall clock on an otherwise idle Radeon 8060S, including the trainer's evaluation.
- The trainer's own table reported 0.749. The server measured 0.752 on the exported fp16 weights.
- Parity on the fine-tune: 18/18 rows pass, worst probability delta 0.001681.
- One run, one seed. The recipe lands within 1.4 points of the published checkpoint and is
  better calibrated on this split.

## Notes

- The temperatures the trainer fits are applied by the server exactly as upstream applies
  them: one per question type, unless the config also has per-option-count buckets.
- `checkpoint_latest/` (the trainer's resume copy) is removed by the export. Keep the train
  log if you need to reproduce a run.
- The base checkpoint's tokenizer and encoder config are copied verbatim, so a fine-tune of
  `laya-multi` stays multilingual.
- PyTorch on gfx1151 uses the plain (math) attention path by default.
  `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 finetune/train.sh ...` switches to AMD's
  AOTriton kernels: attention forward plus backward was 3.2× faster at 8×1024 tokens with the
  same error, but a whole 1-epoch run was only about 15% faster. AMD labels these kernels
  experimental, so the recipe leaves them off.
- Keep the GPU free while training. In our runs, serving or benchmarking on the same chip at
  the same time slowed training several-fold.
