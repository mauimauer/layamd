# Correctness

An inference engine is only useful if it computes the same function as the model it claims
to run. layamd checks this at three levels.

## 1. Golden tensors, layer by layer

`ref/gen_golden.py` runs upstream's own PyTorch code (`ref/rl_common.py` and
`ref/rl_agent_api.py`, unmodified) on CPU in fp32 for nine representative cases:

- every question type (choice, score, noul);
- option counts of 2, 3, 4, 5 and 12;
- short, long and full-context states (512 and 1024 tokens);
- several questions packed into one call.

It records the inputs, the embedding output, the residual stream after **every encoder layer
and head layer**, and the final logits. `laya-parity` (`src/test_parity.cpp`) runs each case
twice, once eagerly and once as a replayed HIP graph (18 rows in all), and compares the
results. A row passes when the option probabilities are within 0.01 and the act-head values
within 0.05. The per-layer dumps (`LAYA_DUMP_DIR`) are for finding where a regression starts.

```
scripts/make_golden.sh laya-en golden      # builds the reference image on first use
scripts/parity.sh laya-en golden
# 18/18 rows pass; worst probability delta = 0.001923
```

All three published checkpoints pass 18/18. A checkpoint you fine-tune gets the same check:
generate its golden tensors, then run parity.

## 2. Against upstream's GPU path

Upstream runs bf16 autocast on CUDA. On the same inputs, both paths compared with fp32
PyTorch (`ref/bf16_drift.py` reproduces the right-hand column):

| Case | layamd abs. probability error | upstream bf16 |
|---|---:|---:|
| choice, 12 options (temperature 0.10) | 0.0057 | 0.0602 |
| score, 5 levels, long state | 0.0002 | 0.0051 |
| everything else | ≤ 0.0011 | 0.0003–0.0052 |

The `choice:11+` calibration bucket has a fitted temperature of 0.10, which amplifies any
numeric noise about 10×. That's the model's calibration, not an engine error.

**Questions with more than 64 options** aren't in the golden set.
`bench/eval/check_many_options.py` compares 20 Banking77 questions (77 options) and one
synthetic 120-option question with PyTorch fp32: 21/21 give the same answer, with a worst
probability delta of 0.0026.

## 3. Whole benchmarks, case for case

`bench/eval` reproduces upstream's published benchmark protocol exactly: same rows, prompts,
seed and metric code. That allows a direct comparison with upstream's Tesla T4 run. For both
checkpoints upstream benchmarked, all 98 suite runs (49 suites, 34,832 questions) are within
one correct answer of upstream's accuracy. The single-answer differences come from near-tie
probabilities, not from a systematic shift. See [benchmarks.md](benchmarks.md).

## Rules for changing the engine

- Run `scripts/parity.sh` on **all three** checkpoints. The English and multilingual encoders
  differ in width, depth, vocabulary and context.
- Compare the worst deltas with the previous run, not only pass or fail. A change that
  doesn't touch numerics should leave them identical.
- If you change packing, gathering or the C API, also run `check_many_options.py`, and a
  benchmark suite or two through the server.
