# bench/eval

Public-benchmark harness and HTML report. Methodology and results are in
[docs/benchmarks.md](../../docs/benchmarks.md).

| File | Does |
|---|---|
| `build_suites.py` | Downloads the datasets (Parquet from the Hugging Face hub) and writes `suites/*.jsonl` |
| `run_bench.py` | Sends every suite to a running server, plus option-order and speed tests → `results/<variant>.json` |
| `run_all.sh` | Starts each checkpoint with `docker compose` and runs `run_bench.py` on it |
| `metrics.py` | Accuracy, macro F1, ECE, Brier, NLL, coverage curves → `report_data.json` |
| `build_report.py` | Embeds `report_data.json` into `report_template.html` → `report.html` |
| `score_suite.py` | Scores a server on your own labelled JSONL and keeps a table of runs |
| `check_many_options.py` | Compares 77- and 120-option questions with PyTorch |
| `upstream/t4_colab_benchmark.json` | Upstream's published T4 results (Apache-2.0, see NOTICE) |

`data/`, `suites/`, `results/` and `.venv/` are generated and not committed.

## Dataset licenses

The datasets are downloaded at run time and never redistributed. Each one keeps its own
license, which you must respect if you use the data:

| Dataset | License (per its Hugging Face card) |
|---|---|
| LocalLLaMA/typed-decisions | Apache-2.0 |
| mteb/amazon_massive_intent, mteb/amazon_massive_scenario | Apache-2.0 (MASSIVE: CC-BY-4.0 upstream) |
| facebook/xnli | not stated on the card (XNLI: CC-BY-NC-4.0 upstream) |
| SetFit/sst5 | not stated on the card |
| dair-ai/emotion | other (research use) |
| deepset/prompt-injections | Apache-2.0 |
| mteb/banking77 | MIT (BANKING77: CC-BY-4.0 upstream) |
| fancyzhx/ag_news | unknown |
| google/boolq | CC-BY-SA-3.0 |
