# finetune

Fine-tune a Laya checkpoint on your own decisions, on gfx1151, and serve it with layamd.
The full walkthrough is [docs/finetuning.md](../docs/finetuning.md).

| File | Does |
|---|---|
| `Dockerfile` | Training image: PyTorch for gfx1151 (AMD wheels) + `laya==0.4.8` (upstream's trainer) |
| `train.sh` | Runs `laya-train` on a read-only copy of the base, then `export.py` |
| `export.py` | Restores the base tokenizer/encoder config and refuses checkpoints the engine can't serve |
| `prepare_typed_decisions.py` | Example data: LocalLLaMA/typed-decisions as laya-train JSONL |

    finetune/train.sh --name my-ft --data path/train.jsonl --eval path/test.jsonl [--base laya-en] [-- laya-train args]
