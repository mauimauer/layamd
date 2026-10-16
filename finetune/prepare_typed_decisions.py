"""Example data for the recipe: LocalLLaMA/typed-decisions (Apache-2.0) as laya-train JSONL.

    python prepare_typed_decisions.py OUT_DIR [--train-limit N] [--test-limit N]
    -> OUT_DIR/train.jsonl, OUT_DIR/test.jsonl   (rows: {"state", "questions", "gold"})

This is the format your own data should take; see finetune/README.md.
"""
import argparse, json, os, urllib.request
import pyarrow.parquet as pq

REPO = "LocalLLaMA/typed-decisions"


def rows(split):
    urls = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/%s/parquet/all/%s" % (REPO, split)))
    out = []
    for url in urls:
        path, _ = urllib.request.urlretrieve(url)
        out += pq.read_table(path).to_pylist()
    return out


def as_json(v):
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except ValueError:
        return v                       # a plain-text state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--train-limit", type=int)
    ap.add_argument("--test-limit", type=int)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for split, limit in (("train", a.train_limit), ("test", a.test_limit)):
        rs = rows(split)[:limit]
        with open(os.path.join(a.out_dir, split + ".jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({"state": as_json(r["state"]), "questions": as_json(r["questions"]),
                                    "gold": as_json(r["gold"])}, ensure_ascii=False) + "\n")
        print("%s: %d rows, %d questions" % (split, len(rs), sum(len(as_json(r["questions"])) for r in rs)))
