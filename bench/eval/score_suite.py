"""Score a running server on your own labelled JSONL (the laya-train format) and compare runs.

    .venv/bin/python score_suite.py DATA.jsonl --label NAME [--url http://localhost:8080]
                                    [--out results/scores.json] [--concurrency 16]

Rows are {"state", "questions", "gold"} or {"state", "questions", "expected"}, as laya-train
reads them: gold[qid] = {"label": ..., "probabilities": {...}} or expected[qid] = the answer
(a choice key, true/false, or a score level). Each run is appended to --out under NAME, and the
table of all runs in that file is printed, so base vs fine-tune is one command each.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_bench import Client, run_cases
from metrics import suite_metrics

HERE = os.path.dirname(os.path.abspath(__file__))


def option_keys(q):
    if q["type"] == "choice":
        return [str(k) for k in q["criteria"]]
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def answer_index(q, val):
    keys = option_keys(q)
    if q["type"] == "noul":
        if isinstance(val, str):
            val = val.strip().lower() in ("true", "yes", "1")
        return int(bool(val))
    if q["type"] == "score" and not (isinstance(val, str) and val in q["criteria"]):
        return int(float(val))
    if q["type"] == "score":
        return q["criteria"].index(val)
    return keys.index(str(val))


def gold_of(row):
    """{qid: {"idx", "soft"?, "gold_score"?}} from either label form."""
    out = {}
    for qid, q in row["questions"].items():
        g, e = (row.get("gold") or {}).get(qid), (row.get("expected") or {}).get(qid)
        if g is not None:
            keys, probs = option_keys(q), g.get("probabilities") or {}
            soft = [float(probs.get(k, 0.0)) for k in keys]
            idx = answer_index(q, g["label"]) if "label" in g else max(range(len(soft)), key=soft.__getitem__)
            out[qid] = {"idx": idx, **({"soft": soft} if sum(soft) > 0 else {})}
        elif e is not None:
            out[qid] = {"idx": answer_index(q, e)}
        else:
            continue
        if q["type"] == "score":
            out[qid]["gold_score"] = float(out[qid]["idx"])
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("--label", required=True)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "scores.json"))
    ap.add_argument("--concurrency", type=int, default=16)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    cases = [{"state": r["state"], "questions": r["questions"], "gold": gold_of(r)} for r in rows]
    cl = Client(a.url)
    import time; t0 = time.perf_counter()
    res = run_cases(cl, [(c["state"], c["questions"]) for c in cases], a.concurrency)
    m = suite_metrics(cases, {"cases": res, "seconds": time.perf_counter() - t0}, soft=True)
    m.pop("reliability", None); m.pop("coverage_curve", None)
    m["model"] = cl.call("GET", "/info")[1].get("model")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    allr = json.load(open(a.out)) if os.path.exists(a.out) else {}
    allr.setdefault(os.path.basename(a.data), {})[a.label] = m
    json.dump(allr, open(a.out, "w"), indent=1)
    runs = allr[os.path.basename(a.data)]
    cols = [("n", "%d"), ("accuracy", "%.3f"), ("macro_f1", "%.3f"), ("ece", "%.3f"), ("brier", "%.3f"),
            ("acc_at_50_coverage", "%.3f"), ("dropped", "%d")]
    print("%-20s" % os.path.basename(a.data) + "".join("%12s" % c.replace("acc_at_50_coverage", "acc@50%") for c, _ in cols))
    for lab, r in runs.items():
        print("%-20s" % lab + "".join("%12s" % (f % r[c] if c in r else "-") for c, f in cols))
        for t, bt in sorted((r.get("by_type") or {}).items()):
            print("  %-18s" % t + "%12d%12.3f" % (bt["n"], bt["accuracy"]))
