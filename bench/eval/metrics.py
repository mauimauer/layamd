"""Turn results/<variant>.json into metrics (same definitions as upstream bench_engine.hard_metrics).

    .venv/bin/python metrics.py  -> report_data.json (consumed by report.html)
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {"laya-en": "laya", "laya-multi": "laya-multilingual", "laya-typed": None}  # ours -> upstream T4 key
P_FLOOR = 5e-5   # the API rounds probabilities to 4 dp; a printed 0.0 is anything below 5e-5


def ece_bins(conf, corr, bins=15):
    edges = np.linspace(0, 1, bins + 1); e = 0.0; out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - corr[sel].mean())
            out.append({"lo": round(lo, 3), "hi": round(hi, 3), "n": int(sel.sum()),
                        "conf": round(float(conf[sel].mean()), 4), "acc": round(float(corr[sel].mean()), 4)})
    return float(e), out


def macro_f1(gold, pred):
    f1 = []
    for c in sorted(set(gold.tolist()) | set(pred.tolist())):
        tp = int(((pred == c) & (gold == c)).sum()); fp = int(((pred == c) & (gold != c)).sum())
        fn = int(((pred != c) & (gold == c)).sum())
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f1))


def hard_metrics(rows, curves=False):
    """rows: [(gold_idx, probs)]"""
    if not rows:
        return {"n": 0}
    gold = np.array([g for g, _ in rows]); probs = [np.asarray(p, float) for _, p in rows]
    pred = np.array([int(np.argmax(p)) for p in probs]); conf = np.array([float(p.max()) for p in probs])
    corr = (pred == gold).astype(float)
    ece, bins = ece_bins(conf, corr)
    o = np.argsort(-conf, kind="stable"); cum_err = np.cumsum(1 - corr[o]) / np.arange(1, len(o) + 1)
    m = {"n": len(rows), "accuracy": float(corr.mean()), "macro_f1": macro_f1(gold, pred), "ece": ece,
         "brier": float(np.mean([((p - np.eye(len(p))[g]) ** 2).sum() for g, p in zip(gold, probs)])),
         "nll": float(np.mean([-math.log(max(float(p[g]), P_FLOOR)) for g, p in zip(gold, probs)])),
         "aurc": float(cum_err.mean()), "mean_confidence": float(conf.mean()),
         "chance": float(np.mean([1 / len(p) for p in probs]))}
    for cov in (0.5, 0.8):
        k = max(1, int(len(conf) * cov)); m["acc_at_%d_coverage" % int(cov * 100)] = float(corr[o[:k]].mean())
    if curves:
        m["reliability"] = bins
        m["coverage_curve"] = [{"coverage": round(c, 2), "accuracy": round(float(corr[o[:max(1, int(len(o) * c))]].mean()), 4)}
                               for c in np.linspace(0.05, 1.0, 20)]
    return m


def suite_metrics(suite_cases, res, soft=False):
    rows, by_type, extra = [], {}, {"soft_acc": [], "tv": [], "score_mae": [], "within_1": []}
    dropped = 0
    for case, r in zip(suite_cases, res["cases"]):
        for qid, g in case["gold"].items():
            if "p" not in r or qid not in r["p"]:
                dropped += 1; continue
            p = np.asarray(r["p"][qid], float); p = p / p.sum()
            rows.append((g["idx"], p)); by_type.setdefault(case["questions"][qid]["type"], []).append((g["idx"], p))
            if soft and "soft" in g and sum(g["soft"]) > 0:
                gp = np.asarray(g["soft"], float); gp = gp / gp.sum()
                extra["soft_acc"].append(float((p * gp).sum())); extra["tv"].append(float(0.5 * np.abs(p - gp).sum()))
            if "gold_score" in g:
                exp = float((np.arange(len(p)) * p).sum())
                extra["score_mae"].append(abs(exp - g["gold_score"])); extra["within_1"].append(float(abs(exp - g["gold_score"]) <= 1))
    m = hard_metrics(rows, curves=True)
    m["dropped"] = dropped
    ok = [r for r in res["cases"] if "p" in r]
    m["questions_per_second"] = round(len(rows) / res["seconds"], 1) if rows else 0
    m["mean_tokens"] = round(float(np.mean([r["tok"] for r in ok])), 1) if ok else 0
    for k, v in extra.items():
        if v: m[k] = float(np.mean(v))
    if len(by_type) > 1:
        m["by_type"] = {t: hard_metrics(rs) for t, rs in sorted(by_type.items())}
    return m


def main():
    index = json.load(open(os.path.join(HERE, "suites", "index.json")))
    cases = {n: [json.loads(l) for l in open(os.path.join(HERE, "suites", n + ".jsonl"))] for n in index}
    wfs = json.load(open(os.path.join(HERE, "suites", "typed_decisions.workflows.json")))
    t4 = json.load(open(os.path.join(HERE, "upstream", "t4_colab_benchmark.json")))
    out = {"suites": index, "variants": {}, "upstream": {"meta": {k: t4["meta"][k] for k in ("gpu", "torch", "timestamp", "seed")},
                                                         "suites": {}, "latency": t4["latency"],
                                                         "option_order_robustness": t4["option_order_robustness"]}}
    for s, per in t4["suites"].items():
        out["upstream"]["suites"][s] = {m: {k: v for k, v in r["calibrated"].items()} | {"questions_per_second": r["questions_per_second"]}
                                        for m, r in per.items() if "calibrated" in r}
    for v in VARIANTS:
        path = os.path.join(HERE, "results", v + ".json")
        if not os.path.exists(path):
            continue
        R = json.load(open(path))
        vm = {"info": R["info"], "timestamp": R["timestamp"], "suites": {}, "speed": R.get("speed"),
              "speed_shipped": R.get("speed_shipped"), "order_robustness": R.get("order_robustness")}
        for n in index:
            q = R["quality"].get(n)
            if q is None: continue
            if all("error" in r for r in q["cases"]):
                vm["suites"][n] = {"n": 0, "error": q["cases"][0]["error"], "dropped": len(q["cases"])}
                continue
            vm["suites"][n] = suite_metrics(cases[n], q, soft=index[n].get("soft", False))
        # typed-decisions per workflow
        td = R["quality"].get("typed_decisions")
        if td:
            per_wf = {}
            for wf, case, r in zip(wfs, cases["typed_decisions"], td["cases"]):
                per_wf.setdefault(wf, ([], []))
                per_wf[wf][0].append(case); per_wf[wf][1].append(r)
            vm["typed_workflows"] = {wf: {k: val for k, val in suite_metrics(c, {"cases": r, "seconds": 1}, soft=True).items()
                                          if k not in ("reliability", "coverage_curve")}
                                     for wf, (c, r) in sorted(per_wf.items())}
        out["variants"][v] = vm
    json.dump(out, open(os.path.join(HERE, "report_data.json"), "w"), separators=(",", ":"))
    # console summary
    print("%-24s" % "suite" + "".join("%22s" % v for v in out["variants"]))
    for n in index:
        line = "%-24s" % n
        for v, vm in out["variants"].items():
            s = vm["suites"].get(n, {}); up = VARIANTS[v] and out["upstream"]["suites"].get(n, {}).get(VARIANTS[v])
            line += "%22s" % (("%.3f (T4 %.3f)" % (s["accuracy"], up["accuracy"])) if s.get("n") and up else
                              ("%.3f" % s["accuracy"]) if s.get("n") else "ERR")
        print(line)
    for v, vm in out["variants"].items():
        if "typed_workflows" in vm:
            print(v, {k: round(w["accuracy"], 3) for k, w in vm["typed_workflows"].items()})


if __name__ == "__main__":
    main()
