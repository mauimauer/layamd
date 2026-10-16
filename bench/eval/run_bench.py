"""Run every suite plus the speed protocols against a running laya server.

    .venv/bin/python run_bench.py --variant laya-en [--url http://localhost:8080] [--only quality|speed]
    -> results/<variant>.json  (per-case probabilities + timings; report.py turns it into metrics)

Each case is one HTTP request carrying all its questions (as upstream scores a case in one pass).
Answers are turned back into a probability vector in option order; a 400 (options do not fit the
head budget) counts as a dropped question, as upstream does.
"""
import argparse, http.client, json, os, random, statistics, threading, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
SUITES_DIR, RESULTS_DIR = os.path.join(HERE, "suites"), os.path.join(HERE, "results")


class Client:
    """One keep-alive connection per thread."""
    def __init__(self, url):
        u = urlparse(url); self.host, self.port = u.hostname, u.port or 80
        self.local = threading.local()

    def _conn(self):
        if getattr(self.local, "c", None) is None:
            self.local.c = http.client.HTTPConnection(self.host, self.port, timeout=120)
        return self.local.c

    def call(self, method, path, obj=None):
        body = json.dumps(obj).encode() if obj is not None else None
        for attempt in (0, 1):
            c = self._conn()
            try:
                t0 = time.perf_counter()
                c.request(method, path, body, {"Content-Type": "application/json"} if body else {})
                r = c.getresponse(); data = r.read()
                return r.status, json.loads(data), (time.perf_counter() - t0) * 1000
            except (http.client.HTTPException, ConnectionError):
                self.local.c = None
                if attempt: raise

    def decide(self, state, questions):
        return self.call("POST", "/v1/decide", {"state": state, "questions": questions})


def prob_vector(qdef, ans):
    if qdef["type"] == "choice":
        keys = list(qdef["criteria"]) if isinstance(qdef["criteria"], dict) else list(qdef["criteria"])
        return [ans["probabilities"][k] for k in keys]
    if qdef["type"] == "score":
        return [ans["probabilities"][str(i)] for i in range(len(qdef["criteria"]))]
    return [round(1 - ans["noul"], 4), ans["noul"]]


def run_cases(cl, cases, conc):
    """cases: [(state, questions)] -> [{"p": {qid: [...]}, "ms": client ms, "srv_ms", "tok"} | {"error"}]"""
    def one(case):
        state, qs = case
        code, out, ms = cl.decide(state, qs)
        if code != 200:
            return {"error": out.get("error", str(code)), "ms": round(ms, 2)}
        return {"p": {qid: prob_vector(qs[qid], a) for qid, a in out["answers"].items()},
                "ms": round(ms, 2), "srv_ms": out["usage"]["latency_ms"], "tok": out["usage"]["input_tokens"]}
    with ThreadPoolExecutor(conc) as ex:
        return list(ex.map(one, cases))


def load_suite(name):
    with open(os.path.join(SUITES_DIR, name + ".jsonl")) as f:
        return [json.loads(l) for l in f]


def quality(cl, conc):
    index = json.load(open(os.path.join(SUITES_DIR, "index.json")))
    out = {}
    for name in index:
        cases = load_suite(name)
        t0 = time.perf_counter()
        res = run_cases(cl, [(c["state"], c["questions"]) for c in cases], conc)
        secs = time.perf_counter() - t0
        out[name] = {"seconds": round(secs, 3), "concurrency": conc, "cases": res}
        nq = sum(len(r["p"]) for r in res if "p" in r)
        print("  %-26s %5d q  %6.1f q/s  %d errors" % (name, nq, nq / secs, sum("error" in r for r in res)), flush=True)
    return out


def order_robustness(cl, conc):
    """Upstream cell 24: shuffle option order (Random(99)) on 200 cases, count changed answers."""
    out = {}
    for name in ("massive_intent.en", "en.emotion", "xnli.en"):
        base = load_suite(name)[:200]
        rng = random.Random(99)
        perm, maps = [], []
        for c in base:
            qid = list(c["questions"])[0]; qdef = c["questions"][qid]
            keys = list(qdef["criteria"]); order = list(range(len(keys))); rng.shuffle(order)
            perm.append((c["state"], {qid: {**qdef, "criteria": {keys[i]: qdef["criteria"][keys[i]] for i in order}}}))
            maps.append((qid, keys, [keys[i] for i in order]))
        a = run_cases(cl, [(c["state"], c["questions"]) for c in base], conc)
        b = run_cases(cl, perm, conc)
        flips = comp = 0
        for ra, rb, (qid, keys, pkeys) in zip(a, b, maps):
            if "p" not in ra or "p" not in rb: continue
            pa, pb = ra["p"][qid], rb["p"][qid]
            comp += 1
            flips += keys[pa.index(max(pa))] != pkeys[pb.index(max(pb))]
        out[name] = {"n": comp, "flip_rate": round(flips / max(1, comp), 4)}
        print("  order robustness %-20s flip rate %.3f" % (name, out[name]["flip_rate"]), flush=True)
    return out


def pct(xs, q):
    xs = sorted(xs); return round(xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))], 2)


def speed(cl, max_len):
    out = {}
    # A. upstream cell 22 protocol: one request with N questions, sequential
    lat_state = {"ticket": {"subject": "Payout failing", "messages": [{"from": "customer",
                 "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6}]}}
    q_noul = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
    q_choice = {"type": "choice", "instructions": "Which team should handle this?",
                "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"}}
    out["questions_per_request"] = {}
    for nq in (1, 2, 5, 10, 20, 50):
        qs = {"q%d" % i: (q_noul if i % 2 else q_choice) for i in range(nq)}
        for _ in range(5): cl.decide(lat_state, qs)
        ts = [cl.decide(lat_state, qs)[2] for _ in range(50)]
        out["questions_per_request"][str(nq)] = {"p50_ms": pct(ts, 50), "p95_ms": pct(ts, 95),
                                                 "ms_per_question": round(pct(ts, 50) / nq, 2), "samples": [round(t, 2) for t in ts]}
        print("  %2d questions/request  p50 %7.2f ms  p95 %7.2f ms" % (nq, pct(ts, 50), pct(ts, 95)), flush=True)

    # B. latency vs input length: one 4-way choice, state grown until it fills max_len
    sentence = "The customer reports that order %d arrived damaged and the replacement never shipped. "
    q = {"route": {"type": "choice", "instructions": "Which queue should take this?",
                   "criteria": {"billing": None, "shipping": None, "tech": None, "other": None}}}
    out["input_length"] = []
    for n in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80):
        state = "".join(sentence % (1000 + i) for i in range(n))
        for _ in range(3): cl.decide(state, q)
        rs = [cl.decide(state, q) for _ in range(30)]
        tok = rs[0][1]["usage"]["input_tokens"]
        if out["input_length"] and tok <= out["input_length"][-1]["tokens"]:
            break                                   # truncated at max_len; longer adds nothing
        ts = [r[2] for r in rs]
        out["input_length"].append({"tokens": tok, "p50_ms": pct(ts, 50), "p95_ms": pct(ts, 95)})
        print("  %4d tokens  p50 %7.2f ms" % (tok, pct(ts, 50)), flush=True)

    # C. concurrency sweep on real benchmark traffic (seeded sample across all suites)
    index = json.load(open(os.path.join(SUITES_DIR, "index.json")))
    pool = [(c["state"], c["questions"]) for n in index for c in load_suite(n)]
    rng = random.Random(7); traffic = [pool[rng.randrange(len(pool))] for _ in range(1600)]
    nq_traffic = [len(t[1]) for t in traffic]
    out["concurrency"] = []
    for conc in (1, 2, 4, 8, 16, 32, 64):
        n = min(len(traffic), max(200, conc * 40))
        run_cases(cl, traffic[:min(64, n)], conc)     # warm connections
        t0 = time.perf_counter(); res = run_cases(cl, traffic[:n], conc); secs = time.perf_counter() - t0
        ts = [r["ms"] for r in res]
        row = {"clients": conc, "requests": n, "req_s": round(n / secs, 1),
               "questions_s": round(sum(nq_traffic[:n]) / secs, 1),
               "p50_ms": pct(ts, 50), "p95_ms": pct(ts, 95), "p99_ms": pct(ts, 99),
               "errors": sum("error" in r for r in res)}
        out["concurrency"].append(row)
        print("  %2d clients  %6.1f req/s  %6.1f q/s  p50 %6.2f  p99 %7.2f ms" %
              (conc, row["req_s"], row["questions_s"], row["p50_ms"], row["p99_ms"]), flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--only", choices=["quality", "speed"])
    ap.add_argument("--speed-key", default="speed", help="result key for the speed run (e.g. speed_shipped)")
    a = ap.parse_args()
    cl = Client(a.url)
    info = cl.call("GET", "/info")[1]
    print("server:", info, flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, a.variant + ".json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    res.update({"variant": a.variant, "info": info, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    if a.only != "speed":
        res["quality"] = quality(cl, a.concurrency)
        res["order_robustness"] = order_robustness(cl, a.concurrency)
    if a.only != "quality":
        res[a.speed_key] = speed(cl, info.get("max_len", 512))
    json.dump(res, open(path, "w"))
    print("wrote", path)
