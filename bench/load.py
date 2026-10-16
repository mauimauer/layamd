"""Concurrent HTTP load: C client threads, each sending single-question requests."""
import json, os, sys, time, threading, urllib.request, statistics
URL = "http://localhost:8080/v1/decide"
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ref', 'gen_golden.py')).read(); ns = {}; exec(src[src.index("LONG_STATE ="):src.index("def main")], ns)
state = ns['LONG_STATE']
import random
MIXED = "--mixed" in sys.argv
def mixed_state(i):
    r = random.Random(i)
    return {"thread": [{"from": "customer", "text": f"message {k}: the unit failed again after the reset, order {r.randint(1000,9999)}."}
                       for k in range(r.randint(2, 40))]}
def body(i): return json.dumps({"state": mixed_state(i) if MIXED else state, "questions": {"q": {"type": "choice", "instructions": f"Q{i}: which team?",
    "criteria": {"billing": None, "shipping": None, "tech": None, "other": None}}}}).encode()
def run(C, N=60):
    lat = []; lock = threading.Lock()
    def worker(w):
        for i in range(N):
            a = time.perf_counter()
            urllib.request.urlopen(urllib.request.Request(URL, body(w*1000+i), {"Content-Type": "application/json"})).read()
            with lock: lat.append((time.perf_counter() - a) * 1000)
    ts = [threading.Thread(target=worker, args=(w,)) for w in range(C)]
    t0 = time.perf_counter(); [t.start() for t in ts]; [t.join() for t in ts]; dt = time.perf_counter() - t0
    lat.sort()
    print(f"clients={C:3d}  {len(lat)/dt:7.1f} req/s   p50 {statistics.median(lat):7.2f} ms   p99 {lat[int(len(lat)*0.99)-1]:7.2f} ms", flush=True)
for C in (1, 4, 16, 32): run(C)
