"""HTTP front-end for the Laya HIP engine.

POST /v1/decide  {"state": ..., "questions": {...}}  -> same shape as rl_agent_api.system_one
GET  /health
GET  /info
"""
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya_server import Batcher, Laya

MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", "/weights/laya-en")
LIB_PATH  = os.environ.get("LAYA_LIB", "/usr/local/lib/liblaya.so")
HOST      = os.environ.get("LAYA_HOST", "0.0.0.0")
PORT      = int(os.environ.get("LAYA_PORT", "8080"))

print(f"[laya] loading {MODEL_DIR}", flush=True)
_t0 = time.perf_counter()
LAYA = Laya(MODEL_DIR, LIB_PATH)
print(f"[laya] ready in {time.perf_counter()-_t0:.1f}s", flush=True)
BATCHER = Batcher(LAYA.engine, int(os.environ.get("LAYA_BATCH_TOKENS", "4096")))
STATS = {"requests": 0, "questions": 0, "total_ms": 0.0}
STATS_LOCK = threading.Lock()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True   # headers and body go out as two writes; without this a
                                     # keep-alive client waits ~40 ms on delayed ACK per request
    def log_message(self, *a): pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok"})
        if self.path == "/info":
            q = STATS["questions"] or 1
            return self._send(200, {
                "model": LAYA.model_name, "max_len": LAYA.cfg["max_len"],
                "head_max_len": LAYA.cfg["head_max_len"], "device": "gfx1151",
                "requests": STATS["requests"], "questions": STATS["questions"],
                "batch_rounds": BATCHER.stats["passes"],
                "questions_per_round": round(BATCHER.stats["questions"] / max(1, BATCHER.stats["passes"]), 2),
                "mean_ms_per_question": round(STATS["total_ms"] / q, 2)})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/v1/decide", "/decide"):
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            state, questions = req.get("state"), req.get("questions")
            if not isinstance(questions, dict) or not questions:
                return self._send(400, {"error": "'questions' must be a non-empty object"})
            t0 = time.perf_counter()
            out = LAYA.system_one(state, questions, BATCHER)
            dt = (time.perf_counter() - t0) * 1000
            with STATS_LOCK:
                STATS["requests"] += 1
                STATS["questions"] += len(questions)
                STATS["total_ms"] += dt
            return self._send(200, out)
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    print(f"[laya] listening on {HOST}:{PORT}", flush=True)
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.request_queue_size = 256
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()
