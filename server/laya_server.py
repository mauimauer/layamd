"""Laya decision server — drop-in replacement for rl_agent_api.RLAgent.system_one.

Tokenization and sequence construction are a faithful port of the upstream rl_common.py
(the BPE tokenizer is the Rust `tokenizers` lib, ~100us); every tensor op runs in the
HIP engine on the iGPU via ctypes.
"""
import ctypes, json, math, os, queue, threading, time
from typing import Dict, List, Optional
import numpy as np
from tokenizers import Tokenizer

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


class Engine:
    def __init__(self, lib_path: str, model_dir: str):
        self.lib = L = ctypes.CDLL(lib_path)
        IP, FP = ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float)
        L.laya_create.restype = ctypes.c_void_p
        L.laya_create.argtypes = [ctypes.c_char_p]
        L.laya_forward_batch.restype = ctypes.c_int
        L.laya_forward_batch.argtypes = [ctypes.c_void_p, ctypes.c_int, IP, IP, IP, IP, IP, FP, FP]
        L.laya_tune_all.argtypes = [ctypes.c_void_p]
        for f in ("laya_n_act", "laya_max_len", "laya_max_markers"):
            getattr(L, f).argtypes = [ctypes.c_void_p]; getattr(L, f).restype = ctypes.c_int
        self.h = L.laya_create(model_dir.encode())
        self.n_act = L.laya_n_act(self.h)
        self.max_markers = L.laya_max_markers(self.h)

    def forward_batch(self, seqs: List[List[int]], markers: List[List[int]], qtypes: List[int]):
        """All questions of a request in one GPU pass. Returns per-question logits and act logits."""
        B = len(seqs)
        arr = lambda t, xs: (t * len(xs))(*xs)
        lens = arr(ctypes.c_int, [len(s) for s in seqs])
        nm = arr(ctypes.c_int, [len(m) for m in markers])
        ids = arr(ctypes.c_int, [t for s in seqs for t in s])
        mk = arr(ctypes.c_int, [p for m in markers for p in m])
        qt = arr(ctypes.c_int, qtypes)
        lo = (ctypes.c_float * sum(len(m) for m in markers))()
        ac = (ctypes.c_float * (B * self.n_act))()
        if self.lib.laya_forward_batch(self.h, B, lens, ids, nm, mk, qt, lo, ac) != 0:
            raise ValueError("engine rejected batch (sequence too long or too many options)")
        lo = np.ctypeslib.as_array(lo).copy(); ac = np.ctypeslib.as_array(ac).reshape(B, self.n_act).copy()
        out, off = [], 0
        for m in markers:
            out.append(lo[off:off + len(m)]); off += len(m)
        return out, ac

    def warmup(self, max_len: int, cls_id: int):
        """Autotune every GEMM shape x M bucket, then capture a graph per single-question length."""
        self.lib.laya_tune_all(self.h)
        for n in range(64, max_len + 1, 64):
            self.forward_batch([[cls_id] * n], [[1, 2]], [0])


# ------------------------------------------------- sequence construction (ports rl_common.py)
def serialize_state(state) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)

def render_options(q: Dict) -> List[str]:
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: "  + (crit.get("true")  or "yes, the statement holds")]

class Tok:
    def __init__(self, path: str, cfg_path: str):
        self.t = Tokenizer.from_file(path)
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        self.mask_token = cfg.get("mask_token", "[MASK]")
        self.mask_token_id = self.t.token_to_id(self.mask_token)
        self.cls_token_id  = self.t.token_to_id(cfg.get("cls_token", "[CLS]"))
        self.sep_token_id  = self.t.token_to_id(cfg.get("sep_token", "[SEP]"))
        self.pad_token_id  = self.t.token_to_id(cfg.get("pad_token", "[PAD]"))
    def ids(self, s: str) -> List[int]:
        return self.t.encode(s, add_special_tokens=False).ids

def build_sequence(tok: Tok, state, q: Dict, max_len: int, head_max_len: int):
    """[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]"""
    opts = render_options(q)
    ins = str(q["ins"]).replace(tok.mask_token, " ")
    head_ids = tok.ids("%s question: %s" % (q["t"], ins))
    opt_ids = [[tok.mask_token_id] + tok.ids(" " + o.replace(tok.mask_token, " "))[:48] for o in opts]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[:max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids)); ids.extend(o)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    st = tok.ids(serialize_state(state))[:room]
    ids = ids + st + [tok.sep_token_id]
    return ids[:max_len], [m for m in markers if m < max_len]

def temp_bucket(qtype: int, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)

def confidence_from_probs(p, k: int) -> float:
    if k < 2: return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1))).sum()
    return float(1 - ent / math.log(k))


class Laya:
    def __init__(self, model_dir: str, lib_path: str):
        self.cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
        self.tok = Tok(os.path.join(model_dir, "tokenizer", "tokenizer.json"),
                       os.path.join(model_dir, "tokenizer", "tokenizer_config.json"))
        self.engine = Engine(lib_path, model_dir)
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temp_by_options = self.cfg.get("temperature_by_options", {})
        self.model_name = self.cfg.get("model_name", "laya")
        # GEMM autotuning + graph capture happen on a shape's first use; pay that at startup
        self.engine.warmup(self.cfg["max_len"], self.tok.cls_token_id)

    @staticmethod
    def _to_internal(qdef):
        t = qdef["type"]; crit = qdef.get("criteria")
        if t == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        ins = qdef["instructions"]
        return {"t": t, "crit": crit,
                "ins": ins if isinstance(ins, str) else json.dumps(ins)}

    def prepare(self, state, questions: Dict):
        """Tokenize and build every question's sequence (CPU; runs in the request thread)."""
        qids, qs, items = [], [], []
        for qid, qdef in questions.items():
            q = self._to_internal(qdef)
            seq, markers = build_sequence(self.tok, state, q, self.cfg["max_len"], self.cfg["head_max_len"])
            if len(markers) != len(render_options(q)):
                raise ValueError("question %r: options do not fit in head_max_len=%d tokens"
                                 % (qid, self.cfg["head_max_len"]))
            if len(markers) > self.engine.max_markers:
                raise ValueError("question %r: more than %d options" % (qid, self.engine.max_markers))
            qids.append(qid); qs.append(q); items.append((seq, markers, QTYPES[q["t"]]))
        return qids, qs, items

    def finish(self, qids, qs, items, logits, act, t0) -> Dict:
        answers = {}
        for qid, q, (seq, _, qt), lg, ac in zip(qids, qs, items, logits, act):
            k = len(lg)
            z = lg / self.temp_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            p = np.exp(z - z.max()); p = p / p.sum()
            ea = np.exp(ac - ac.max()); ea = ea / ea.sum()
            ext = {"act_probability": float(ea[0])}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
            elif q["t"] == "score":
                answers[qid] = {"type": "score",
                    "score": round(float((np.arange(k) * p).sum()), 4),
                    "legend": {str(i): c for i, c in enumerate(q["crit"])},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": round(confidence_from_probs(p, k), 4), "rl_agent": ext}
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4), "rl_agent": ext}
        return {"model": self.model_name, "answers": answers,
                "usage": {"input_tokens": sum(len(it[0]) for it in items), "output_tokens": 0,
                          "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}}

    def system_one(self, state, questions: Dict, batcher: Optional["Batcher"] = None) -> Dict:
        t0 = time.perf_counter()
        qids, qs, items = self.prepare(state, questions)
        if batcher is not None:
            logits, act = batcher.run(items)
        else:
            logits, act = self.engine.forward_batch([i[0] for i in items], [i[1] for i in items], [i[2] for i in items])
        return self.finish(qids, qs, items, logits, act, t0)


class Batcher:
    """Continuous batching across concurrent requests.

    Request threads enqueue their (already tokenized) questions and block. One worker drains
    everything queued -- up to `max_tokens` -- and hands it to the engine in one call (the engine
    packs mixed lengths back to back and splits into LLC-sized passes). At low load a lone request
    goes straight through; under load, requests arriving during a round form the next one.
    """
    def __init__(self, engine: Engine, max_tokens: int = 4096):
        self.engine, self.max_tokens = engine, max_tokens
        self.q: "queue.Queue" = queue.Queue()
        self.carry = None
        self.stats = {"passes": 0, "questions": 0}
        threading.Thread(target=self._loop, daemon=True, name="laya-batcher").start()

    def run(self, items):
        job = {"items": items, "ev": threading.Event(), "logits": [None] * len(items),
               "act": [None] * len(items), "err": None}
        self.q.put(job)
        job["ev"].wait()
        if job["err"] is not None:
            raise job["err"]
        return job["logits"], job["act"]

    def _loop(self):
        while True:
            jobs = [self.carry if self.carry is not None else self.q.get()]
            self.carry = None
            tokens = sum(len(it[0]) for it in jobs[0]["items"])
            while True:
                try: j = self.q.get_nowait()
                except queue.Empty: break
                t = sum(len(it[0]) for it in j["items"])
                if tokens + t > self.max_tokens: self.carry = j; break
                jobs.append(j); tokens += t
            flat = [(ji, ii, it) for ji, j in enumerate(jobs) for ii, it in enumerate(j["items"])]
            try:   # one call: the engine packs mixed lengths back to back and splits into ~1k-row passes
                lg, ac = self.engine.forward_batch([f[2][0] for f in flat], [f[2][1] for f in flat],
                                                   [f[2][2] for f in flat])
                for (ji, ii, _), l_, a_ in zip(flat, lg, ac):
                    jobs[ji]["logits"][ii] = l_; jobs[ji]["act"][ii] = a_
                self.stats["passes"] += 1; self.stats["questions"] += len(flat)
            except Exception as e:                      # inputs are validated in prepare(); defensive
                for j in jobs: j["err"] = e
            for j in jobs: j["ev"].set()
