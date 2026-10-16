"""How far does upstream's own GPU path (bf16 autocast, see rl_agent_api.py) drift from fp32?"""
import json, os, sys, numpy as np, torch
sys.path.insert(0, "/ref")
from rl_common import QTYPES, build_sequence, collate_items, temp_bucket
from rl_agent_api import RLAgent
src = open("/ref/gen_golden.py").read(); ns = {}; exec(src[src.index("LONG_STATE ="):src.index("def main")], ns)
d = os.environ["LAYA_MODEL_DIR"]; torch.set_grad_enabled(False)
ag = RLAgent(d, device="cpu"); ag.model.float().eval()
print(f"{'case':14s} {'q':8s} {'k':>3s} {'bf16 dlogit':>12s} {'bf16 dprob(API)':>16s}")
for name, state, qs in ns["CASES"]:
    items = []
    for qid, qd in qs.items():
        q = ag._to_internal(qd); seq, mk = build_sequence(ag.tok, state, q, ag.cfg["max_len"], ag.cfg["head_max_len"])
        items.append({"ids": seq, "markers": mk, "qtype": QTYPES[q["t"]], "target": [0.0]*len(mk), "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "api"})
    b = collate_items([items], ag.tok.pad_token_id)
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    l32, _ = ag.model(*args)
    with torch.autocast("cpu", dtype=torch.bfloat16): l16, _ = ag.model(*args)
    for r, qid in enumerate(qs):
        k = int(b["marker_mask"][r].sum()); qt = int(b["qtype"][r])
        T = ag.temperature_by_options.get(temp_bucket(qt, k), ag.temperature[qt])
        def p(z): z = z[:k].double().numpy() / T; e = np.exp(z - z.max()); return e / e.sum()
        print(f"{name:14s} {qid:8s} {k:3d} {float((l16[r,:k]-l32[r,:k]).abs().max()):12.4f} {float(np.abs(p(l16[r])-p(l32[r])).max()):16.4f}")
