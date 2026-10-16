"""Generate golden tensors from the reference PyTorch implementation (CPU, fp32).

Dumps inputs + final outputs + per-layer intermediate activations so the HIP engine
can be validated layer-by-layer, not just end-to-end.
"""
import json, os, sys, struct
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_common import QTYPES, build_sequence, collate_items, render_options, temp_bucket, confidence_from_probs
from rl_agent_api import RLAgent

MODEL_DIR = os.environ.get("LAYA_MODEL_DIR", "/weights/laya-en")
OUT_DIR   = os.environ.get("LAYA_GOLDEN_DIR", "/golden")

# --- representative cases: every qtype, several option cardinalities, short + long state ---
LONG_STATE = {"ticket": {"id": 4471, "subject": "refund not received",
              "body": "I returned the item three weeks ago and tracking shows delivered. " * 12,
              "customer_tier": "gold", "prior_contacts": 3}}
XL_STATE = {"thread": [{"from": "customer" if i % 2 == 0 else "agent",
             "text": f"Message {i}: the replacement unit also failed after two days, order 88{i:02d}, "
                     "I have already reset it twice and followed every step in the manual."} for i in range(60)]}
CASES = [
  ("fill_ctx", XL_STATE,
   {"q": {"type": "choice", "instructions": "What should happen next?",
          "criteria": {"refund": None, "replace": None, "escalate": None, "troubleshoot": None}}}),
  ("noul2_short", "The user is asking for a refund.",
   {"q": {"type": "noul", "instructions": "Does the statement hold?", "criteria": None}}),
  ("choice3_short", "User says: my package never arrived and I want my money back.",
   {"q": {"type": "choice", "instructions": "Route this ticket to the right queue.",
          "criteria": {"billing": "refunds and charges", "shipping": "delivery issues", "technical": "product faults"}}}),
  ("choice5_mid", {"msg": "I was charged twice for order 8812 and the second one never shipped."},
   {"q": {"type": "choice", "instructions": "Pick the single best next action.",
          "criteria": {"refund": "issue a full refund", "partial": "issue a partial refund",
                       "reship": "send a replacement", "investigate": "ask for more detail",
                       "escalate": "hand to a human agent"}}}),
  ("score5_long", LONG_STATE,
   {"q": {"type": "score", "instructions": "How urgent is this ticket?",
          "criteria": ["not urgent at all", "low", "moderate", "high", "drop everything"]}}),
  ("choice12_long", LONG_STATE,
   {"q": {"type": "choice", "instructions": "Assign the most specific category.",
          "criteria": {f"cat{i}": f"category number {i} description text" for i in range(12)}}}),
  ("multi_q", LONG_STATE, {
     "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["none","low","med","high","critical"]},
     "route":   {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": None, "shipping": None, "tech": None}},
     "refund":  {"type": "noul", "instructions": "Should we issue a refund?", "criteria": None}}),
]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_grad_enabled(False)
    agent = RLAgent(MODEL_DIR, device="cpu")
    agent.model.float().eval()
    m = agent.model

    acts = {}
    hooks = []
    def cap(name):
        def fn(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t): acts[name] = t.detach().float().cpu().numpy()
        return fn
    enc = m.encoder
    hooks.append(enc.embeddings.register_forward_hook(cap("emb")))
    for i, lyr in enumerate(enc.layers):
        hooks.append(lyr.register_forward_hook(cap(f"enc{i}")))
    hooks.append(enc.final_norm.register_forward_hook(cap("final_norm")))
    for i, lyr in enumerate(m.head.layers):
        hooks.append(lyr.register_forward_hook(cap(f"head{i}")))

    manifest = {"model_dir": MODEL_DIR, "cases": []}
    for name, state, questions in CASES:
        ids_list, items = list(questions.keys()), []
        for qid in ids_list:
            q = agent._to_internal(questions[qid])
            seq, markers = build_sequence(agent.tok, state, q, agent.cfg["max_len"], agent.cfg["head_max_len"])
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]],
                          "target": [0.0]*len(markers), "label": -1, "episode": 0,
                          "ep_step": 0, "ep_len": 1, "src": "api"})
        b = collate_items([items], agent.tok.pad_token_id)
        acts.clear()
        logits, act_logits = m(b["input_ids"], b["attention_mask"], b["marker_pos"],
                               b["marker_mask"], b["qtype"])
        cdir = os.path.join(OUT_DIR, name); os.makedirs(cdir, exist_ok=True)
        def save(fn, arr, dtype=np.float32):
            arr = np.ascontiguousarray(arr.astype(dtype)); arr.tofile(os.path.join(cdir, fn)); return list(arr.shape)
        rec = {"name": name, "shapes": {}}
        rec["shapes"]["input_ids"]     = save("input_ids.bin", b["input_ids"].numpy(), np.int32)
        rec["shapes"]["attention_mask"]= save("attention_mask.bin", b["attention_mask"].numpy(), np.int32)
        rec["shapes"]["marker_pos"]    = save("marker_pos.bin", b["marker_pos"].numpy(), np.int32)
        rec["shapes"]["marker_mask"]   = save("marker_mask.bin", b["marker_mask"].numpy().astype(np.int32), np.int32)
        rec["shapes"]["qtype"]         = save("qtype.bin", b["qtype"].numpy(), np.int32)
        rec["shapes"]["logits"]        = save("logits.bin", logits.numpy())
        rec["shapes"]["act_logits"]    = save("act_logits.bin", act_logits.numpy())
        for k in ("emb", "final_norm", "head0", "head1"):
            if k in acts: rec["shapes"][k] = save(f"{k}.bin", acts[k])
        for i in range(len(enc.layers)):
            k = f"enc{i}"
            if k in acts: rec["shapes"][k] = save(f"{k}.bin", acts[k])
        # also the public API answer, so the server can be diffed against it
        rec["api"] = agent.system_one(state, questions)
        rec["seq_lens"] = [int(x) for x in b["attention_mask"].sum(-1).tolist()]
        manifest["cases"].append(rec)
        print(f"[golden] {name:16s} n={b['input_ids'].shape[0]} L={b['input_ids'].shape[1]} "
              f"lens={rec['seq_lens']} logits={logits.shape}", flush=True)
    for h in hooks: h.remove()
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[golden] wrote {len(manifest['cases'])} cases -> {OUT_DIR}")

if __name__ == "__main__":
    main()
