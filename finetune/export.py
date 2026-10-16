"""Make a laya-train output servable by the HIP engine, and refuse it if it is not.

    python export.py BASE_DIR OUT_DIR

laya-train saves fp16 weights plus a re-saved tokenizer and encoder config. Fine-tuning never
changes the tokenizer or the encoder shape, so this restores the base checkpoint's exact
tokenizer/ and encoder/config.json (the files the server and engine were validated with), then
checks what the engine requires: same tensors and shapes as the base, fp16/bf16/fp32 dtypes,
the sequential option layout, and token budgets the base supports.
"""
import json, os, shutil, struct, sys

ENGINE_DTYPES = {"F16", "BF16", "F32"}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def main(base, out):
    errors = []
    shutil.rmtree(os.path.join(out, "checkpoint_latest"), ignore_errors=True)   # trainer's resume copy
    for rel in ("tokenizer", os.path.join("encoder", "config.json")):
        src, dst = os.path.join(base, rel), os.path.join(out, rel)
        if os.path.isdir(src):
            shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    hb, ho = header(os.path.join(base, "model.safetensors")), header(os.path.join(out, "model.safetensors"))
    if set(hb) != set(ho):
        errors.append("tensor names differ from the base: missing %s, extra %s"
                      % (sorted(set(hb) - set(ho))[:5], sorted(set(ho) - set(hb))[:5]))
    for k in set(hb) & set(ho):
        if hb[k]["shape"] != ho[k]["shape"]:
            errors.append("%s: shape %s, base %s" % (k, ho[k]["shape"], hb[k]["shape"]))
    bad = sorted({v["dtype"] for v in ho.values()} - ENGINE_DTYPES)
    if bad:
        errors.append("dtypes %s are not loadable (engine takes %s)" % (bad, sorted(ENGINE_DTYPES)))

    cb = json.load(open(os.path.join(base, "rl_agent_config.json")))
    co = json.load(open(os.path.join(out, "rl_agent_config.json")))
    if co.get("option_layout", "sequential") != "sequential":
        errors.append("option_layout %r: the engine implements the sequential layout only" % co["option_layout"])
    enc = json.load(open(os.path.join(base, "encoder", "config.json")))
    if co.get("max_len", 0) > enc.get("max_position_embeddings", 1 << 30):
        errors.append("max_len %s exceeds the encoder's %s positions" % (co["max_len"], enc["max_position_embeddings"]))
    if co.get("head_layers", cb.get("head_layers")) != cb.get("head_layers"):
        errors.append("head_layers changed from the base")

    if errors:
        print("NOT servable by the engine:\n  " + "\n  ".join(errors)); sys.exit(1)
    temps = co.get("temperature"); by_opt = co.get("temperature_by_options") or {}
    print("ok: %d tensors, max_len %s, head_max_len %s, temperatures %s%s"
          % (len(ho), co.get("max_len"), co.get("head_max_len"), temps,
             ", per-option-count buckets %s" % sorted(by_opt) if by_opt else ""))


if __name__ == "__main__":
    main(*sys.argv[1:3])
