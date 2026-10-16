"""Download public benchmark sets and build Laya request suites.

Mirrors upstream's research/scripts/laya_benchmark_colab.ipynb (cells 10-16): same datasets,
splits, row order, instructions, option rendering, SEED=13 and sample sizes, so results line up
case-for-case with research/results/t4_colab_benchmark.json.

    .venv/bin/python build_suites.py        -> data/*.parquet (cache), suites/<name>.jsonl, suites/index.json
"""
import gzip, json, os, random, sys, time, urllib.error, urllib.request
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data"), os.path.join(HERE, "suites")
SEED, N_OPTS, PER_LANG = 13, 20, 300
MASSIVE_LANGS = ["en", "de", "fr", "es", "pt", "ru", "tr", "ar", "hi", "ta", "zh-CN", "ja", "ko", "sw"]
XNLI_LANGS = ["en", "de", "fr", "es", "ru", "tr", "ar", "hi", "ur", "vi", "th", "el", "bg", "zh", "sw"]


def rows(repo, config, split):
    """All rows of one split, in dataset order (HF parquet conversion)."""
    if repo.startswith("mteb/amazon_massive"):
        return massive_rows(repo, config, split)
    return parquet_rows(repo, config, split)


def massive_rows(repo, config, split):
    """The hub's parquet conversion of some MASSIVE configs 403s on its CDN; the repo's own
    {split}/{lang}.json.gz files hold the same rows in the same order."""
    path = os.path.join(DATA, "%s__%s__%s.json.gz" % (repo.replace("/", "__"), config, split))
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        urllib.request.urlretrieve("https://huggingface.co/datasets/%s/resolve/main/%s/%s.json.gz"
                                   % (repo, split, config), path + ".part")
        os.replace(path + ".part", path)
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def parquet_rows(repo, config, split):
    path = os.path.join(DATA, "%s__%s__%s.parquet" % (repo.replace("/", "__"), config, split))
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        for attempt in range(6):            # the hub answers bursts with 403/429
            try:
                urls = json.load(urllib.request.urlopen(
                    "https://huggingface.co/api/datasets/%s/parquet/%s/%s" % (repo, config, split)))
                if len(urls) != 1:
                    raise RuntimeError("%s/%s/%s: expected 1 parquet shard, got %d" % (repo, config, split, len(urls)))
                urllib.request.urlretrieve(urls[0], path + ".part")
                break
            except urllib.error.HTTPError as e:
                if e.code not in (403, 429, 502, 503) or attempt == 5:
                    raise
                time.sleep(10 * 2 ** attempt)
        os.replace(path + ".part", path)
    return pq.read_table(path).to_pylist()


SUITES = {}

def register(name, cases, **meta):
    SUITES[name] = meta
    with open(os.path.join(OUT, name + ".jsonl"), "w") as f:
        for state, questions, gold in cases:
            f.write(json.dumps({"state": state, "questions": questions, "gold": gold}, ensure_ascii=False) + "\n")
    print("  %-26s %5d cases" % (name, len(cases)), flush=True)


def build_choice_questions(rng, gold, all_labels, n_opts, instructions, qid="label"):
    pool = [x for x in all_labels if x != gold]
    keys = [gold] + rng.sample(pool, min(n_opts - 1, len(pool)))
    rng.shuffle(keys)
    crit = {k: k.replace("_", " ").replace(".", ": ") for k in keys}
    return {qid: {"type": "choice", "instructions": instructions, "criteria": crit}}, keys.index(gold)


def typed_decisions():
    cases, wfs = [], []
    for r in rows("LocalLLaMA/typed-decisions", "all", "test"):
        questions = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        state = r["state"]
        try:
            state = json.loads(state)
        except Exception:
            pass
        gmap = {}
        for qid, qdef in questions.items():
            g = gold[qid]
            if qdef["type"] == "choice":
                keys = list(qdef["criteria"].keys())
                gmap[qid] = {"idx": keys.index(str(g["label"])),
                             "soft": [float(g.get("probabilities", {}).get(k, 0.0)) for k in keys]}
            elif qdef["type"] == "noul":
                pt = float(g.get("probabilities", {}).get("true", g.get("noul", 0.5)))
                gmap[qid] = {"idx": 1 if str(g["label"]).lower() == "true" else 0, "soft": [1 - pt, pt]}
            else:
                n = len(qdef["criteria"])
                gmap[qid] = {"idx": int(g["label"]),
                             "soft": [float(g.get("probabilities", {}).get(str(i), 0.0)) for i in range(n)],
                             "gold_score": float(g.get("score", float(g["label"])))}
        cases.append((state, questions, gmap)); wfs.append(r["workflow"])
    with open(os.path.join(OUT, "typed_decisions.workflows.json"), "w") as f:
        json.dump(wfs, f)
    register("typed_decisions", cases, family="typed_decisions", soft=True, held_out=True,
             note="4 workflows, soft teacher labels")


def multilingual():
    for repo, short, instr in [("mteb/amazon_massive_intent", "massive_intent",
                                "What is the user asking for in `utterance`?"),
                               ("mteb/amazon_massive_scenario", "massive_scenario",
                                "Which domain does `utterance` belong to?")]:
        for lg in MASSIVE_LANGS:
            d = rows(repo, lg, "test")
            labels = sorted(set(r["label_text"] for r in d))
            rng = random.Random(SEED)
            cases = []
            for r in d[:PER_LANG]:
                qs, gi = build_choice_questions(rng, r["label_text"], labels, N_OPTS, instr)
                cases.append(({"utterance": r["text"]}, qs, {"label": {"idx": gi}}))
            register("%s.%s" % (short, lg), cases, family=short, lang=lg, n_options=N_OPTS,
                     label_space=len(labels), held_out=True)

    crit = {"entailment": "the premise implies the hypothesis is true",
            "neutral": "the premise neither implies nor contradicts the hypothesis",
            "contradiction": "the premise implies the hypothesis is false"}
    for lg in XNLI_LANGS:
        cases = []
        for r in rows("facebook/xnli", lg, "test")[:PER_LANG]:
            qs = {"relation": {"type": "choice",
                               "instructions": "What is the relationship between `premise` and `hypothesis`?",
                               "criteria": dict(crit)}}
            cases.append(({"premise": r["premise"], "hypothesis": r["hypothesis"]}, qs,
                          {"relation": {"idx": int(r["label"])}}))
        register("xnli.%s" % lg, cases, family="xnli", lang=lg, n_options=3, held_out=True)


def english():
    crit = ["very negative", "negative", "neutral", "positive", "very positive"]
    register("en.sst5", [({"text": r["text"]},
                          {"sentiment": {"type": "score", "instructions": "How positive is the sentiment of `text`?",
                                         "criteria": crit}},
                          {"sentiment": {"idx": int(r["label"]), "gold_score": float(r["label"])}})
                         for r in rows("SetFit/sst5", "default", "test")[:600]],
             family="english", held_out=True, note="ordinal score, 5 levels")

    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    register("en.emotion", [({"text": r["text"]},
                             {"emotion": {"type": "choice",
                                          "instructions": "Which emotion is most strongly expressed in `text`?",
                                          "criteria": {n: None for n in names}}},
                             {"emotion": {"idx": int(r["label"])}})
                            for r in rows("dair-ai/emotion", "split", "test")[:600]],
             family="english", held_out=True, note="6-way choice")

    register("en.prompt_injections", [({"text": r["text"]},
                                       {"injection": {"type": "noul", "instructions":
                                           "Does `text` try to inject or override instructions given to an AI system?"}},
                                       {"injection": {"idx": int(r["label"])}})
                                      for r in rows("deepset/prompt-injections", "default", "test")],
             family="english", held_out=True, note="yes/no guardrail")

    # PolyAI/banking77 is script-only on the hub; mteb/banking77 is the same test split with label_text
    d = rows("mteb/banking77", "default", "test")
    names = [t for _, t in sorted({(r["label"], r["label_text"]) for r in d})]
    assert len(names) == 77, len(names)
    register("en.banking77_full", [({"message": r["text"]},
                                    {"intent": {"type": "choice", "instructions": "Which banking intent does `message` express?",
                                                "criteria": {n.replace("_", " "): None for n in names}}},
                                    {"intent": {"idx": int(r["label"])}})
                                   for r in d[:500]],
             family="english", held_out=True, n_options=77, note="77 options in one question (stress)")

    crit = {"world": "world news and international politics", "sports": "sports",
            "business": "business and economy", "sci_tech": "science and technology"}
    register("en.ag_news", [({"article": r["text"]},
                             {"topic": {"type": "choice", "instructions": "What is the topic of `article`?",
                                        "criteria": dict(crit)}},
                             {"topic": {"idx": int(r["label"])}})
                            for r in rows("fancyzhx/ag_news", "default", "test")[:600]],
             family="english", held_out=False, note="in training mix")

    register("en.boolq", [({"passage": r["passage"], "question": r["question"]},
                           {"answer": {"type": "noul", "instructions": "Based on `passage`, is the answer to `question` yes?"}},
                           {"answer": {"idx": int(bool(r["answer"]))}})
                          for r in rows("google/boolq", "default", "validation")[:600]],
             family="english", held_out=False, note="in training mix")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (typed_decisions, multilingual, english):
        try:
            fn()
        except Exception as e:
            print("FAILED %s: %s: %s" % (fn.__name__, type(e).__name__, e), file=sys.stderr)
    with open(os.path.join(OUT, "index.json"), "w") as f:
        json.dump(SUITES, f, indent=1)
    print("%d suites" % len(SUITES))
