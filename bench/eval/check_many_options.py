"""Check questions with more than 64 options against upstream's PyTorch RLAgent (fp32, CPU).

    python3 check_many_options.py ref  <model_dir> <out.json>    # inside laya-ref:cpu (ref/ on PYTHONPATH)
    python3 check_many_options.py diff <ref.json> [url]          # against a running server

Cases: the first 20 Banking77 test questions (77 options) and one synthetic 120-option question.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def cases():
    out = []
    with open(os.path.join(HERE, "suites", "en.banking77_full.jsonl")) as f:
        for line in list(f)[:20]:
            c = json.loads(line); out.append({"state": c["state"], "questions": c["questions"]})
    crit = {"route_%03d" % i: "queue %d" % i for i in range(120)}
    out.append({"state": "Ticket 4411: invoice paid twice, please refund to the original card.",
                "questions": {"q": {"type": "choice", "instructions": "Which queue?", "criteria": crit}}})
    return out


def probs(answers, questions):
    return {qid: [a["probabilities"][k] for k in questions[qid]["criteria"]] for qid, a in answers.items()}


if __name__ == "__main__":
    if sys.argv[1] == "ref":
        from rl_agent_api import RLAgent
        agent = RLAgent(sys.argv[2], device="cpu")
        res = [probs(agent.system_one(c["state"], c["questions"])["answers"], c["questions"]) for c in cases()]
        json.dump({"cases": cases(), "probs": res}, open(sys.argv[3], "w"))
    else:
        import http.client
        ref = json.load(open(sys.argv[2])); url = sys.argv[3] if len(sys.argv) > 3 else "localhost:8080"
        conn = http.client.HTTPConnection(url); worst = 0.0; agree = 0
        for c, rp in zip(ref["cases"], ref["probs"]):
            conn.request("POST", "/v1/decide", json.dumps(c), {"Content-Type": "application/json"})
            r = conn.getresponse(); out = json.loads(r.read())
            if r.status != 200:
                sys.exit("server rejected a case: %s" % out)
            ours = probs(out["answers"], c["questions"])
            for qid in rp:
                worst = max(worst, max(abs(a - b) for a, b in zip(ours[qid], rp[qid])))
                agree += ours[qid].index(max(ours[qid])) == rp[qid].index(max(rp[qid]))
        n = sum(len(p) for p in ref["probs"])
        print("%d questions (77 and 120 options): argmax agrees %d/%d, worst probability delta %.5f" % (n, agree, n, worst))
        sys.exit(0 if agree == n and worst < 0.01 else 1)
