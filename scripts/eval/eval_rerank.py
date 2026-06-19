#!/usr/bin/env python3
"""
End-to-end evaluation of the world-model rerank (System-2 planner).

Question: does generate-K-then-rerank actually serve clinically better replies
than the bare DPO Qwen (System 1, take the first sample)?

Method (honest, held-out):
  1. Build the empirical change-talk rate per (client_state, MI_action) from the
     AnnoMI VAL split — this is the ground-truth "what actually evokes change".
  2. For a sample of VAL turns, take the real client message + context, and have
     the live DPO Qwen generate K candidate replies (same settings as /api/chat).
  3. Tag each candidate's MI action. Two policies:
       - System 1 (no rerank): keep candidate[0].
       - System 2 (rerank):    keep the candidate whose action the planner ranks
                               highest (Q-value) for the estimated client state.
  4. Score each policy by the EMPIRICAL change rate of the action it chose,
     measured on held-out val. Report the difference.

This never lets the model grade itself: the reward is the held-out empirical
change rate from gold AnnoMI.

  python -m scripts.eval.eval_rerank --n 50 --k 4
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from scripts.world_model.transition_model import TALKS
from scripts.world_model import planner as P
from scripts.world_model.counterfactual import tag_action, estimate_state, _model

SYSTEM_KERRIO = (
    "You are Kerrio, a supportive coach using Motivational Interviewing. "
    "Be non-judgmental; use open questions, reflective listening, and affirmations; "
    "ask permission before advice; avoid directives. Keep it concise."
)


def empirical_change_rate(val, min_support=8):
    """(state, action) -> (change_rate, n) on the held-out val split."""
    c = defaultdict(Counter)
    for r in val:
        c[(r["prev_talk"], r["action"])][r["next_talk"]] += 1
    out = {}
    for key, cnt in c.items():
        n = sum(cnt.values())
        if n >= min_support:
            out[key] = (cnt["change"] / n, n)
    return out


def last_client_msg(history):
    for turn in reversed(history or []):
        if turn.get("role") == "client" and turn.get("text", "").strip():
            return turn["text"].strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/world_model/transitions.jsonl")
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH",
                    "runs/qwen2p5-3b-mi-dpo-merged"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-support", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--out", default="reports/rerank_eval.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    val = [r for r in rows if r["split"] == "val"]
    emp = empirical_change_rate(val, args.min_support)

    # Candidate turns: have a client message AND the (state, *) cell has support
    # for at least 2 actions so action choice can actually matter.
    states_with_support = defaultdict(set)
    for (s, a) in emp:
        states_with_support[s].add(a)
    cand = [r for r in val
            if last_client_msg(r.get("history"))
            and len(states_with_support[r["prev_talk"]]) >= 2]
    # deterministic sample (no RNG): evenly spaced across the pool
    if len(cand) > args.n:
        step = len(cand) / args.n
        cand = [cand[int(i * step)] for i in range(args.n)]
    print(f"[rerank-eval] scoring {len(cand)} val turns, K={args.k}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, trust_remote_code=True).to("cuda").eval()

    wm = _model()
    eos = tok.eos_token_id

    def gen_candidates(client_msg, history):
        msgs = [{"role": "system", "content": SYSTEM_KERRIO}]
        for t in (history or [])[-6:]:
            role = "user" if t["role"] == "client" else "assistant"
            if t.get("text"):
                msgs.append({"role": role, "content": t["text"]})
        # ensure the last message is the client turn we're responding to
        if not msgs or msgs[-1]["role"] != "user":
            msgs.append({"role": "user", "content": client_msg})
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=160, do_sample=True,
                                 temperature=0.7, top_p=0.9,
                                 num_return_sequences=args.k, pad_token_id=eos)
        plen = inp["input_ids"].shape[-1]
        return [tok.decode(out[i][plen:], skip_special_tokens=True).strip()
                for i in range(out.shape[0])]

    n_changed = 0
    s1_rates, s2_rates = [], []   # empirical change rate of chosen action
    per = []
    for i, r in enumerate(cand):
        state = r["prev_talk"]                     # gold state: isolate ACTION selection
        cmsg = last_client_msg(r["history"])
        cands = [c for c in gen_candidates(cmsg, r["history"]) if c] or ["(empty)"]
        ranked = P.rank_actions(wm, state, args.horizon)
        q = {x["action"]: x["Q"] for x in ranked}
        acts = [tag_action(c) for c in cands]
        a_first = acts[0]
        a_rerank = max(range(len(acts)), key=lambda j: q.get(acts[j], -9))
        a_rerank = acts[a_rerank]
        # held-out empirical change rate of each chosen action (skip if unsupported)
        rf = emp.get((state, a_first), (None, 0))[0]
        rr = emp.get((state, a_rerank), (None, 0))[0]
        if rf is not None:
            s1_rates.append(rf)
        if rr is not None:
            s2_rates.append(rr)
        if a_first != a_rerank:
            n_changed += 1
        per.append({"state": state, "first": a_first, "rerank": a_rerank,
                    "emp_first": rf, "emp_rerank": rr})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(cand)} done")

    res = {
        "n": len(cand), "k": args.k,
        "rerank_changed_action_pct": round(100 * n_changed / len(cand), 1),
        "mean_emp_change_first": round(sum(s1_rates) / len(s1_rates), 3) if s1_rates else None,
        "mean_emp_change_rerank": round(sum(s2_rates) / len(s2_rates), 3) if s2_rates else None,
    }
    if res["mean_emp_change_first"] is not None and res["mean_emp_change_rerank"] is not None:
        res["rerank_uplift"] = round(res["mean_emp_change_rerank"] - res["mean_emp_change_first"], 3)
    os.makedirs("reports", exist_ok=True)
    json.dump({"summary": res, "per_turn": per}, open(args.out, "w"), indent=2)
    print("\n[rerank-eval] " + json.dumps(res, indent=2))
    print(f"[rerank-eval] wrote -> {args.out}")


if __name__ == "__main__":
    main()
