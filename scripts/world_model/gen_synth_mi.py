#!/usr/bin/env python3
"""
In-domain synthetic MI data for the online active loop.

Use the DPO Qwen coach + a simulated client (same model, roleplay prompts) to
generate short MI exchanges TARGETED at the weak talk-types (sustain / change),
then silver-label both client turns with the talk-type classifier and tag the
counselor action. Output grounded (prev_talk, action, next_talk) transitions with
a confidence score for rejection sampling in active_loop.py.

In-domain (MI-style) generation should yield HIGHER classifier confidence than the
out-of-domain ESConv text -> cleaner silver labels -> bigger loop gain (hypothesis).

Self-training caveat: data + labeler both trace to AnnoMI-trained models, so the
real-AnnoMI-val gate in active_loop.py is what keeps this honest.

  python -m scripts.world_model.gen_synth_mi --n 600 --out data/world_model/synth_silver_transitions.jsonl
"""
import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

TALKS = ["change", "sustain", "neutral"]
BEHAVIORS = ["drinking", "smoking", "exercising regularly", "eating healthier",
             "procrastination", "screen time", "taking my medication",
             "managing my anger", "gambling", "my sleep habits"]

CLIENT_SYS = {
    "change": "You are roleplaying a therapy CLIENT who is becoming motivated to change. "
              "Express desire, ability, reasons, or need to change. Reply in ONE short sentence, first person.",
    "sustain": "You are roleplaying a therapy CLIENT who is ambivalent/resistant. "
               "Express reasons to stay the same, doubt, or that change is too hard. Reply in ONE short sentence, first person.",
}
COACH_SYS = ("You are a motivational interviewing coach. Respond to the client with ONE short, "
             "MI-consistent turn (reflection, open question, affirmation, or asking permission). Do not lecture.")


def gen(model, tok, prompts, max_new=64, temp=0.8):
    """Batched chat generation; returns list of decoded completions."""
    texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in prompts]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temp,
                             top_p=0.9, pad_token_id=tok.eos_token_id)
    res = []
    for i in range(out.shape[0]):
        new = out[i][enc["input_ids"].shape[1]:]
        res.append(tok.decode(new, skip_special_tokens=True).strip().split("\n")[0])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/qwen2p5-3b-mi-dpo-merged")
    ap.add_argument("--talk-clf", default="runs/talktype_clf")
    ap.add_argument("--action-clf", default="runs/action_clf")
    ap.add_argument("--out", default="data/world_model/synth_silver_transitions.jsonl")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--bs", type=int, default=40)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).eval().to(dev)

    ttok = AutoTokenizer.from_pretrained(args.talk_clf)
    tmdl = AutoModelForSequenceClassification.from_pretrained(args.talk_clf).eval().to(dev)
    atok = AutoTokenizer.from_pretrained(args.action_clf)
    amdl = AutoModelForSequenceClassification.from_pretrained(args.action_clf).eval().to(dev)

    def classify(mdl, t, texts, ctxs=None):
        ins = texts if ctxs is None else [f"Counselor: {c}\nClient: {x}" for c, x in zip(ctxs, texts)]
        enc = t(ins, return_tensors="pt", truncation=True, max_length=192, padding=True).to(dev)
        enc.pop("token_type_ids", None)
        with torch.no_grad():
            p = F.softmax(mdl(**enc).logits, -1)
        conf, idx = p.max(-1)
        return [mdl.config.id2label[i.item()] for i in idx], conf.tolist()

    records = []
    import itertools
    stances = list(itertools.islice(itertools.cycle(["sustain", "change"]), args.n))
    for s in range(0, args.n, args.bs):
        batch_st = stances[s:s + args.bs]
        beh = [BEHAVIORS[(s + i) % len(BEHAVIORS)] for i in range(len(batch_st))]
        # 1) client opening
        op_prompts = [[{"role": "system", "content": CLIENT_SYS[st]},
                       {"role": "user", "content": f"Open up about {b} to your counselor."}]
                      for st, b in zip(batch_st, beh)]
        openings = gen(model, tok, op_prompts, 48)
        # 2) counselor reply
        co_prompts = [[{"role": "system", "content": COACH_SYS},
                       {"role": "user", "content": o}] for o in openings]
        coaches = gen(model, tok, co_prompts, 64, temp=0.7)
        # 3) client follow-up (lean target stance)
        fu_prompts = [[{"role": "system", "content": CLIENT_SYS[st]},
                       {"role": "user", "content": f"Counselor said: {c}\nRespond as the client."}]
                      for st, c in zip(batch_st, coaches)]
        followups = gen(model, tok, fu_prompts, 48)

        # silver-label both client turns (with counselor context) + tag action
        prev_lab, prev_conf = classify(tmdl, ttok, openings, ctxs=[""] * len(openings))
        next_lab, next_conf = classify(tmdl, ttok, followups, ctxs=coaches)
        act_lab, _ = classify(amdl, atok, coaches)
        for i in range(len(batch_st)):
            records.append({
                "prev_talk": prev_lab[i], "action": act_lab[i], "next_talk": next_lab[i],
                "conf": round(min(prev_conf[i], next_conf[i]), 4),
                "target_stance": batch_st[i], "source": "synth_qwen", "split": "train",
                "_text": {"opening": openings[i], "coach": coaches[i], "followup": followups[i]},
            })
        print(f"[synth] {min(s+args.bs,args.n)}/{args.n} done")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[synth] n={len(records)} next_talk={dict(Counter(r['next_talk'] for r in records))}")
    cs = sorted(r["conf"] for r in records)
    print(f"[synth] conf quartiles: {cs[len(cs)//4]:.2f}/{cs[len(cs)//2]:.2f}/{cs[3*len(cs)//4]:.2f}")
    print(f"[synth] wrote -> {args.out}")


if __name__ == "__main__":
    main()
