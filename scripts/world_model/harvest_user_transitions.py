#!/usr/bin/env python3
"""
Continuous-improvement step — harvest REAL user transitions from chat logs.

As users chat (runs/chat_logs/<user>.jsonl: each turn has `user` = client text,
`coach` = model reply), reconstruct (prev_client_talk, coach_action, next_client_talk)
transitions, silver-labeled by the talk-type classifier + action tagger.

These are REAL in-domain MI exchanges (the actual product traffic) — the highest-
value augmentation source. Labels are silver (confidence kept for rejection sampling);
the AnnoMI-val gate downstream stays the gold judge.

  python -m scripts.world_model.harvest_user_transitions --clf runs/talktype_clf_mpnet
"""
import argparse
import glob
import json
import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="runs/chat_logs")
    ap.add_argument("--talk-clf", default="runs/talktype_clf_mpnet")
    ap.add_argument("--action-clf", default="runs/action_clf")
    ap.add_argument("--out", default="data/world_model/user_transitions.jsonl")
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir(args.talk_clf):
        args.talk_clf = "runs/talktype_clf"
    ttok = AutoTokenizer.from_pretrained(args.talk_clf)
    tmdl = AutoModelForSequenceClassification.from_pretrained(args.talk_clf).eval().to(dev)
    atok = AutoTokenizer.from_pretrained(args.action_clf)
    amdl = AutoModelForSequenceClassification.from_pretrained(args.action_clf).eval().to(dev)

    def classify(mdl, t, texts):
        out_lab, out_conf = [], []
        for i in range(0, len(texts), args.bs):
            enc = t(texts[i:i + args.bs], return_tensors="pt", truncation=True,
                    max_length=192, padding=True).to(dev)
            enc.pop("token_type_ids", None)
            with torch.no_grad():
                p = F.softmax(mdl(**enc).logits, -1)
            conf, idx = p.max(-1)
            out_lab += [mdl.config.id2label[i.item()] for i in idx]
            out_conf += conf.tolist()
        return out_lab, out_conf

    transitions = []
    for fp in glob.glob(os.path.join(args.logs, "*.jsonl")):
        turns = []
        for line in open(fp, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "user" in r and "coach" in r:
                turns.append(r)
        turns.sort(key=lambda r: r.get("turn_id", 0))
        if len(turns) < 2:
            continue
        # silver-label each client (user) utterance with preceding coach as context
        client_ctx = [f"Counselor: {turns[i-1]['coach']}\nClient: {turns[i]['user']}" if i > 0
                      else f"Client: {turns[i]['user']}" for i in range(len(turns))]
        talk, tconf = classify(tmdl, ttok, client_ctx)
        act, _ = classify(amdl, atok, [t["coach"] for t in turns])
        for i in range(len(turns) - 1):
            transitions.append({
                "prev_talk": talk[i], "action": act[i], "next_talk": talk[i + 1],
                "conf": round(min(tconf[i], tconf[i + 1]), 4),
                "source": "user", "split": "train",
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in transitions:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[harvest] user transitions={len(transitions)} from {args.logs}")
    if transitions:
        print(f"[harvest] next_talk={dict(Counter(r['next_talk'] for r in transitions))}")
    print(f"[harvest] wrote -> {args.out}")


if __name__ == "__main__":
    main()
