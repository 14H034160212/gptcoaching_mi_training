#!/usr/bin/env python3
"""
Online-loop step 1 — silver-label the real ESConv corpus with the talk-type
classifier, producing grounded (prev_talk, action, next_talk) transitions with
a confidence score (min of the two endpoints' classifier confidence).

Why real text + silver labels (not model-generated): avoids the self-training
confirmation-bias trap. The text is real human counseling dialogue; only the
talk-type label is predicted. Rejection sampling on confidence handles label noise.

  python -m scripts.world_model.silver_label_esconv --out data/world_model/esconv_silver_transitions.jsonl
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.world_model.build_esconv_pairs import STRAT2ACTION

TALKS = ["change", "sustain", "neutral"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf", default="runs/talktype_clf")
    ap.add_argument("--out", default="data/world_model/esconv_silver_transitions.jsonl")
    ap.add_argument("--bs", type=int, default=128)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.clf)
    mdl = AutoModelForSequenceClassification.from_pretrained(args.clf).eval().to(dev)
    id2lab = mdl.config.id2label

    ds = load_dataset("thu-coai/esconv")

    # 1) collect every client utterance with its preceding counselor turn (context)
    #    and the counselor action that elicited it.
    items = []  # {dialog_id, idx, ctx_text, action}
    dialogs = []  # list of list-of-client-turn-meta for sequencing
    for split in ds:
        for di, row in enumerate(ds[split]):
            convo = json.loads(row["text"])["dialog"]
            prev_ther, last_action = "", None
            seq = []
            for turn in convo:
                spk, text = turn.get("speaker"), (turn.get("text") or "").strip()
                if not text:
                    continue
                if spk == "sys":
                    prev_ther = text
                    last_action = STRAT2ACTION.get(turn.get("strategy"), "other")
                else:
                    seq.append({"ctx": f"Counselor: {prev_ther}\nClient: {text}",
                                "action": last_action})
            if seq:
                dialogs.append(seq)

    # 2) batch-classify talk-type + confidence
    flat = [s for seq in dialogs for s in seq]
    texts = [s["ctx"] for s in flat]
    for i in range(0, len(texts), args.bs):
        enc = tok(texts[i:i + args.bs], return_tensors="pt", truncation=True,
                  max_length=192, padding=True)
        enc.pop("token_type_ids", None)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            probs = F.softmax(mdl(**enc).logits, dim=-1)
        conf, idx = probs.max(-1)
        for j in range(idx.shape[0]):
            flat[i + j]["talk"] = id2lab[idx[j].item()]
            flat[i + j]["conf"] = round(conf[j].item(), 4)

    # 3) build transitions: (prev client talk, action that follows, next client talk)
    out = []
    for seq in dialogs:
        for k in range(1, len(seq)):
            prev, cur = seq[k - 1], seq[k]
            if cur["action"] is None:
                continue
            out.append({
                "prev_talk": prev["talk"], "action": cur["action"], "next_talk": cur["talk"],
                "conf": round(min(prev["conf"], cur["conf"]), 4),
                "source": "esconv_silver", "split": "train",
            })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[silver] transitions={len(out)}  next_talk={dict(Counter(r['next_talk'] for r in out))}")
    print(f"[silver] conf quartiles: "
          f"{sorted(r['conf'] for r in out)[len(out)//4]:.2f} / "
          f"{sorted(r['conf'] for r in out)[len(out)//2]:.2f} / "
          f"{sorted(r['conf'] for r in out)[3*len(out)//4]:.2f}")
    print(f"[silver] wrote -> {args.out}")


if __name__ == "__main__":
    main()
