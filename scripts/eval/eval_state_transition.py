#!/usr/bin/env python3
"""
B1 — State + transition evaluation.

Two things the world model must get right, both on held-out (by-transcript) data:

  (1) STATE ESTIMATION: the talk-type classifier vs gold AnnoMI client_talk_type
      -> accuracy, macro-F1, per-class F1, confusion matrix.
  (2) TRANSITION MODEL: P(next_talk | prev_talk, action) vs baselines
      -> accuracy + macro-F1 lift over the momentum baseline (does the ACTION help?).

Returns a dict (also runnable standalone). No training here — eval only.
"""
import json
import os
from collections import Counter, defaultdict

TALKS = ["change", "sustain", "neutral"]


def _macro_f1(preds, golds, classes):
    f1 = {}
    for c in classes:
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1[c] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return f1


def eval_state_estimator(val_path="data/world_model/talktype_val.jsonl",
                         clf_path=None):
    # prefer the production mpnet labeler if present
    if clf_path is None:
        clf_path = next((p for p in ("runs/talktype_clf_mpnet", "runs/talktype_clf")
                         if os.path.isdir(p)), "runs/talktype_clf")
    if not os.path.isdir(clf_path):
        return {"status": "skipped (no talk-type classifier)"}
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(clf_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(clf_path).eval().to(dev)
    rows = [json.loads(l) for l in open(val_path, encoding="utf-8")]
    preds, golds = [], []
    confusion = defaultdict(Counter)
    bs = 64
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]
        enc = tok([r["text"] for r in batch], return_tensors="pt", truncation=True,
                  max_length=192, padding=True)
        enc.pop("token_type_ids", None)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            ids = mdl(**enc).logits.argmax(-1).tolist()
        for r, idx in zip(batch, ids):
            p = mdl.config.id2label[idx]
            preds.append(p); golds.append(r["label"]); confusion[r["label"]][p] += 1
    f1 = _macro_f1(preds, golds, TALKS)
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    return {
        "n": len(golds), "accuracy": round(acc, 4),
        "macro_f1": round(sum(f1.values()) / len(f1), 4),
        "per_class_f1": {k: round(v, 4) for k, v in f1.items()},
        "confusion": {g: dict(c) for g, c in confusion.items()},
    }


def eval_transition_model(data="data/world_model/transitions.jsonl", alpha=1.0, min_cell=3):
    from scripts.world_model.transition_model import TransitionModel, _laplace
    train = [json.loads(l) for l in open(data, encoding="utf-8") if json.loads(l)["split"] == "train"]
    val = [json.loads(l) for l in open(data, encoding="utf-8") if json.loads(l)["split"] == "val"]
    wm = TransitionModel(alpha=alpha, min_cell=min_cell).fit(train)

    # baselines
    c_next = Counter(r["next_talk"] for r in train)
    c_prev = defaultdict(Counter)
    for r in train:
        c_prev[r["prev_talk"]][r["next_talk"]] += 1
    prior_pred = max(c_next, key=lambda k: c_next[k])

    def mom_pred(pt):
        d = _laplace(c_prev.get(pt, Counter()), TALKS, alpha)
        return max(d, key=lambda k: d[k])

    def wm_pred(pt, a):
        d = wm.predict_dist(pt, a)
        return max(d, key=lambda k: d[k])

    def acc_f1(predfn):
        preds = [predfn(r) for r in val]
        golds = [r["next_talk"] for r in val]
        f1 = _macro_f1(preds, golds, TALKS)
        return (round(sum(p == g for p, g in zip(preds, golds)) / len(golds), 4),
                round(sum(f1.values()) / len(f1), 4))

    b0 = acc_f1(lambda r: prior_pred)
    b1 = acc_f1(lambda r: mom_pred(r["prev_talk"]))
    wmr = acc_f1(lambda r: wm_pred(r["prev_talk"], r["action"]))
    return {
        "n_val": len(val),
        "prior":     {"acc": b0[0], "macro_f1": b0[1]},
        "momentum":  {"acc": b1[0], "macro_f1": b1[1]},
        "world_model": {"acc": wmr[0], "macro_f1": wmr[1]},
        "action_helps_f1_lift": round(wmr[1] - b1[1], 4),
    }


def run():
    return {
        "state_estimator": eval_state_estimator(),
        "transition_model": eval_transition_model(),
    }


if __name__ == "__main__":
    import json as _j
    print(_j.dumps(run(), indent=2))
