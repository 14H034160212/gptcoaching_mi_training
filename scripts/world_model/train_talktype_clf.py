#!/usr/bin/env python3
"""
Phase 1b — Train the client talk-type classifier (change/sustain/neutral).

This is the STATE ESTIMATOR for the world model: it lets us estimate the client
latent state (talk-type) on live dialogue, not just on gold AnnoMI. We report
macro-F1 on a held-out (by-transcript) split so the estimator's quality is a
real, citable number.

Class-weighted cross-entropy handles the neutral-heavy imbalance.

Usage:
  python scripts/world_model/train_talktype_clf.py \
      --model distilbert-base-uncased \
      --out runs/talktype_clf --epochs 4
"""
import argparse
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)

LABELS = ["change", "sustain", "neutral"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def macro_f1(preds, golds, n=3):
    f1s = []
    for c in range(n):
        tp = sum((p == c and g == c) for p, g in zip(preds, golds))
        fp = sum((p == c and g != c) for p, g in zip(preds, golds))
        fn = sum((p != c and g == c) for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return f1s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/world_model/talktype_train.jsonl")
    ap.add_argument("--val", default="data/world_model/talktype_val.jsonl")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--out", default="runs/talktype_clf")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=192)
    args = ap.parse_args()

    train_rows, val_rows = load(args.train), load(args.val)
    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(rows):
        ds = Dataset.from_dict({
            "text": [r["text"] for r in rows],
            "labels": [LAB2ID[r["label"]] for r in rows],
        })
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len),
                      batched=True)

    train_ds, val_ds = encode(train_rows), encode(val_rows)

    # class weights (inverse freq) for the neutral-heavy imbalance
    freq = Counter(LAB2ID[r["label"]] for r in train_rows)
    total = sum(freq.values())
    weights = torch.tensor([total / (len(LABELS) * freq[i]) for i in range(len(LABELS))],
                           dtype=torch.float)
    print(f"[clf] class weights: {dict(zip(LABELS, weights.tolist()))}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(LABELS),
        id2label={i: l for l, i in LAB2ID.items()}, label2id=LAB2ID)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = nn.functional.cross_entropy(out.logits, labels,
                                               weight=weights.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1).tolist()
        f1s = macro_f1(preds, list(labels))
        acc = sum(p == g for p, g in zip(preds, labels)) / len(labels)
        return {"accuracy": acc, "macro_f1": sum(f1s) / len(f1s),
                "f1_change": f1s[0], "f1_sustain": f1s[1], "f1_neutral": f1s[2]}

    from transformers import DataCollatorWithPadding
    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=32,
        learning_rate=args.lr, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        logging_steps=50, report_to=[],
    )
    trainer = WTrainer(model=model, args=targs, train_dataset=train_ds,
                       eval_dataset=val_ds, data_collator=DataCollatorWithPadding(tok),
                       compute_metrics=metrics)
    trainer.train()
    res = trainer.evaluate()
    print("\n=== talk-type classifier (val) ===")
    for k in ["eval_accuracy", "eval_macro_f1", "eval_f1_change", "eval_f1_sustain", "eval_f1_neutral"]:
        print(f"  {k:<18}{res[k]:.4f}")
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    json.dump(res, open(f"{args.out}/val_metrics.json", "w"), indent=2)
    print(f"\n[clf] saved -> {args.out}")


if __name__ == "__main__":
    main()
