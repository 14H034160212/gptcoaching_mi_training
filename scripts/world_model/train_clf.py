#!/usr/bin/env python3
"""
Generic single-label text classifier trainer (class-weighted, macro-F1).
Labels are inferred from the training file. Used for the MI action tagger
and reusable for any {text,label} jsonl.

Usage:
  python scripts/world_model/train_clf.py \
      --train data/world_model/action_train.jsonl \
      --val   data/world_model/action_val.jsonl \
      --out   runs/action_clf --epochs 5
"""
import argparse
import json
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def macro_f1(preds, golds, n):
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
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    args = ap.parse_args()

    train_rows, val_rows = load(args.train), load(args.val)
    labels = sorted({r["label"] for r in train_rows})
    lab2id = {l: i for i, l in enumerate(labels)}
    print(f"[clf] {len(labels)} labels: {labels}")

    tok = AutoTokenizer.from_pretrained(args.model)

    def encode(rows):
        ds = Dataset.from_dict({"text": [r["text"] for r in rows],
                                "labels": [lab2id[r["label"]] for r in rows]})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True)

    train_ds, val_ds = encode(train_rows), encode(val_rows)

    freq = Counter(lab2id[r["label"]] for r in train_rows)
    total = sum(freq.values())
    weights = torch.tensor([total / (len(labels) * max(freq[i], 1)) for i in range(len(labels))], dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(labels),
        id2label={i: l for l, i in lab2id.items()}, label2id=lab2id)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            lab = inputs.pop("labels")
            out = model(**inputs)
            loss = nn.functional.cross_entropy(out.logits, lab, weight=weights.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def metrics(eval_pred):
        logits, labs = eval_pred
        preds = np.argmax(logits, axis=-1).tolist()
        f1s = macro_f1(preds, list(labs), len(labels))
        acc = sum(p == g for p, g in zip(preds, labs)) / len(labs)
        return {"accuracy": acc, "macro_f1": sum(f1s) / len(f1s)}

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs, per_device_eval_batch_size=32,
        learning_rate=args.lr, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        logging_steps=50, report_to=[])
    trainer = WTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                       data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    res = trainer.evaluate()
    print(f"\n=== {args.out} (val) ===  acc={res['eval_accuracy']:.4f}  macro_f1={res['eval_macro_f1']:.4f}")
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    json.dump({"labels": labels, **res}, open(f"{args.out}/val_metrics.json", "w"), indent=2)
    print(f"[clf] saved -> {args.out}")


if __name__ == "__main__":
    main()
