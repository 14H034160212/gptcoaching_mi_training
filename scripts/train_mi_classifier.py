#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_mi_classifier.py
Train a multi-label MI behavior classifier on unified JSONL (use normalized labels if available).

Assumptions:
- Input JSONL has fields: coach_utt (text), mi_tags (list[str]).
- We'll do multi-label classification over a fixed label set passed via --labels.
- This produces a Hugging Face checkpoint usable by `metrics_mi_model.py`.

Example:
python scripts/train_mi_classifier.py \
  --train_file data/mi_unified_from_annomi.norm.jsonl \
  --labels open_question,reflection_simple,reflection_complex,affirm,summary,provide_info,ask_permission,directive \
  --model_name_or_path distilbert-base-uncased \
  --output_dir runs/mi_classifier \
  --num_train_epochs 3 --per_device_train_batch_size 16
"""
import argparse, json, os
from dataclasses import dataclass
from typing import List, Dict, Any
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments)
import torch
import numpy as np

@dataclass
class Config:
    labels: List[str]

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--eval_file")
    ap.add_argument("--labels", required=True, help="comma-separated label list")
    ap.add_argument("--model_name_or_path", default="distilbert-base-uncased")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    label_list = [s.strip() for s in args.labels.split(",") if s.strip()]
    lab2id = {l:i for i,l in enumerate(label_list)}

    train_rows = load_jsonl(args.train_file)
    if args.eval_file:
        eval_rows = load_jsonl(args.eval_file)
    else:
        # simple split
        split = int(0.9 * len(train_rows)) if len(train_rows) > 100 else max(1, int(0.8*len(train_rows)))
        eval_rows = train_rows[split:]
        train_rows = train_rows[:split]

    def to_examples(rows):
        X, Y = [], []
        for r in rows:
            text = r.get("coach_utt") or r.get("coach_text") or ""
            tags = r.get("mi_tags", [])
            y = [0]*len(label_list)
            for t in tags:
                if t in lab2id:
                    y[lab2id[t]] = 1
            X.append(text)
            Y.append(y)
        return X, Y

    Xtr, Ytr = to_examples(train_rows)
    Xev, Yev = to_examples(eval_rows)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    def enc(texts):
        return tok(texts, truncation=True, padding=True, max_length=512)

    enc_tr = enc(Xtr)
    enc_ev = enc(Xev)
    ds_tr = Dataset.from_dict({**enc_tr, "labels": Ytr})
    ds_ev = Dataset.from_dict({**enc_ev, "labels": Yev})

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        problem_type="multi_label_classification"
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1/(1+np.exp(-logits))
        preds = (probs>=0.5).astype(int)
        # micro-F1
        tp = (preds*labels).sum()
        fp = (preds*(1-labels)).sum()
        fn = ((1-preds)*labels).sum()
        precision = tp/(tp+fp+1e-9)
        recall = tp/(tp+fn+1e-9)
        f1 = 2*precision*recall/(precision+recall+1e-9)
        return {"micro_f1": float(f1), "precision": float(precision), "recall": float(recall)}

    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1"
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds_tr,
        eval_dataset=ds_ev,
        tokenizer=tok,
        compute_metrics=compute_metrics
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
        json.dump(label_list, f)

if __name__ == "__main__":
    main()
