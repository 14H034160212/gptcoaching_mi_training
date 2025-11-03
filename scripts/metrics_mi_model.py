#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics_mi_model.py
Model-based MI/MITI-style scoring using a Hugging Face text classifier.
You must provide a local HF checkpoint fine-tuned to predict MI behavior labels.
Example classes expected (customize via --labels):
  open_question, reflection_simple, reflection_complex, affirm, summary, provide_info, ask_permission, directive

Usage:
python scripts/metrics_mi_model.py \
  --pred_file runs/sample_outputs.jsonl \
  --checkpoint /path/to/hf-mi-classifier \
  --labels open_question,reflection_simple,reflection_complex,affirm,summary,provide_info,ask_permission,directive \
  --out_csv runs/mi_scores.csv

Input JSONL (minimal fields):
{"dialog_id":"...", "turn_id":1, "coach_text":"..."}
"""
import argparse, json, csv, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--labels", default="open_question,reflection_simple,reflection_complex,affirm,summary,provide_info,ask_permission,directive")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    tok = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rows, texts = [], []
    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            rows.append(ex)
            texts.append(ex.get("coach_text",""))

    out_rows = []
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i+args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1).cpu().tolist()
        for j, p in enumerate(probs):
            record = {"dialog_id": rows[i+j].get("dialog_id",""), "turn_id": rows[i+j].get("turn_id",0), "coach_text": texts[i+j]}
            for k, lab in enumerate(labels):
                record[lab] = float(p[k]) if k < len(p) else 0.0
            # Example composite MI score (favor MI behaviors, penalize directive)
            record["mi_score"] = sum(record.get(l,0.0) for l in labels if l != "directive") - 0.7*record.get("directive",0.0)
            out_rows.append(record)

    # write CSV
    import csv, os
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows -> {args.out_csv}")
if __name__ == "__main__":
    main()
