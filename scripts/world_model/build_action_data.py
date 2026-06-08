#!/usr/bin/env python3
"""
A1 — Build the MI action-classification dataset from AnnoMI.

For the world model's ACTION TAGGER: map a counselor utterance -> one of the
9 MI actions (open_question, closed_question, simple/complex_reflection,
information, advice, negotiation, options, other). Labels are derived from
AnnoMI's behaviour + subtype columns (same `derive_action` used for transitions),
majority-voted across annotators, split by transcript.

Usage:
  python scripts/world_model/build_action_data.py --csv data/AnnoMI-full.csv --out-dir data/world_model
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from scripts.world_model.build_transition_data import collapse_utterances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/AnnoMI-full.csv")
    ap.add_argument("--out-dir", default="data/world_model")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["transcript_id"]].append(r)

    def in_val(tid):
        return (hash((args.seed, tid)) % 1000) / 1000.0 < args.val_frac

    train, val = [], []
    for tid, trows in by_t.items():
        for u in collapse_utterances(trows):
            if u["role"] != "therapist" or u["action"] == "n/a":
                continue
            rec = {"text": u["text"], "label": u["action"], "transcript_id": tid}
            (val if in_val(tid) else train).append(rec)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for name, data in [("action_train.jsonl", train), ("action_val.jsonl", val)]:
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[action] train={len(train)} val={len(val)}")
    print(f"[action] train label dist: {dict(Counter(r['label'] for r in train))}")
    print(f"[action] wrote -> {args.out_dir}/action_{{train,val}}.jsonl")


if __name__ == "__main__":
    main()
