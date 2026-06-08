#!/usr/bin/env python3
"""
Phase 1a — Build the client talk-type classification dataset from AnnoMI.

Label: client_talk_type in {change, sustain, neutral}  (we drop "n/a").
Each example = the client utterance text, optionally prefixed with the
preceding therapist turn as context (helps disambiguate change vs sustain).

Majority-vote labels across annotators per utterance. Split by transcript.

Usage:
  python scripts/world_model/build_talktype_data.py \
      --csv data/AnnoMI-full.csv --out-dir data/world_model --context
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

TALK_TYPES = {"change", "sustain", "neutral"}


def majority(values):
    c = Counter(v for v in values if v not in (None, "", "n/a"))
    if not c:
        return "n/a"
    top = c.most_common()
    best = top[0][1]
    return sorted([k for k, v in top if v == best])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/AnnoMI-full.csv")
    ap.add_argument("--out-dir", default="data/world_model")
    ap.add_argument("--context", action="store_true",
                    help="prefix preceding therapist turn as context")
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
        by_uid = defaultdict(list)
        for r in trows:
            by_uid[int(r["utterance_id"])].append(r)
        ordered = sorted(by_uid)
        prev_ther_text = None
        for uid in ordered:
            grp = by_uid[uid]
            role = majority([r["interlocutor"] for r in grp])
            text = grp[0]["utterance_text"]
            if role == "therapist":
                prev_ther_text = text
                continue
            talk = majority([r["client_talk_type"] for r in grp])
            if talk not in TALK_TYPES:
                continue
            if args.context and prev_ther_text:
                inp = f"Counselor: {prev_ther_text}\nClient: {text}"
            else:
                inp = text
            rec = {"text": inp, "label": talk, "transcript_id": tid}
            (val if in_val(tid) else train).append(rec)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    for name, data in [("talktype_train.jsonl", train), ("talktype_val.jsonl", val)]:
        with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[talktype] train={len(train)} val={len(val)} context={args.context}")
    print(f"[talktype] train label dist: {dict(Counter(r['label'] for r in train))}")
    print(f"[talktype] val   label dist: {dict(Counter(r['label'] for r in val))}")
    print(f"[talktype] wrote -> {args.out_dir}/talktype_{{train,val}}.jsonl")


if __name__ == "__main__":
    main()
