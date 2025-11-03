#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple MI-style metrics: heuristic coverage and ratios for open questions, reflections, affirmations, summaries.
"""
import argparse, json, re
from collections import Counter

OPEN_Q = re.compile(r"\?$")
AFFIRM = re.compile(r"\b(good job|that makes sense|you've been|sounds like|i appreciate)\b", re.I)
REFLECT = re.compile(r"\b(you feel|you're saying|it sounds like|so you|you seem)\b", re.I)
SUMMARY = re.compile(r"\b(let me summarize|to recap|what we discussed|summary)\b", re.I)

def score_line(text: str):
    flags = []
    if OPEN_Q.search(text.strip()): flags.append("open_q")
    if AFFIRM.search(text): flags.append("affirm")
    if REFLECT.search(text): flags.append("reflect")
    if SUMMARY.search(text): flags.append("summary")
    return flags or ["other"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_file", required=True, help="JSONL with fields: dialog_id, turn_id, coach_text")
    args = ap.parse_args()

    counts = Counter()
    n = 0
    with open(args.pred_file, "r", encoding="utf-8") as f:
        for line in f:
            n += 1
            ex = json.loads(line)
            tags = score_line(ex.get("coach_text",""))
            counts.update(tags)

    for k,v in counts.items():
        print(f"{k}: {v} ({v/max(1,n):.2%})")
    print(f"Total: {n}")

if __name__ == "__main__":
    main()
