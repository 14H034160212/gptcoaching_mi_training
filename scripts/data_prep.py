#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalize MI datasets (MI-TAGS / AnnoMI / MI-Dataset) into a common JSONL.
Usage:
  python scripts/data_prep.py --mi_tags_csv PATH --annomi_csv PATH --mi_dataset_json PATH --out_jsonl data/mi_unified_train.jsonl
You can pass any subset of inputs.
"""
import argparse, json, csv, re, os, sys
from typing import Dict, Iterable

def write_jsonl(records: Iterable[Dict], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def from_mi_tags(path: str):
    # Placeholder parser: adapt columns to your MI-TAGS CSV
    # Expected columns (example): session_id,turn_id,speaker,utterance,mi_label
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        session_state = {}
        for row in reader:
            if row.get("speaker","").lower() in ("therapist","coach","counselor"):
                yield {
                    "dialog_id": row.get("session_id",""),
                    "turn_id": int(row.get("turn_id","0")),
                    "user_utt": "",   # will be filled by pairing if available
                    "coach_utt": norm_text(row.get("utterance","")),
                    "mi_tags": [row.get("mi_label","")],
                    "state_before": session_state.get(row.get("session_id",""), {}),
                    "state_after": {}
                }

def from_annomi(path: str):
    # Placeholder parser for AnnoMI CSV
    # Expected columns: dialog_id,turn_id,role,utterance,tags
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = (row.get("interlocutor", "") or "").lower()
            utt = norm_text(row.get("utterance_text", ""))
            tags = [t.strip() for t in (row.get("main_therapist_behaviour", "") or "").split("|") if t.strip()]

            # Therapist = coach turn
            if role in ("therapist", "coach", "counselor"):
                yield {
                    "dialog_id": row.get("transcript_id", ""),
                    "turn_id": int(row.get("utterance_", "0")),
                    "user_utt": "",
                    "coach_utt": utt,
                    "mi_tags": tags or ["unknown"],
                    "state_before": {},
                    "state_after": {}
                }

            # Client = user turn (optional, if you want to preserve)
            elif role == "client":
                yield {
                    "dialog_id": row.get("transcript_id", ""),
                    "turn_id": int(row.get("utterance_", "0")),
                    "user_utt": utt,
                    "coach_utt": "",
                    "mi_tags": ["client"],
                    "state_before": {},
                    "state_after": {}
                }

def from_mi_dataset(path: str):
    # Placeholder for a JSON list of conversations with MITI-derived tags
    data = json.load(open(path, "r", encoding="utf-8"))
    for conv in data:
        did = conv.get("dialog_id","")
        for t in conv.get("turns", []):
            if t.get("role") in ("therapist","coach","counselor"):
                yield {
                    "dialog_id": did,
                    "turn_id": int(t.get("turn_id",0)),
                    "user_utt": t.get("prev_user_utt",""),
                    "coach_utt": norm_text(t.get("text","")),
                    "mi_tags": t.get("mi_tags",["unknown"]),
                    "state_before": t.get("state_before",{}),
                    "state_after": t.get("state_after",{})
                }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mi_tags_csv")
    ap.add_argument("--annomi_csv")
    ap.add_argument("--mi_dataset_json")
    ap.add_argument("--extra_jsonl", help="Any existing JSONL already in unified format")
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    records = []
    if args.mi_tags_csv and os.path.exists(args.mi_tags_csv):
        records += list(from_mi_tags(args.mi_tags_csv))
    if args.annomi_csv and os.path.exists(args.annomi_csv):
        records += list(from_annomi(args.annomi_csv))
    if args.mi_dataset_json and os.path.exists(args.mi_dataset_json):
        records += list(from_mi_dataset(args.mi_dataset_json))
    if args.extra_jsonl and os.path.exists(args.extra_jsonl):
        with open(args.extra_jsonl, "r", encoding="utf-8") as f:
            records += [json.loads(line) for line in f]

    if not records:
        print("No input files found. Writing example dataset instead.", file=sys.stderr)
        with open("data/example_mi_dialogs.jsonl","r",encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

    write_jsonl(records, args.out_jsonl)
    print(f"Wrote {len(records)} records to {args.out_jsonl}")

if __name__ == "__main__":
    main()
