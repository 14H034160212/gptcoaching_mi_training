#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
normalize_labels.py
Apply a mapping to mi_tags in a unified JSONL to consolidate fine-grained labels.
Usage:
python scripts/normalize_labels.py \
  --in_jsonl data/mi_unified_from_annomi.jsonl \
  --map_json data/mi_label_map.json \
  --out_jsonl data/mi_unified_from_annomi.norm.jsonl
"""
import argparse, json, re

def load_map(path):
    raw = json.load(open(path, "r", encoding="utf-8"))
    canon = {}
    for k, variants in raw.items():
        canon[k] = set([k.lower()])
        for v in variants:
            canon[k].add(str(v).lower())
    return canon

def norm_tags(tags, cmap):
    out = set()
    for t in tags or []:
        t_low = str(t).strip().lower()
        for canon, variants in cmap.items():
            if t_low in variants:
                out.add(canon)
                break
    return sorted(out) if out else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--map_json", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()

    cmap = load_map(args.map_json)
    w = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fout, open(args.in_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)
            ex["mi_tags"] = norm_tags(ex.get("mi_tags", []), cmap)
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
            w += 1
    print(f"Wrote {w} records -> {args.out_jsonl}")

if __name__ == "__main__":
    main()
