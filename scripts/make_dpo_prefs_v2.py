#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build DPO preferences from SFT jsonl using AntiMIResponderV2.
Output lines: {"prompt","chosen","rejected"}
"""

import argparse, json
from collections import defaultdict
from anti_mi_responder_v2 import AntiMIResponderV2

SYS_PROMPT = (
    "You are a supportive health coach using Motivational Interviewing (MI). "
    "Be non-judgmental; prefer open questions, reflective listening, and affirmations. "
    "Use the structured state and wearable summary when relevant."
)

def build_prompt_prefix(state_before, user_utt) -> str:
    return (
        f"<|system|>\n{SYS_PROMPT}\n</|system|>\n"
        f"<|context|>\nSTATE={json.dumps(state_before or {}, ensure_ascii=False)}\n</|context|>\n"
        f"<|user|>\n{(user_utt or '').strip()}\n</|user|>\n"
        f"<|assistant|>\n"
    )

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def group_by_dialog(rows):
    dialogs = defaultdict(list)
    for r in rows:
        did = str(r.get("dialog_id", "0"))
        dialogs[did].append(r)
    return dialogs

def make_pairs(sft_path, out_path, seed=123, max_samples=None):
    anti = AntiMIResponderV2(seed=seed)
    rows = list(read_jsonl(sft_path))
    dialogs = group_by_dialog(rows)
    count = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for did, turns in dialogs.items():
            # Create user->next coach pairs
            for i in range(len(turns)-1):
                cur = turns[i]
                nxt = turns[i+1]
                user = (cur.get("user_utt") or "").strip()
                coach = (nxt.get("coach_utt") or "").strip()
                if not (user and coach):
                    continue
                prompt = build_prompt_prefix(cur.get("state_before", {}), user)
                rejected = anti.generate(user, cur.get("state_before", {}))
                if rejected.strip() == coach.strip():
                    continue
                w.write(json.dumps({"prompt": prompt, "chosen": coach, "rejected": rejected}, ensure_ascii=False) + "\n")
                count += 1
                if max_samples and count >= max_samples:
                    return count
    return count

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    n = make_pairs(args.sft_file, args.out_file, seed=args.seed, max_samples=args.max_samples)
    print(f"✅ Wrote {n} pairs to {args.out_file}")

if __name__ == "__main__":
    main()
