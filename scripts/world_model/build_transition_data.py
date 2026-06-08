#!/usr/bin/env python3
"""
Tier 1a — Build the MI world-model transition dataset from AnnoMI.

Core claim we want to test/validate:
    (prev client talk-type, therapist action) -> next client talk-type

AnnoMI-full.csv ships real labels per utterance:
  - interlocutor               : therapist | client
  - main_therapist_behaviour   : question | reflection | therapist_input | other
  - {question,reflection,therapist_input}_subtype : refines the action
  - client_talk_type           : change | sustain | neutral | n/a
  - mi_quality                 : high | low      (reward signal)

Each utterance has 1..10 annotator rows -> we MAJORITY-VOTE per utterance.

Output (dual-use):
  - transitions.jsonl : one record per state->action->state transition.
      tabular fields  : prev_talk, action, behaviour, next_talk, mi_quality
      JEPA fields     : history (list of {role,text}), action, future_text
  - split is by transcript_id (no leakage between train/val).

Usage:
  python scripts/world_model/build_transition_data.py \
      --csv data/AnnoMI-full.csv \
      --out data/world_model/transitions.jsonl \
      --val-frac 0.2
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

TALK_TYPES = {"change", "sustain", "neutral"}  # we drop "n/a"

# Refined MI action space derived from AnnoMI behaviour + subtype columns.
ACTIONS = [
    "open_question",
    "closed_question",
    "simple_reflection",
    "complex_reflection",
    "information",
    "advice",
    "negotiation",
    "options",
    "other",
]


def majority(values):
    """Majority vote over a list of strings, ignoring empties. Stable on ties."""
    c = Counter(v for v in values if v not in (None, "", "n/a"))
    if not c:
        return "n/a"
    top = c.most_common()
    best = top[0][1]
    # tie-break deterministically by label to stay reproducible
    return sorted([k for k, v in top if v == best])[0]


def derive_action(rows):
    """Map a therapist utterance's annotator rows to one refined MI action."""
    behaviour = majority([r["main_therapist_behaviour"] for r in rows])
    if behaviour == "question":
        sub = majority([r["question_subtype"] for r in rows])
        return {"open": "open_question", "closed": "closed_question"}.get(sub, "open_question"), behaviour
    if behaviour == "reflection":
        sub = majority([r["reflection_subtype"] for r in rows])
        return {"simple": "simple_reflection", "complex": "complex_reflection"}.get(sub, "simple_reflection"), behaviour
    if behaviour == "therapist_input":
        sub = majority([r["therapist_input_subtype"] for r in rows])
        return {
            "information": "information",
            "advice": "advice",
            "negotiation": "negotiation",
            "options": "options",
        }.get(sub, "information"), behaviour
    return "other", behaviour if behaviour != "n/a" else "other"


def collapse_utterances(rows):
    """Group annotator rows by utterance_id, majority-vote the labels.

    Returns list of utterances ordered by utterance_id, each:
      {uid, role, text, talk_type, action, behaviour, mi_quality}
    """
    by_uid = defaultdict(list)
    for r in rows:
        by_uid[int(r["utterance_id"])].append(r)

    utts = []
    for uid in sorted(by_uid):
        grp = by_uid[uid]
        role = majority([r["interlocutor"] for r in grp])
        text = grp[0]["utterance_text"]  # identical across annotators
        mi_quality = majority([r["mi_quality"] for r in grp])
        if role == "therapist":
            action, behaviour = derive_action(grp)
            talk_type = "n/a"
        else:
            action, behaviour = "n/a", "n/a"
            talk_type = majority([r["client_talk_type"] for r in grp])
        utts.append({
            "uid": uid,
            "role": role,
            "text": text,
            "talk_type": talk_type,
            "action": action,
            "behaviour": behaviour,
            "mi_quality": mi_quality,
        })
    return utts


def build_transitions(utts, transcript_id, max_history=8):
    """Walk an ordered utterance list and emit state->action->state transitions.

    Transition = (prev client talk-type) --[immediately preceding therapist action]--> (next client talk-type).
    The therapist action used is the one closest before the responding client turn
    (most-proximate stimulus), which is the standard turn-level MI analysis.
    """
    out = []
    prev_talk = None          # last client talk-type seen (state s)
    last_action = None        # most recent therapist action between client turns
    last_behaviour = None
    last_quality = None

    for i, u in enumerate(utts):
        if u["role"] == "therapist":
            if u["action"] != "n/a":
                last_action = u["action"]
                last_behaviour = u["behaviour"]
                last_quality = u["mi_quality"]
        elif u["role"] == "client":
            cur_talk = u["talk_type"]
            if (prev_talk in TALK_TYPES
                    and last_action is not None
                    and cur_talk in TALK_TYPES):
                history = [{"role": x["role"], "text": x["text"]}
                           for x in utts[max(0, i - max_history):i]]
                out.append({
                    "transcript_id": transcript_id,
                    "uid": u["uid"],
                    # --- tabular fields ---
                    "prev_talk": prev_talk,
                    "action": last_action,
                    "behaviour": last_behaviour,
                    "next_talk": cur_talk,
                    "mi_quality": last_quality,
                    # --- JEPA / sequence dual-use fields ---
                    "history": history,
                    "future_text": u["text"],
                })
            if cur_talk in TALK_TYPES:
                prev_talk = cur_talk
            last_action = None  # consume the action once the client responds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/AnnoMI-full.csv")
    ap.add_argument("--out", default="data/world_model/transitions.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-history", type=int, default=8)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    by_transcript = defaultdict(list)
    for r in rows:
        by_transcript[r["transcript_id"]].append(r)

    # deterministic transcript-level split (no leakage)
    tids = sorted(by_transcript)
    # simple reproducible hash split
    def in_val(tid):
        return (hash((args.seed, tid)) % 1000) / 1000.0 < args.val_frac

    all_records = []
    n_val_t = 0
    for tid in tids:
        utts = collapse_utterances(by_transcript[tid])
        trans = build_transitions(utts, tid, max_history=args.max_history)
        split = "val" if in_val(tid) else "train"
        if split == "val":
            n_val_t += 1
        for t in trans:
            t["split"] = split
        all_records.extend(trans)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in all_records:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # summary
    n_train = sum(1 for t in all_records if t["split"] == "train")
    n_val = len(all_records) - n_train
    act_dist = Counter(t["action"] for t in all_records)
    talk_dist = Counter(t["next_talk"] for t in all_records)
    print(f"[build] transcripts={len(tids)} (val transcripts={n_val_t})")
    print(f"[build] transitions={len(all_records)}  train={n_train}  val={n_val}")
    print(f"[build] action dist: {dict(act_dist)}")
    print(f"[build] next_talk dist: {dict(talk_dist)}")
    print(f"[build] wrote -> {args.out}")


if __name__ == "__main__":
    main()
