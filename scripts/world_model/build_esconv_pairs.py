#!/usr/bin/env python3
"""
JEPA data lever — build self-supervised (context, action, future_text) pairs
from ESConv (thu-coai/esconv), a 1.3k multi-turn counseling-style corpus.

NO talk-type labels (ESConv has none) — these are for SELF-SUPERVISED predictor
pretraining only. The talk-type probe stays on AnnoMI (which has gold labels).

ESConv supporter ("sys") strategies map to our MI action space; the client
("usr") response is the prediction target. Same shape as AnnoMI transitions.

  python -m scripts.world_model.build_esconv_pairs --out data/world_model/esconv_pairs.jsonl
"""
import argparse
import json
from pathlib import Path

# ESConv strategy -> our 9-action MI space (build_transition_data.ACTIONS)
STRAT2ACTION = {
    "Question": "open_question",
    "Restatement or Paraphrasing": "simple_reflection",
    "Reflection of feelings": "complex_reflection",
    "Self-disclosure": "other",
    "Affirmation and Reassurance": "other",
    "Providing Suggestions": "advice",
    "Information": "information",
    "Others": "other",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/world_model/esconv_pairs.jsonl")
    ap.add_argument("--max-history", type=int, default=6)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("thu-coai/esconv")

    pairs = []
    for split in ds:
        for row in ds[split]:
            d = json.loads(row["text"])
            convo = d["dialog"]
            history = []          # list of {role, text}
            last_action = None
            for turn in convo:
                spk = turn.get("speaker")
                text = (turn.get("text") or "").strip()
                if not text:
                    continue
                if spk == "sys":  # counselor
                    strat = turn.get("strategy")
                    last_action = STRAT2ACTION.get(strat, "other")
                    if strat and strat.startswith("Question") and not text.endswith("?"):
                        pass
                    history.append({"role": "therapist", "text": text})
                else:             # client (usr)
                    if last_action is not None and history:
                        pairs.append({
                            "history": history[-args.max_history:],
                            "action": last_action,
                            "future_text": text,
                        })
                    history.append({"role": "client", "text": text})
                    last_action = None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"[esconv] pairs={len(pairs)}  action dist={dict(Counter(p['action'] for p in pairs))}")
    print(f"[esconv] wrote -> {args.out}")


if __name__ == "__main__":
    main()
