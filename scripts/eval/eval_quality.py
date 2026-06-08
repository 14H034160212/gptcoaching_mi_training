#!/usr/bin/env python3
"""
B5 — Quality / clinical-validity eval dimensions (beyond raw accuracy).

  (1) MI-consistency of recommendations: over real AnnoMI-val states, how often is
      the planner's TOP action MI-adherent (open question / reflection / affirmation-
      like / negotiation / options) vs directive (advice / closed question). A good
      MI world model should mostly recommend evoking, not directing.
  (2) Calibration (Brier): mean Brier score of the transition model's predicted
      talk-type distribution vs the one-hot gold next_talk on val. Lower = better.
  (3) Clinical validity (action-effect directionality): does the model agree with MI
      theory that EVOKING actions raise change talk more than DIRECTIVE actions?
      Reports mean P(change) gap (evoking - directive); should be > 0.

Model-agnostic: works on the tabular or the JEPA-backed transition model.
"""
import json
from collections import Counter

from scripts.world_model.transition_model import TransitionModel, TALKS, ACTIONS
from scripts.world_model.planner import rank_actions

EVOKING = {"open_question", "complex_reflection", "simple_reflection", "negotiation", "options"}
DIRECTIVE = {"advice", "closed_question", "information"}


def _load(data):
    return [json.loads(l) for l in open(data, encoding="utf-8")]


def run(data="data/world_model/transitions.jsonl", prod="data/world_model/transitions_prod.jsonl"):
    import os
    train_path = prod if os.path.exists(prod) else data
    wm = TransitionModel().fit([r for r in _load(train_path) if r.get("split", "train") == "train"])
    val = [r for r in _load(data) if r["split"] == "val"]

    # (1) MI-consistency of the planner's top recommendation per distinct val state
    states = sorted({r["prev_talk"] for r in val})
    adherent = 0
    rec = {}
    for s in states:
        top = rank_actions(wm, s, horizon=3)[0]["action"]
        rec[s] = top
        if top in EVOKING:
            adherent += 1
    mi_consistency = adherent / len(states)

    # (2) calibration: Brier score on val
    brier = 0.0
    for r in val:
        d = wm.predict_dist(r["prev_talk"], r["action"])
        for t in TALKS:
            y = 1.0 if t == r["next_talk"] else 0.0
            brier += (d[t] - y) ** 2
    brier /= len(val)

    # (3) clinical validity: mean P(change|action) for evoking vs directive (marginal over states)
    def mean_pchange(action_set):
        vals = []
        for s in TALKS:
            for a in action_set:
                vals.append(wm.predict_dist(s, a)["change"])
        return sum(vals) / len(vals)
    ev, di = mean_pchange(EVOKING), mean_pchange(DIRECTIVE)

    return {
        "mi_consistency_of_recommendations": round(mi_consistency, 3),
        "recommended_top_action_by_state": rec,
        "brier_score (lower better)": round(brier, 4),
        "clinical_validity_change_gap (evoking - directive, >0 good)": round(ev - di, 3),
        "mean_Pchange_evoking": round(ev, 3),
        "mean_Pchange_directive": round(di, 3),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
