#!/usr/bin/env python3
"""
Reusable tabular MI world model:  P(next_talk | prev_talk, action)

Loaded by the MPC planner / counterfactual modules. Fitting logic mirrors
transition_matrix.py (Tier 1b) but packaged as a class with rollout + reward.
"""
import json
from collections import defaultdict, Counter

TALKS = ["change", "sustain", "neutral"]

# Refined MI action space (must match build_transition_data.ACTIONS).
ACTIONS = [
    "open_question", "closed_question", "simple_reflection", "complex_reflection",
    "information", "advice", "negotiation", "options", "other",
]


def _laplace(counter, classes, alpha=1.0):
    total = sum(counter.get(c, 0) for c in classes)
    denom = total + alpha * len(classes)
    return {c: (counter.get(c, 0) + alpha) / denom for c in classes}


class TransitionModel:
    """Count-based P(next_talk | prev_talk, action) with momentum back-off."""

    def __init__(self, alpha=1.0, min_cell=3):
        self.alpha = alpha
        self.min_cell = min_cell
        self.c_next = Counter()
        self.c_prev_next = defaultdict(Counter)
        self.c_pa_next = defaultdict(Counter)
        self.c_act_next = defaultdict(Counter)
        self._fitted = False

    def fit(self, records):
        for r in records:
            pt, a, nt = r["prev_talk"], r["action"], r["next_talk"]
            self.c_next[nt] += 1
            self.c_prev_next[pt][nt] += 1
            self.c_pa_next[(pt, a)][nt] += 1
            self.c_act_next[a][nt] += 1
        self._fitted = True
        return self

    @classmethod
    def from_jsonl(cls, path, split="train", alpha=1.0, min_cell=3):
        recs = []
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if split is None or r.get("split") == split:
                recs.append(r)
        return cls(alpha=alpha, min_cell=min_cell).fit(recs)

    # ---- core dynamics ----
    def predict_dist(self, prev_talk, action):
        """P(next_talk | prev_talk, action), backing off to momentum then prior."""
        cell = self.c_pa_next.get((prev_talk, action))
        if cell and sum(cell.values()) >= self.min_cell:
            return _laplace(cell, TALKS, self.alpha)
        mom = self.c_prev_next.get(prev_talk)
        if mom and sum(mom.values()) >= self.min_cell:
            return _laplace(mom, TALKS, self.alpha)
        return _laplace(self.c_next, TALKS, self.alpha)

    def prior(self):
        return _laplace(self.c_next, TALKS, self.alpha)

    def base_change_rate(self):
        return self.prior()["change"]
