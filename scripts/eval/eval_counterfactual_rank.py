#!/usr/bin/env python3
"""
B2 — Counterfactual ranking evaluation.

The planner (fit on TRAIN) ranks MI actions per client state. We check that
ranking against what actually happened on the held-out VAL split:

  (1) RANK AGREEMENT: per state, Spearman correlation between the planner's
      predicted P(change|state,action) and the EMPIRICAL change-talk rate of
      each action measured on val (actions with enough support).
  (2) TOP-1 AGREEMENT: how often the planner's top action equals the action
      with the highest empirical change rate on val.
  (3) OFFLINE POLICY UPLIFT: empirical change rate of the planner-recommended
      action minus the val base change rate for that state — an estimate of the
      change-talk gain from following the planner.

Held-out & empirical, so this is a real test, not the model judging itself.
"""
import json
from collections import Counter, defaultdict

TALKS = ["change", "sustain", "neutral"]


def _spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else None


def empirical_change_rate(rows, min_support=8):
    """P(change | prev_talk, action) measured empirically, cells with >= min_support."""
    cell = defaultdict(Counter)
    for r in rows:
        cell[(r["prev_talk"], r["action"])][r["next_talk"]] += 1
    out = {}
    for key, c in cell.items():
        n = sum(c.values())
        if n >= min_support:
            out[key] = (c["change"] / n, n)
    return out


def run(data="data/world_model/transitions.jsonl", horizon=3, gamma=0.9, min_support=8):
    from scripts.world_model.transition_model import TransitionModel
    from scripts.world_model.planner import rank_actions

    all_rows = [json.loads(l) for l in open(data, encoding="utf-8")]
    train = [r for r in all_rows if r["split"] == "train"]
    val = [r for r in all_rows if r["split"] == "val"]
    wm = TransitionModel().fit(train)

    emp = empirical_change_rate(val, min_support)          # (state,action) -> (rate,n)
    val_base = defaultdict(Counter)                        # base change rate per state on val
    for r in val:
        val_base[r["prev_talk"]][r["next_talk"]] += 1

    spearmans, top1_hits, top1_total = [], 0, 0
    uplifts = []
    per_state = {}
    for state in TALKS:
        # planner ranking (predicted P_change per action)
        ranked = rank_actions(wm, state, horizon, gamma)
        planner_pc = {r["action"]: r["P_change"] for r in ranked}
        planner_best = ranked[0]["action"]
        # actions with empirical support in this state
        supported = [(a, emp[(state, a)][0]) for a in planner_pc if (state, a) in emp]
        if len(supported) >= 2:
            xs = [planner_pc[a] for a, _ in supported]
            ys = [rate for _, rate in supported]
            rho = _spearman(xs, ys)
            emp_best = max(supported, key=lambda kv: kv[1])[0]
            top1_total += 1
            if planner_best == emp_best or planner_best not in dict(supported):
                # count a hit only when planner_best is itself supported & matches
                if planner_best == emp_best:
                    top1_hits += 1
            # offline uplift: emp change-rate of planner_best (if supported) vs val base
            base = (val_base[state]["change"] / max(sum(val_base[state].values()), 1))
            if (state, planner_best) in emp:
                uplifts.append(emp[(state, planner_best)][0] - base)
            per_state[state] = {
                "planner_best": planner_best,
                "empirical_best": emp_best,
                "spearman": round(rho, 3) if rho is not None else None,
                "n_supported_actions": len(supported),
            }
            if rho is not None:
                spearmans.append(rho)

    return {
        "min_support": min_support,
        "mean_spearman_planner_vs_empirical": round(sum(spearmans) / len(spearmans), 3) if spearmans else None,
        "top1_agreement": f"{top1_hits}/{top1_total}" if top1_total else "n/a",
        "mean_offline_change_uplift": round(sum(uplifts) / len(uplifts), 3) if uplifts else None,
        "per_state": per_state,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
