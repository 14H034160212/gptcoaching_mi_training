#!/usr/bin/env python3
"""
Tier 2 — MPC counterfactual planner over the MI action space.

Given the current client state (talk-type) we score every MI action by a
finite-horizon lookahead in the tabular world model (PlaNet/TD-MPC-style
planning-without-policy; the H-step value lookahead stands in for TD-MPC's
learned value head). This is the backend of the counterfactual feedback panel:

    "You used <closed_question>  -> P(change)=0.18  Q=-0.05
     Better: <open_question>     -> P(change)=0.37  Q=+0.41
     i.e. predicted change-talk uplift +0.19"

Reward shaping (per resulting client talk-type):
    change=+1, sustain=-1, neutral=0     (we want to EVOKE change talk)

Usage:
  python scripts/world_model/planner.py --state neutral --horizon 3
"""
import argparse
import json
from scripts.world_model.transition_model import TransitionModel, TALKS, ACTIONS

TALK_REWARD = {"change": 1.0, "sustain": -1.0, "neutral": 0.0}

# Example phrasings so the UI can show a concrete "say something like..." per action.
ACTION_TEMPLATES = {
    "open_question": "What would make a change feel worth it to you?",
    "closed_question": "Do you want to cut down — yes or no?",
    "simple_reflection": "So you're feeling stuck right now.",
    "complex_reflection": "Part of you wants this to change, and part of you isn't sure it's possible.",
    "information": "Here are the facts about what this does to your body.",
    "advice": "You should start by cutting back this week.",
    "negotiation": "Could we agree on one small step you'd be willing to try?",
    "options": "We could try A, B, or C — which feels most doable?",
    "other": "(non-MI / filler turn)",
}


def immediate_reward(model, state, action):
    """E_{s'~P(.|s,a)} reward(s')."""
    d = model.predict_dist(state, action)
    return sum(d[s2] * TALK_REWARD[s2] for s2 in TALKS), d


def finite_horizon_values(model, horizon, gamma):
    """Backward induction:  V_h(s) = max_a [ R(s,a) + gamma * E V_{h-1}(s') ]."""
    V = {s: 0.0 for s in TALKS}
    for _ in range(horizon):
        newV = {}
        for s in TALKS:
            best = float("-inf")
            for a in ACTIONS:
                r, d = immediate_reward(model, s, a)
                q = r + gamma * sum(d[s2] * V[s2] for s2 in TALKS)
                best = max(best, q)
            newV[s] = best
        V = newV
    return V


def rank_actions(model, state, horizon=3, gamma=0.9):
    """Q-value of taking each action NOW, with (H-1)-step lookahead afterward."""
    Vnext = finite_horizon_values(model, max(0, horizon - 1), gamma)
    rows = []
    for a in ACTIONS:
        r, d = immediate_reward(model, state, a)
        q = r + gamma * sum(d[s2] * Vnext[s2] for s2 in TALKS)
        rows.append({
            "action": a,
            "P_change": round(d["change"], 3),
            "P_sustain": round(d["sustain"], 3),
            "P_neutral": round(d["neutral"], 3),
            "immediate_reward": round(r, 3),
            "Q": round(q, 3),
            "template": ACTION_TEMPLATES[a],
        })
    rows.sort(key=lambda x: -x["Q"])
    return rows


def counterfactual(model, state, actual_action, horizon=3, gamma=0.9):
    """Compare the counselor's actual action against the model-optimal one."""
    ranked = rank_actions(model, state, horizon, gamma)
    by_action = {r["action"]: r for r in ranked}
    actual = by_action.get(actual_action, ranked[-1])
    best = ranked[0]
    return {
        "state": state,
        "actual": actual,
        "best": best,
        "change_uplift": round(best["P_change"] - actual["P_change"], 3),
        "Q_gain": round(best["Q"] - actual["Q"], 3),
        "is_optimal": actual["action"] == best["action"],
        "ranked": ranked,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/world_model/transitions.jsonl")
    ap.add_argument("--state", default="neutral", choices=TALKS)
    ap.add_argument("--actual-action", default=None, choices=ACTIONS + [None])
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.9)
    args = ap.parse_args()

    model = TransitionModel.from_jsonl(args.data, split="train")

    print(f"\n=== MPC action ranking | state='{args.state}' "
          f"horizon={args.horizon} gamma={args.gamma} ===\n")
    print(f"{'action':<20}{'P(change)':>10}{'P(sustain)':>11}{'imm_R':>8}{'Q':>8}")
    for r in rank_actions(model, args.state, args.horizon, args.gamma):
        print(f"{r['action']:<20}{r['P_change']:>10.2f}{r['P_sustain']:>11.2f}"
              f"{r['immediate_reward']:>8.2f}{r['Q']:>8.2f}")

    if args.actual_action:
        cf = counterfactual(model, args.state, args.actual_action, args.horizon, args.gamma)
        print(f"\n--- Counterfactual feedback ---")
        print(f"You used      : {cf['actual']['action']:<20} P(change)={cf['actual']['P_change']:.2f}  Q={cf['actual']['Q']:.2f}")
        print(f"Model suggests: {cf['best']['action']:<20} P(change)={cf['best']['P_change']:.2f}  Q={cf['best']['Q']:.2f}")
        print(f"  -> predicted change-talk uplift {cf['change_uplift']:+.2f},  Q gain {cf['Q_gain']:+.2f}")
        print(f"  -> try: \"{cf['best']['template']}\"")
        print(f"  (already optimal)" if cf["is_optimal"] else "")
    print()


if __name__ == "__main__":
    main()
