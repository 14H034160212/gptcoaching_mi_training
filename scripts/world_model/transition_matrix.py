#!/usr/bin/env python3
"""
Tier 1b — Tabular Markov world model:  P(next_talk | prev_talk, action)

This is the simplest possible MI world model and the FIRST validatable test of
the whole world-model thesis:

    Do therapist actions predict client talk-type transitions ABOVE baselines?

We compare four predictors on a held-out (by-transcript) val split:
  B0 prior          : argmax P(next_talk)                      [ignores everything]
  B1 momentum       : argmax P(next_talk | prev_talk)          [ignores the action]
  WM full           : argmax P(next_talk | prev_talk, action)  [the world model]
  WM action-only    : argmax P(next_talk | action)             [action, no state]

If WM(full) > B1(momentum), the action carries information beyond conversational
momentum -> the world-model thesis holds and Tiers 2/3 are worth building.

Also emits the teachable artifact: per-action change-talk / sustain uplift table.

Usage:
  python scripts/world_model/transition_matrix.py \
      --data data/world_model/transitions.jsonl \
      --report reports/transition_report.json
"""
import argparse
import json
import math
import os
from collections import defaultdict, Counter
from pathlib import Path

TALKS = ["change", "sustain", "neutral"]


def laplace_dist(counter, classes, alpha=1.0):
    total = sum(counter.get(c, 0) for c in classes)
    denom = total + alpha * len(classes)
    return {c: (counter.get(c, 0) + alpha) / denom for c in classes}


def argmax_dist(dist):
    return max(dist.items(), key=lambda kv: (kv[1], kv[0]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/world_model/transitions.jsonl")
    ap.add_argument("--report", default="reports/transition_report.json")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing")
    args = ap.parse_args()

    train, val = [], []
    for line in open(args.data, encoding="utf-8"):
        r = json.loads(line)
        (train if r["split"] == "train" else val).append(r)

    # ---- fit counts on train ----
    c_next = Counter()                                   # P(next)
    c_prev_next = defaultdict(Counter)                   # P(next | prev)
    c_act_next = defaultdict(Counter)                    # P(next | action)
    c_pa_next = defaultdict(Counter)                     # P(next | prev, action)
    for r in train:
        nt, pt, a = r["next_talk"], r["prev_talk"], r["action"]
        c_next[nt] += 1
        c_prev_next[pt][nt] += 1
        c_act_next[a][nt] += 1
        c_pa_next[(pt, a)][nt] += 1

    prior = laplace_dist(c_next, TALKS, args.alpha)
    prior_pred = argmax_dist(prior)

    def pred_momentum(pt):
        return argmax_dist(laplace_dist(c_prev_next.get(pt, Counter()), TALKS, args.alpha))

    def pred_action(a):
        return argmax_dist(laplace_dist(c_act_next.get(a, Counter()), TALKS, args.alpha))

    def pred_full(pt, a):
        cell = c_pa_next.get((pt, a))
        if cell and sum(cell.values()) >= 3:             # back off if cell too sparse
            return argmax_dist(laplace_dist(cell, TALKS, args.alpha))
        return pred_momentum(pt)                         # back off to momentum

    # ---- evaluate on val ----
    def evaluate(predfn):
        correct = 0
        per_class = {c: {"tp": 0, "fp": 0, "fn": 0} for c in TALKS}
        nll = 0.0
        for r in val:
            gold = r["next_talk"]
            pred = predfn(r)
            if pred == gold:
                correct += 1
                per_class[gold]["tp"] += 1
            else:
                per_class[pred]["fp"] += 1
                per_class[gold]["fn"] += 1
        # macro-F1
        f1s = []
        for c in TALKS:
            tp, fp, fn = per_class[c]["tp"], per_class[c]["fp"], per_class[c]["fn"]
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            f1s.append(f1)
        return {"acc": correct / len(val), "macro_f1": sum(f1s) / len(f1s)}

    def nll_of(distfn):
        s = 0.0
        for r in val:
            d = distfn(r)
            s += -math.log(max(d.get(r["next_talk"], 1e-9), 1e-9))
        return s / len(val)

    results = {
        "B0_prior":       evaluate(lambda r: prior_pred),
        "B1_momentum":    evaluate(lambda r: pred_momentum(r["prev_talk"])),
        "WM_action_only": evaluate(lambda r: pred_action(r["action"])),
        "WM_full":        evaluate(lambda r: pred_full(r["prev_talk"], r["action"])),
    }
    nlls = {
        "B0_prior":    nll_of(lambda r: prior),
        "B1_momentum": nll_of(lambda r: laplace_dist(c_prev_next.get(r["prev_talk"], Counter()), TALKS, args.alpha)),
        "WM_full":     nll_of(lambda r: laplace_dist(c_pa_next.get((r["prev_talk"], r["action"]), Counter()), TALKS, args.alpha)
                              if sum(c_pa_next.get((r["prev_talk"], r["action"]), Counter()).values()) >= 3
                              else laplace_dist(c_prev_next.get(r["prev_talk"], Counter()), TALKS, args.alpha)),
    }

    # ---- teachable artifact: per-action talk-type distribution (marginal) ----
    action_effect = {}
    for a in sorted(c_act_next):
        d = laplace_dist(c_act_next[a], TALKS, args.alpha)
        action_effect[a] = {
            "n": sum(c_act_next[a].values()),
            "P_change": round(d["change"], 3),
            "P_sustain": round(d["sustain"], 3),
            "P_neutral": round(d["neutral"], 3),
        }
    base_change = prior["change"]

    report = {
        "n_train": len(train), "n_val": len(val),
        "val_accuracy": {k: round(v["acc"], 4) for k, v in results.items()},
        "val_macro_f1": {k: round(v["macro_f1"], 4) for k, v in results.items()},
        "val_nll": {k: round(v, 4) for k, v in nlls.items()},
        "action_effect": action_effect,
        "base_rate_change": round(base_change, 3),
    }
    Path(os.path.dirname(args.report)).mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.report, "w"), indent=2)

    # ---- pretty print ----
    print(f"\n=== Tabular world model  (train={len(train)}  val={len(val)}) ===\n")
    print(f"{'predictor':<16}{'val_acc':>10}{'macro_F1':>10}")
    for k in ["B0_prior", "B1_momentum", "WM_action_only", "WM_full"]:
        print(f"{k:<16}{results[k]['acc']*100:>9.1f}%{results[k]['macro_f1']:>10.3f}")
    lift = (results["WM_full"]["acc"] - results["B1_momentum"]["acc"]) * 100
    print(f"\n  WM_full vs B1_momentum (does the ACTION help?):  {lift:+.1f} acc pts")
    print(f"  val NLL  prior={nlls['B0_prior']:.3f}  momentum={nlls['B1_momentum']:.3f}  WM_full={nlls['WM_full']:.3f}")

    print(f"\n=== Per-action talk-type effect  (base change-rate = {base_change:.1%}) ===\n")
    print(f"{'action':<20}{'n':>6}{'P(change)':>11}{'P(sustain)':>12}{'change uplift':>15}")
    for a, e in sorted(action_effect.items(), key=lambda kv: -kv[1]["P_change"]):
        uplift = (e["P_change"] - base_change) * 100
        print(f"{a:<20}{e['n']:>6}{e['P_change']:>11.2f}{e['P_sustain']:>12.2f}{uplift:>+14.1f}%")
    print(f"\n[report] wrote -> {args.report}\n")


if __name__ == "__main__":
    main()
