#!/usr/bin/env python3
"""
Online active-learning loop (the user's "synthesize-where-weak, retrain" idea).

Each round:
  1. evaluate the tabular transition model on REAL AnnoMI-val  -> per-class recall
  2. find the weakest talk-type (lowest recall)
  3. pull silver-labeled REAL ESConv transitions that target that class AND pass a
     confidence threshold (rejection sampling)
  4. add them, refit, re-evaluate on REAL AnnoMI-val
  5. KEEP the round only if val macro-F1 improved (reward = held-out real F1);
     otherwise revert and raise the confidence bar.

This is DAgger/active-learning/STaR-flavored. Honest guardrails:
  - text is REAL (ESConv), only the talk-type label is silver -> avoids the worst
    self-generation confirmation-bias trap (label noise remains, hence rejection sampling)
  - the reward is ALWAYS real AnnoMI-val, never synthetic
  - a round that doesn't improve real-val is rejected

  python -m scripts.world_model.active_loop --rounds 6 --conf 0.8 --budget 400
"""
import argparse
import json
from collections import Counter, defaultdict

from scripts.world_model.transition_model import TransitionModel, TALKS, _laplace


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def eval_model(wm, val):
    preds, golds = [], []
    for r in val:
        d = wm.predict_dist(r["prev_talk"], r["action"])
        preds.append(max(d, key=lambda k: d[k])); golds.append(r["next_talk"])
    per = {}
    for c in TALKS:
        tp = sum(p == c and g == c for p, g in zip(preds, golds))
        fp = sum(p == c and g != c for p, g in zip(preds, golds))
        fn = sum(p != c and g == c for p, g in zip(preds, golds))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        per[c] = {"recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
    macro_f1 = sum(per[c]["f1"] for c in TALKS) / len(TALKS)
    return {"acc": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "per_class": {c: {k: round(v, 4) for k, v in per[c].items()} for c in TALKS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default="data/world_model/transitions.jsonl")
    ap.add_argument("--silver", default="data/world_model/esconv_silver_transitions.jsonl")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--budget", type=int, default=400, help="max silver triples added per round")
    ap.add_argument("--report", default="reports/active_loop.json")
    ap.add_argument("--export-train", default=None,
                    help="if set, write the final kept-augmented training set here for production")
    args = ap.parse_args()

    anno = load_jsonl(args.anno)
    train = [r for r in anno if r["split"] == "train"]
    val = [r for r in anno if r["split"] == "val"]
    silver_all = load_jsonl(args.silver)

    cur_train = list(train)
    wm = TransitionModel().fit(cur_train)
    base = eval_model(wm, val)
    print(f"[loop] baseline (AnnoMI only): acc={base['acc']} macro_f1={base['macro_f1']} "
          f"recall={{c: round(base['per_class'][c]['recall'],2) for c in TALKS}}")

    history = [{"round": 0, "added": 0, "pool_conf": args.conf, **base}]
    best_f1 = base["macro_f1"]
    used = set()
    conf = args.conf

    for rnd in range(1, args.rounds + 1):
        # 1-2) weakest class by recall on real val
        weak = min(TALKS, key=lambda c: history[-1]["per_class"][c]["recall"])
        # 3) candidate REAL ESConv silver triples targeting the weak class, high conf, unused
        cands = [i for i, r in enumerate(silver_all)
                 if r["next_talk"] == weak and r["conf"] >= conf and i not in used]
        cands = cands[:args.budget]
        if not cands:
            conf = round(conf - 0.05, 2)  # relax the bar if dry
            print(f"[loop] round {rnd}: no candidates for '{weak}' at conf>={conf+0.05}; lowering to {conf}")
            if conf < 0.5:
                print("[loop] confidence floor reached; stopping.")
                break
            continue
        add = [silver_all[i] for i in cands]
        trial_train = cur_train + add
        wm_trial = TransitionModel().fit(trial_train)
        res = eval_model(wm_trial, val)
        improved = res["macro_f1"] > best_f1 + 1e-4
        tag = "KEEP" if improved else "reject"
        print(f"[loop] round {rnd}: weak='{weak}' +{len(add)} silver(conf>={conf}) "
              f"-> val_f1={res['macro_f1']} ({tag}; best={best_f1})")
        if improved:
            cur_train = trial_train
            wm = wm_trial
            best_f1 = res["macro_f1"]
            used.update(cands)
            history.append({"round": rnd, "added": len(add), "weak_class": weak,
                            "pool_conf": conf, **res})
        else:
            conf = round(conf + 0.05, 2)  # raise the bar; demand cleaner labels
            history.append({"round": rnd, "added": 0, "weak_class": weak,
                            "pool_conf": conf, "rejected_f1": res["macro_f1"],
                            "acc": history[-1]["acc"], "macro_f1": best_f1,
                            "per_class": history[-1]["per_class"]})

    final = eval_model(wm, val)
    if args.export_train:
        from pathlib import Path as _P
        _P(args.export_train).parent.mkdir(parents=True, exist_ok=True)
        with open(args.export_train, "w", encoding="utf-8") as f:
            for r in cur_train:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[loop] exported {len(cur_train)} augmented training rows -> {args.export_train}")
    report = {"baseline": base, "final": final,
              "delta_macro_f1": round(final["macro_f1"] - base["macro_f1"], 4),
              "total_silver_added": len(used), "history": history}
    json.dump(report, open(args.report, "w"), indent=2)
    print(f"\n[loop] baseline macro_f1={base['macro_f1']} -> final macro_f1={final['macro_f1']} "
          f"(delta {report['delta_macro_f1']:+}), silver added={len(used)}")
    print(f"[loop] wrote -> {args.report}")


if __name__ == "__main__":
    main()
