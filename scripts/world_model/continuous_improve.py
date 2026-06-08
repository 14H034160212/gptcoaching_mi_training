#!/usr/bin/env python3
"""
Continuous improvement cycle — the data flywheel.

One cycle (run on a schedule, e.g. cron / Claude Code /schedule):
  1. HARVEST real user transitions from chat logs (product traffic).
  2. POOL augmentation candidates: harvested-user + ESConv-silver + Qwen-synth.
  3. IMPROVE the production (tabular) model via the real-val-gated active loop:
     target the weakest class, add high-conf silver, KEEP only if AnnoMI-val
     macro-F1 improves. Export the winner to transitions_prod.jsonl, which the
     live server hot-reloads (mtime) — NO restart needed.
  4. CHALLENGER CHECK (model-agnostic gate): score the Tier-3 MI-JEPA on the same
     real AnnoMI-val. Report tabular vs JEPA. When JEPA crosses the tabular score,
     it should be promoted to the production transition model (see note).

INTEGRITY RULE: the gate is ALWAYS real gold AnnoMI-val. User/silver data only
ever enters the *training* pool, never the judge — so the flywheel cannot drift
or self-deceive. A cycle that doesn't beat the current prod model changes nothing.

  python -m scripts.world_model.continuous_improve
"""
import argparse
import json
import os
import subprocess
import sys

PY = sys.executable


def run(mod, *cliargs):
    subprocess.run([PY, "-m", mod, *cliargs], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default="data/world_model/transitions.jsonl")
    ap.add_argument("--prod", default="data/world_model/transitions_prod.jsonl")
    ap.add_argument("--talk-clf", default="runs/talktype_clf_mpnet")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--skip-harvest", action="store_true")
    args = ap.parse_args()

    dw = "data/world_model"
    # 1) harvest real user data
    if not args.skip_harvest:
        run("scripts.world_model.harvest_user_transitions", "--talk-clf", args.talk_clf,
            "--out", f"{dw}/user_transitions.jsonl")

    # 2) pool all silver augmentation sources that exist
    pool = f"{dw}/continuous_pool.jsonl"
    sources = [f"{dw}/user_transitions.jsonl", f"{dw}/esconv_silver_mpnet.jsonl",
               f"{dw}/synth_silver_v2.jsonl"]
    n = 0
    with open(pool, "w", encoding="utf-8") as out:
        for s in sources:
            if os.path.exists(s):
                for line in open(s, encoding="utf-8"):
                    out.write(line); n += 1
    print(f"[continuous] augmentation pool = {n} silver transitions ({[os.path.basename(s) for s in sources if os.path.exists(s)]})")

    # 3) real-val-gated improve; export winner to production (hot-reloaded by server)
    run("scripts.world_model.active_loop", "--silver", pool, "--rounds", "12",
        "--conf", str(args.conf), "--budget", str(args.budget),
        "--report", "reports/continuous_loop.json",
        "--export-train", args.prod)

    loop = json.load(open("reports/continuous_loop.json"))
    tab_f1 = loop["final"]["macro_f1"]
    base_f1 = loop["baseline"]["macro_f1"]

    # 4) challenger: score MI-JEPA on the SAME real val (model-agnostic gate)
    jepa_f1 = None
    for rep in ("reports/mi_jepa_frozen_aug_eval.json", "reports/mi_jepa_readout.json"):
        if os.path.exists(rep):
            d = json.load(open(rep))
            for k in d:
                if "dynamics" in k and isinstance(d[k], dict):
                    jepa_f1 = d[k]["macro_f1"]; break
            if jepa_f1 is not None:
                break

    winner = "tabular(+aug)"
    if jepa_f1 is not None and jepa_f1 > tab_f1:
        winner = "MI-JEPA"
    summary = {
        "production_model": "tabular (transitions_prod.jsonl)",
        "tabular_baseline_f1": base_f1, "tabular_augmented_f1": tab_f1,
        "jepa_challenger_f1": jepa_f1, "current_best": winner,
        "note": ("MI-JEPA still below tabular -> tabular stays in production."
                 if winner != "MI-JEPA" else
                 "MI-JEPA now beats tabular on real val -> wire JEPA-backed predict_dist "
                 "into the planner and promote it."),
    }
    json.dump(summary, open("reports/continuous_summary.json", "w"), indent=2)
    print("\n[continuous] ==== cycle summary ====")
    print(json.dumps(summary, indent=2))
    print(f"[continuous] production model exported -> {args.prod} "
          f"(server hot-reloads on next /api/counterfactual call)")


if __name__ == "__main__":
    main()
