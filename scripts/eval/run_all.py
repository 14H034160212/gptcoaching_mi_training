#!/usr/bin/env python3
"""
B4 — Run the full world-model eval suite and write reports/world_model_eval.json.

  python -m scripts.eval.run_all
"""
import json
import os
from pathlib import Path

from scripts.eval import eval_state_transition, eval_counterfactual_rank, eval_safety


def main(out="reports/world_model_eval.json"):
    report = {
        "state_and_transition": eval_state_transition.run(),
        "counterfactual_ranking": eval_counterfactual_rank.run(),
        "safety": eval_safety.run(),
    }
    Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2)

    print("\n================ WORLD MODEL EVAL ================\n")
    st = report["state_and_transition"]
    se = st["state_estimator"]; tm = st["transition_model"]
    print("STATE ESTIMATOR (talk-type classifier vs gold)")
    if "accuracy" in se:
        print(f"  acc={se['accuracy']}  macro_f1={se['macro_f1']}  per_class={se['per_class_f1']}")
    else:
        print(f"  {se.get('status')}")
    print("\nTRANSITION MODEL (predict next talk-type)")
    print(f"  prior     acc={tm['prior']['acc']}  f1={tm['prior']['macro_f1']}")
    print(f"  momentum  acc={tm['momentum']['acc']}  f1={tm['momentum']['macro_f1']}")
    print(f"  worldmodel acc={tm['world_model']['acc']}  f1={tm['world_model']['macro_f1']}")
    print(f"  -> action helps macro_f1 by {tm['action_helps_f1_lift']:+}")
    cr = report["counterfactual_ranking"]
    print("\nCOUNTERFACTUAL RANKING (planner vs held-out empirical)")
    print(f"  mean Spearman(planner P_change, empirical change rate) = {cr['mean_spearman_planner_vs_empirical']}")
    print(f"  top-1 agreement = {cr['top1_agreement']}")
    print(f"  mean offline change uplift = {cr['mean_offline_change_uplift']}")
    sf = report["safety"]
    print("\nSAFETY")
    for k, v in sf.items():
        print(f"  {k}: {v}")
    print(f"\n[eval] wrote -> {out}\n")


if __name__ == "__main__":
    main()
