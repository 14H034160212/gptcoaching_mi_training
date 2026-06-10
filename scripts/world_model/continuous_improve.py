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
    # Empirical: conf=0.4 lets the active loop accept a second batch at higher
    # conf in a later round (warm-up then refine), yielding macro_f1=0.5892 vs
    # 0.5837 at conf=0.5 on AnnoMI val. See reports/exp/active_loop_b400_c0.4.json.
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--skip-harvest", action="store_true")
    ap.add_argument("--regen-synth", action="store_true",
                    help="(weekly) regenerate in-domain synthetic data targeting the current weak class")
    ap.add_argument("--synth-n", type=int, default=500)
    args = ap.parse_args()

    dw = "data/world_model"
    # 1) harvest real user data
    if not args.skip_harvest:
        run("scripts.world_model.harvest_user_transitions", "--talk-clf", args.talk_clf,
            "--out", f"{dw}/user_transitions.jsonl")

    # 1b) OPTIONAL (weekly): regenerate synthetic MI data targeting the CURRENT weak class,
    #     then ACCUMULATE it (append) so the synthetic corpus grows over time.
    synth_pool = f"{dw}/synth_accumulated.jsonl"
    if args.regen_synth:
        from scripts.world_model.transition_model import TransitionModel, TALKS
        from scripts.eval.eval_state_transition import _macro_f1  # noqa: F401 (ensure import path ok)
        from scripts.world_model.active_loop import eval_model, load_jsonl
        cur = args.prod if os.path.exists(args.prod) else args.anno
        wm = TransitionModel().fit([r for r in load_jsonl(cur) if r.get("split", "train") == "train"])
        val = [r for r in load_jsonl(args.anno) if r["split"] == "val"]
        rec = eval_model(wm, val)["per_class"]
        # weakest of the two generatable stances (neutral is not a generation target)
        weak = min(["sustain", "change"], key=lambda c: rec[c]["recall"])
        sustain_frac = 0.85 if weak == "sustain" else 0.15
        print(f"[continuous] regen-synth: current weak class = {weak} -> sustain_frac={sustain_frac}")
        new_synth = f"{dw}/synth_new.jsonl"
        run("scripts.world_model.gen_synth_mi", "--talk-clf", args.talk_clf,
            "--n", str(args.synth_n), "--sustain-frac", str(sustain_frac), "--out", new_synth,
            "--keep-on-target")
        # accumulate (append) so the synthetic corpus grows
        with open(synth_pool, "a", encoding="utf-8") as out:
            for line in open(new_synth, encoding="utf-8"):
                out.write(line)
        print(f"[continuous] appended new synth -> {synth_pool} (total {sum(1 for _ in open(synth_pool))})")
    elif not os.path.exists(synth_pool) and os.path.exists(f"{dw}/synth_silver_v2.jsonl"):
        # seed the accumulating pool from the existing v2 synth on first run
        import shutil
        shutil.copy(f"{dw}/synth_silver_v2.jsonl", synth_pool)

    # 2) pool all silver augmentation sources that exist
    pool = f"{dw}/continuous_pool.jsonl"
    sources = [f"{dw}/user_transitions.jsonl", f"{dw}/esconv_silver_mpnet.jsonl",
               synth_pool]
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
    champion = "tabular"
    if jepa_f1 is not None and jepa_f1 > tab_f1:
        winner = "MI-JEPA"
        champion = "jepa"
    # write the champion the live server reads (auto-promotion); tabular stays until JEPA wins
    os.makedirs("runs", exist_ok=True)
    with open("runs/world_model_champion.txt", "w") as f:
        f.write(champion)
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
