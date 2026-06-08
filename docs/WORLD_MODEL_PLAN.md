# Kerrio World Model for MI — Implementation Plan

> **Reframing:** Upgrade the existing SFT/DPO MI coach + 7-stage Kerrio journey
> into a **world-model-inspired, counterfactual MI policy evaluator**.
> Keep the world-model narrative, but **anchor every component to a metric that
> is measurable on AnnoMI ground truth** — not LLM self-consistency.

## Why a world model on top of SFT/DPO (not instead of)

SFT/DPO trains a **model-free reflex policy** (System 1): context → next
utterance, optimizing immediate preference, no notion of "3 turns later".
The world model is a **model-based planner + critic** (System 2) on top:

| | SFT/DPO Qwen | World model (Tiers 1–3) |
|---|---|---|
| role | **generate** fluent candidate replies | **plan / evaluate / rank** them |
| horizon | one step | 2–3 turns + value |
| explainable | black box | counterfactual + predicted consequences |
| change behavior | retrain 3B | re-weight reward (seconds) |
| safety | soft (less likely) | **hard constraint at decision time** |
| validate | trust preference pairs | **ground-truth transition accuracy on AnnoMI** |
| simulate client | cannot (only acts as agent) | yes → offline policy evaluation |

The world model does **not** generate text. Qwen (SFT/DPO) generates K
candidates; the world model does **generate-K-then-evaluate** on top.
The product is a *training tool for human counselors* — showing predicted
consequences of alternatives is the core value DPO structurally cannot deliver.

## The MDP (nail this down — every method maps to it)

- **state `s`** = `ClientLatentState`.
  - observable: client talk-type ∈ {change, sustain, neutral} — **AnnoMI gold**.
  - latent: readiness / resistance / alliance / self_efficacy — LLM-estimated, **unvalidated** (kept honestly separate).
- **action `a`** = MI tactic (~13). `main_therapist_behaviour` ∈ {question, reflection, therapist_input, other} gives **gold action labels**.
- **reward `r`** = `mi_quality` (high/low) + change-talk uplift + safety.
- **horizon `h`** = 2–3 turns (dialogue is short → no MCTS needed).

## The killer asset ChatGPT missed

`data/AnnoMI-full.csv` (13,551 utterances, 133 transcripts) ships real labels:
- `main_therapist_behaviour` → action labels
- `client_talk_type` (change 1668 / sustain 840 / neutral 4217 / n.a. 6826) → **gold transition signal**
- `mi_quality` → reward signal

So the world model's core claim is **trainable and validatable** on real data:
`(prev client talk-type, therapist behaviour) → next client talk-type`.

## Which world models we use (3 tiers)

**Tier 1 — Count-based Markov / tabular MDP (BUILD NOW).**
`P(s'|s,a)` from AnnoMI sequences. Answers the make-or-break question cheaply:
*do therapist actions predict client talk-type transitions above baseline?*
Reports accuracy vs majority-class. This is the baseline every later tier must beat.

**Tier 2 — PlaNet / TD-MPC-style MPC (the counterfactual evaluator itself).**
Sample K candidate MI actions → roll out predicted talk-type with the Tier 1
model → rank by predicted change-talk uplift. TD-MPC value head bootstraps
long-term readiness so we don't reward manipulative short-term talk.
Planning-without-policy (PlaNet) fits: small discrete action space, short horizon.

**Tier 3 — Neural latent dynamics (only after Tier 1 shows signal).**
Architecture = **TD-MPC2 skeleton** (latent dynamics + reward + value) +
**MuZero value-equivalent principle** (do NOT reconstruct utterance text, only
predict latent talk-type/reward) + **MDN-RNN multimodal output** (same action →
multimodal client reaction).

**JEPA option (Tier 3 ideal engine).** Action-conditioned **MI-JEPA** modeled on
**V-JEPA 2-AC**: context encoder + EMA target encoder + predictor `P(s_t, a_t, z)
→ ŝ_{t+1}` in embedding space; plan via latent MPC. Predicts in *representation*
space, not token space — the purest form of "don't predict the text".
Big win: **self-supervised** → learns dynamics without labels (solves AnnoMI
label scarcity); validate via **linear probe of latent → talk-type** on gold.
Risks: text/dialogue JEPA is exploratory, representation collapse (needs EMA +
VICReg), needs pretrained text encoder init (don't train from scratch).
Does NOT replace Tier 1 — Tier 1 is the baseline JEPA must beat.

**Decision Transformer** = strong baseline / control (condition on high
change-talk return → predict action). Used to check MPC actually beats imitation.

**Deliberately NOT used:** full RSSM/Dreamer pixel reconstruction (no images,
wastes capacity on wording); MCTS (h=3 too shallow); Dreamer-style imagination
policy training (compounding model error on small data — prefer per-step MPC).

## Phased build order

- **Phase 0** — `ClientLatentState` + `MIAction` schema; transition dataset from AnnoMI. *(designed dual-use for JEPA `(history, action, future)` so we don't redo data later.)*
- **Phase 1** — state estimator: reuse MI classifier as therapist-action tagger; **new client talk-type classifier** trained on AnnoMI gold → report F1.
- **Phase 2** — tabular transition `P(next_talk|prev_talk,action)` + baseline; then MPC counterfactual.
- **Phase 3** — reward (mi_quality + change-talk uplift) + counterfactual ranking; Decision Transformer baseline.
- **Phase 4** — Web World Model Panel: current latent state, candidate-action comparison, 3-turn imagined rollout.
- **Phase 5 (optional)** — neural latent dynamics (TD-MPC2 skeleton) or MI-JEPA, only after Tier 1 beats baseline.

## Target file layout

```
scripts/world_model/
  state_schema.py        # ClientLatentState, MIAction, RewardVector
  build_transition_data.py  # AnnoMI -> (prev_talk, action, next_talk); dual-use JEPA samples
  transition_matrix.py   # P(next_talk|prev_talk,action) + baseline report
  reward_model.py
  planner.py             # MPC counterfactual ranking
  rollout.py
  counterfactual.py
scripts/eval/            # eval_state_transition, eval_counterfactual_rank, eval_safety
data/schemas/            # mi_state_schema.json, mi_action_schema.json
reports/                 # world_model_eval.json, transition_report.*
```

## Status / current starting point (corrected vs README)

- `state_before`/`state_after` exist only as example fields in `data/schema.json` — **not wired into any code**.
- Current "state extraction" in `kerrio_journey.py` is **regex/keyword only**; no numeric latent state exists yet.
- An MI behaviour classifier already exists (`train_mi_classifier.py` + `/score`) — reuse as action tagger.
- FastAPI (`app_demo.py`, ~30 endpoints) + 7-stage journey state machine + profile JSON persistence are solid infra to build on.

## Tier 1 RESULTS (2026-06-08) — thesis HOLDS ✅

Built `scripts/world_model/build_transition_data.py` (4,669 transitions from 133
transcripts, 9-action space, transcript-level train/val split) and
`scripts/world_model/transition_matrix.py`. Report: `reports/transition_report.json`.

Held-out val (563 transitions):

| predictor | val acc | macro-F1 |
|---|---|---|
| B0 prior (always neutral) | 60.2% | 0.251 |
| B1 momentum P(next\|prev) | 65.9% | 0.442 |
| WM action-only P(next\|action) | 59.9% | 0.254 |
| **WM full P(next\|prev,action)** | **67.1%** | **0.555** |

- Action carries info **beyond conversational momentum**: macro-F1 **0.442 → 0.555**
  (big jump on the clinically important rare change/sustain classes), NLL 0.809 → 0.798.
  Accuracy lift is modest (+1.2pt) only because "neutral" dominates (60%) and momentum
  already nails it — the value is in the minority classes, which macro-F1 captures.
- **Per-action change-talk uplift table is clinically sensible (= validation):**
  negotiation +27%, open_question +12.7% **evoke** change talk; information −10.5%,
  closed_question −7% **suppress** it. Matches MI theory → real, teachable signal.

**Conclusion:** therapist actions predict client talk-type shifts above baseline.
Tiers 2/3 are worth building.

## Tier 2 + Phase 1 RESULTS (2026-06-08) — counterfactual demo works end-to-end ✅

- `scripts/world_model/transition_model.py` — reusable `TransitionModel` (counts + momentum back-off).
- `scripts/world_model/planner.py` — MPC: finite-horizon value lookahead over the 9 MI actions
  (PlaNet/TD-MPC planning-without-policy; H-step lookahead = stand-in for TD-MPC value head).
  Reward: change=+1, sustain=-1, neutral=0.
- `scripts/world_model/build_talktype_data.py` + `train_talktype_clf.py` — **state estimator**:
  distilbert 3-class talk-type classifier (class-weighted), **val macro-F1 0.55, acc 0.65**
  (change 0.43 / sustain 0.44 / neutral 0.78) — hard imbalanced task, citable baseline.
  Saved to `runs/talktype_clf`.
- `scripts/world_model/counterfactual.py` — ties it together: estimate state -> tag MI action
  (rule-based, swappable for the MI classifier) -> MPC counterfactual.
- `/api/counterfactual` endpoint added to `app_demo.py` (backend for the World Model Panel).

**End-to-end sanity (clinically correct MI emerged from data):**
client "I drink too much but I don't think I can stop" -> estimated **sustain**;
coach "You should cut down" -> tagged **advice** -> P(change)=0.22, P(sustain)=0.39;
model suggests **negotiation** ("Could we agree on one small step?") -> P(change)=0.38,
uplift **+0.15**. The model learned "don't advise a resistant client, negotiate instead"
from AnnoMI alone.

## Web World Model Panel (2026-06-08) — DONE ✅

Added a "World Model" tab to `web/index.html` (served from disk by the FastAPI
StaticFiles mount, so a server restart deploys it on the tunnel URL):
- inputs auto-fill from the latest chat turn (tracked via `lastClientMsg`/`lastCoachReply`);
- calls `/api/counterfactual`, renders estimated client state badge, your-action vs
  model-suggested cards with change/sustain bars, an uplift banner, and a full
  ranked-actions bar chart (★ = best, ▶ = yours).
- Cross-version fix in `counterfactual.py`: `enc.pop("token_type_ids")` (DistilBERT
  checkpoint saved under transformers 5.2, server runs base-python transformers 4.57).
- Verified end-to-end under the server's base-python env.

Deploy = restart `uvicorn scripts.app_demo:app` on port 8081 (live demo, ask first).
If the Cloudflare Pages copy of index.html is used, also `git push` for Pages rebuild.

## Next steps (not yet built)

- Swap rule-based action tagger for the existing MI behaviour classifier.
- Tier 3 / MI-JEPA only if we want learned latent dynamics (the dual-use `history`/`future_text`
  fields in `transitions.jsonl` are already there for it).
- `scripts/eval/` proper eval harness (state-transition acc, counterfactual ranking, safety).
