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

## Action tagger + Eval suite (2026-06-08) — DONE ✅

**Action tagger (learned).** `build_action_data.py` (therapist utterances → 9-class
action, AnnoMI-derived) + generic `train_clf.py` → `runs/action_clf` (distilbert,
val acc 0.67 / macro-F1 0.49). Wired into `counterfactual.tag_action`: uses the
classifier when present, falls back to the rule-based `_tag_action_rule`.

**Safety layer.** `scripts/world_model/safety.py` — high-recall crisis screen.
`coach_feedback` now short-circuits to an escalation message (no MI tips) on a
crisis hit, and flags risky counselor replies.

**Eval suite** (`scripts/eval/`, run via `python -m scripts.eval.run_all` →
`reports/world_model_eval.json`):
- `eval_state_transition` — state estimator acc 0.65 / macro-F1 0.55;
  transition model macro-F1 0.56 vs momentum 0.44 (**action helps +0.12**).
- `eval_counterfactual_rank` — planner vs held-out empirical:
  mean Spearman 0.38, top-1 agreement 1/3, **mean offline change uplift +0.13**.
- `eval_safety` — crisis recall **5/5**, benign FP **0/5**, risky-reply recall 2/2.

Note: `/api/counterfactual` lazy-imports `counterfactual` and Python caches it, so
**the live server needs a restart** to pick up the learned tagger + safety layer.

## Tier 3 — MI-JEPA (2026-06-08) — implemented; HONEST result: does NOT beat tabular yet

`scripts/world_model/mi_jepa.py` — action-conditioned MI-JEPA (V-JEPA 2-AC style):
trainable context encoder + EMA target encoder (distilbert backbone) + action-
conditioned predictor, VICReg loss (invariance + variance + covariance). Trained
8 epochs on the AnnoMI transitions (self-supervised, no talk-type labels).
Validation by linear probe (`mi_jepa.py` built-in + `mi_jepa_probe.py` standard readout).

Results (val, 563), reports/mi_jepa_eval.json + mi_jepa_readout.json:

| eval | acc | macro-F1 |
|---|---|---|
| target-probe (sees the reply — upper bound) | 0.56 | 0.44 |
| JEPA dynamics, probe on target basis | 0.36 | 0.28 |
| **JEPA dynamics, standard readout (probe on predicted latents)** | **0.45** | **0.38** |
| **tabular transition baseline** | **0.67** | **0.56** |

- No collapse (variance term held ~1.55, inv plateaued ~0.13).
- The predicted latent linearly encodes next-talk-type at F1 0.38 — real signal, but
  **below the count-based tabular model (0.56)** at this data scale (4,106 train).
- This is the documented Tier-3 risk: dialogue-JEPA shines at scale; AnnoMI is tiny.
  Methodology held — Tier 1 is the bar, and JEPA hasn't cleared it, so the tabular
  model + MPC stays the production dynamics engine.

### Scaling lever #1 — frozen strong encoder (`mi_jepa_frozen.py`, all-mpnet-base-v2)

Tried the top lever: freeze a sentence encoder pretrained on ~1B pairs (scale via
pretraining), train only the predictor. Result (reports/mi_jepa_frozen_eval.json):

| eval | acc | macro-F1 |
|---|---|---|
| target-probe (mpnet, sees reply — upper bound) | 0.62 | **0.56** |
| JEPA dynamics readout (mpnet) | 0.46 | 0.40 |
| JEPA dynamics (distilbert) | 0.45 | 0.38 |
| tabular transition baseline | 0.67 | 0.56 |

**Finding — the lever fixed the representation, not the dynamics:**
- target-probe jumped 0.44 → **0.56** F1 = matches the tabular model's predictive F1
  and the supervised talk-type classifier (0.55). **Representation is no longer the bottleneck.**
- dynamics prediction barely moved (0.38 → 0.40), still << tabular 0.56. So the
  bottleneck is the **predictor**: forecasting a 768-d free-form next-utterance
  embedding from (context, action) is far harder than modeling the discrete 3-way
  talk-type transition the tabular model targets directly. At 4k examples the
  discrete model is much more sample-efficient.
- **Useful byproduct:** frozen mpnet + linear probe matches the fine-tuned
  talk-type classifier (F1 ~0.56) for free — a cheaper state estimator.

**Verdict:** tabular + MPC stays the production dynamics engine. The remaining
untested big lever is **data scale** (the predictor is data-starved, not mis-specified):
self-supervised pretrain the predictor on a much larger corpus (e.g. tens of
thousands of LLM-simulated MI turns) then probe AnnoMI. That's expensive (3B-model
generation) and needs a go-decision; deferred.

### Scaling lever #2 — more data (ESConv corpus + online active loop)

Fetched **thu-coai/esconv** (1.3k multi-turn counseling dialogues). Two uses:

**(a) Self-supervised predictor data for JEPA** (`build_esconv_pairs.py` → 14.4k
(context, action, future_text) pairs; `mi_jepa_frozen.py --extra`):
JEPA dynamics readout 0.395 → **0.408** F1 (18.5k predictor-train). Real but small
gain; still << tabular 0.56. Confirms the predictor is data-hungry, but forecasting
free-form next-utterance embeddings is intrinsically hard — more data helps slowly.

**(b) Online active-learning loop** (the user's "synthesize-where-weak, retrain"
idea — `silver_label_esconv.py` + `active_loop.py`, reports/active_loop.json):
- silver-label REAL ESConv text with the talk-type classifier (real text + silver
  label → avoids self-generation confirmation bias; labels are noisy since ESConv
  is out-of-domain, top-quartile conf only 0.53 → rejection sampling needed).
- each round: find weakest class on real AnnoMI-val → add high-conf silver triples
  for it → refit → **KEEP only if real-val macro-F1 improves**, else raise conf bar.
- Result: macro-F1 **0.5606 → 0.5686 (+0.008)**; round 1 (+400 'sustain') kept,
  rounds 2-8 rejected by the gate. Modest lift, self-correcting, no degradation.
- This is DAgger/STaR-flavored active learning. Honest framing: gain is small and
  bounded by silver-label noise + domain gap; the value is the *mechanism* (real-val
  reward gate that refuses harmful data), which scales to better labelers / in-domain
  synthesis.

### Scaling lever #3 — in-domain synthetic MI (DPO Qwen self-play)

`gen_synth_mi.py`: DPO Qwen generates MI exchanges (coach + simulated client)
targeting the weak talk-types; silver-label both client turns + tag action.
- **In-domain helped label quality:** conf quartiles 0.47/0.54/**0.64** vs ESConv's
  0.43/0.47/0.53 — cleaner silver labels, as hypothesized.
- **But generation skewed to change-talk** (422 change vs 160 sustain) while the
  weak class is *sustain* → starved for high-conf sustain examples. Standalone synth
  loop gain = **0** (the 4 high-conf sustain triples lowered real-val → gate rejected).
- **Combined ESConv+synth pool**: same **+0.008** as ESConv alone — synth added
  nothing beyond ESConv at this scale (reports/active_loop_synth.json, _combined.json).

**Net lesson (the useful negative result):** the online loop's gain is bottlenecked
by high-confidence examples *of the specific weak class*, not raw data volume. To move
the needle: (1) target generation hard at the weak class (sustain), (2) improve the
*labeler* (talk-type classifier, currently F1 0.55) since label noise caps the whole
loop, (3) human-verify a small high-value slice instead of pure silver. The real-val
gate worked perfectly throughout — it never let a non-improving batch through.

### Scaling lever #4 — act on the lesson: better labeler + sustain-targeted (the win)

Applied (1) and (2) directly:
- **Better labeler:** retrained the talk-type classifier on **all-mpnet-base-v2**
  (`train_talktype_clf.py --model sentence-transformers/all-mpnet-base-v2`) →
  macro-F1 **0.55 → 0.59**, **sustain F1 0.44 → 0.52**, acc 0.70. Now also the
  production state estimator (`counterfactual.py` prefers `runs/talktype_clf_mpnet`).
- **Sustain-targeted synthesis** (`gen_synth_mi.py --sustain-frac 0.8`, stronger
  client prompt) → 288 sustain synth (was 160); re-silver-labeled ESConv with the
  mpnet labeler (5.7k sustain, was 4.7k).
- Online loop on the combined pool (`reports/active_loop_v2.json`):
  macro-F1 **0.5606 → 0.5837 (+0.0231, ~3× the prior +0.008)**. The gain is entirely
  in the targeted weak class: **sustain recall 0.26 → 0.44, F1 0.36 → 0.45** (small
  trade-off on change/neutral). Round 1's 400 high-conf sustain triples kept; rest
  rejected by the real-val gate.

**Confirmed:** the two predicted levers (cleaner labels + target the actual weak
class) are what move the online loop — together they ~tripled the gain, and it is
interpretable (the weak class improved, gated on real data).

## Production deployment + continuous-improvement flywheel (2026-06-08)

**Best model is live.** `counterfactual._model()` now prefers
`data/world_model/transitions_prod.jsonl` (the loop-augmented tabular model,
macro-F1 0.58) over the AnnoMI-only one, with **mtime-based hot-reload** — future
retrains swap in with NO server restart. Live server confirmed loading it.

**The flywheel** (`scripts/world_model/continuous_improve.py`, runnable on a schedule):
1. `harvest_user_transitions.py` — turn real chat-log traffic into silver-labeled transitions.
2. pool harvested-user + ESConv-silver + Qwen-synth.
3. `active_loop.py --export-train` — real-val-gated improve → export winner to `transitions_prod.jsonl` (hot-reloaded).
4. **model-agnostic gate** — also score MI-JEPA on the same real AnnoMI-val; promote whichever wins. Currently tabular 0.58 > JEPA 0.41 → tabular stays; when the data flywheel pushes JEPA past tabular, it gets flagged for promotion.

**INTEGRITY RULE (enforced):** the gate is ALWAYS real gold AnnoMI-val; user/silver
data only enters the *training* pool, never the judge → the flywheel cannot drift.

**On "RL / Tier-3 will ultimately be best" (correct asymptotically):** tabular+MPC is
already model-based RL with a tabular world model; swapping in the JEPA neural world
model = full RL. JEPA loses now only on data scale (4k). The continuous flywheel is the
mechanism that pushes data past the crossover, and the model-agnostic gate auto-promotes
JEPA when it wins — no manual bet, data decides. (Wiring a JEPA-backed `predict_dist`
into the MPC planner is the follow-up for when that crossover happens.)

## Next steps (not yet built)

- Tier 3 / MI-JEPA only if we want learned latent dynamics (the dual-use `history`/`future_text`
  fields in `transitions.jsonl` are already there for it).
- `scripts/eval/` proper eval harness (state-transition acc, counterfactual ranking, safety).
