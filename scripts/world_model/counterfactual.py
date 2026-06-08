#!/usr/bin/env python3
"""
Counterfactual MI feedback — ties the world model to real utterances.

Pipeline:
  1. estimate client state (talk-type) from the client's message   [state estimator]
  2. tag the counselor's reply with an MI action                   [action tagger]
  3. run the MPC planner to compare that action vs the optimal one [Tier 2]
  -> structured counterfactual feedback for the UI / API.

State estimator uses the trained talk-type classifier (runs/talktype_clf) if
present, else a transparent keyword fallback so the module always runs.
Action tagger is rule-based with a hook to swap in the MI behaviour classifier.
"""
import os
import re
from functools import lru_cache

from scripts.world_model.transition_model import TransitionModel, ACTIONS
from scripts.world_model import planner as P

# ----------------------------- action tagger -----------------------------
_ADVICE = re.compile(r"\b(you should|you need to|you have to|you ought to|i'?d recommend|try to|make sure)\b", re.I)
_REFLECT = re.compile(r"^(so|it sounds like|you'?re feeling|you feel|what i'?m hearing|part of you)\b", re.I)
_OPTIONS = re.compile(r"\b(option|either|or we could|we could try|a, b)\b", re.I)
_NEGOTIATE = re.compile(r"\b(could we|would you be willing|can we agree|shall we|let'?s agree|how about we)\b", re.I)
_OPEN_Q = re.compile(r"^(what|how|why|tell me|describe|in what way)\b", re.I)


@lru_cache(maxsize=1)
def _load_action_clf(path="runs/action_clf"):
    if not os.path.isdir(path):
        return None
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSequenceClassification.from_pretrained(path)
        mdl.eval()
        return tok, mdl
    except Exception as e:
        print(f"[counterfactual] action clf load failed: {e}")
        return None


def tag_action(text: str) -> str:
    """MI action tag for a counselor utterance (1 of ACTIONS).

    Uses the trained action classifier (runs/action_clf) if present, else the
    transparent rule-based fallback below.
    """
    clf = _load_action_clf()
    if clf is not None:
        import torch
        tok, mdl = clf
        enc = tok(text, return_tensors="pt", truncation=True, max_length=128)
        enc.pop("token_type_ids", None)
        with torch.no_grad():
            pred = mdl(**enc).logits.argmax(-1).item()
        lab = mdl.config.id2label[pred]
        if lab in ACTIONS:
            return lab
    return _tag_action_rule(text)


def _tag_action_rule(text: str) -> str:
    """Rule-based MI action tag for a counselor utterance (1 of ACTIONS)."""
    t = text.strip()
    low = t.lower()
    is_q = t.endswith("?")
    if _NEGOTIATE.search(low):
        return "negotiation"
    if _OPTIONS.search(low):
        return "options"
    if is_q:
        # open vs closed
        if _OPEN_Q.match(low) or low.startswith(("what", "how", "why")):
            return "open_question"
        return "closed_question"
    if _ADVICE.search(low):
        return "advice"
    if _REFLECT.match(low):
        # complex if it links two ideas / ambivalence
        return "complex_reflection" if (" but " in low or " and " in low or "part of you" in low) else "simple_reflection"
    # informational statement
    if len(t.split()) > 6:
        return "information"
    return "other"


# --------------------------- state estimator ---------------------------
@lru_cache(maxsize=1)
def _load_talktype_clf(path="runs/talktype_clf"):
    if not os.path.isdir(path):
        return None
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForSequenceClassification.from_pretrained(path)
        mdl.eval()
        return tok, mdl
    except Exception as e:
        print(f"[counterfactual] talk-type clf load failed: {e}")
        return None


_CHANGE_KW = re.compile(r"\b(i want|i'?d like|i need to|i should|i'?m going to|i could|i will|ready to|i hope)\b", re.I)
_SUSTAIN_KW = re.compile(r"\b(i can'?t|i don'?t want|too hard|no point|i like|not ready|but i|i won'?t)\b", re.I)


def estimate_state(client_text: str, context: str = "") -> str:
    """Estimate client talk-type {change,sustain,neutral}."""
    clf = _load_talktype_clf()
    if clf is not None:
        import torch
        tok, mdl = clf
        inp = (f"Counselor: {context}\nClient: {client_text}" if context else client_text)
        enc = tok(inp, return_tensors="pt", truncation=True, max_length=192)
        enc.pop("token_type_ids", None)  # DistilBERT.forward has no token_type_ids
        with torch.no_grad():
            pred = mdl(**enc).logits.argmax(-1).item()
        return mdl.config.id2label[pred]
    # transparent fallback
    if _CHANGE_KW.search(client_text):
        return "change"
    if _SUSTAIN_KW.search(client_text):
        return "sustain"
    return "neutral"


# --------------------------- feedback assembly ---------------------------
@lru_cache(maxsize=1)
def _model(data="data/world_model/transitions.jsonl"):
    return TransitionModel.from_jsonl(data, split="train")


def coach_feedback(client_msg: str, coach_reply: str, horizon=3, gamma=0.9, context=""):
    """Full counterfactual feedback for one (client_msg, coach_reply) pair."""
    from scripts.world_model.safety import safety_screen
    safety = safety_screen(client_msg, coach_reply)
    if safety["crisis_detected"]:
        # In a crisis, suppress MI coaching tips and escalate instead.
        return {
            "safety": safety,
            "estimated_client_state": "crisis",
            "your_action": tag_action(coach_reply),
            "your_reply": coach_reply,
            "escalation": safety["escalation"],
            "ranked_actions": [],
        }
    state = estimate_state(client_msg, context)
    action = tag_action(coach_reply)
    cf = P.counterfactual(_model(), state, action, horizon, gamma)
    return {
        "safety": safety,
        "estimated_client_state": state,
        "your_action": action,
        "your_reply": coach_reply,
        "predicted_effect": {
            "P_change": cf["actual"]["P_change"],
            "P_sustain": cf["actual"]["P_sustain"],
            "Q": cf["actual"]["Q"],
        },
        "is_optimal": cf["is_optimal"],
        "better_action": cf["best"]["action"],
        "better_example": cf["best"]["template"],
        "better_effect": {
            "P_change": cf["best"]["P_change"],
            "P_sustain": cf["best"]["P_sustain"],
            "Q": cf["best"]["Q"],
        },
        "change_uplift": cf["change_uplift"],
        "ranked_actions": cf["ranked"],
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", default="I know I drink too much but I don't think I can stop.")
    ap.add_argument("--coach", default="You should really cut down starting this week.")
    ap.add_argument("--horizon", type=int, default=3)
    args = ap.parse_args()
    fb = coach_feedback(args.client, args.coach, horizon=args.horizon)
    print(json.dumps({k: v for k, v in fb.items() if k != "ranked_actions"}, indent=2))
    print("\nTop-3 alternatives:")
    for r in fb["ranked_actions"][:3]:
        print(f"  {r['action']:<20} P(change)={r['P_change']:.2f} Q={r['Q']:.2f}  \"{r['template']}\"")
