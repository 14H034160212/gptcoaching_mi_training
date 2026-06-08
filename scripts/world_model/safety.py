#!/usr/bin/env python3
"""
Safety layer for the MI world model.

The world model is a *coaching* tool, not a crisis service. Before offering
MI counterfactual feedback we screen the client message for crisis / clinical-
risk signals. On a hit we surface an ESCALATION rather than a coaching tip,
and we flag counselor replies that give risky directives in those moments.

This is intentionally high-recall (better to over-flag) and rule-based so it is
auditable. It is a safety net, not a diagnostic instrument.
"""
import re

# Crisis / self-harm / harm-to-others signals (high recall).
CRISIS_PATTERNS = [
    r"\bkill(?:ing)? myself\b", r"\bend(?:ing)? (?:my life|it all)\b",
    r"\bsuicid", r"\bself[- ]?harm", r"\bharm(?:ing)? myself\b",
    r"\bhurt(?:ing)? myself\b", r"\bcut(?:ting)? myself\b",
    r"\bdon'?t want to (?:be alive|live)\b", r"\bno reason to live\b",
    r"\bwant(?:ing)? to die\b", r"\bbetter off dead\b", r"\boverdos",
    r"\bhurt(?:ing)? (?:someone|him|her|them|people)\b",
    r"\bkill(?:ing)? (?:him|her|them|someone)\b",
]

# Counselor replies that are inappropriate in a crisis moment (e.g. casual advice
# or minimizing instead of escalating).
UNSAFE_COACH_PATTERNS = [
    r"\byou should just\b", r"\bcalm down\b", r"\bget over it\b",
    r"\bit'?s not a big deal\b", r"\bstop worrying\b",
]

ESCALATION_MESSAGE = (
    "This message contains possible crisis or self-harm signals. This tool is a "
    "motivational-interviewing coach, not a crisis service. The counselor should "
    "pause MI technique, respond with direct concern, assess safety, and connect "
    "the client to emergency/crisis resources (e.g. a local emergency number or a "
    "suicide/crisis hotline)."
)


def detect_crisis(client_text: str) -> bool:
    low = (client_text or "").lower()
    return any(re.search(p, low) for p in CRISIS_PATTERNS)


def coach_reply_unsafe_in_crisis(coach_text: str) -> bool:
    low = (coach_text or "").lower()
    return any(re.search(p, low) for p in UNSAFE_COACH_PATTERNS)


def safety_screen(client_text: str, coach_text: str = "") -> dict:
    """Return a safety verdict for a turn."""
    crisis = detect_crisis(client_text)
    return {
        "crisis_detected": crisis,
        "coach_reply_risky": crisis and coach_reply_unsafe_in_crisis(coach_text),
        "escalation": ESCALATION_MESSAGE if crisis else None,
    }
