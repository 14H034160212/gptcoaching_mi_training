#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AntiMIResponderV2: Generates intentionally non‑MI, prescriptive, low‑empathy replies.
Design goals:
- Topic aware (alcohol, smoking, work, diet, exercise, sleep, stress).
- Uses several anti‑MI tactics with controllable diversity.
- Mirrors a detail from user text, then pivots to orders.
- Avoids unsafe content: no promotion of self‑harm or dangerous instructions.
"""

import re, random, json
from typing import List, Dict, Optional

TOPIC_PATTERNS = {
    "alcohol": r"\b(alcohol|drink(?:s|ing)?|beer|wine|liquor|hangover)\b",
    "smoking": r"\b(smok(?:e|ing)|cigarette(?:s)?|vape|vaping|nicotine)\b",
    "work": r"\b(work|job|boss|meeting|deadline|office)\b",
    "diet": r"\b(diet|eat(?:ing)?|food|snack|sugar|calori(?:e|es)|weight)\b",
    "exercise": r"\b(exercise|workout|gym|run(?:ning)?|walk(?:ing)?|training|yoga)\b",
    "sleep": r"\b(sleep|insomnia|awake|asleep|bedtime|nap|night)\b",
    "stress": r"\b(stress|stressed|anxiety|anxious|overwhelmed|pressure|burnout)\b",
}

# Tactic banks
CLOSED_OPENERS = [
    "Is that clear?",
    "Can you follow basic instructions?",
    "Do you understand what needs to happen now?",
    "Are you willing to comply starting today?",
    "Will you stop debating this and stick to the plan?",
]

MORALIZING = [
    "You should know better by now.",
    "This is about discipline, not feelings.",
    "It’s irresponsible to keep repeating this.",
    "Excuses won’t change outcomes.",
]

MINIMIZING = [
    "This isn’t complicated.",
    "People manage this all the time.",
    "You’re overcomplicating a simple issue.",
    "Plenty of others do this without fuss.",
]

CONSEQUENCES = [
    "If you don’t, expect things to get worse quickly.",
    "Otherwise you’ll regret it sooner than you think.",
    "Failing to comply will only prolong the problem.",
    "If you ignore this, you’re choosing the consequences.",
]

GENERIC_ORDERS = [
    "Write down strict rules for yourself and follow them without exceptions.",
    "Cut out anything that gets in the way and report back only if you comply.",
    "Stop negotiating with yourself and execute the plan as written.",
]

TOPIC_ORDERS = {
    "alcohol": [
        "Stop drinking entirely for the next 30 days and avoid any situation with alcohol.",
        "Remove all alcohol from your home today and decline any invitations that involve drinks.",
        "Set your intake to zero on weekdays and one maximum on weekends—no exceptions.",
    ],
    "smoking": [
        "Quit immediately and get rid of all smoking/vaping materials.",
        "Avoid any environment where others smoke and tell friends you won’t discuss it.",
        "When a craving hits, ignore it and continue with your tasks—don’t indulge it.",
    ],
    "work": [
        "Create an hour‑by‑hour schedule for tomorrow and stick to it exactly.",
        "Turn off all notifications and do not switch tasks until the hour ends.",
        "Submit your deliverables first, then revisit feelings later.",
    ],
    "diet": [
        "Cut all sugary snacks now and log everything you eat.",
        "Pre‑plan three simple meals and repeat them daily—no deviations.",
        "Stop eating after 7 PM, period.",
    ],
    "exercise": [
        "Complete a 30‑minute workout every day this week—no rest days.",
        "Walk 12,000 steps before any screen time.",
        "Join a class today and attend tonight.",
    ],
    "sleep": [
        "Go to bed at 10 PM and wake up at 6 AM every day without exception.",
        "No screens after 8:30 PM; put devices in another room.",
        "Skip naps entirely for the next two weeks.",
    ],
    "stress": [
        "Stop fixating on stress and follow a strict schedule.",
        "Plan tomorrow in 30‑minute blocks and don’t deviate.",
        "Push emotions aside and focus on output only.",
    ],
    "generic": GENERIC_ORDERS,
}

def guess_topic(text: str) -> str:
    t = (text or "").lower()
    for topic, pat in TOPIC_PATTERNS.items():
        if re.search(pat, t):
            return topic
    return "generic"

def extract_detail(text: str, max_words: int = 14) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # Prefer grabbing the last clause users mentioned
    t = re.split(r"[\.!?]", t)[-1].strip() or t
    words = t.split()
    if len(words) <= max_words:
        return t
    return " ".join(words[-max_words:])

def detect_numbers(text: str) -> Optional[str]:
    # Capture small factual hooks to sound “certain” while being directive
    m = re.findall(r"\b(\d{1,3})(?:\s*(?:x|times|drinks?|minutes?|hours?|days?))?\b", text or "", flags=re.I)
    if m:
        val = m[-1]
        return val
    return None

class AntiMIResponderV2:
    def __init__(self, seed: int = 123):
        self.rng = random.Random(seed)

    def _pick(self, arr: List[str]) -> str:
        return self.rng.choice(arr)

    def generate(self, user_utt: str, state_before: Optional[Dict] = None, max_len: int = 220) -> str:
        topic = guess_topic(user_utt or "")
        detail = extract_detail(user_utt or "")
        number = detect_numbers(user_utt or "")

        parts: List[str] = []

        # Closed, compliance‑checking opener (anti‑collaboration)
        if self.rng.random() < 0.75:
            parts.append(self._pick(CLOSED_OPENERS))

        # Faux acknowledgement → pivot to orders (anti‑empathy)
        if detail:
            parts.append(f"I heard \"{detail}\", but that doesn’t change what needs to happen.")

        # Certainty inflation using a detected number, if any
        if number and self.rng.random() < 0.5:
            parts.append(f"You’ve already mentioned {number}; that’s precisely why we must act decisively.")

        # Hard directive (anti‑autonomy)
        topic_orders = TOPIC_ORDERS.get(topic, TOPIC_ORDERS["generic"])
        parts.append(self._pick(topic_orders))

        # Moralizing or minimizing (invalidating feelings)
        if self.rng.random() < 0.7:
            parts.append(self._pick(MINIMIZING))
        if self.rng.random() < 0.6:
            parts.append(self._pick(MORALIZING))

        # Consequences / threat framing (pressure instead of partnership)
        if self.rng.random() < 0.55:
            parts.append(self._pick(CONSEQUENCES))

        # Close with a binary compliance check
        parts.append("Will you follow this exactly, yes or no?")

        text = " ".join(parts)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > max_len:
            text = text[:max_len].rstrip() + "..."
        return text
