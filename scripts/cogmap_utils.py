#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cogmap_utils.py

Heuristic cognitive-map builder for MI-style coaching dialogs.

Input  : list of turns [{"user": "...", "coach": "..."}, ...]
Output : dict { "nodes": [...], "edges": [...], "summary": "..." }

Nodes:
    {
        "id": "n1",
        "label": "Be healthier to play with kids",
        "type": "goal",          # goal | emotion | barrier | behaviour | value | outcome | other
        "mentions": [1, 2]       # user turn indices where this idea appears
    }

Edges:
    {
        "source": "n1",
        "target": "n2",
        "type": "supports_goal", # supports_goal | barrier_to | influences | leads_to | related_to
        "evidence_turns": [2],   # user turns where this relation is implied
        "weight": 1.0
    }

This module is intentionally lightweight: no external LLM calls, no embeddings.
You can later swap the heuristics with a more powerful model if you like.
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any


# --------- basic structures ---------


@dataclass
class Node:
    id: str
    label: str
    type: str = "other"
    mentions: List[int] = field(default_factory=list)


@dataclass
class Edge:
    source: str
    target: str
    type: str = "related_to"
    evidence_turns: List[int] = field(default_factory=list)
    weight: float = 1.0


# --------- small vocab for heuristics ---------
# Extended for Kerrio.AI requirements (Mayo Clinic diagnostic model)

EMOTION_WORDS = {
    "tired": "Tiredness",
    "exhausted": "Tiredness",
    "stressed": "Stress",
    "anxious": "Anxiety",
    "sad": "Sadness",
    "down": "Low mood",
    "guilty": "Guilt",
    "frustrated": "Frustration",
    "overwhelmed": "Overwhelm",
    "burned out": "Burnout",
    "unfulfilled": "Unfulfillment",
    "stuck": "Feeling stuck",
    "lost": "Feeling lost",
}

HEALTH_WORDS = {"health", "healthy", "fitter", "fitness"}

BEHAVIOUR_PATTERNS = [
    (r"\bwalk(ing)?\b", "Walking"),
    (r"\brun(ning)?\b", "Running / jogging"),
    (r"\bexercise\b", "Exercise"),
    (r"\bdrink(ing)?\b", "Drinking alcohol"),
    (r"\bsmok(ing|e)\b", "Smoking"),
    (r"\bscroll(ing)? (on )?(my |the )?phone\b", "Phone scrolling"),
    (r"\boverwork(ing)?\b", "Overworking"),
    (r"\bprocrastinat(e|ing)\b", "Procrastination"),
    (r"\bavoid(ing)?\b", "Avoidance"),
]

BARRIER_PHRASES = [
    "too tired",
    "no time",
    "too busy",
    "hard to",
    "difficult to",
    "can't seem to",
    "don't know how",
    "afraid to",
    "scared to",
]

GOAL_PATTERNS = [
    r"\bI want to ([^.,;]+)",
    r"\bI'd like to ([^.,;]+)",
    r"\bmy goal is to ([^.,;]+)",
    r"\bI need to ([^.,;]+)",
    r"\bI'm trying to ([^.,;]+)",
]

SO_I_CAN_PATTERN = r"\bso I can ([^.,;]+)"

# === Kerrio-specific: Core Assumptions & Beliefs ===
CORE_ASSUMPTION_PATTERNS = [
    (r"\bI(?:'m| am) (?:not |never )?(?:good enough|worthy|capable|smart enough)\b", "core_assumption"),
    (r"\bI (?:always|never) ([^.,;]+)", "recurrent_pattern"),
    (r"\bI(?:'ve| have) always been ([^.,;]+)", "core_assumption"),
    (r"\bI should(?:n't)? ([^.,;]+)", "belief"),
    (r"\bI must ([^.,;]+)", "belief"),
    (r"\bpeople (?:always|never) ([^.,;]+)", "belief"),
    (r"\bif I (?:don't|do) ([^.,;]+) then ([^.,;]+)", "core_assumption"),
]

# === Kerrio-specific: Recurrent Patterns ===
RECURRENT_PATTERN_SIGNALS = [
    "i always",
    "every time",
    "i keep",
    "i tend to",
    "this happens whenever",
    "the same thing",
    "pattern",
    "cycle",
    "i find myself",
]

# === Kerrio-specific: Hidden Constraints ===
HIDDEN_CONSTRAINT_PATTERNS = [
    (r"(?:without realizing|unconsciously|automatically)[^.]*", "hidden_constraint"),
    (r"I don't know why I ([^.,;]+)", "hidden_constraint"),
    (r"for some reason I ([^.,;]+)", "hidden_constraint"),
]

# === Kerrio-specific: Triggers ===
TRIGGER_PATTERNS = [
    (r"when(?:ever)? ([^,]+), I ([^.,;]+)", "trigger"),
    (r"([^.,;]+) makes me ([^.,;]+)", "trigger"),
    (r"([^.,;]+) triggers ([^.,;]+)", "trigger"),
]

# === Kerrio-specific: Strengths ===
STRENGTH_PATTERNS = [
    (r"I(?:'m| am) good at ([^.,;]+)", "strength"),
    (r"my strength is ([^.,;]+)", "strength"),
    (r"I excel at ([^.,;]+)", "strength"),
    (r"people say I(?:'m| am) ([^.,;]+)", "strength"),
]


# --------- text helpers ---------


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def clean_label(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text[:120]


def jaccard_sim(a: str, b: str) -> float:
    """Lexical similarity used for cheap node clustering."""
    toks_a = set(re.findall(r"[a-z0-9]+", a.lower()))
    toks_b = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not toks_a or not toks_b:
        return 1.0 if a.lower() == b.lower() else 0.0
    return len(toks_a & toks_b) / len(toks_a | toks_b)


# --------- concept extraction from one utterance ---------


def extract_concepts_from_user_text(text: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Heuristically extract concepts from a single user utterance.

    Extended for Kerrio.AI to support Mayo Clinic diagnostic model.
    Returns dict with keys for all Kerrio node types.
    """
    concepts: Dict[str, List[Dict[str, str]]] = {
        "goals": [],
        "emotions": [],
        "behaviours": [],
        "barriers": [],
        "values": [],
        "outcomes": [],
        # Kerrio-specific additions
        "core_assumptions": [],
        "recurrent_patterns": [],
        "hidden_constraints": [],
        "triggers": [],
        "strengths": [],
        "beliefs": [],
    }

    low = text.lower()

    # 1) emotions
    for w, label in EMOTION_WORDS.items():
        if w in low:
            concepts["emotions"].append({"label": label, "type": "emotion"})

    # 2) health-related values
    if any(w in low for w in HEALTH_WORDS):
        concepts["values"].append({"label": "Health", "type": "value"})
        # also sometimes explicit motivation
        concepts["values"].append({"label": "Motivation to be healthy", "type": "value"})

    # 3) behaviours
    for pat, label in BEHAVIOUR_PATTERNS:
        if re.search(pat, low):
            concepts["behaviours"].append({"label": label, "type": "behaviour"})

    # more specific behaviour phrase if present
    m_walk_after = re.search(r"walk (after|in) dinner[^.]*", low)
    if m_walk_after:
        phr = clean_label(m_walk_after.group(0))
        concepts["behaviours"].append({"label": phr, "type": "behaviour"})

    # 4) goals (I want to..., I'd like to..., my goal is to...)
    for pat in GOAL_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            goal_text = clean_label(m.group(1))
            concepts["goals"].append({"label": goal_text, "type": "goal"})

    # also "so I can ..."
    m = re.search(SO_I_CAN_PATTERN, text, flags=re.IGNORECASE)
    if m:
        goal_text = "So I can " + clean_label(m.group(1))
        concepts["goals"].append({"label": goal_text, "type": "goal"})

    # 5) barriers
    for ph in BARRIER_PHRASES:
        if ph in low:
            concepts["barriers"].append({"label": ph.capitalize(), "type": "barrier"})

    # 6) outcome phrases like "feel better", "more energy"
    m = re.search(r"feel(ing)? better[^.]*", low)
    if m:
        concepts["outcomes"].append(
            {"label": clean_label(m.group(0).capitalize()), "type": "outcome"}
        )

    # === Kerrio-specific extractions ===

    # 7) Core assumptions and beliefs
    for pat, node_type in CORE_ASSUMPTION_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            label = clean_label(m.group(0))
            if node_type == "core_assumption":
                concepts["core_assumptions"].append({"label": label, "type": "core_assumption"})
            elif node_type == "recurrent_pattern":
                concepts["recurrent_patterns"].append({"label": label, "type": "recurrent_pattern"})
            elif node_type == "belief":
                concepts["beliefs"].append({"label": label, "type": "belief"})

    # 8) Recurrent patterns
    for signal in RECURRENT_PATTERN_SIGNALS:
        if signal in low:
            # Extract the sentence containing the signal
            sentences = text.split(".")
            for sent in sentences:
                if signal in sent.lower():
                    label = clean_label(sent.strip())
                    if label and len(label) > 10:
                        concepts["recurrent_patterns"].append({
                            "label": f"Pattern: {label}",
                            "type": "recurrent_pattern"
                        })
                    break
            break

    # 9) Hidden constraints
    for pat, node_type in HIDDEN_CONSTRAINT_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            label = clean_label(m.group(0))
            concepts["hidden_constraints"].append({"label": label, "type": "hidden_constraint"})

    # 10) Triggers
    for pat, node_type in TRIGGER_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            trigger_text = m.group(1) if m.lastindex >= 1 else m.group(0)
            label = f"Trigger: {clean_label(trigger_text)}"
            concepts["triggers"].append({"label": label, "type": "trigger"})

    # 11) Strengths
    for pat, node_type in STRENGTH_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            strength_text = m.group(1) if m.lastindex >= 1 else m.group(0)
            label = f"Strength: {clean_label(strength_text)}"
            concepts["strengths"].append({"label": label, "type": "strength"})

    return concepts


# --------- node / edge building ---------


def build_raw_nodes_edges(session: List[Dict[str, str]]) -> (List[Node], List[Edge]):
    """
    Build *uncollapsed* nodes and edges from a sequence of turns.
    Node ids are temporary; we'll reassign after clustering.

    Extended for Kerrio.AI Mayo Clinic diagnostic model.
    """
    nodes: List[Node] = []
    edges: List[Edge] = []

    def add_node(lbl: str, typ: str, turn_idx: int) -> Node:
        node = Node(id=f"tmp_{len(nodes)}", label=clean_label(lbl), type=typ, mentions=[turn_idx])
        nodes.append(node)
        return node

    for i, turn in enumerate(session, start=1):
        user_txt = turn.get("user", "") or ""
        if not user_txt.strip():
            continue

        conc = extract_concepts_from_user_text(user_txt)

        # Original node types
        emo_nodes = [add_node(c["label"], c["type"], i) for c in conc["emotions"]]
        beh_nodes = [add_node(c["label"], c["type"], i) for c in conc["behaviours"]]
        goal_nodes = [add_node(c["label"], c["type"], i) for c in conc["goals"]]
        barrier_nodes = [add_node(c["label"], c["type"], i) for c in conc["barriers"]]
        value_nodes = [add_node(c["label"], c["type"], i) for c in conc["values"]]
        outcome_nodes = [add_node(c["label"], c["type"], i) for c in conc["outcomes"]]

        # Kerrio-specific node types
        assumption_nodes = [add_node(c["label"], c["type"], i) for c in conc["core_assumptions"]]
        pattern_nodes = [add_node(c["label"], c["type"], i) for c in conc["recurrent_patterns"]]
        hidden_nodes = [add_node(c["label"], c["type"], i) for c in conc["hidden_constraints"]]
        trigger_nodes = [add_node(c["label"], c["type"], i) for c in conc["triggers"]]
        strength_nodes = [add_node(c["label"], c["type"], i) for c in conc["strengths"]]
        belief_nodes = [add_node(c["label"], c["type"], i) for c in conc["beliefs"]]

        # === Original relational heuristics ===

        # emotion -> behaviour (influences)
        for e in emo_nodes:
            for b in beh_nodes:
                edges.append(
                    Edge(
                        source=e.id,
                        target=b.id,
                        type="influences",
                        evidence_turns=[i],
                    )
                )
        # behaviour -> goal (supports_goal)
        for b in beh_nodes:
            for g in goal_nodes:
                edges.append(
                    Edge(
                        source=b.id,
                        target=g.id,
                        type="supports_goal",
                        evidence_turns=[i],
                    )
                )
        # barrier -> goal (barrier_to)
        for br in barrier_nodes:
            for g in goal_nodes:
                edges.append(
                    Edge(
                        source=br.id,
                        target=g.id,
                        type="barrier_to",
                        evidence_turns=[i],
                    )
                )
        # outcome -> goal (leads_to or supports)
        for o in outcome_nodes:
            for g in goal_nodes:
                edges.append(
                    Edge(
                        source=o.id,
                        target=g.id,
                        type="leads_to",
                        evidence_turns=[i],
                    )
                )
        # value -> goal (related_to/supports)
        for v in value_nodes:
            for g in goal_nodes:
                edges.append(
                    Edge(
                        source=v.id,
                        target=g.id,
                        type="supports_goal",
                        evidence_turns=[i],
                    )
                )

        # === Kerrio-specific relational heuristics ===

        # core_assumption -> recurrent_pattern (causes)
        for a in assumption_nodes:
            for p in pattern_nodes:
                edges.append(
                    Edge(source=a.id, target=p.id, type="causes", evidence_turns=[i])
                )

        # core_assumption -> barrier (causes)
        for a in assumption_nodes:
            for br in barrier_nodes:
                edges.append(
                    Edge(source=a.id, target=br.id, type="causes", evidence_turns=[i])
                )

        # recurrent_pattern -> emotion (leads_to)
        for p in pattern_nodes:
            for e in emo_nodes:
                edges.append(
                    Edge(source=p.id, target=e.id, type="leads_to", evidence_turns=[i])
                )

        # recurrent_pattern -> barrier (maintains)
        for p in pattern_nodes:
            for br in barrier_nodes:
                edges.append(
                    Edge(source=p.id, target=br.id, type="maintains", evidence_turns=[i])
                )

        # hidden_constraint -> recurrent_pattern (causes)
        for h in hidden_nodes:
            for p in pattern_nodes:
                edges.append(
                    Edge(source=h.id, target=p.id, type="causes", evidence_turns=[i])
                )

        # trigger -> behaviour (triggers)
        for t in trigger_nodes:
            for b in beh_nodes:
                edges.append(
                    Edge(source=t.id, target=b.id, type="triggers", evidence_turns=[i])
                )

        # trigger -> emotion (triggers)
        for t in trigger_nodes:
            for e in emo_nodes:
                edges.append(
                    Edge(source=t.id, target=e.id, type="triggers", evidence_turns=[i])
                )

        # strength -> goal (supports_goal)
        for s in strength_nodes:
            for g in goal_nodes:
                edges.append(
                    Edge(source=s.id, target=g.id, type="supports_goal", evidence_turns=[i])
                )

        # belief -> behaviour (influences)
        for bl in belief_nodes:
            for b in beh_nodes:
                edges.append(
                    Edge(source=bl.id, target=b.id, type="influences", evidence_turns=[i])
                )

        # belief -> barrier (causes)
        for bl in belief_nodes:
            for br in barrier_nodes:
                edges.append(
                    Edge(source=bl.id, target=br.id, type="causes", evidence_turns=[i])
                )

    return nodes, edges


def cluster_nodes(nodes: List[Node], sim_threshold: float = 0.7) -> (List[Node], Dict[str, str]):
    """
    Deduplicate semantically similar nodes using Jaccard similarity over tokens.
    Returns:
      - new_nodes: clustered nodes with fresh ids n1, n2, ...
      - id_map: mapping old_id -> new_id
    """
    clusters: List[Node] = []
    id_map: Dict[str, str] = {}

    for node in nodes:
        matched_cluster = None
        for c in clusters:
            sim = jaccard_sim(node.label, c.label)
            if sim >= sim_threshold and node.type == c.type:
                matched_cluster = c
                break
        if matched_cluster is None:
            new_id = f"n{len(clusters) + 1}"
            new_node = Node(id=new_id, label=node.label, type=node.type, mentions=list(node.mentions))
            clusters.append(new_node)
            id_map[node.id] = new_id
        else:
            id_map[node.id] = matched_cluster.id
            # merge mentions
            for t in node.mentions:
                if t not in matched_cluster.mentions:
                    matched_cluster.mentions.append(t)

    return clusters, id_map


def remap_and_merge_edges(edges: List[Edge], id_map: Dict[str, str]) -> List[Edge]:
    """
    Remap edge source/target using id_map and merge duplicates by (src, tgt, type).
    """
    merged: Dict[tuple, Edge] = {}
    for e in edges:
        src = id_map.get(e.source)
        tgt = id_map.get(e.target)
        if not src or not tgt or src == tgt:
            continue
        key = (src, tgt, e.type)
        if key not in merged:
            merged[key] = Edge(source=src, target=tgt, type=e.type,
                               evidence_turns=list(e.evidence_turns), weight=e.weight)
        else:
            edge = merged[key]
            for t in e.evidence_turns:
                if t not in edge.evidence_turns:
                    edge.evidence_turns.append(t)
            edge.weight += e.weight
    return list(merged.values())


def build_summary(nodes: List[Node], edges: List[Edge]) -> str:
    """
    Build a structured diagnostic summary for Kerrio.AI.
    Organized by Mayo Clinic diagnostic categories.
    """
    # Original categories
    goals = [n.label for n in nodes if n.type == "goal"]
    behaviours = [n.label for n in nodes if n.type == "behaviour"]
    barriers = [n.label for n in nodes if n.type == "barrier"]
    values = [n.label for n in nodes if n.type == "value"]
    outcomes = [n.label for n in nodes if n.type == "outcome"]

    # Kerrio-specific categories
    core_assumptions = [n.label for n in nodes if n.type == "core_assumption"]
    recurrent_patterns = [n.label for n in nodes if n.type == "recurrent_pattern"]
    hidden_constraints = [n.label for n in nodes if n.type == "hidden_constraint"]
    triggers = [n.label for n in nodes if n.type == "trigger"]
    strengths = [n.label for n in nodes if n.type == "strength"]
    beliefs = [n.label for n in nodes if n.type == "belief"]
    emotions = [n.label for n in nodes if n.type == "emotion"]

    parts: List[str] = []

    # Structured Diagnostic Summary (per Kerrio User Journey PDF)
    parts.append("=== STRUCTURED DIAGNOSTIC SUMMARY ===")

    if goals:
        parts.append("\n[Goals]: " + "; ".join(goals))

    if core_assumptions:
        parts.append("\n[Core Assumptions]: " + "; ".join(core_assumptions))

    if recurrent_patterns:
        parts.append("\n[Recurrent Patterns]: " + "; ".join(recurrent_patterns))

    if beliefs:
        parts.append("\n[Beliefs]: " + "; ".join(beliefs))

    if barriers:
        parts.append("\n[Barriers]: " + "; ".join(barriers))

    if hidden_constraints:
        parts.append("\n[Hidden Constraints]: " + "; ".join(hidden_constraints))

    if triggers:
        parts.append("\n[Triggers]: " + "; ".join(triggers))

    if emotions:
        parts.append("\n[Emotional Patterns]: " + "; ".join(sorted(set(emotions))))

    if behaviours:
        parts.append("\n[Behaviours]: " + "; ".join(behaviours))

    if strengths:
        parts.append("\n[Strengths/Resources]: " + "; ".join(strengths))

    if values:
        parts.append("\n[Underlying Values]: " + "; ".join(sorted(set(values))))

    if outcomes:
        parts.append("\n[Desired Outcomes]: " + "; ".join(outcomes))

    if len(parts) <= 1:
        return "Cognitive Wiring Map in progress (continue gathering history across three pillars: History, Psychology/Philosophy, Physiology)."

    return "".join(parts)


def build_cognitive_map_from_session(session: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Main entry point used by FastAPI endpoint.

    session: list of {"user": "...", "coach": "..."}
    """
    if not session:
        return {"nodes": [], "edges": [], "summary": "No conversation yet."}

    raw_nodes, raw_edges = build_raw_nodes_edges(session)
    nodes, id_map = cluster_nodes(raw_nodes)
    edges = remap_and_merge_edges(raw_edges, id_map)
    summary = build_summary(nodes, edges)

    return {
        "nodes": [asdict(n) for n in nodes],
        "edges": [asdict(e) for e in edges],
        "summary": summary,
    }


# Optional: small CLI for offline testing
if __name__ == "__main__":
    # Example mini-dialog for quick test
    example_session = [
        {
            "user": "I've been feeling really tired and unmotivated to exercise lately.",
            "coach": "",
        },
        {
            "user": "I want to be healthier so I can play with my kids without getting so exhausted.",
            "coach": "",
        },
        {
            "user": "I was thinking about going for a walk after dinner a few times a week.",
            "coach": "",
        },
        {
            "user": "Mostly work and feeling too tired in the evening. Sometimes I just end up on my phone instead.",
            "coach": "",
        },
    ]
    cm = build_cognitive_map_from_session(example_session)
    print(json.dumps(cm, indent=2, ensure_ascii=False))
