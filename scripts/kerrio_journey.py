#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
kerrio_journey.py — Kerrio.AI User Journey Management Module

Implements the Mayo Clinic-inspired 7-stage clinical journey:
1. Registration - Validated as invited guest
2. History Collection - Three pillars (History, Psychology/Philosophy, Physiology)
3. Consultation - Clarify ambiguities, uncover blind spots
4. Diagnosis - Build Cognitive Wiring Map, explain WHY
5. Proposal - Personalized treatment plan
6. Treatment - Cognitive Rewiring Maps
7. Monitoring - Longitudinal progress assessment

Key Design Principles (from Kerrio.AI User Journey PDF):
- "Accurate diagnosis is the foundation of effective treatment"
- "Understanding is a prerequisite for permanent change"
- "Client History and Clinician's Notes are maintained separately"
- "Diagnostic-first, not motivation/engagement focused"
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class JourneyStage(str, Enum):
    """The 7 stages of Kerrio clinical journey (Mayo Clinic model)."""
    REGISTRATION = "registration"
    HISTORY_COLLECTION = "history_collection"
    CONSULTATION = "consultation"
    DIAGNOSIS = "diagnosis"
    PROPOSAL = "proposal"
    TREATMENT = "treatment"
    MONITORING = "monitoring"

    @classmethod
    def next_stage(cls, current: "JourneyStage") -> Optional["JourneyStage"]:
        """Get the next stage in the journey."""
        stages = list(cls)
        idx = stages.index(current)
        if idx + 1 < len(stages):
            return stages[idx + 1]
        return None


# === Three Pillars of History Collection ===

@dataclass
class HistoryPillar:
    """Pillar 1: Life events, formative experiences, patterns."""
    life_events: List[str] = field(default_factory=list)
    formative_experiences: List[str] = field(default_factory=list)
    recurrent_patterns: List[str] = field(default_factory=list)
    background_summary: str = ""


@dataclass
class PsychologyPhilosophyPillar:
    """Pillar 2: Beliefs, values, meaning structures, emotional wiring."""
    beliefs: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    meaning_structures: List[str] = field(default_factory=list)
    emotional_wiring: List[str] = field(default_factory=list)
    core_assumptions: List[str] = field(default_factory=list)


@dataclass
class PhysiologyPillar:
    """Pillar 3: Sleep, stress, health, energy, constraints."""
    sleep_quality: str = ""
    sleep_hours: float = 0
    stress_level: str = ""  # low, moderate, high, severe
    health_conditions: List[str] = field(default_factory=list)
    energy_patterns: List[str] = field(default_factory=list)
    physical_constraints: List[str] = field(default_factory=list)


@dataclass
class ClientHistory:
    """
    Raw data across three pillars.
    This is the client's verbatim narrative - their unfiltered story.
    Maintained SEPARATELY from Clinician Notes.
    """
    history: HistoryPillar = field(default_factory=HistoryPillar)
    psychology_philosophy: PsychologyPhilosophyPillar = field(
        default_factory=PsychologyPhilosophyPillar
    )
    physiology: PhysiologyPillar = field(default_factory=PhysiologyPillar)
    raw_narrative_turns: List[Dict[str, str]] = field(default_factory=list)


# === Clinician Notes (AI's clinical observations) ===

@dataclass
class SessionInsight:
    """A single insight discovered during consultation."""
    turn_id: int
    observation: str
    category: str  # blind_spot, resistance, pattern, breakthrough
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class ClinicianNotes:
    """
    AI's clinical observations - maintained SEPARATELY from Client History.
    Any new insights uncovered are:
    - Explained to the client
    - Added back into the formal history
    - Used to refine the diagnostic model
    """
    session_insights: List[SessionInsight] = field(default_factory=list)
    emerging_patterns: List[str] = field(default_factory=list)
    diagnostic_hypotheses: List[str] = field(default_factory=list)
    blind_spots_identified: List[str] = field(default_factory=list)
    ambiguities_to_clarify: List[str] = field(default_factory=list)


# === Cognitive Wiring Map (Diagnosis output) ===

@dataclass
class CognitiveNode:
    """
    A node in the Cognitive Wiring Map (personalized brain model).
    Extended node types per Kerrio requirements.
    """
    id: str
    label: str
    type: str  # See NODE_TYPES below
    evidence_turns: List[int] = field(default_factory=list)
    confidence: float = 0.5
    explanation: str = ""  # WHY this node exists


@dataclass
class CognitiveEdge:
    """A relationship in the Cognitive Wiring Map."""
    source: str
    target: str
    type: str  # See EDGE_TYPES below
    evidence_turns: List[int] = field(default_factory=list)
    weight: float = 1.0
    explanation: str = ""


# Extended node types per Kerrio requirements
NODE_TYPES = [
    "goal",
    "value",
    "emotion",
    "barrier",
    "behaviour",
    "outcome",
    "core_assumption",      # Beliefs about self/world that drive behavior
    "core_constraint",      # Fundamental limitations
    "bottleneck",           # Specific blocking points
    "hidden_constraint",    # Unconscious limitations
    "recurrent_pattern",    # Repeating cycles
    "belief",               # General beliefs
    "strength",             # Resources and capabilities
    "trigger",              # What activates patterns
    "coping_mechanism",     # How client deals with stress
]

# Extended edge types
EDGE_TYPES = [
    "supports_goal",
    "barrier_to",
    "influences",
    "leads_to",
    "causes",
    "maintains",            # What keeps a pattern going
    "conflicts_with",
    "reinforces",
    "triggers",             # What activates something
    "masks",                # What hides underlying issues
    "compensates_for",
    "related_to",
]


@dataclass
class CognitiveWiringMap:
    """
    Full personalized brain model built during diagnosis.
    This is unique to each client and forms the foundation for treatment.
    """
    nodes: List[CognitiveNode] = field(default_factory=list)
    edges: List[CognitiveEdge] = field(default_factory=list)
    summary: str = ""
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now(timezone.utc).isoformat()

    def add_node(self, node: CognitiveNode) -> None:
        """Add a node, avoiding duplicates by id."""
        if not any(n.id == node.id for n in self.nodes):
            self.nodes.append(node)

    def add_edge(self, edge: CognitiveEdge) -> None:
        """Add an edge, avoiding duplicates."""
        key = (edge.source, edge.target, edge.type)
        if not any((e.source, e.target, e.type) == key for e in self.edges):
            self.edges.append(edge)


# === Diagnosis ===

@dataclass
class RecommendedVideo:
    """Educational video recommendation based on diagnosis."""
    video_id: str
    title: str
    relevance: str
    url: str = ""


@dataclass
class Diagnosis:
    """
    The diagnostic output - the most important phase.
    Explains WHY the problem exists, not just what it looks like.
    """
    core_constraints: List[str] = field(default_factory=list)
    bottlenecks: List[str] = field(default_factory=list)
    root_causes: List[str] = field(default_factory=list)
    explanation: str = ""  # Clear explanation for the client
    recommended_videos: List[RecommendedVideo] = field(default_factory=list)
    client_understood: bool = False  # Must be True before proceeding


# === Treatment Proposal ===

@dataclass
class Intervention:
    """A specific treatment intervention."""
    id: str
    name: str
    target_node_id: str  # Which cognitive node this addresses
    description: str
    frequency: str  # daily, weekly, as_needed
    duration_weeks: int = 4
    progress: float = 0.0


@dataclass
class RewiringStep:
    """A single step in a Cognitive Rewiring Map."""
    id: str
    name: str
    description: str
    neuroscience_rationale: str = ""
    completed: bool = False
    completed_at: str = ""

@dataclass
class CognitiveRewiringMap:
    """
    Patent Pending Cognitive Rewiring Maps.
    Maps current wiring to target wiring with specific steps.
    """
    current_wiring: str  # Description of current cognitive pattern
    target_wiring: str   # Desired cognitive pattern
    rewiring_steps: List[RewiringStep] = field(default_factory=list)
    progress: float = 0.0

    def update_progress(self) -> None:
        """Recalculate progress based on completed steps."""
        if not self.rewiring_steps:
            self.progress = 0.0
            return
        completed = [s for s in self.rewiring_steps if s.completed]
        self.progress = len(completed) / len(self.rewiring_steps)


@dataclass
class TreatmentProposal:
    """
    Personalized treatment plan grounded in diagnosis.
    Only presented after client understands diagnosis.
    """
    interventions: List[Intervention] = field(default_factory=list)
    rewiring_map: Optional[CognitiveRewiringMap] = None
    estimated_duration_weeks: int = 12
    client_accepted: bool = False


# === Monitoring ===

@dataclass
class Assessment:
    """A single progress assessment."""
    date: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    map_delta: str = ""  # Changes in cognitive map
    notes: str = ""


@dataclass
class MonitoringRecord:
    """Longitudinal progress tracking."""
    assessments: List[Assessment] = field(default_factory=list)
    map_evolution: List[Dict[str, Any]] = field(default_factory=list)


# === Main Client Profile ===

@dataclass
class KerriClientProfile:
    """
    Complete client profile implementing the Kerrio journey.
    This is the digital cognitive clinic's patient record.
    """
    user_id: str
    stage: JourneyStage = JourneyStage.REGISTRATION
    registered_at: str = ""

    is_validated_guest: bool = False

    # Separated as per Kerrio requirements
    client_history: ClientHistory = field(default_factory=ClientHistory)
    clinician_notes: ClinicianNotes = field(default_factory=ClinicianNotes)

    # Built during diagnosis
    cognitive_wiring_map: CognitiveWiringMap = field(default_factory=CognitiveWiringMap)
    diagnosis: Diagnosis = field(default_factory=Diagnosis)

    # Treatment
    treatment_proposal: TreatmentProposal = field(default_factory=TreatmentProposal)

    # Monitoring
    monitoring: MonitoringRecord = field(default_factory=MonitoringRecord)

    # Session tracking
    conversation_history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if not self.registered_at:
            self.registered_at = datetime.now(timezone.utc).isoformat()

    def validate_invite_code(self, code: str) -> bool:
        """Validate the user's invite code."""
        # Demo logic: accept any code starting with "KERRIO-"
        if code.startswith("KERRIO-") or code == "DEMO-VIP":
            self.is_validated_guest = True
            return True
        return False

    def can_advance_stage(self) -> bool:
        """Check if client can advance to next stage based on requirements."""
        if self.stage == JourneyStage.REGISTRATION:
            # Must be validated guest to proceed
            return self.is_validated_guest

        elif self.stage == JourneyStage.HISTORY_COLLECTION:
            # Need substantial history across pillars
            h = self.client_history
            has_history = bool(h.history.life_events or h.history.formative_experiences)
            has_psych = bool(h.psychology_philosophy.beliefs or h.psychology_philosophy.values)
            return has_history and has_psych

        elif self.stage == JourneyStage.CONSULTATION:
            # Need clinician insights
            return len(self.clinician_notes.session_insights) >= 2

        elif self.stage == JourneyStage.DIAGNOSIS:
            # Client must understand diagnosis before proposal
            return self.diagnosis.client_understood

        elif self.stage == JourneyStage.PROPOSAL:
            # Client must accept treatment
            return self.treatment_proposal.client_accepted

        elif self.stage == JourneyStage.TREATMENT:
            # Must complete at least one rewiring step or reach 50% progress
            if self.treatment_proposal.rewiring_map:
                return self.treatment_proposal.rewiring_map.progress >= 0.5
            return True

        return False

    def advance_stage(self) -> bool:
        """Attempt to advance to next stage. Returns True if successful."""
        if not self.can_advance_stage():
            return False
        next_stage = JourneyStage.next_stage(self.stage)
        if next_stage:
            self.stage = next_stage
            return True
        return False


# === Journey Manager ===

class KerriJourneyManager:
    """
    Manages the Kerrio client journey lifecycle.
    Handles persistence, stage transitions, and clinical logic.
    """

    def __init__(self, storage_dir: str = "runs/kerrio_profiles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: Dict[str, KerriClientProfile] = {}

    def get_profile(self, user_id: str) -> Optional[KerriClientProfile]:
        """Return existing profile (from memory or disk), or None if unknown."""
        if user_id in self._profiles:
            return self._profiles[user_id]

        profile_path = self.storage_dir / f"{user_id}.json"
        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                profile = self._deserialize_profile(data)
                self._profiles[user_id] = profile
                return profile
            except Exception as e:
                print(f"[warn] Failed to load profile {user_id}: {e}")
        return None

    def get_or_create_profile(self, user_id: str) -> KerriClientProfile:
        """Get existing profile or create new one."""
        existing = self.get_profile(user_id)
        if existing is not None:
            return existing

        profile = KerriClientProfile(user_id=user_id)
        self._profiles[user_id] = profile
        self.save_profile(user_id)
        return profile

    def save_profile(self, user_id: str) -> None:
        """Persist profile to disk."""
        if user_id not in self._profiles:
            return
        profile = self._profiles[user_id]
        profile_path = self.storage_dir / f"{user_id}.json"

        data = self._serialize_profile(profile)
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _serialize_profile(self, profile: KerriClientProfile) -> Dict[str, Any]:
        """Convert profile to JSON-serializable dict."""
        return {
            "user_id": profile.user_id,
            "stage": profile.stage.value,
            "registered_at": profile.registered_at,
            "client_history": asdict(profile.client_history),
            "clinician_notes": asdict(profile.clinician_notes),
            "cognitive_wiring_map": asdict(profile.cognitive_wiring_map),
            "diagnosis": asdict(profile.diagnosis),
            "treatment_proposal": asdict(profile.treatment_proposal),
            "monitoring": asdict(profile.monitoring),
            "conversation_history": profile.conversation_history,
        }
    def _deserialize_profile(self, data: Dict[str, Any]) -> KerriClientProfile:
        """Reconstruct profile from dict."""
        profile = KerriClientProfile(
            user_id=data["user_id"],
            stage=JourneyStage(data.get("stage", "registration")),
            registered_at=data.get("registered_at", ""),
        )
        profile.conversation_history = data.get("conversation_history", [])

        # Reconstruct nested dataclasses
        if "client_history" in data:
             # Manual reconstruction to handle nested structures properly
             ch = data["client_history"]
             profile.client_history = ClientHistory(
                 history=HistoryPillar(**ch.get("history", {})),
                 psychology_philosophy=PsychologyPhilosophyPillar(**ch.get("psychology_philosophy", {})),
                 physiology=PhysiologyPillar(**ch.get("physiology", {})),
                 raw_narrative_turns=ch.get("raw_narrative_turns", [])
             )

        if "clinician_notes" in data:
            cn = data["clinician_notes"]
            profile.clinician_notes = ClinicianNotes(
                session_insights=[SessionInsight(**i) for i in cn.get("session_insights", [])],
                emerging_patterns=cn.get("emerging_patterns", []),
                diagnostic_hypotheses=cn.get("diagnostic_hypotheses", []),
                blind_spots_identified=cn.get("blind_spots_identified", []),
                ambiguities_to_clarify=cn.get("ambiguities_to_clarify", [])
            )

        if "cognitive_wiring_map" in data:
            cm = data["cognitive_wiring_map"]
            profile.cognitive_wiring_map = CognitiveWiringMap(
                nodes=[CognitiveNode(**n) for n in cm.get("nodes", [])],
                edges=[CognitiveEdge(**e) for e in cm.get("edges", [])],
                summary=cm.get("summary", ""),
                last_updated=cm.get("last_updated", "")
            )

        if "diagnosis" in data:
             d = data["diagnosis"]
             profile.diagnosis = Diagnosis(
                 core_constraints=d.get("core_constraints", []),
                 bottlenecks=d.get("bottlenecks", []),
                 root_causes=d.get("root_causes", []),
                 explanation=d.get("explanation", ""),
                 recommended_videos=[RecommendedVideo(**v) for v in d.get("recommended_videos", [])],
                 client_understood=d.get("client_understood", False)
             )

        if "treatment_proposal" in data:
            tp = data["treatment_proposal"]
            rm = None
            if tp.get("rewiring_map"):
                rm_data = tp["rewiring_map"]
                rm = CognitiveRewiringMap(
                    current_wiring=rm_data.get("current_wiring", ""),
                    target_wiring=rm_data.get("target_wiring", ""),
                    rewiring_steps=[RewiringStep(**s) for s in rm_data.get("rewiring_steps", [])],
                    progress=rm_data.get("progress", 0.0)
                )
            profile.treatment_proposal = TreatmentProposal(
                interventions=[Intervention(**i) for i in tp.get("interventions", [])],
                rewiring_map=rm,
                estimated_duration_weeks=tp.get("estimated_duration_weeks", 12),
                client_accepted=tp.get("client_accepted", False)
            )

        return profile

    def get_stage_system_prompt(self, profile: KerriClientProfile) -> str:
        """
        Get the appropriate system prompt for the current journey stage.
        This ensures the AI behaves correctly for each phase.
        """
        stage = profile.stage

        base = (
            "You are Kerrio, a digital twin of Dr Kerry Spackman, an expert in "
            "neuroscience-based cognitive optimization for elite-performance clients. "
            "You are NOT a chatbot or motivation app. You are a digital cognitive clinic. "
            "Your purpose is permanent human optimization through diagnosis and cognitive rewiring. "
        )

        if stage == JourneyStage.REGISTRATION:
            return base + (
                "\n\nCurrent Stage: REGISTRATION\n"
                "Welcome the client. Explain that Kerrio is different from typical coaching apps - "
                "it's a structured clinical process based on the Mayo Clinic model. "
                "Set expectations: this is a journey of diagnosis before treatment. "
                "Ask if they're ready to begin with detailed history collection."
            )

        elif stage == JourneyStage.HISTORY_COLLECTION:
            return base + (
                "\n\nCurrent Stage: HISTORY COLLECTION (Three Pillars)\n"
                "Gather comprehensive personal history across three pillars:\n"
                "1. HISTORY: Life events, formative experiences, recurrent patterns\n"
                "2. PSYCHOLOGY & PHILOSOPHY: Beliefs, values, meaning structures, emotional wiring\n"
                "3. PHYSIOLOGY: Sleep, stress, health, energy, physical constraints\n\n"
                "This is NOT a questionnaire. Guide a narrative-rich conversation. "
                "This raw data forms the foundation for the Cognitive Wiring Map."
            )

        elif stage == JourneyStage.CONSULTATION:
            return base + (
                "\n\nCurrent Stage: CONSULTATION (Clinician's Notes)\n"
                "Through structured interaction:\n"
                "- Clarify ambiguities in the history\n"
                "- Uncover blind spots the client may not see\n"
                "- Identify patterns that static history cannot reveal\n\n"
                "IMPORTANT: Maintain separation between Client History and Clinician Notes. "
                "When you discover new insights, explain them to the client first, "
                "then add them to the formal diagnostic model."
            )

        elif stage == JourneyStage.DIAGNOSIS:
            return base + (
                "\n\nCurrent Stage: DIAGNOSIS\n"
                "This is the most important phase. You must:\n"
                "1. Identify core constraints and bottlenecks\n"
                "2. Construct the Cognitive Wiring Map (personalized brain model)\n"
                "3. Explain WHY the problem exists, not just what it looks like\n\n"
                "Direct the client to relevant educational content that explains:\n"
                "- The relevant neuroscience\n"
                "- The mechanisms maintaining their patterns\n"
                "- Why specific interventions will work\n\n"
                "Understanding is a PREREQUISITE for permanent change. "
                "Do NOT proceed until the client demonstrates understanding of their diagnosis."
            )

        elif stage == JourneyStage.PROPOSAL:
            return base + (
                "\n\nCurrent Stage: TREATMENT PROPOSAL\n"
                "Only now that the client understands their diagnosis, propose treatment.\n"
                "The plan must be:\n"
                "- Personalized to their Cognitive Wiring Map\n"
                "- Structured, sequenced, and deliberate\n"
                "- Include Cognitive Rewiring Maps (specific pattern transformations)\n\n"
                "There is NO generic coaching advice. Every intervention targets "
                "specific nodes in their cognitive map."
            )

        elif stage == JourneyStage.TREATMENT:
            return base + (
                "\n\nCurrent Stage: TREATMENT\n"
                "Guide the client through neuroscience-based interventions:\n"
                "- Cognitive Rewiring exercises\n"
                "- Structured cognitive and behavioral recalibration\n"
                "- Targeted psychological rewiring\n\n"
                "The goal is STRUCTURAL CHANGE, not motivational compliance. "
                "Track progress against the original Cognitive Wiring Map."
            )

        elif stage == JourneyStage.MONITORING:
            return base + (
                "\n\nCurrent Stage: MONITORING & REASSESSMENT\n"
                "Monitor longitudinal progress against:\n"
                "- The original Cognitive Wiring Map\n"
                "- The original constraints\n"
                "- Performance and wellbeing indicators\n\n"
                "Update the map, refine interventions, and reassess progress. "
                "Close the clinical loop: Diagnosis -> Treatment -> Reassessment"
            )

        return base

    def add_turn_and_extract(
        self,
        profile: KerriClientProfile,
        user_msg: str,
        coach_reply: str
    ) -> None:
        """
        Add a conversation turn and extract relevant information
        based on the current journey stage.
        """
        turn = {"user": user_msg, "coach": coach_reply}
        profile.conversation_history.append(turn)
        turn_id = len(profile.conversation_history)

        # Stage-specific extraction
        if profile.stage == JourneyStage.HISTORY_COLLECTION:
            self._extract_history_data(profile, user_msg, turn_id)
        elif profile.stage == JourneyStage.CONSULTATION:
            self._extract_clinician_insights(profile, user_msg, coach_reply, turn_id)
        elif profile.stage in (JourneyStage.DIAGNOSIS, JourneyStage.TREATMENT):
            self._update_cognitive_map(profile, user_msg, turn_id)

        self.save_profile(profile.user_id)

    def _extract_history_data(
        self, profile: KerriClientProfile, user_msg: str, turn_id: int
    ) -> None:
        """Extract history pillar data from user message."""
        lower = user_msg.lower()
        h = profile.client_history

        # Store raw narrative
        h.raw_narrative_turns.append({"turn_id": turn_id, "text": user_msg})

        # Simple pattern matching for history extraction
        # Life events
        life_patterns = [
            r"when i was (?:a child|young|growing up)[^.]*",
            r"my (?:father|mother|parents)[^.]*",
            r"i (?:grew up|was raised)[^.]*",
        ]
        for pat in life_patterns:
            m = re.search(pat, lower)
            if m:
                h.history.life_events.append(m.group(0).strip())

        # Values
        value_patterns = [
            r"(?:i value|important to me|what matters)[^.]*",
            r"(?:family|success|health|freedom|security)[^.]*is important",
        ]
        for pat in value_patterns:
            m = re.search(pat, lower)
            if m:
                h.psychology_philosophy.values.append(m.group(0).strip())

        # Beliefs
        belief_patterns = [
            r"i (?:believe|think|feel) that[^.]*",
            r"i'?ve always (?:thought|believed)[^.]*",
        ]
        for pat in belief_patterns:
            m = re.search(pat, lower)
            if m:
                h.psychology_philosophy.beliefs.append(m.group(0).strip())

        # Physiology
        if any(w in lower for w in ["sleep", "tired", "exhausted", "insomnia"]):
            if "poor" in lower or "bad" in lower or "trouble" in lower:
                h.physiology.sleep_quality = "poor"
            elif "good" in lower or "well" in lower:
                h.physiology.sleep_quality = "good"

        if any(w in lower for w in ["stress", "anxious", "overwhelmed"]):
            h.physiology.stress_level = "high"

    def _extract_clinician_insights(
        self,
        profile: KerriClientProfile,
        user_msg: str,
        coach_reply: str,
        turn_id: int
    ) -> None:
        """Extract clinician observations during consultation."""
        notes = profile.clinician_notes
        lower = user_msg.lower()

        # Detect resistance
        resistance_signals = ["i don't think", "that's not", "but", "however", "i disagree"]
        if any(sig in lower for sig in resistance_signals):
            insight = SessionInsight(
                turn_id=turn_id,
                observation=f"Client showed resistance: '{user_msg[:100]}...'",
                category="resistance"
            )
            notes.session_insights.append(insight)

        # Detect blind spots (things client avoids or deflects)
        deflection_signals = ["anyway", "let's move on", "that's not important", "i don't want to"]
        if any(sig in lower for sig in deflection_signals):
            insight = SessionInsight(
                turn_id=turn_id,
                observation=f"Possible blind spot - client deflected: '{user_msg[:100]}...'",
                category="blind_spot"
            )
            notes.session_insights.append(insight)

    def _update_cognitive_map(
        self, profile: KerriClientProfile, user_msg: str, turn_id: int
    ) -> None:
        """Update cognitive wiring map based on new information."""
        cmap = profile.cognitive_wiring_map
        lower = user_msg.lower()

        # Pattern detection
        pattern_signals = ["i always", "every time", "i keep", "i tend to"]
        for sig in pattern_signals:
            if sig in lower:
                node = CognitiveNode(
                    id=f"rp_{len(cmap.nodes)+1}",
                    label=f"Recurrent pattern: {user_msg[:80]}",
                    type="recurrent_pattern",
                    evidence_turns=[turn_id],
                    confidence=0.6
                )
                cmap.add_node(node)
                break

    def submit_monitoring_feedback(
        self,
        profile: KerriClientProfile,
        metrics: Dict[str, Any],
        notes: str
    ) -> bool:
        """
        Submit longitudinal monitoring data.
        If significant issues are detected, may trigger a return to Diagnosis stage.
        """
        assessment = Assessment(
            date=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            notes=notes
        )
        profile.monitoring.assessments.append(assessment)

        # Closed-loop logic: check if we need to return to diagnosis
        # If stress is severe or core constraints are re-activated
        should_rediagnose = False
        if metrics.get("stress_level") == "severe":
            should_rediagnose = True

        if should_rediagnose:
            # Loop back to Consultation or Diagnosis
            profile.stage = JourneyStage.CONSULTATION
            profile.clinician_notes.ambiguities_to_clarify.append(
                f"Monitoring revealed concern: {notes}"
            )

        self.save_profile(profile.user_id)
        return should_rediagnose


# === Stage-specific prompts for history collection ===

HISTORY_COLLECTION_PROMPTS = {
    "history_pillar": [
        "Tell me about your early life. What was your childhood like?",
        "What significant life events have shaped who you are today?",
        "Do you notice any patterns that keep repeating in your life?",
        "What formative experiences do you think influenced your current situation?",
    ],
    "psychology_philosophy_pillar": [
        "What do you believe about yourself that feels deeply true?",
        "What values are most important to you?",
        "What gives your life meaning?",
        "How do you typically respond when you're under pressure emotionally?",
    ],
    "physiology_pillar": [
        "How would you describe your sleep patterns?",
        "How do you experience stress in your body?",
        "Are there any physical health considerations I should know about?",
        "What does your energy typically look like throughout the day?",
    ],
}


# === Export for use in app_demo.py ===

__all__ = [
    "JourneyStage",
    "KerriClientProfile",
    "KerriJourneyManager",
    "CognitiveWiringMap",
    "CognitiveNode",
    "CognitiveEdge",
    "NODE_TYPES",
    "EDGE_TYPES",
    "HISTORY_COLLECTION_PROMPTS",
]


# === LLM-Powered Diagnostic Tools ===

# Prompts for LLM-based extraction
LLM_HISTORY_EXTRACTION_PROMPT = """
You are a clinical psychologist assistant helping extract structured information from a client conversation.

Analyze the following conversation and extract information into the THREE PILLARS:

1. HISTORY PILLAR (Life events, formative experiences, recurrent patterns)
2. PSYCHOLOGY & PHILOSOPHY PILLAR (Beliefs, values, meaning structures, emotional wiring, core assumptions)
3. PHYSIOLOGY PILLAR (Sleep, stress, health, energy, physical constraints)

Conversation:
{conversation}

Return a JSON object with this exact structure:
{{
  "history": {{
    "life_events": ["event1", "event2"],
    "formative_experiences": ["exp1", "exp2"],
    "recurrent_patterns": ["pattern1", "pattern2"],
    "background_summary": "Brief summary of background"
  }},
  "psychology_philosophy": {{
    "beliefs": ["belief1", "belief2"],
    "values": ["value1", "value2"],
    "meaning_structures": ["structure1"],
    "emotional_wiring": ["pattern1"],
    "core_assumptions": ["assumption1"]
  }},
  "physiology": {{
    "sleep_quality": "good/poor/variable",
    "stress_level": "low/moderate/high/severe",
    "health_conditions": ["condition1"],
    "energy_patterns": ["pattern1"],
    "physical_constraints": ["constraint1"]
  }}
}}

Only include items that are explicitly mentioned or strongly implied in the conversation.
Return ONLY valid JSON, no additional text.
"""

LLM_DIAGNOSIS_PROMPT = """
You are Dr Kerry Spackman's digital twin, an expert in neuroscience-based cognitive optimization.

Based on the client's history across three pillars, generate a comprehensive diagnosis.

CLIENT HISTORY:
{history_summary}

COGNITIVE WIRING MAP:
{cognitive_map}

Generate a diagnosis that:
1. Identifies CORE CONSTRAINTS (fundamental limitations blocking progress)
2. Identifies BOTTLENECKS (specific points where progress gets stuck)
3. Explains ROOT CAUSES (WHY the problem exists, not just what it looks like)
4. Provides a clear EXPLANATION for the client (they must understand before treatment)

Return JSON with this structure:
{{
  "core_constraints": [
    {{"id": "cc1", "description": "...", "evidence": "...", "neuroscience_basis": "..."}}
  ],
  "bottlenecks": [
    {{"id": "bn1", "description": "...", "upstream_cause": "...", "downstream_effect": "..."}}
  ],
  "root_causes": [
    {{"id": "rc1", "description": "...", "mechanism": "..."}}
  ],
  "explanation": "Clear, client-friendly explanation of the diagnosis",
  "recommended_videos": [
    {{"topic": "...", "relevance": "..."}}
  ]
}}

Return ONLY valid JSON, no additional text.
"""

LLM_REWIRING_MAP_PROMPT = """
You are designing a Cognitive Rewiring Map - a personalized transformation plan.

CURRENT DIAGNOSIS:
{diagnosis}

TARGET STATE:
{target_state}

Design a Cognitive Rewiring Map that shows:
1. CURRENT WIRING: The existing cognitive patterns that maintain the problem
2. TARGET WIRING: The desired cognitive patterns
3. REWIRING STEPS: Specific, actionable steps to transform from current to target

Return JSON:
{{
  "current_wiring": {{
    "pattern_description": "...",
    "maintaining_factors": ["factor1", "factor2"],
    "automatic_thoughts": ["thought1", "thought2"],
    "behavioral_loops": ["loop1", "loop2"]
  }},
  "target_wiring": {{
    "pattern_description": "...",
    "new_beliefs": ["belief1", "belief2"],
    "new_behaviors": ["behavior1", "behavior2"],
    "expected_outcomes": ["outcome1", "outcome2"]
  }},
  "rewiring_steps": [
    {{
      "step_number": 1,
      "name": "...",
      "description": "...",
      "neuroscience_rationale": "...",
      "exercises": ["exercise1", "exercise2"],
      "duration": "1-2 weeks"
    }}
  ],
  "monitoring_metrics": ["metric1", "metric2"]
}}

Return ONLY valid JSON, no additional text.
"""


# === Video Recommendation Database ===
# Pre-recorded videos by Dr Spackman (simulated for demo)

VIDEO_DATABASE = {
    "core_beliefs": {
        "id": "v001",
        "title": "Understanding Core Beliefs and How They Shape Your Reality",
        "topics": ["beliefs", "assumptions", "self-worth", "identity"],
        "duration_minutes": 15,
        "description": "How your unconscious beliefs about yourself were formed and how they drive behavior."
    },
    "stress_response": {
        "id": "v002",
        "title": "The Neuroscience of Stress and Performance",
        "topics": ["stress", "anxiety", "cortisol", "performance"],
        "duration_minutes": 12,
        "description": "Understanding how stress affects your brain and what you can do about it."
    },
    "habit_formation": {
        "id": "v003",
        "title": "The Science of Habit Formation and Change",
        "topics": ["habits", "behavior", "patterns", "change"],
        "duration_minutes": 18,
        "description": "How neural pathways create habits and the science of rewiring them."
    },
    "emotional_regulation": {
        "id": "v004",
        "title": "Emotional Regulation: The Prefrontal Cortex Connection",
        "topics": ["emotions", "regulation", "anxiety", "overwhelm"],
        "duration_minutes": 14,
        "description": "Understanding emotional reactions and building regulation capacity."
    },
    "motivation_science": {
        "id": "v005",
        "title": "Intrinsic vs Extrinsic Motivation: Why Willpower Fails",
        "topics": ["motivation", "willpower", "intrinsic", "goals"],
        "duration_minutes": 16,
        "description": "Why traditional motivation strategies fail and what actually works."
    },
    "cognitive_distortions": {
        "id": "v006",
        "title": "Cognitive Distortions: How Your Brain Tricks You",
        "topics": ["thinking", "distortions", "perception", "reality"],
        "duration_minutes": 13,
        "description": "Common thinking errors and how to recognize them in yourself."
    },
    "neuroplasticity": {
        "id": "v007",
        "title": "Neuroplasticity: Your Brain Can Change at Any Age",
        "topics": ["change", "neuroplasticity", "learning", "growth"],
        "duration_minutes": 11,
        "description": "The science of brain change and why permanent transformation is possible."
    },
    "sleep_cognition": {
        "id": "v008",
        "title": "Sleep and Cognitive Performance",
        "topics": ["sleep", "cognition", "performance", "health"],
        "duration_minutes": 10,
        "description": "How sleep affects your brain function and decision-making."
    },
    "self_sabotage": {
        "id": "v009",
        "title": "Understanding Self-Sabotage: The Hidden Logic",
        "topics": ["sabotage", "patterns", "protection", "fear"],
        "duration_minutes": 17,
        "description": "Why we undermine ourselves and the unconscious logic behind it."
    },
    "perfectionism": {
        "id": "v010",
        "title": "Perfectionism: The Hidden Constraint",
        "topics": ["perfectionism", "achievement", "fear", "standards"],
        "duration_minutes": 14,
        "description": "How perfectionism limits performance and creates anxiety."
    },
}


def recommend_videos_for_diagnosis(diagnosis: Diagnosis) -> List[RecommendedVideo]:
    """
    Recommend relevant educational videos based on the diagnosis.
    Maps diagnostic findings to relevant neuroscience education.
    """
    recommendations = []
    seen_ids = set()

    # Analyze diagnosis content for keyword matching
    all_text = " ".join([
        " ".join(diagnosis.core_constraints),
        " ".join(diagnosis.bottlenecks),
        " ".join(diagnosis.root_causes),
        diagnosis.explanation
    ]).lower()

    # Score each video by relevance
    video_scores = []
    for key, video in VIDEO_DATABASE.items():
        score = 0
        matched_topics = []
        for topic in video["topics"]:
            if topic in all_text:
                score += 2
                matched_topics.append(topic)
            # Also check for related words
            related = {
                "beliefs": ["assume", "think", "belief"],
                "stress": ["anxious", "overwhelm", "pressure"],
                "habits": ["pattern", "behavior", "routine"],
                "emotions": ["feel", "emotion", "mood"],
                "motivation": ["unmotivated", "lazy", "procrastin"],
                "sleep": ["tired", "exhausted", "fatigue"],
                "perfectionism": ["perfect", "standard", "enough"],
            }
            for word in related.get(topic, []):
                if word in all_text:
                    score += 1

        if score > 0:
            video_scores.append((score, video, matched_topics))

    # Sort by score and take top recommendations
    video_scores.sort(key=lambda x: x[0], reverse=True)

    for score, video, topics in video_scores[:3]:
        if video["id"] not in seen_ids:
            rec = RecommendedVideo(
                video_id=video["id"],
                title=video["title"],
                relevance=f"Relevant to: {', '.join(topics)}",
                url=f"/videos/{video['id']}"
            )
            recommendations.append(rec)
            seen_ids.add(video["id"])

    return recommendations


class DiagnosticEngine:
    """
    Engine for generating structured diagnoses from client history.
    This is the core of the Mayo Clinic model - accurate diagnosis before intervention.
    """

    def __init__(self):
        self.diagnosis_templates = {
            "perfectionism": {
                "constraint": "Perfectionism as a protection mechanism",
                "mechanism": "Fear of judgment leads to impossibly high standards, which paradoxically reduces output",
                "bottleneck": "Unable to start or complete tasks due to fear of imperfection",
            },
            "self_worth": {
                "constraint": "Self-worth tied to external achievement",
                "mechanism": "Core belief that value comes from accomplishment creates chronic inadequacy",
                "bottleneck": "Success never feels sufficient; rest feels undeserved",
            },
            "avoidance": {
                "constraint": "Avoidance as primary coping strategy",
                "mechanism": "Short-term anxiety relief reinforces long-term avoidance patterns",
                "bottleneck": "Important tasks remain undone, creating more anxiety",
            },
            "burnout": {
                "constraint": "Chronic overextension without recovery",
                "mechanism": "Inability to set boundaries depletes resources faster than they regenerate",
                "bottleneck": "Physical and mental exhaustion prevents effective action",
            },
        }

    def generate_diagnosis_from_history(
        self,
        client_history: ClientHistory,
        clinician_notes: ClinicianNotes,
        cognitive_map: CognitiveWiringMap
    ) -> Diagnosis:
        """
        Generate a structured diagnosis based on collected data.
        This is a heuristic implementation - can be enhanced with LLM.
        """
        diagnosis = Diagnosis()

        # Analyze patterns in history
        patterns = client_history.history.recurrent_patterns
        beliefs = client_history.psychology_philosophy.beliefs
        assumptions = client_history.psychology_philosophy.core_assumptions
        stress = client_history.physiology.stress_level

        # Identify core constraints
        if any("perfect" in p.lower() for p in patterns + beliefs):
            diagnosis.core_constraints.append(
                "Perfectionism creating paralysis and chronic dissatisfaction"
            )
            diagnosis.root_causes.append(
                "Learned belief that self-worth depends on flawless performance"
            )

        if any("enough" in b.lower() or "worthy" in b.lower() for b in beliefs + assumptions):
            diagnosis.core_constraints.append(
                "Core belief of inadequacy driving compensatory behaviors"
            )
            diagnosis.root_causes.append(
                "Early experiences created a schema of conditional self-worth"
            )

        if stress in ["high", "severe"]:
            diagnosis.bottlenecks.append(
                "Chronic stress depleting cognitive resources needed for change"
            )

        if any("avoid" in p.lower() for p in patterns):
            diagnosis.core_constraints.append(
                "Avoidance pattern preventing exposure and growth"
            )
            diagnosis.root_causes.append(
                "Avoidance provides short-term relief but maintains the underlying fear"
            )

        # Generate explanation
        if diagnosis.core_constraints:
            diagnosis.explanation = self._generate_explanation(diagnosis)
        else:
            diagnosis.explanation = (
                "Further history collection needed. Continue exploring the three pillars: "
                "History, Psychology/Philosophy, and Physiology."
            )

        # Recommend relevant videos
        diagnosis.recommended_videos = recommend_videos_for_diagnosis(diagnosis)

        return diagnosis

    def _generate_explanation(self, diagnosis: Diagnosis) -> str:
        """Generate a clear, client-friendly explanation of the diagnosis."""
        parts = [
            "Based on what you've shared, here's what I'm seeing:\n"
        ]

        if diagnosis.core_constraints:
            parts.append("\n**Core Constraints:**")
            for i, constraint in enumerate(diagnosis.core_constraints, 1):
                parts.append(f"\n{i}. {constraint}")

        if diagnosis.root_causes:
            parts.append("\n\n**Why This Happens:**")
            for cause in diagnosis.root_causes:
                parts.append(f"\n- {cause}")

        if diagnosis.bottlenecks:
            parts.append("\n\n**Where You Get Stuck:**")
            for bottleneck in diagnosis.bottlenecks:
                parts.append(f"\n- {bottleneck}")

        parts.append(
            "\n\nUnderstanding WHY these patterns exist is the first step to permanent change. "
            "The goal is not to fight against yourself, but to rewire the underlying patterns."
        )

        return "".join(parts)


class CognitiveRewiringEngine:
    """
    Engine for generating Cognitive Rewiring Maps.
    Patent Pending - transforms current wiring to target wiring.
    """

    def generate_rewiring_map(
        self,
        diagnosis: Diagnosis,
        cognitive_map: CognitiveWiringMap,
        target_goals: List[str]
    ) -> CognitiveRewiringMap:
        """
        Generate a personalized Cognitive Rewiring Map based on diagnosis.
        """
        # Identify current wiring pattern
        current_patterns = []
        for constraint in diagnosis.core_constraints:
            current_patterns.append(f"Pattern: {constraint}")

        current_wiring = "Current cognitive wiring: " + "; ".join(current_patterns) if current_patterns else "Pattern analysis in progress"

        # Define target wiring
        target_patterns = []
        for goal in target_goals[:3]:
            target_patterns.append(f"Aligned with: {goal}")
        target_wiring = "Target state: " + "; ".join(target_patterns) if target_patterns else "Define desired outcomes"

        # Generate rewiring steps
        steps = self._generate_rewiring_steps_v2(diagnosis, target_goals)

        return CognitiveRewiringMap(
            current_wiring=current_wiring,
            target_wiring=target_wiring,
            rewiring_steps=steps,
            progress=0.0
        )

    def _generate_rewiring_steps_v2(
        self,
        diagnosis: Diagnosis,
        target_goals: List[str]
    ) -> List[RewiringStep]:
        """Generate interactive RewiringStep objects based on diagnosis."""
        raw_steps = []

        # Base steps
        raw_steps.append({
            "name": "Awareness Training",
            "description": "Notice when the old pattern activates without trying to change it.",
            "rationale": "Breaking automatic behavior starts with conscious observation."
        })

        # Pattern-specific steps
        if any("perfectionism" in c.lower() for c in diagnosis.core_constraints):
            raw_steps.append({
                "name": "The 80% Experiment",
                "description": "Complete one task at 80% and observe the outcome.",
                "rationale": "Testing the belief that anything less than 100% is failure."
            })
            raw_steps.append({
                "name": "Self-Worth Decoupling",
                "description": "Practice 5 minutes of self-compassion meditation daily.",
                "rationale": "Neuroplasticity requires repeating new emotional associations."
            })

        if any("avoidance" in c.lower() for c in diagnosis.core_constraints):
            raw_steps.append({
                "name": "Micro-Exposure",
                "description": "Take the smallest possible action (2 mins) toward the avoided task.",
                "rationale": "Overcoming the amygdala's threat response through tiny wins."
            })

        # Convert to objects
        steps = []
        for i, s in enumerate(raw_steps):
            steps.append(RewiringStep(
                id=f"step_{i+1}",
                name=s["name"],
                description=s["description"],
                neuroscience_rationale=s["rationale"]
            ))

        return steps

    def _generate_rewiring_steps(
        self,
        diagnosis: Diagnosis,
        target_goals: List[str]
    ) -> List[str]:
        """Generate specific rewiring steps based on diagnosis."""
        steps = []

        # Base steps for any rewiring
        steps.append(
            "Step 1: Awareness - Notice when the old pattern activates without trying to change it"
        )

        # Pattern-specific steps
        if any("perfectionism" in c.lower() for c in diagnosis.core_constraints):
            steps.extend([
                "Step 2: Experiment with 'good enough' - Complete one task at 80% and observe the outcome",
                "Step 3: Separate self-worth from output - Practice self-compassion exercises",
                "Step 4: Reframe mistakes as data - When errors occur, extract learning without self-criticism",
            ])

        if any("avoidance" in c.lower() for c in diagnosis.core_constraints):
            steps.extend([
                "Step 2: Micro-exposure - Take the smallest possible action toward the avoided task",
                "Step 3: Discomfort tolerance - Sit with anxiety for 2 minutes before deciding to avoid",
                "Step 4: Success logging - Record each time you chose action over avoidance",
            ])

        if any("stress" in c.lower() or "overwhelm" in c.lower() for c in diagnosis.bottlenecks):
            steps.extend([
                "Step 2: Nervous system regulation - Practice 4-7-8 breathing twice daily",
                "Step 3: Boundary setting - Identify one commitment to reduce or remove",
                "Step 4: Recovery scheduling - Block protected time for genuine rest",
            ])

        # Default steps if nothing specific matched
        if len(steps) == 1:
            steps.extend([
                "Step 2: Pattern interruption - When you notice the old pattern, pause for 3 breaths",
                "Step 3: Alternative response - Practice the new response in low-stakes situations",
                "Step 4: Reinforcement - Acknowledge progress, no matter how small",
            ])

        steps.append(
            f"Step {len(steps)+1}: Integration - Practice new patterns until they become automatic"
        )

        return steps


# === Initialize engines ===
diagnostic_engine = DiagnosticEngine()
rewiring_engine = CognitiveRewiringEngine()


if __name__ == "__main__":
    # Quick test
    manager = KerriJourneyManager()
    profile = manager.get_or_create_profile("test_user")
    print(f"Stage: {profile.stage.value}")
    print(f"System prompt:\n{manager.get_stage_system_prompt(profile)[:500]}...")

    # Test diagnostic engine
    print("\n--- Testing Diagnostic Engine ---")
    profile.client_history.psychology_philosophy.beliefs = [
        "I'm not good enough",
        "I must be perfect to be accepted"
    ]
    profile.client_history.history.recurrent_patterns = [
        "I always procrastinate on important tasks",
        "I avoid situations where I might fail"
    ]
    profile.client_history.physiology.stress_level = "high"

    diagnosis = diagnostic_engine.generate_diagnosis_from_history(
        profile.client_history,
        profile.clinician_notes,
        profile.cognitive_wiring_map
    )
    print(f"Core Constraints: {diagnosis.core_constraints}")
    print(f"Root Causes: {diagnosis.root_causes}")
    print(f"Explanation:\n{diagnosis.explanation}")
    print(f"Recommended Videos: {[v.title for v in diagnosis.recommended_videos]}")
