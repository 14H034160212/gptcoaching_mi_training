#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app_demo.py — FastAPI demo for MI-style chat with short-term memory + per-user logging + cognitive maps.

Run:
  export MODEL_PATH=/abs/path/to/your/merged-model  # or HF repo id (public or private)
  # optional: export CLASSIFIER_PATH=/abs/path/to/mi-classifier
  # optional: export HF_TOKEN=hf_xxx   (if MODEL_PATH is a private HF repo)
  # optional: export LOG_DIR=runs/chat_logs
  uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8080

API:
  POST /api/chat   {user_id, history?, user_msg}  ->  {reply}
  POST /api/journey/validate {user_id, invite_code} -> {success: bool}
  POST /api/score  {user_id, history?, user_msg?} ->  {labels, probs}
  POST /api/reset  {user_id} -> {ok: true}
  GET  /api/health -> model + device info
  GET  /api/map/{user_id}?max_turns=20 -> cognitive map JSON

Static:
  "/" serves web/index.html
"""
import os, json, re, torch
from typing import List, Dict, Optional
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, APIRouter, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel, EmailStr

import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from typing import List, Dict, Any
from pydantic import BaseModel

from scripts.auth import AuthStore, send_magic_link_email
from scripts.cogmap_utils import build_cognitive_map_from_session
from scripts.kerrio_journey import (
    KerriJourneyManager,
    KerriClientProfile,
    JourneyStage,
    HISTORY_COLLECTION_PROMPTS,
    diagnostic_engine,
    rewiring_engine,
    VIDEO_DATABASE,
    Diagnosis,
    CognitiveRewiringMap,
)

# Initialize Kerrio Journey Manager
journey_manager = KerriJourneyManager()

# Original MI system prompt (kept for backward compatibility)
SYSTEM_MI = (
    "You are a supportive health coach using Motivational Interviewing (MI). "
    "Be non-judgmental; use open questions, reflective listening, and affirmations; "
    "ask permission before giving advice; avoid directives. Keep it concise."
)

# Kerrio system prompt (Mayo Clinic diagnostic model)
SYSTEM_KERRIO = (
    "You are Kerrio, a digital twin of Dr Kerry Spackman. "
    "You are NOT a chatbot or motivation app. You are a digital cognitive clinic. "
    "Your purpose is permanent human optimization through diagnosis and cognitive rewiring. "
    "Follow the Mayo Clinic model: Accurate diagnosis before intervention. "
    "Understanding is a prerequisite for permanent change."
)

# Default to Kerrio mode
SYSTEM = SYSTEM_KERRIO

# ===== Env =====
MODEL_PATH = os.environ.get("MODEL_PATH", "").strip()
CLASSIFIER_PATH = os.environ.get("CLASSIFIER_PATH", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
LOG_DIR = os.environ.get("LOG_DIR", "runs/chat_logs")
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "").strip()
MAIL_FROM = os.environ.get("MAIL_FROM", "onboarding@resend.dev").strip()
APP_URL = os.environ.get("APP_URL", "https://gptcoaching-mi-training.pages.dev").strip().rstrip("/")
# World-model rerank: number of candidate replies the DPO Qwen proposes per turn
# before the planner picks the one that best evokes change talk. 1 disables it.
RERANK_K = int(os.environ.get("RERANK_K", "4"))

assert MODEL_PATH, "MODEL_PATH is empty. export MODEL_PATH=/abs/path/to/model or HF repo id"
assert RESEND_API_KEY, "RESEND_API_KEY is empty. export RESEND_API_KEY=re_xxx for magic-link email"

os.makedirs(LOG_DIR, exist_ok=True)

# Auth kwargs for private HF repos (Transformers >= 4.41 uses `token`)
hf_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

# ===== Load main model =====
print(f"[boot] MODEL_PATH={MODEL_PATH}")

# Check if MODEL_PATH is a PEFT adapter directory
is_peft = os.path.exists(os.path.join(MODEL_PATH, "adapter_config.json"))

if is_peft:
    print("[boot] Detected PEFT adapter. Loading base model first.")
    from peft import PeftModel
    with open(os.path.join(MODEL_PATH, "adapter_config.json"), "r") as f:
        import json
        adapter_cfg = json.load(f)
        base_model_path = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-3B-Instruct")
    
    # Load tokenizer from adapter path (it usually has the correct overrides)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True, **hf_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    # Load base model
    # Note: Removed device_map="auto" to prevent PEFT offloading errors on CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True, **hf_kwargs
    )
    # Load adapter
    model = PeftModel.from_pretrained(base_model, MODEL_PATH).eval()
else:
    # Standard full model load
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True, **hf_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map="auto", trust_remote_code=True, **hf_kwargs
    ).eval()

# ===== Optional classifier (for MI behavior scoring) =====
cls_tok = cls_model = None
cls_labels = None
if CLASSIFIER_PATH:
    try:
        print(f"[boot] CLASSIFIER_PATH={CLASSIFIER_PATH}")
        cls_tok = AutoTokenizer.from_pretrained(CLASSIFIER_PATH, use_fast=True, **hf_kwargs)
        cls_model = AutoModelForSequenceClassification.from_pretrained(
            CLASSIFIER_PATH, device_map="auto", **hf_kwargs
        ).eval()
        # optional label names
        try:
            with open(os.path.join(CLASSIFIER_PATH, "labels.json"), "r", encoding="utf-8") as f:
                cls_labels = json.load(f)
        except Exception:
            pass
    except Exception as e:
        print(f"[warn] failed to load classifier: {e}")

# ===== FastAPI =====
app = FastAPI(title="GPTCoach MI Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

api = APIRouter()

# ===== Auth =====
auth_store = AuthStore()

# Open demo mode: when AUTH_REQUIRED is off, anyone who opens the site can use it
# without signing in, and all activity is attributed to DEMO_USER. The magic-link
# auth system stays intact — flip AUTH_REQUIRED back on for real multi-user use.
AUTH_REQUIRED = os.environ.get("AUTH_REQUIRED", "1").lower() not in ("0", "false", "no", "off")
DEMO_USER = {
    "email": os.environ.get("DEMO_USER_EMAIL", "demo@kerrio.ai"),
    "user_id": os.environ.get("DEMO_USER_ID", "demo_user"),
}


def get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """Resolve `Authorization: Bearer <session_token>` to a user record.

    A valid session always wins. If none is present and AUTH_REQUIRED is off
    (demo mode), fall back to the shared DEMO_USER instead of rejecting.
    """
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(None, 1)[1].strip()
        user = auth_store.get_session_user(token)
        if user is not None:
            return user
    if not AUTH_REQUIRED:
        return DEMO_USER
    raise HTTPException(status_code=401, detail="Invalid or missing session")

# Server-side per-user memory (lives for the process lifetime)
SESSIONS: Dict[str, List[dict]] = defaultdict(list)

# ===== Schemas =====
class Turn(BaseModel):
    user: str
    coach: str

class ChatRequest(BaseModel):
    user_id: str = "anon"
    history: List[Turn] = []   # optional client-provided history (seed only)
    user_msg: str
    rerank: bool = True        # world-model generate-K-then-rerank (System-2 planner)

class ResetReq(BaseModel):
    user_id: str

class CogMapReq(BaseModel):
    user_id: str = "anon"

class JourneyStatusReq(BaseModel):
    user_id: str = "anon"

class AdvanceStageReq(BaseModel):
    user_id: str = "anon"

class ValidateInviteReq(BaseModel):
    user_id: str
    invite_code: str

class StepCompleteReq(BaseModel):
    user_id: str
    step_id: str

class MonitoringReq(BaseModel):
    user_id: str
    metrics: Dict[str, Any]
    notes: str

class CounterfactualReq(BaseModel):
    user_id: str = "anon"
    client_msg: str          # the client's last message (state is estimated from this)
    coach_reply: str         # the counselor reply to evaluate
    horizon: int = 3


# ===== Helpers =====
def append_jsonl(user_id: str, payload: dict):
    """Append one record to LOG_DIR/<user_id>.jsonl"""
    fp = os.path.join(LOG_DIR, f"{user_id}.jsonl")
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_messages(
    history: List[Turn],
    user_msg: str,
    map_summary: Optional[str] = None,
    kerrio_profile: Optional[KerriClientProfile] = None
):
    """
    Build chat messages; if map_summary is provided, inject it into the system prompt
    as a kind of 'Graph-of-Thoughts' memory.

    If kerrio_profile is provided, use stage-specific system prompt.
    """
    # Use Kerrio stage-specific prompt if profile is available
    if kerrio_profile:
        sys_content = journey_manager.get_stage_system_prompt(kerrio_profile)
    else:
        sys_content = SYSTEM

    if map_summary:
        sys_content += (
            "\n\n=== COGNITIVE WIRING MAP SUMMARY ===\n"
            f"{map_summary}\n"
            "Use this diagnostic context to guide the conversation. "
            "Do not repeat this summary verbatim to the client."
        )

    # Add client history context for diagnosis/treatment stages
    if kerrio_profile and kerrio_profile.stage in (
        JourneyStage.CONSULTATION,
        JourneyStage.DIAGNOSIS,
        JourneyStage.PROPOSAL,
        JourneyStage.TREATMENT,
    ):
        h = kerrio_profile.client_history
        if h.psychology_philosophy.beliefs or h.psychology_philosophy.values:
            sys_content += (
                "\n\n=== CLIENT HISTORY SUMMARY ===\n"
                f"Beliefs: {', '.join(h.psychology_philosophy.beliefs[:3]) or 'Not yet identified'}\n"
                f"Values: {', '.join(h.psychology_philosophy.values[:3]) or 'Not yet identified'}\n"
                f"Patterns: {', '.join(h.history.recurrent_patterns[:3]) or 'Not yet identified'}\n"
            )

    msgs = [{"role": "system", "content": sys_content}]
    for t in history:
        if t.user:
            msgs.append({"role": "user", "content": t.user})
        if t.coach:
            msgs.append({"role": "assistant", "content": t.coach})
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def render_prompt(
    history: List[Turn],
    user_msg: str,
    map_summary: Optional[str] = None,
    kerrio_profile: Optional[KerriClientProfile] = None
) -> str:
    msgs = build_messages(history, user_msg, map_summary=map_summary, kerrio_profile=kerrio_profile)
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True  # start a new assistant turn
    )

# ===== CogMap Extraction Helpers =====

def load_dialog_for_user(user_id: str, max_turns: int = 20):
    """Load last N turns from logs/<user_id>.jsonl"""
    path = Path(LOG_DIR) / f"{user_id}.jsonl"
    if not path.exists():
        return []
    turns = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("user") is None and rec.get("coach") is None:
                continue
            turns.append(rec)
    return turns[-max_turns:]


def turns_to_plaintext(turns):
    lines = []
    for t in turns:
        if t.get("user"):
            lines.append(f"USER: {t['user']}")
        if t.get("coach"):
            lines.append(f"COACH: {t['coach']}")
    return "\n".join(lines)


COGMAP_SYSTEM = """
You are an expert motivational interviewing coach and knowledge-mapping assistant.

Your job:
- Read the conversation between USER and COACH.
- Extract a cognitive map that summarizes the USER's goals, values, strengths, barriers, beliefs, actions, and outcomes.

Return JSON ONLY with the structure:

{
  "nodes": [
    {"id": "n1", "type": "goal|value|strength|barrier|belief|action|outcome", "label": "...", "evidence": "..."}
  ],
  "edges": [
    {"source": "n1", "target": "n2", "type": "supports|blocks|explains|leads_to|part_of"}
  ]
}
""".strip()


def build_cogmap_prompt(dialogue_text: str):
    msgs = [
        {"role": "system", "content": COGMAP_SYSTEM},
        {
            "role": "user",
            "content": (
                "Here is the conversation:\n\n"
                f"{dialogue_text}\n\n"
                "Extract the cognitive map JSON."
            ),
        },
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def extract_json_block(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON found in model output.")
    return text[start : end + 1]


def generate_cogmap_from_turns(turns, max_new_tokens=512):
    """Use the tuned Qwen model to extract cognitive maps from dialogue."""
    dialogue = turns_to_plaintext(turns)
    prompt = build_cogmap_prompt(dialogue)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    eos_id = tok.eos_token_id
    try:
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
    except Exception:
        im_end = None
    stop_ids = [i for i in (eos_id, im_end) if i is not None]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=eos_id,
            eos_token_id=stop_ids[0] if stop_ids else eos_id,
        )

    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    try:
        js = extract_json_block(text)
        data = json.loads(js)
    except Exception:
        cleaned = re.sub(r"<\|im_(start|end)\|>", " ", text)
        cleaned = cleaned.strip()
        js = extract_json_block(cleaned)
        data = json.loads(js)

    return data


def summarize_cogmap_for_prompt(cmap: dict, max_items: int = 4) -> str:
    """
    Turn the cognitive map JSON into a short text summary
    that can be injected into the system prompt (Graph-of-Thoughts style).
    """
    buckets: Dict[str, List[str]] = {
        "goal": [],
        "value": [],
        "strength": [],
        "barrier": [],
        "belief": [],
        "action": [],
        "outcome": [],
    }
    for n in cmap.get("nodes", []):
        t = (n.get("type") or "").lower()
        label = (n.get("label") or "").strip()
        if t in buckets and label:
            if len(buckets[t]) < max_items:
                buckets[t].append(label)

    lines = []
    if buckets["goal"]:
        lines.append("Goals: " + "; ".join(buckets["goal"]))
    if buckets["value"]:
        lines.append("Values/Why it matters: " + "; ".join(buckets["value"]))
    if buckets["barrier"]:
        lines.append("Barriers: " + "; ".join(buckets["barrier"]))
    if buckets["strength"]:
        lines.append("Strengths/Resources: " + "; ".join(buckets["strength"]))
    if buckets["belief"]:
        lines.append("Key beliefs: " + "; ".join(buckets["belief"]))
    if buckets["action"]:
        lines.append("Actions/Strategies mentioned: " + "; ".join(buckets["action"]))
    if buckets["outcome"]:
        lines.append("Outcomes/Feedback: " + "; ".join(buckets["outcome"]))

    return "\n".join(lines)

# ===== Auth endpoints =====
class AuthRequestReq(BaseModel):
    email: EmailStr

class AuthVerifyReq(BaseModel):
    token: str


@api.post("/auth/request")
def auth_request(req: AuthRequestReq):
    """Send a magic-link email. Always returns 200 so the endpoint doesn't
    leak whether an email is registered."""
    token = auth_store.issue_magic_token(req.email)
    link = f"{APP_URL}/?verify_token={token}"
    try:
        send_magic_link_email(RESEND_API_KEY, MAIL_FROM, req.email, link)
        print(f"[auth] magic link sent to {req.email}")
    except Exception as e:
        print(f"[auth] failed to send magic link to {req.email}: {e}")
        raise HTTPException(status_code=502, detail="Email delivery failed")
    return {"ok": True}


@api.post("/auth/verify")
def auth_verify(req: AuthVerifyReq):
    """Exchange a magic-link token for a long-lived session token."""
    user = auth_store.consume_magic_token(req.token)
    if user is None:
        raise HTTPException(status_code=400, detail="Invalid or expired magic link")
    session_token = auth_store.issue_session(user["user_id"])
    return {
        "session_token": session_token,
        "user": {"email": user["email"], "user_id": user["user_id"]},
    }


@api.get("/auth/me")
def auth_me(current_user: dict = Depends(get_current_user)):
    return {
        "email": current_user["email"],
        "user_id": current_user["user_id"],
        "demo_mode": not AUTH_REQUIRED,
    }


@api.post("/auth/logout")
def auth_logout(authorization: Optional[str] = Header(default=None)):
    if authorization and authorization.lower().startswith("bearer "):
        auth_store.revoke_session(authorization.split(None, 1)[1].strip())
    return {"ok": True}


# ===== Endpoints =====
@api.get("/health")
def health():
    return {
        "model_path": MODEL_PATH,
        "classifier_path": CLASSIFIER_PATH or None,
        "device": str(model.device),
        "log_dir": LOG_DIR,
    }

@api.post("/counterfactual")
def counterfactual_endpoint(req: CounterfactualReq, current_user: dict = Depends(get_current_user)):
    """World-model counterfactual MI feedback.

    Estimates the client's talk-type state, tags the counselor's MI action, and
    runs the MPC planner to compare it against the model-optimal intervention.
    Backend for the World Model Panel in the UI.
    """
    req.user_id = current_user["user_id"]
    try:
        from scripts.world_model.counterfactual import coach_feedback
        return coach_feedback(req.client_msg, req.coach_reply, horizon=req.horizon)
    except Exception as e:
        print(f"[counterfactual] failed: {e}")
        return {"error": str(e)}

@api.post("/chat")
def chat_endpoint(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    req.user_id = current_user["user_id"]
    try:
        # 0) Get or create Kerrio client profile for journey management
        kerrio_profile = journey_manager.get_or_create_profile(req.user_id)

        # 1) Use server-side memory; seed once from client if provided
        mem = SESSIONS[req.user_id]
        if req.history and not mem:
            for t in req.history:
                mem.append({"user": t.user, "coach": t.coach})

        # 2) Optional: build a cognitive map summary (Graph-of-Thoughts memory).
        # This is an EXTRA full generation per turn — disabled by default because
        # combined with rerank it pushes total latency past the tunnel's ~100s
        # timeout on later turns. The cognitive-map panel has its own /cogmap and
        # /map endpoints, so the map is still available on demand.
        map_summary: Optional[str] = None
        if COGMAP_IN_CHAT:
            try:
                if len(mem) >= 3:
                    turns_for_map = load_dialog_for_user(req.user_id, max_turns=12)
                    if turns_for_map:
                        cmap = generate_cogmap_from_turns(turns_for_map, max_new_tokens=400)
                        map_summary = summarize_cogmap_for_prompt(cmap)
            except Exception as e:
                print(f"[warn] cogmap summary failed: {e}")

        # 3) Prompt with last N turns + optional map summary + Kerrio profile
        recent_hist = [Turn(user=h["user"], coach=h["coach"]) for h in mem][-6:]
        prompt = render_prompt(
            recent_hist,
            req.user_msg,
            map_summary=map_summary,
            kerrio_profile=kerrio_profile
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        # 4) Generate (decode only new tokens)
        eos_id = tok.eos_token_id
        im_end_id = None
        try:
            im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            pass
        stop_ids = [i for i in [eos_id, im_end_id] if i is not None]

        # System 1 proposes K candidates; System 2 (world model) reranks them.
        n_seq = RERANK_K if (req.rerank and RERANK_K > 1) else 1

        def _decode(seq_ids):
            r = tok.decode(seq_ids, skip_special_tokens=True).strip()
            if not r:  # rare fallback clean
                raw = tok.decode(seq_ids, skip_special_tokens=False)
                r = re.sub(r"<\|im_(start|end)\|>|\s+", " ", raw).strip()
            return r

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=n_seq,
                pad_token_id=eos_id,
                eos_token_id=stop_ids[0] if stop_ids else eos_id,
            )

        plen = inputs["input_ids"].shape[-1]
        candidates = [_decode(out[i][plen:]) for i in range(out.shape[0])]
        candidates = [c for c in candidates if c] or ["(no content)"]

        # 4b) World-model rerank: pick the candidate whose MI action the planner
        #     predicts best evokes change talk for this client's estimated state.
        rerank_info = None
        reply = candidates[0]
        if n_seq > 1 and len(candidates) > 1:
            try:
                from scripts.world_model.counterfactual import rerank_replies
                ctx = recent_hist[-1].coach if recent_hist else ""
                rr = rerank_replies(req.user_msg, candidates, context=ctx)
                reply = rr["chosen_reply"] or reply
                chosen = rr["scored"][rr["chosen_index"]]
                rerank_info = {
                    "enabled": True,
                    "n_candidates": len(candidates),
                    "client_state": rr["client_state"],
                    "chosen_action": chosen["action"],
                    "chosen_Q": chosen["Q"],
                    "best_action": rr["best_action"],
                    "candidates": [
                        {"action": s["action"], "Q": s["Q"],
                         "P_change": s["P_change"], "chosen": s["index"] == rr["chosen_index"]}
                        for s in rr["scored"]
                    ],
                }
                print(f"[rerank] state={rr['client_state']} "
                      f"picked '{chosen['action']}' (Q={chosen['Q']}) "
                      f"from {len(candidates)} candidates")
            except Exception as e:
                print(f"[rerank] failed, using first candidate: {e}")

        # 5) Update server memory
        mem.append({"user": req.user_msg, "coach": reply})

        # 6) Persist one turn
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "user_id": req.user_id,
            "model_path": MODEL_PATH,
            "turn_id": len(mem),                 # 1-based
            "user": req.user_msg,
            "coach": reply,
            "history_len": len(mem),
            "journey_stage": kerrio_profile.stage.value,  # Track journey stage
            "rerank": rerank_info,
        }
        append_jsonl(req.user_id, record)

        # 7) Update Kerrio profile with extracted insights
        journey_manager.add_turn_and_extract(kerrio_profile, req.user_msg, reply)

        print("[chat] reply=", reply[:200].replace("\n", "\\n"))
        print(f"[kerrio] stage={kerrio_profile.stage.value}")

        return {
            "reply": reply or "(no content)",
            "journey_stage": kerrio_profile.stage.value,
            "can_advance": kerrio_profile.can_advance_stage(),
            "rerank": rerank_info,
        }

    except Exception as e:
        print(f"[error] chat generation failed: {e}")
        return {"reply": f"(backend error: {e})"}

@api.post("/cogmap")
def cogmap_endpoint(req: CogMapReq, current_user: dict = Depends(get_current_user)):
    """
    Build / refresh the cognitive map for a given user_id based on the
    server-side session memory (SESSIONS[user_id]).
    """
    req.user_id = current_user["user_id"]
    session = SESSIONS.get(req.user_id, [])
    # we only use user utterances + coach replies; structure already matches
    cm = build_cognitive_map_from_session(session)

    # Optional: you can also persist this map per user for history / profiling
    try:
        cm_rec = dict(cm)
        cm_rec["user_id"] = req.user_id
        cm_rec["turns"] = len(session)
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        cm_rec["ts"] = ts

        import os, json
        from pathlib import Path

        log_dir = os.environ.get("LOG_DIR", "runs/chat_logs")
        maps_dir = os.path.join(log_dir, "cogmaps")
        Path(maps_dir).mkdir(parents=True, exist_ok=True)
        fp = os.path.join(maps_dir, f"{req.user_id}.jsonl")
        with open(fp, "a", encoding="utf-8") as f:
            f.write(json.dumps(cm_rec, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[warn] failed to persist cognitive map: {e}")

    return cm

@api.post("/score")
def score_endpoint(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    req.user_id = current_user["user_id"]
    if not (cls_model and cls_tok):
        return {"labels": [], "probs": []}
    # Score the last coach reply (or user_msg as a fallback)
    text = req.user_msg
    mem = SESSIONS.get(req.user_id) or []
    if mem:
        text = mem[-1]["coach"]
    enc = cls_tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(cls_model.device)
    with torch.no_grad():
        logits = cls_model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().tolist()[0]
    labels = cls_labels if cls_labels else list(range(len(probs)))
    return {"labels": labels, "probs": probs}

@api.post("/reset")
def reset_endpoint(req: ResetReq, backup: bool = False, current_user: dict = Depends(get_current_user)):
    """Reset user session, optionally backup data"""
    req.user_id = current_user["user_id"]
    import shutil
    
    if backup:
        # Backup current data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = Path(LOG_DIR) / f"{req.user_id}_backup_{timestamp}.jsonl"
        
        original_file = Path(LOG_DIR) / f"{req.user_id}.jsonl"
        if original_file.exists():
            shutil.copy(original_file, backup_file)
    
    # Clear in-memory session
    SESSIONS.pop(req.user_id, None)
    
    # Clear journey manager profile
    if hasattr(journey_manager, 'profiles') and req.user_id in journey_manager.profiles:
        del journey_manager.profiles[req.user_id]
    
    # Clear log file
    log_file = Path(LOG_DIR) / f"{req.user_id}.jsonl"
    if log_file.exists():
        log_file.unlink()
    
    return {
        "ok": True,
        "message": f"Session reset for {req.user_id}",
        "backup_created": backup
    }

@api.get("/users/{user_id}/history")
def get_user_history(user_id: str, current_user: dict = Depends(get_current_user)):
    """Get complete conversation history for a user"""
    user_id = current_user["user_id"]
    from dataclasses import asdict
    
    profile = journey_manager.get_profile(user_id)
    
    if not profile:
        return {
            "error": "User not found",
            "user_id": user_id,
            "conversation_history": [],
            "stage": "unknown"
        }
    
    # Load from JSONL file
    log_history = []
    log_file = Path(LOG_DIR) / f"{user_id}.jsonl"
    
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    log_history.append(record)
                except:
                    pass
    
    return {
        "user_id": user_id,
        "stage": profile.stage.value,
        "conversation_turns": len(profile.conversation_history),
        "conversation_history": profile.conversation_history,
        "log_history": log_history,
        "pillars": {
            "history": asdict(profile.client_history.history),
            "psychology_philosophy": asdict(profile.client_history.psychology_philosophy),
            "physiology": asdict(profile.client_history.physiology)
        } if profile.client_history else None,
        "diagnosis": asdict(profile.diagnosis) if profile.diagnosis else None,
        "treatment_plan": asdict(profile.treatment_plan) if profile.treatment_plan else None,
        "monitoring_history": profile.monitoring_history if hasattr(profile, 'monitoring_history') else []
    }


# === Kerrio Journey Management Endpoints ===

@api.get("/journey/{user_id}")
def get_journey_status(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get the current Kerrio journey status for a user.
    Returns stage, progress, and what's needed to advance.
    """
    user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(user_id)

    # Build stage requirements message
    stage_info = {
        JourneyStage.REGISTRATION: "Welcome phase. Ready to begin history collection.",
        JourneyStage.HISTORY_COLLECTION: "Gathering history across three pillars: History, Psychology/Philosophy, Physiology.",
        JourneyStage.CONSULTATION: "Clarifying ambiguities and uncovering blind spots.",
        JourneyStage.DIAGNOSIS: "Building Cognitive Wiring Map and explaining root causes.",
        JourneyStage.PROPOSAL: "Presenting personalized treatment plan.",
        JourneyStage.TREATMENT: "Implementing cognitive rewiring interventions.",
        JourneyStage.MONITORING: "Tracking progress and refining interventions.",
    }

    return {
        "user_id": user_id,
        "current_stage": profile.stage.value,
        "stage_description": stage_info.get(profile.stage, ""),
        "can_advance": profile.can_advance_stage(),
        "registered_at": profile.registered_at,
        "conversation_turns": len(profile.conversation_history),
        "conversation_history": profile.conversation_history,  # Add full conversation history
        "client_history_summary": {
            "life_events_count": len(profile.client_history.history.life_events),
            "beliefs_count": len(profile.client_history.psychology_philosophy.beliefs),
            "values_count": len(profile.client_history.psychology_philosophy.values),
        },
        "clinician_insights_count": len(profile.clinician_notes.session_insights),
        "is_validated": profile.is_validated_guest
    }


@api.post("/journey/validate")
def validate_invite(req: ValidateInviteReq, current_user: dict = Depends(get_current_user)):
    """
    Validate a user's invitation code to allow them to proceed.
    """
    req.user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(req.user_id)
    if profile.validate_invite_code(req.invite_code):
        journey_manager.save_profile(req.user_id)
        return {"success": True, "message": "Invitation validated. Welcome to Kerrio."}
    else:
        return {"success": False, "message": "Invalid invitation code."}


@api.post("/journey/proposal/accept")
def accept_proposal(req: AdvanceStageReq, current_user: dict = Depends(get_current_user)):
    """
    Client accepts the treatment proposal.
    """
    req.user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(req.user_id)
    profile.treatment_proposal.client_accepted = True
    journey_manager.save_profile(req.user_id)
    return {"success": True, "message": "Treatment proposal accepted. Ready to begin."}


@api.post("/journey/advance")
def advance_journey_stage(req: AdvanceStageReq, current_user: dict = Depends(get_current_user)):
    """
    Attempt to advance the user to the next journey stage.
    Returns success/failure and the new stage.
    """
    req.user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(req.user_id)
    old_stage = profile.stage.value

    if profile.advance_stage():
        journey_manager.save_profile(req.user_id)
        return {
            "success": True,
            "old_stage": old_stage,
            "new_stage": profile.stage.value,
            "message": f"Advanced from {old_stage} to {profile.stage.value}",
        }
    else:
        return {
            "success": False,
            "current_stage": profile.stage.value,
            "message": "Cannot advance yet. Complete current stage requirements first.",
            "can_advance": False,
        }


@api.get("/journey/prompts/{user_id}")
def get_stage_prompts(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get suggested prompts/questions for the current journey stage.
    Useful for guiding the conversation.
    """
    user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(user_id)

    if profile.stage == JourneyStage.HISTORY_COLLECTION:
        return {
            "stage": profile.stage.value,
            "prompts": HISTORY_COLLECTION_PROMPTS,
            "instruction": "Guide the client through these three pillars of history collection.",
        }
    elif profile.stage == JourneyStage.CONSULTATION:
        return {
            "stage": profile.stage.value,
            "prompts": {
                "clarification": [
                    "Can you tell me more about what you meant when you said...",
                    "I noticed you mentioned X but didn't elaborate. Can we explore that?",
                    "What was that experience like for you emotionally?",
                ],
                "blind_spot_probing": [
                    "How do others close to you see this situation?",
                    "Is there anything you've been avoiding thinking about?",
                    "What would you not want me to ask about?",
                ],
            },
            "instruction": "Uncover blind spots and clarify ambiguities. Maintain separation between client history and clinician notes.",
        }
    elif profile.stage == JourneyStage.DIAGNOSIS:
        return {
            "stage": profile.stage.value,
            "prompts": {
                "explanation": [
                    "Based on what you've shared, here's what I'm seeing...",
                    "The pattern that emerges is...",
                    "The root cause appears to be...",
                ],
                "education": [
                    "Let me explain the neuroscience behind this...",
                    "Understanding WHY this happens is crucial before we can change it...",
                ],
            },
            "instruction": "Explain the diagnosis clearly. Client must understand before proceeding to treatment.",
        }
    else:
        return {
            "stage": profile.stage.value,
            "prompts": {},
            "instruction": f"Continue with {profile.stage.value} phase.",
        }


@api.get("/journey/history/{user_id}")
def get_client_history(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get the client's collected history (three pillars).
    Separate from clinician notes per Kerrio requirements.
    """
    user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(user_id)
    h = profile.client_history

    return {
        "user_id": user_id,
        "history_pillar": {
            "life_events": h.history.life_events,
            "formative_experiences": h.history.formative_experiences,
            "recurrent_patterns": h.history.recurrent_patterns,
            "background_summary": h.history.background_summary,
        },
        "psychology_philosophy_pillar": {
            "beliefs": h.psychology_philosophy.beliefs,
            "values": h.psychology_philosophy.values,
            "meaning_structures": h.psychology_philosophy.meaning_structures,
            "emotional_wiring": h.psychology_philosophy.emotional_wiring,
            "core_assumptions": h.psychology_philosophy.core_assumptions,
        },
        "physiology_pillar": {
            "sleep_quality": h.physiology.sleep_quality,
            "sleep_hours": h.physiology.sleep_hours,
            "stress_level": h.physiology.stress_level,
            "health_conditions": h.physiology.health_conditions,
            "energy_patterns": h.physiology.energy_patterns,
            "physical_constraints": h.physiology.physical_constraints,
        },
    }


@api.get("/journey/notes/{user_id}")
def get_clinician_notes(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get the clinician's notes (AI observations).
    Maintained separately from client history per Kerrio requirements.
    """
    user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(user_id)
    n = profile.clinician_notes

    return {
        "user_id": user_id,
        "session_insights": [
            {
                "turn_id": i.turn_id,
                "observation": i.observation,
                "category": i.category,
                "timestamp": i.timestamp,
            }
            for i in n.session_insights
        ],
        "emerging_patterns": n.emerging_patterns,
        "diagnostic_hypotheses": n.diagnostic_hypotheses,
        "blind_spots_identified": n.blind_spots_identified,
        "ambiguities_to_clarify": n.ambiguities_to_clarify,
    }


# === Diagnosis and Treatment Endpoints ===

@api.get("/journey/diagnosis/{user_id}")
def get_diagnosis(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Generate or retrieve the diagnosis for a user.
    This is the most important phase - explaining WHY the problem exists.
    """
    user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(user_id)

    # Generate diagnosis from collected history
    diagnosis = diagnostic_engine.generate_diagnosis_from_history(
        profile.client_history,
        profile.clinician_notes,
        profile.cognitive_wiring_map
    )

    # Store diagnosis in profile
    profile.diagnosis = diagnosis
    journey_manager.save_profile(user_id)

    return {
        "user_id": user_id,
        "stage": profile.stage.value,
        "diagnosis": {
            "core_constraints": diagnosis.core_constraints,
            "bottlenecks": diagnosis.bottlenecks,
            "root_causes": diagnosis.root_causes,
            "explanation": diagnosis.explanation,
            "client_understood": diagnosis.client_understood,
            "recommended_videos": [
                {
                    "video_id": v.video_id,
                    "title": v.title,
                    "relevance": v.relevance,
                    "url": v.url
                }
                for v in diagnosis.recommended_videos
            ]
        }
    }


@api.post("/journey/diagnosis/confirm/{user_id}")
def confirm_diagnosis_understood(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Client confirms they understand the diagnosis.
    This is required before proceeding to treatment proposal.
    'Understanding is a prerequisite for permanent change.'
    """
    user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(user_id)
    profile.diagnosis.client_understood = True
    journey_manager.save_profile(user_id)

    return {
        "success": True,
        "message": "Diagnosis understanding confirmed. Ready for treatment proposal.",
        "can_advance": profile.can_advance_stage()
    }


@api.get("/journey/treatment/{user_id}")
def get_treatment_proposal(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get the treatment proposal including Cognitive Rewiring Map.
    Only available after diagnosis is understood.
    """
    user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(user_id)

    if not profile.diagnosis.client_understood:
        return {
            "error": "Diagnosis must be understood before treatment proposal.",
            "message": "Please confirm understanding of the diagnosis first."
        }

    # Extract goals from cognitive map or history
    goals = []
    for node in profile.cognitive_wiring_map.nodes:
        if node.type == "goal":
            goals.append(node.label)

    # Generate rewiring map if not already done
    if not profile.treatment_proposal.rewiring_map:
        rewiring_map = rewiring_engine.generate_rewiring_map(
            profile.diagnosis,
            profile.cognitive_wiring_map,
            goals
        )
        profile.treatment_proposal.rewiring_map = rewiring_map
        journey_manager.save_profile(user_id)

    rm = profile.treatment_proposal.rewiring_map
    return {
        "user_id": user_id,
        "stage": profile.stage.value,
        "treatment_proposal": {
            "rewiring_map": {
                "current_wiring": rm.current_wiring if rm else "",
                "target_wiring": rm.target_wiring if rm else "",
                "progress": rm.progress if rm else 0.0,
                "steps": [
                    {
                        "id": s.id,
                        "name": s.name,
                        "description": s.description,
                        "rationale": s.neuroscience_rationale,
                        "completed": s.completed,
                        "completed_at": s.completed_at
                    }
                    for s in (rm.rewiring_steps if rm else [])
                ]
            },
            "interventions": [
                {
                    "id": i.id,
                    "name": i.name,
                    "description": i.description,
                    "frequency": i.frequency,
                    "progress": i.progress
                }
                for i in profile.treatment_proposal.interventions
            ],
            "client_accepted": profile.treatment_proposal.client_accepted
        }
    }


@api.post("/journey/treatment/step/complete")
def complete_rewiring_step(req: StepCompleteReq, current_user: dict = Depends(get_current_user)):
    """Mark a specific rewiring step as complete."""
    req.user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(req.user_id)
    if not profile.treatment_proposal.rewiring_map:
        return {"success": False, "message": "No rewiring map found."}

    rm = profile.treatment_proposal.rewiring_map
    step_found = False
    for s in rm.rewiring_steps:
        if s.id == req.step_id:
            s.completed = True
            s.completed_at = datetime.now(timezone.utc).isoformat()
            step_found = True
            break

    if step_found:
        rm.update_progress()
        journey_manager.save_profile(req.user_id)
        return {
            "success": True,
            "progress": rm.progress,
            "can_advance": profile.can_advance_stage()
        }
    return {"success": False, "message": "Step not found."}


@api.post("/journey/monitoring/submit")
def submit_monitoring(req: MonitoringReq, current_user: dict = Depends(get_current_user)):
    """Submit monitoring feedback and trigger closed-loop logic."""
    req.user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(req.user_id)
    should_rediagnose = journey_manager.submit_monitoring_feedback(
        profile, req.metrics, req.notes
    )

    return {
        "success": True,
        "should_rediagnose": should_rediagnose,
        "new_stage": profile.stage.value,
        "message": "Diagnosis updated" if should_rediagnose else "Monitoring recorded"
    }


@api.post("/journey/treatment/accept/{user_id}")
def accept_treatment(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Client accepts the treatment proposal.
    This advances them to the active treatment phase.
    """
    user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(user_id)
    profile.treatment_proposal.client_accepted = True
    journey_manager.save_profile(user_id)

    return {
        "success": True,
        "message": "Treatment accepted. Beginning cognitive rewiring process.",
        "can_advance": profile.can_advance_stage()
    }


@api.get("/journey/videos")
def get_all_videos():
    """
    Get the complete video library for educational content.
    Videos are assigned based on diagnosis.
    """
    return {
        "videos": [
            {
                "id": video["id"],
                "title": video["title"],
                "topics": video["topics"],
                "duration_minutes": video["duration_minutes"],
                "description": video["description"]
            }
            for video in VIDEO_DATABASE.values()
        ]
    }


@api.get("/journey/videos/{video_id}")
def get_video_details(video_id: str):
    """
    Get details for a specific video.
    """
    for video in VIDEO_DATABASE.values():
        if video["id"] == video_id:
            return video
    return {"error": f"Video {video_id} not found"}


@api.post("/journey/treatment/progress/{user_id}")
def update_treatment_progress(user_id: str, step_index: int = 0, completed: bool = True, current_user: dict = Depends(get_current_user)):
    """
    Update progress on treatment steps.
    Used for monitoring and reassessment.
    """
    user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(user_id)

    if profile.treatment_proposal.rewiring_map:
        total_steps = len(profile.treatment_proposal.rewiring_map.rewiring_steps)
        if total_steps > 0:
            completed_steps = min(step_index + 1 if completed else step_index, total_steps)
            profile.treatment_proposal.rewiring_map.progress = completed_steps / total_steps

    journey_manager.save_profile(user_id)

    return {
        "success": True,
        "progress": profile.treatment_proposal.rewiring_map.progress if profile.treatment_proposal.rewiring_map else 0
    }


@api.get("/journey/full-profile/{user_id}")
def get_full_profile(user_id: str, current_user: dict = Depends(get_current_user)):
    """
    Get the complete Kerrio profile for a user.
    Includes all journey data: history, notes, diagnosis, treatment.
    """
    user_id = current_user["user_id"]
    profile = journey_manager.get_or_create_profile(user_id)
    h = profile.client_history
    n = profile.clinician_notes
    d = profile.diagnosis
    t = profile.treatment_proposal

    return {
        "user_id": user_id,
        "stage": profile.stage.value,
        "registered_at": profile.registered_at,
        "conversation_turns": len(profile.conversation_history),

        "client_history": {
            "history_pillar": {
                "life_events": h.history.life_events,
                "formative_experiences": h.history.formative_experiences,
                "recurrent_patterns": h.history.recurrent_patterns,
            },
            "psychology_philosophy_pillar": {
                "beliefs": h.psychology_philosophy.beliefs,
                "values": h.psychology_philosophy.values,
                "core_assumptions": h.psychology_philosophy.core_assumptions,
            },
            "physiology_pillar": {
                "sleep_quality": h.physiology.sleep_quality,
                "stress_level": h.physiology.stress_level,
            }
        },

        "clinician_notes": {
            "insights_count": len(n.session_insights),
            "blind_spots": n.blind_spots_identified,
            "patterns": n.emerging_patterns,
        },

        "diagnosis": {
            "core_constraints": d.core_constraints,
            "bottlenecks": d.bottlenecks,
            "root_causes": d.root_causes,
            "client_understood": d.client_understood,
        },

        "treatment": {
            "accepted": t.client_accepted,
            "progress": t.rewiring_map.progress if t.rewiring_map else 0,
            "steps_count": len(t.rewiring_map.rewiring_steps) if t.rewiring_map else 0,
        },

        "cognitive_map": {
            "nodes_count": len(profile.cognitive_wiring_map.nodes),
            "edges_count": len(profile.cognitive_wiring_map.edges),
        }
    }


@api.get("/map/{user_id}")
def get_cognitive_map(user_id: str, max_turns: int = 20, current_user: dict = Depends(get_current_user)):
    """
    Generate a cognitive map JSON from the user's dialogue history.
    If the LLM returns no edges, auto-generate simple reasonable edges
    so the map is not empty and the UI will show connections.
    """
    user_id = current_user["user_id"]
    try:
        turns = load_dialog_for_user(user_id, max_turns=max_turns)
        if not turns:
            return {"error": f"No dialog found for user_id={user_id}"}

        data = generate_cogmap_from_turns(turns)
        nodes = data.get("nodes", []) or []
        edges = data.get("edges", []) or []

        # -------------------------------------------
        # Minimal heuristic: auto-add edges if missing
        # -------------------------------------------
        if not edges and len(nodes) >= 2:
            # First: try to find a goal to link everything to
            goal_nodes = [n for n in nodes if (n.get("type") or "").lower() == "goal"]

            if goal_nodes:
                goal_id = goal_nodes[0]["id"]
                for n in nodes:
                    nid = n["id"]
                    if nid == goal_id:
                        continue

                    ntype = (n.get("type") or "").lower()
                    rel = "supports"
                    if ntype == "barrier":
                        rel = "blocks"
                    elif ntype in ("belief", "strength"):
                        rel = "explains"
                    elif ntype == "action":
                        rel = "leads_to"

                    edges.append({
                        "source": nid,
                        "target": goal_id,
                        "type": rel
                    })
            else:
                # Fallback: connect all nodes to first node
                root = nodes[0]["id"]
                for n in nodes[1:]:
                    edges.append({
                        "source": root,
                        "target": n["id"],
                        "type": "related"
                    })

        # Update and return final map
        data["nodes"] = nodes
        data["edges"] = edges
        return data

    except Exception as e:
        return {"error": str(e)}

# Provide a null favicon to silence 404s
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# Mount API first, then static site at root
app.include_router(api, prefix="/api")
app.mount("/", StaticFiles(directory="web", html=True), name="web")
