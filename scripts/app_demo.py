#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app_demo.py — FastAPI demo for MI-style chat with short-term memory + per-user logging + cognitive maps.

Run:
  export MODEL_PATH=/abs/path/to/your/merged-model  # or HF repo id (public or private)
  # optional: export CLASSIFIER_PATH=/abs/path/to/mi-classifier
  # optional: export HF_TOKEN=hf_xxx   (if MODEL_PATH is a private HF repo)
  # optional: export LOG_DIR=runs/chat_logs
  uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8000

API:
  POST /api/chat   {user_id, history?, user_msg}  ->  {reply}
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

from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response
from pydantic import BaseModel

import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from typing import List, Dict
from pydantic import BaseModel

from scripts.cogmap_utils import build_cognitive_map_from_session


SYSTEM = (
    "You are a supportive health coach using Motivational Interviewing (MI). "
    "Be non-judgmental; use open questions, reflective listening, and affirmations; "
    "ask permission before giving advice; avoid directives. Keep it concise."
)

# ===== Env =====
MODEL_PATH = os.environ.get("MODEL_PATH", "").strip()
CLASSIFIER_PATH = os.environ.get("CLASSIFIER_PATH", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
LOG_DIR = os.environ.get("LOG_DIR", "runs/chat_logs")

assert MODEL_PATH, "MODEL_PATH is empty. export MODEL_PATH=/abs/path/to/model or HF repo id"

os.makedirs(LOG_DIR, exist_ok=True)

# Auth kwargs for private HF repos (Transformers >= 4.41 uses `token`)
hf_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

# ===== Load main model =====
print(f"[boot] MODEL_PATH={MODEL_PATH}")
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

class ResetReq(BaseModel):
    user_id: str

class CogMapReq(BaseModel):
    user_id: str = "anon"


# ===== Helpers =====
def append_jsonl(user_id: str, payload: dict):
    """Append one record to LOG_DIR/<user_id>.jsonl"""
    fp = os.path.join(LOG_DIR, f"{user_id}.jsonl")
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_messages(history: List[Turn], user_msg: str, map_summary: Optional[str] = None):
    """
    Build chat messages; if map_summary is provided, inject it into the system prompt
    as a kind of 'Graph-of-Thoughts' memory.
    """
    sys_content = SYSTEM
    if map_summary:
        sys_content += (
            "\n\nHere is a brief summary of the user's situation based on previous dialogue "
            "(goals, values, barriers, actions):\n"
            f"{map_summary}\n"
            "Use this context to ask better MI-style questions, but do not repeat this summary verbatim."
        )

    msgs = [{"role": "system", "content": sys_content}]
    for t in history:
        if t.user:
            msgs.append({"role": "user", "content": t.user})
        if t.coach:
            msgs.append({"role": "assistant", "content": t.coach})
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def render_prompt(history: List[Turn], user_msg: str, map_summary: Optional[str] = None) -> str:
    msgs = build_messages(history, user_msg, map_summary=map_summary)
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

# ===== Endpoints =====
@api.get("/health")
def health():
    return {
        "model_path": MODEL_PATH,
        "classifier_path": CLASSIFIER_PATH or None,
        "device": str(model.device),
        "log_dir": LOG_DIR,
    }

@api.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        # 1) Use server-side memory; seed once from client if provided
        mem = SESSIONS[req.user_id]
        if req.history and not mem:
            for t in req.history:
                mem.append({"user": t.user, "coach": t.coach})

        # 2) Optional: build a cognitive map summary (Graph-of-Thoughts memory)
        map_summary: Optional[str] = None
        try:
            # Only start building maps after a few turns to avoid overkill
            if len(mem) >= 3:
                turns_for_map = load_dialog_for_user(req.user_id, max_turns=12)
                if turns_for_map:
                    cmap = generate_cogmap_from_turns(turns_for_map, max_new_tokens=400)
                    map_summary = summarize_cogmap_for_prompt(cmap)
        except Exception as e:
            print(f"[warn] cogmap summary failed: {e}")

        # 3) Prompt with last N turns + optional map summary
        recent_hist = [Turn(user=h["user"], coach=h["coach"]) for h in mem][-6:]
        prompt = render_prompt(recent_hist, req.user_msg, map_summary=map_summary)
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        # 4) Generate (decode only new tokens)
        eos_id = tok.eos_token_id
        im_end_id = None
        try:
            im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            pass
        stop_ids = [i for i in [eos_id, im_end_id] if i is not None]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=220,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=eos_id,
                eos_token_id=stop_ids[0] if stop_ids else eos_id,
            )

        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        reply = tok.decode(gen_ids, skip_special_tokens=True).strip()
        if not reply:  # rare fallback clean
            raw = tok.decode(gen_ids, skip_special_tokens=False)
            reply = re.sub(r"<\|im_(start|end)\|>|\s+", " ", raw).strip()

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
        }
        append_jsonl(req.user_id, record)

        print("[chat] reply=", reply[:200].replace("\n", "\\n"))
        return {"reply": reply or "(no content)"}

    except Exception as e:
        print(f"[error] chat generation failed: {e}")
        return {"reply": f"(backend error: {e})"}

@api.post("/cogmap")
def cogmap_endpoint(req: CogMapReq):
    """
    Build / refresh the cognitive map for a given user_id based on the
    server-side session memory (SESSIONS[user_id]).
    """
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
def score_endpoint(req: ChatRequest):
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
def reset_endpoint(req: ResetReq):
    SESSIONS.pop(req.user_id, None)
    # also clear log file if you want (optional)
    return {"ok": True}

@api.get("/map/{user_id}")
def get_cognitive_map(user_id: str, max_turns: int = 20):
    """
    Generate a cognitive map JSON from the user's dialogue history.
    If the LLM returns no edges, auto-generate simple reasonable edges
    so the map is not empty and the UI will show connections.
    """
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
