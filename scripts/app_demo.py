#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app_demo.py — FastAPI demo for MI-style chat with short-term memory + per-user logging.

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

Static:
  "/" serves web/index.html
"""
import os, json, re, torch
from typing import List, Dict
from datetime import datetime, timezone
from collections import defaultdict

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

# ===== Helpers =====
def append_jsonl(user_id: str, payload: dict):
    """Append one record to LOG_DIR/<user_id>.jsonl"""
    fp = os.path.join(LOG_DIR, f"{user_id}.jsonl")
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def build_messages(history: List[Turn], user_msg: str):
    msgs = [{"role": "system", "content": SYSTEM}]
    for t in history:
        if t.user:
            msgs.append({"role": "user", "content": t.user})
        if t.coach:
            msgs.append({"role": "assistant", "content": t.coach})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

def render_prompt(history: List[Turn], user_msg: str) -> str:
    msgs = build_messages(history, user_msg)
    return tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True  # start a new assistant turn
    )

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

        # 2) Prompt with last N turns
        recent_hist = [Turn(user=h["user"], coach=h["coach"]) for h in mem][-6:]
        prompt = render_prompt(recent_hist, req.user_msg)
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        # 3) Generate (decode only new tokens)
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

        # 4) Update server memory
        mem.append({"user": req.user_msg, "coach": reply})

        # 5) Persist one turn
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
    return {"ok": True}

# Provide a null favicon to silence 404s
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

# Mount API first, then static site at root
app.include_router(api, prefix="/api")
app.mount("/", StaticFiles(directory="web", html=True), name="web")