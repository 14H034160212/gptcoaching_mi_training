#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app_demo.py — Local FastAPI demo for MI-style chat with short-term memory.

Run:
  export MODEL_PATH=/abs/path/to/your/merged-model
  # optional: export CLASSIFIER_PATH=/abs/path/to/mi-classifier
  uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8000 --reload

Notes:
- API routes under /api  →  POST /api/chat, POST /api/score, GET /api/health
- Static front-end (web/index.html) is mounted at "/"
"""
import os, json, re, torch
from typing import List
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
import torch.nn.functional as F

SYSTEM = (
    "You are a supportive health coach using Motivational Interviewing (MI). "
    "Be non-judgmental; use open questions, reflective listening, and affirmations; "
    "ask permission before giving advice; avoid directives. Keep it concise."
)

# ===== Env =====
MODEL_PATH = os.environ.get("MODEL_PATH", "").strip()
CLASSIFIER_PATH = os.environ.get("CLASSIFIER_PATH", "").strip()
assert MODEL_PATH, "MODEL_PATH is empty. export MODEL_PATH=/abs/path/to/model before launching."

# ===== Load main model =====
print(f"[boot] MODEL_PATH={MODEL_PATH}")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map="auto", trust_remote_code=True
).eval()

# ===== Optional classifier (for MI behavior scoring) =====
cls_tok = cls_model = None
cls_labels = None
if CLASSIFIER_PATH:
    try:
        print(f"[boot] CLASSIFIER_PATH={CLASSIFIER_PATH}")
        cls_tok = AutoTokenizer.from_pretrained(CLASSIFIER_PATH, use_fast=True)
        cls_model = AutoModelForSequenceClassification.from_pretrained(
            CLASSIFIER_PATH, device_map="auto"
        ).eval()
        # optional labels.json
        try:
            with open(os.path.join(CLASSIFIER_PATH, "labels.json"), "r") as f:
                cls_labels = json.load(f)
        except Exception:
            pass
    except Exception as e:
        print(f"[warn] failed to load classifier: {e}")

# ===== FastAPI =====
app = FastAPI(title="GPTCoach MI Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

api = APIRouter()

class Turn(BaseModel):
    user: str
    coach: str

class ChatRequest(BaseModel):
    history: List[Turn] = []
    user_msg: str

# ---- Build prompt using model's chat template to avoid leaking <|system|> tags ----
def build_messages(history: List[Turn], user_msg: str):
    msgs = [{"role": "system", "content": SYSTEM}]
    for t in history:
        if t.user:
            msgs.append({"role": "user", "content": t.user})
        if t.coach:
            msgs.append({"role": "assistant", "content": t.coach})
    msgs.append({"role": "user", "content": user_msg})
    return msgs

def render_prompt(history: list[Turn], user_msg: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}]
    for t in history:
        if t.user:  msgs.append({"role": "user", "content": t.user})
        if t.coach: msgs.append({"role": "assistant", "content": t.coach})
    msgs.append({"role": "user", "content": user_msg})
    # last is user → start a new assistant turn
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@api.get("/health")
def health():
    return {
        "model_path": MODEL_PATH,
        "classifier_path": CLASSIFIER_PATH or None,
        "device": str(model.device),
    }

@api.post("/chat")
def chat_endpoint(req: ChatRequest):
    try:
        prompt = render_prompt(req.history[-6:], req.user_msg)
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        # === Insert the new generation block here ===
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
                eos_token_id=stop_ids[0] if stop_ids else eos_id
            )

        # 🔑 Decode only new tokens (avoid echoing the prompt)
        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        reply = tok.decode(gen_ids, skip_special_tokens=True).strip()

        # Fallback clean-up (rare)
        if not reply:
            import re
            raw = tok.decode(gen_ids, skip_special_tokens=False)
            reply = re.sub(r"<\|im_(start|end)\|>|\s+", " ", raw).strip()

        print("[chat] reply=", reply[:200].replace("\n", "\\n"))
        return {"reply": reply or "(no content)"}

    except Exception as e:
        print(f"[error] chat generation failed: {e}")
        return {"reply": f"(backend error: {e})"}

@api.post("/score")
def score_endpoint(req: ChatRequest):
    if not (cls_model and cls_tok):
        return {"labels": [], "probs": []}
    text = req.user_msg
    if req.history and req.history[-1].coach:
        text = req.history[-1].coach
    enc = cls_tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(cls_model.device)
    with torch.no_grad():
        logits = cls_model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().tolist()[0]
    labels = cls_labels if cls_labels else list(range(len(probs)))
    return {"labels": labels, "probs": probs}

# Mount API first, then static site at root (prevents POST /chat being intercepted by StaticFiles)
app.include_router(api, prefix="/api")
app.mount("/", StaticFiles(directory="web", html=True), name="web")
