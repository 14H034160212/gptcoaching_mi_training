#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app_demo.py
Local FastAPI demo for MI-style chat with short-term memory.
Usage:
  uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8000 --reload
Env:
  MODEL_PATH=runs/dpo-llama3-mi-annomi  (or any HF model path)
"""
import os, json, torch
from typing import List, Dict, Any
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
import torch.nn.functional as F

SYSTEM = (
  "You are a supportive health coach using Motivational Interviewing (MI). "
  "Be non-judgmental; use open questions, reflective listening, and affirmations; "
  "ask permission before giving advice; avoid directives. Keep it concise."
)

MODEL_PATH = os.environ.get("MODEL_PATH", "runs/dpo-llama3-mi-annomi")
CLASSIFIER_PATH = os.environ.get("CLASSIFIER_PATH")

tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
cls_tok = None
cls_model = None
cls_labels = None
if CLASSIFIER_PATH:
    try:
        cls_tok = AutoTokenizer.from_pretrained(CLASSIFIER_PATH, use_fast=True)
        cls_model = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH, device_map="auto").eval()
        # optional labels.json
        try:
            import json, os
            with open(os.path.join(CLASSIFIER_PATH, 'labels.json'), 'r') as f:
                cls_labels = json.load(f)
        except Exception:
            pass
    except Exception as e:
        print(f"[warn] failed to load classifier: {e}")

app = FastAPI(title="GPTCoach MI Demo")
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
app.mount('/', StaticFiles(directory='web', html=True), name='web')

class Turn(BaseModel):
    user: str
    coach: str

class ChatRequest(BaseModel):
    history: List[Turn] = []
    user_msg: str

def render_prompt(history: List[Turn], user_msg: str) -> str:
    ctx = "\n".join([f"USER: {t.user}\nCOACH: {t.coach}" for t in history])
    return f"<|system|>\n{SYSTEM}\n</|system|>\n{ctx}\nUSER: {user_msg}\nASSISTANT:"

@app.post("/chat")
def score(req: ChatRequest):
    prompt = render_prompt(req.history[-6:], req.user_msg)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=220, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    reply = text.split("ASSISTANT:")[-1].strip()
    return {"reply": reply}

@app.post("/score")
def score(req: ChatRequest):
    if not cls_model or not cls_tok:
        return {"labels": [], "probs": []}
    # Score the COACH side for the prospective reply given the user_msg (or last bot msg if in history)
    text = req.user_msg
    if req.history:
        text = req.history[-1].coach
    enc = cls_tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512).to(cls_model.device)
    with torch.no_grad():
        logits = cls_model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().tolist()[0]
    labels = cls_labels if cls_labels else list(range(len(probs)))
    return {"labels": labels, "probs": probs}
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=220, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    # best-effort to extract assistant reply
    reply = text.split("ASSISTANT:")[-1].strip()
    return {"reply": reply}
