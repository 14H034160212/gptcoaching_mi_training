#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SYSTEM = (
    "You are a supportive health coach using Motivational Interviewing (MI). "
    "Be non-judgmental; prefer open questions, reflective listening, affirmations; ask permission before giving advice. "
    "Use provided STATE (profile, wearable summary) to personalize."
)

def render_prompt(state, history, user_msg):
    ctx = "\n".join([f"USER: {u}\nCOACH: {a}" for u,a in history])
    return f"<|system|>\n{SYSTEM}\n</|system|>\n" \
           f"<|context|>\nSTATE={json.dumps(state, ensure_ascii=False)}\n</|context|>\n" \
           f"{ctx}\nUSER: {user_msg}\nASSISTANT:"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")

    state = {"goal":"increase_steps","week_steps_mean":5200,"barriers":["time"]}
    history = [("I want to be more active.","What matters most to you about being more active?")]
    user_msg = "I just feel too busy this week."

    prompt = render_prompt(state, history, user_msg)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    print(tok.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
