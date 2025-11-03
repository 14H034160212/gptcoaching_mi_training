#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
policy_bridge.py
A minimal policy bridge to use the MI LLM as a dialogue policy inside a ConvLab-style loop.
This does not require ConvLab at runtime; it's a template to adapt into ConvLab-3's policy interface.

Usage:
python integrations/convlab/policy_bridge.py --model_path runs/dpo-llama3-mi-annomi
"""
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM = (
  "You are a supportive health coach using Motivational Interviewing (MI). "
  "Be non-judgmental; use open questions, reflective listening, and affirmations; "
  "ask permission before giving advice; avoid directives. Keep it concise."
)

def render_prompt(state, history, user_msg):
    ctx = "\n".join([f"USER: {u}\\nCOACH: {a}" for u,a in history])
    return f"<|system|>\\n{SYSTEM}\\n</|system|>\\n" \
           f"<|context|>\\nSTATE={json.dumps(state, ensure_ascii=False)}\\n</|context|>\\n" \
           f"{ctx}\\nUSER: {user_msg}\\nASSISTANT:"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")

    # Example DST state and history (to be replaced by ConvLab trackers)
    state = {"goal":"increase_steps", "barriers":["time"], "preference":[]}
    history = [("I want to be more active.","What matters most about being active for you?")]
    user_msg = "I'm busy this week."
    prompt = render_prompt(state, history, user_msg)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=180, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    reply = text.split("ASSISTANT:")[-1].strip()
    print(reply)

if __name__ == "__main__":
    main()
