#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_models_mi.py — simple MI-Coach vs Base model comparison
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPTS = [
    "I’ve been trying to get back into running, but I keep losing motivation after a few days.",
    "I know drinking less would be good for me, but it’s the only way I can unwind after work.",
    "I try to eat healthier, but when I don’t see progress, I give up and go back to old habits.",
    "I’m always tired because I stay up late scrolling on my phone, but I can’t seem to stop.",
    "Work has been overwhelming lately, and I feel like I’m just running on empty.",
    "I know I should be kinder to myself, but I just keep criticizing everything I do.",
    "My partner says I don’t listen enough, but I feel like they don’t understand me either.",
    "I quit smoking for two months, but last week I started again. I feel like I failed.",
    "I want to meet new people, but I get so nervous that I avoid social events altogether.",
    "I want to build better habits, but I never stick with them long enough to see results.",
]

SYSTEM_PROMPT = (
    "You are a supportive health coach using Motivational Interviewing (MI). "
    "Be non-judgmental, empathetic, and collaborative. Use open questions, reflections, and affirmations."
)

TUNED_PATH = "/mnt/gptcoaching_mi_training/runs/qwen2p5-3b-mi-dpo-merged"
BASE_PATH = "Qwen/Qwen2.5-3B-Instruct"

def load_model(path):
    tok = AutoTokenizer.from_pretrained(path, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True).eval()
    return tok, model

def generate_reply(tok, model, user_msg):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=180, temperature=0.7, top_p=0.9)
    # Strip special tokens properly
    reply = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    # Remove any leftover chat tags if model still generates them
    reply = re.sub(r"<\|.*?\|>", "", reply).strip()
    return reply

def main():
    tuned_tok, tuned_model = load_model(TUNED_PATH)
    base_tok, base_model = load_model(BASE_PATH)

    print("# MI-Coach vs Base Model — Expanded Comparison\n")
    for i, p in enumerate(PROMPTS, 1):
        tuned = generate_reply(tuned_tok, tuned_model, p)
        base = generate_reply(base_tok, base_model, p)
        print(f"## Prompt {i}\n> **User:** {p}\n")
        print(f"**Tuned (SFT+DPO)**\n\n{tuned}\n")
        print(f"**Base (Qwen2.5-3B-Instruct)**\n\n{base}\n")
        print("---\n")

if __name__ == "__main__":
    main()
