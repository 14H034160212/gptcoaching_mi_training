#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_prefs_from_annomi.py
Generate preference pairs for DPO from unified MI JSONL (e.g., mi_unified_from_annomi.jsonl).

Two modes:
1) "dual-style": produce A=MI-compliant (persona prompt), B=anti-MI (anti-persona) as the contrast.
2) "dual-sample": produce two samples from the same MI persona (different sampling seeds/temperatures).

Then score both with MI heuristics and pick chosen/rejected automatically.
If the scores tie, fall back to a lexicographic tiebreaker for determinism.

Usage:
python scripts/make_prefs_from_annomi.py \
  --model_path runs/sft-llama3-mi-annomi \
  --input_jsonl /path/to/mi_unified_from_annomi.jsonl \
  --out_jsonl data/mi_prefs.jsonl \
  --mode dual-style \
  --limit 2000

Requirements: transformers, torch
"""
import argparse, json, random, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM

SYSTEM_MI = (
  "You are a supportive health coach using Motivational Interviewing (MI). "
  "Be non-judgmental; prefer open questions, reflective listening, affirmations; "
  "ask permission before advice; avoid directives. Keep responses concise."
)
SYSTEM_ANTI = (
  "You are a directive advisor. Give prescriptive, imperative instructions. "
  "Avoid open questions and reflections. Be brief and to-the-point."
)

def render_prompt(system, state, user):
    return f"<|system|>\n{system}\n</|system|>\n" \
           f"<|context|>\nSTATE={json.dumps(state, ensure_ascii=False)}\n</|context|>\n" \
           f"<|user|>\n{user}\n</|user|>\n<|assistant|>\n"

# Simple heuristics borrowed from metrics_mi.py
OPEN_Q = re.compile(r"\?\s*$")
AFFIRM = re.compile(r"\b(good job|that makes sense|you've been|sounds like|i appreciate)\b", re.I)
REFLECT = re.compile(r"\b(you feel|you're saying|it sounds like|so you|you seem)\b", re.I)
SUMMARY = re.compile(r"\b(let me summarize|to recap|what we discussed|summary)\b", re.I)
DIRECTIVE = re.compile(r"\b(you must|you should|do this|just do|stop|start immediately)\b", re.I)

def mi_score(text: str) -> float:
    score = 0.0
    if OPEN_Q.search(text.strip()): score += 1.0
    if AFFIRM.search(text): score += 0.7
    if REFLECT.search(text): score += 0.8
    if SUMMARY.search(text): score += 0.5
    if DIRECTIVE.search(text): score -= 0.8
    # length penalty for rambling
    if len(text.split()) > 200: score -= 0.3
    return score

def generate(model, tok, prompt, temperature=0.7, top_p=0.9, max_new_tokens=200, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, pad_token_id=tok.eos_token_id
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    # Extract assistant segment if special tokens were included
    return text.split("</|assistant|>")[0].split("<|assistant|>")[-1].strip() if "<|assistant|>" in text else text.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input_jsonl", required=True, help="Unified JSONL (e.g., mi_unified_from_annomi.jsonl)")
    ap.add_argument("--out_jsonl", required=True, help="Output preference pairs JSONL")
    ap.add_argument("--mode", choices=["dual-style","dual-sample"], default="dual-style")
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--min_margin", type=float, default=0.3, help="minimum MI-score margin between chosen and rejected; skip if below")
    ap.add_argument("--skip_empty_user", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")

    written = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as fout, open(args.input_jsonl, "r", encoding="utf-8") as fin:
        for line in fin:
            ex = json.loads(line)
            if args.skip_empty_user and not ex.get("user_utt"):
                continue
            user = ex.get("user_utt","") or "I'd like to discuss my activity goals."
            state = ex.get("state_before",{}) or {}

            if args.mode == "dual-style":
                p_mi = render_prompt(SYSTEM_MI, state, user)
                p_anti = render_prompt(SYSTEM_ANTI, state, user)
                resp_a = generate(model, tok, p_mi, temperature=0.7, top_p=0.9, max_new_tokens=180, seed=42)
                resp_b = generate(model, tok, p_anti, temperature=0.9, top_p=0.95, max_new_tokens=180, seed=43)
            else:
                # dual-sample - same MI persona, two different samplings
                p = render_prompt(SYSTEM_MI, state, user)
                resp_a = generate(model, tok, p, temperature=0.7, top_p=0.9, max_new_tokens=180, seed=42)
                resp_b = generate(model, tok, p, temperature=1.0, top_p=0.95, max_new_tokens=180, seed=43)

            sa, sb = mi_score(resp_a), mi_score(resp_b)
            margin = abs(sa - sb)
            if margin < args.min_margin:
                continue
            if sa > sb:
                chosen, rejected = resp_a, resp_b
            elif sb > sa:
                chosen, rejected = resp_b, resp_a
            else:
                chosen, rejected = (resp_a, resp_b) if resp_a < resp_b else (resp_b, resp_a)

            prompt = render_prompt(SYSTEM_MI, state, user)
            fout.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.limit:
                break

    print(f"Wrote {written} preference pairs to {args.out_jsonl}")

if __name__ == "__main__":
    main()
