#!/usr/bin/env python3
"""
Is the merged DPO Qwen a better talk-type labeler than the trained classifier?

The whole synth/tabular pipeline is capped by the silver labeler's quality
(mpnet talk-type clf ~0.59 macro-F1). If an LLM-as-judge (few-shot) beats that
on gold AnnoMI val, then re-labeling synthetic data with it raises the data
ceiling. If not, more synthetic data labeled the same way can't break 0.59.

Measures few-shot LLM-judge macro-F1 on AnnoMI val (client response + coach
context -> change/sustain/neutral) against gold labels.

  python -m scripts.world_model.llm_judge_talktype --n 200
"""
import argparse
import json
import re

import torch

TALKS = ["change", "sustain", "neutral"]

JUDGE_SYS = (
    "You are an expert annotator of Motivational Interviewing sessions. "
    "Classify the CLIENT's utterance into exactly one talk-type:\n"
    "- change: language favoring change (desire, ability, reasons, need, commitment, steps taken)\n"
    "- sustain: language favoring the status quo (reasons not to change, can't, won't, it's fine)\n"
    "- neutral: neither — small talk, factual, off-topic, or ambivalent with no lean\n"
    "Answer with ONLY one word: change, sustain, or neutral."
)
FEWSHOT = [
    ("Coach: What would be different if you cut back?\nClient: I really want to be there for my kids, so I think I need to stop.", "change"),
    ("Coach: How do you feel about your drinking?\nClient: Honestly it's fine, I don't see the problem and I'm not changing anything.", "sustain"),
    ("Coach: Thanks for coming in today.\nClient: Sure, no problem, traffic wasn't too bad.", "neutral"),
]


def coach_utt(r):
    for t in reversed(r["history"]):
        if t["role"] == "therapist" and t.get("text", "").strip():
            return t["text"]
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/qwen2p5-3b-mi-dpo-merged")
    ap.add_argument("--data", default="data/world_model/transitions.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    val = [r for r in rows if r["split"] == "val" and r.get("future_text", "").strip()]
    if len(val) > args.n:
        step = len(val) / args.n
        val = [val[int(i * step)] for i in range(args.n)]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).eval().to("cuda")

    def build(r):
        msgs = [{"role": "system", "content": JUDGE_SYS}]
        for ctx, lab in FEWSHOT:
            msgs.append({"role": "user", "content": ctx})
            msgs.append({"role": "assistant", "content": lab})
        msgs.append({"role": "user", "content": f"Coach: {coach_utt(r)}\nClient: {r['future_text']}"})
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    preds, golds = [], []
    for s in range(0, len(val), args.bs):
        batch = val[s:s + args.bs]
        texts = [build(r) for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        for i, r in enumerate(batch):
            new = tok.decode(out[i][enc["input_ids"].shape[1]:], skip_special_tokens=True).lower()
            lab = next((t for t in TALKS if t in new), "neutral")
            preds.append(lab); golds.append(r["next_talk"])
        print(f"  judged {min(s+args.bs,len(val))}/{len(val)}")

    def macro_f1(preds, golds):
        per = {}
        for c in TALKS:
            tp = sum(p == c and g == c for p, g in zip(preds, golds))
            fp = sum(p == c and g != c for p, g in zip(preds, golds))
            fn = sum(p != c and g == c for p, g in zip(preds, golds))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            per[c] = round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0
        acc = sum(p == g for p, g in zip(preds, golds)) / len(golds)
        return round(acc, 4), round(sum(per.values()) / 3, 4), per

    acc, f1, per = macro_f1(preds, golds)
    print(f"\n[llm-judge] n={len(val)} acc={acc} macro_f1={f1} per_class_f1={per}")
    print(f"[llm-judge] vs mpnet talk-type clf 0.59 -> delta {f1 - 0.59:+.3f}")


if __name__ == "__main__":
    main()
