#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_experiment.py
Minimal conversation loop to evaluate the MI LLM policy using a toy user simulator,
and compute metrics (open-question rate, heuristic MI coverage, and optional classifier-based score).

Usage:
python integrations/convlab/run_experiment.py \
  --model_path runs/dpo-llama3-mi-annomi \
  --n_dialogs 20 --max_turns 8 \
  --classifier_path runs/mi_classifier  # optional
"""
import argparse, json, random, re, os
from typing import List, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

SYSTEM = (
  "You are a supportive health coach using Motivational Interviewing (MI). "
  "Be non-judgmental; use open questions, reflective listening, and affirmations; "
  "ask permission before giving advice; avoid directives. Keep it concise."
)

OPEN_Q = re.compile(r"\?\s*$")
AFFIRM = re.compile(r"\b(good job|that makes sense|you've been|sounds like|i appreciate)\b", re.I)
REFLECT = re.compile(r"\b(you feel|you're saying|it sounds like|so you|you seem)\b", re.I)
SUMMARY = re.compile(r"\b(let me summarize|to recap|what we discussed|summary)\b", re.I)

def mi_flags(text: str):
    flags = {"open_q":0, "affirm":0, "reflect":0, "summary":0}
    if OPEN_Q.search(text.strip()): flags["open_q"]=1
    if AFFIRM.search(text): flags["affirm"]=1
    if REFLECT.search(text): flags["reflect"]=1
    if SUMMARY.search(text): flags["summary"]=1
    return flags

def build_prompt(history: List[Tuple[str,str]], user_msg: str):
    ctx = "\n".join([f"USER: {u}\nCOACH: {a}" for u,a in history])
    return f"<|system|>\n{SYSTEM}\n</|system|>\n{ctx}\nUSER: {user_msg}\nASSISTANT:"

def user_simulator(state):
    # toy simulator that alternates goals/barriers
    intent = random.choice(["report_progress","discuss_barrier","set_goal","reflect_feelings"])
    if intent=="report_progress":
        return "I managed a short walk yesterday."
    if intent=="discuss_barrier":
        return "My schedule is hectic and I feel tired."
    if intent=="set_goal":
        return "I'd like to set a small goal for this week."
    return "I feel stuck and unsure where to begin."
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_dialogs", type=int, default=20)
    ap.add_argument("--max_turns", type=int, default=8)
    ap.add_argument("--classifier_path")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto").eval()

    cls_tok, cls_model, cls_labels = None, None, None
    if args.classifier_path:
        try:
            cls_tok = AutoTokenizer.from_pretrained(args.classifier_path, use_fast=True)
            cls_model = AutoModelForSequenceClassification.from_pretrained(args.classifier_path, device_map="auto").eval()
            try:
                with open(os.path.join(args.classifier_path,"labels.json"),"r") as f:
                    cls_labels = json.load(f)
            except Exception:
                pass
        except Exception as e:
            print(f"[warn] classifier not loaded: {e}")

    stats = {"turns":0, "open_q":0, "affirm":0, "reflect":0, "summary":0, "cls_mi_score_sum":0.0, "cls_count":0}

    for d in range(args.n_dialogs):
        history = []
        user_msg = "I want to be more active."
        for t in range(args.max_turns):
            # coach
            prompt = build_prompt(history, user_msg)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=160, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0], skip_special_tokens=True)
            coach = text.split("ASSISTANT:")[-1].strip()

            flags = mi_flags(coach)
            for k,v in flags.items():
                stats[k]+=v
            stats["turns"]+=1

            if cls_model and cls_tok:
                enc = cls_tok([coach], return_tensors="pt", padding=True, truncation=True, max_length=512).to(cls_model.device)
                with torch.no_grad():
                    logits = cls_model(**enc).logits
                    probs = F.softmax(logits, dim=-1).cpu().tolist()[0]
                # simple MI score: sum of non-directive probs - 0.7*directive
                if cls_labels and "directive" in cls_labels:
                    did = cls_labels.index("directive")
                    mi_score = sum(p for i,p in enumerate(probs) if i!=did) - 0.7*probs[did]
                else:
                    mi_score = float(sum(probs)/len(probs))
                stats["cls_mi_score_sum"] += mi_score
                stats["cls_count"] += 1

            history.append((user_msg, coach))
            # user simulator reacts
            user_msg = user_simulator({})

    # report
    oq_rate = stats["open_q"]/max(1,stats["turns"])
    affirm_rate = stats["affirm"]/max(1,stats["turns"])
    reflect_rate = stats["reflect"]/max(1,stats["turns"])
    summary_rate = stats["summary"]/max(1,stats["turns"])
    cls_avg = stats["cls_mi_score_sum"]/max(1,stats["cls_count"])
    print(json.dumps({
        "n_dialogs": args.n_dialogs,
        "turns": stats["turns"],
        "open_question_rate": oq_rate,
        "affirm_rate": affirm_rate,
        "reflect_rate": reflect_rate,
        "summary_rate": summary_rate,
        "classifier_mi_score_avg": cls_avg
    }, indent=2))

if __name__ == "__main__":
    main()
