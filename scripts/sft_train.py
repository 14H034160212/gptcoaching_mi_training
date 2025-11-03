#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFT for MI-style multi-turn coach with TRL.
"""
import argparse, json, os, random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

def build_prompt(ex):
    sys_prompt = (
        "You are a supportive health coach using Motivational Interviewing (MI). "
        "Be non-judgmental; prefer open questions, reflective listening, and affirmations. "
        "Use the structured state and wearable summary when relevant."
    )
    state = ex.get("state_before", {})
    user = ex.get("user_utt","")
    target = ex.get("coach_utt","")
    return f"<|system|>\n{sys_prompt}\n</|system|>\n" \
           f"<|context|>\nSTATE={json.dumps(state, ensure_ascii=False)}\n</|context|>\n" \
           f"<|user|>\n{user}\n</|user|>\n<|assistant|>\n{target}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--bnb_4bit", action="store_true")
    args = ap.parse_args()

    bnb_config = None
    if args.bnb_4bit:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.lora:
        peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj","k_proj","o_proj"])
        model = get_peft_model(model, peft_cfg)

    ds = load_dataset("json", data_files={"train": args.train_file})
    def map_fn(ex):
        return {"text": build_prompt(ex)}
    ds = ds.map(map_fn)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        bf16=True
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds["train"],
        args=cfg,
        packing=False,
        dataset_text_field="text"
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
