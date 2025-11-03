#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DPO training on MI preference pairs.
pref_file JSONL format:
{"prompt":"...", "chosen":"...", "rejected":"..."}
"""
import argparse, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--pref_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--bnb_4bit", action="store_true")
    args = ap.parse_args()

    bnb_config = None
    if args.bnb_4bit:
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, quantization_config=bnb_config, device_map="auto")
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.lora:
        peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","v_proj","k_proj","o_proj"])
        model = get_peft_model(model, peft_cfg)

    ds = load_dataset("json", data_files={"train": args.pref_file})

    cfg = DPOConfig(output_dir=args.output_dir, per_device_train_batch_size=2, num_train_epochs=1, logging_steps=10, save_steps=200, bf16=True)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # implicit reference
        args=cfg,
        beta=0.1,
        train_dataset=ds["train"],
        tokenizer=tok,
        max_length=2048,
        max_prompt_length=1024
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
