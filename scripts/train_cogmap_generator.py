#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_cogmap_generator.py

Supervised fine-tuning: dialogue -> cognitive map JSON.

Assumes a JSONL train file where each line is:
{
  "dialogue": "USER: ...\\nCOACH: ...",
  "map": { "nodes": [...], "edges": [...] }
}

Usage:
  python scripts/train_cogmap_generator.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --train_file data/cogmap_train.jsonl \
    --output_dir runs/qwen2p5-cogmap-sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2
"""
import argparse, json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

SYSTEM = """
You are an expert MI coach and cognitive mapping model.
Given a dialogue between USER and COACH, output ONLY a JSON object with:
{
  "nodes": [...],
  "edges": [...]
}
No explanations, no extra text.
""".strip()


def format_example(tok, ex):
    dialogue = ex["dialogue"]
    map_json = ex["map"]
    target = json.dumps(map_json, ensure_ascii=False)

    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",
         "content": "Here is the conversation:\n\n" + dialogue + "\n\nExtract the cognitive map JSON."},
        {"role": "assistant", "content": target},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
    )

    ds = load_dataset("json", data_files={"train": args.train_file})["train"]
    ds = ds.map(lambda ex: format_example(tok, ex), remove_columns=ds.column_names)

    def tokenize_fn(ex):
        return tok(ex["text"], truncation=True, max_length=2048)
    ds_tok = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        evaluation_strategy="no",
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
