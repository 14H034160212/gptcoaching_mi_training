#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DPO training on MI preference pairs.

Each line in the preference file (JSONL) should have:
{"prompt": "...", "chosen": "...", "rejected": "..."}

Example usage:
python scripts/dpo_train.py \
  --model_name_or_path outputs/qwen2p5-3b-mi-sft \
  --pref_file data/mi_prefs.jsonl \
  --output_dir runs/qwen2p5-3b-mi-dpo \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --lr 2e-5 \
  --logging_steps 10 \
  --save_steps 200 \
  --wandb --wandb_project mi-coach-dpo \
  --lora --bnb_4bit
"""
import argparse, json, time
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model

try:
    import wandb
except ImportError:
    wandb = None


def main():
    ap = argparse.ArgumentParser()
    # ---------------- Required ----------------
    ap.add_argument("--model_name_or_path", required=True, help="Path or model name (e.g., runs/sft-llama3-mi)")
    ap.add_argument("--pref_file", required=True, help="JSONL file with prompt/chosen/rejected triples")
    ap.add_argument("--output_dir", required=True, help="Directory to save model and logs")

    # ---------------- Optional Training Args ----------------
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    # ---------------- LoRA / Quantization ----------------
    ap.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning")
    ap.add_argument("--bnb_4bit", action="store_true", help="Use 4-bit quantization (QLoRA)")

    # ---------------- WandB Logging ----------------
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="mi-coach-dpo")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()

    # ---------------- Model loading ----------------
    bnb_config = None
    if args.bnb_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

    # Trainable model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Frozen reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto"
    )
    ref_model.requires_grad_(False)

    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Apply LoRA if requested
    if args.lora:
        peft_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        model = get_peft_model(model, peft_cfg)

    # ---------------- Dataset loading ----------------
    ds = load_dataset("json", data_files={"train": args.pref_file})
    print(f"Loaded {len(ds['train'])} preference pairs from {args.pref_file}")

    # ---------------- WandB setup ----------------
    report_to = ["wandb"] if args.wandb else ["none"]
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Run `pip install wandb`.")
        run_name = args.wandb_run_name or f"dpo_{args.model_name_or_path.split('/')[-1]}_{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
        wandb.config.update({
            "lora": args.lora,
            "bnb_4bit": args.bnb_4bit,
            "training_type": "DPO"
        })
        print(f"🔗 WandB run: {wandb.run.url}")

    # ---------------- DPO Configuration ----------------
    cfg = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=report_to,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        bf16=True,               # or set fp16=True if hardware doesn’t support bf16
        max_length=args.max_length,
        beta=0.1,                # regularization strength
        loss_type="sigmoid",     # can also try "hinge"
    )

    # ---------------- Trainer ----------------
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=ds["train"],
    )

    # ---------------- Training ----------------
    print("🚀 Starting DPO fine-tuning ...")
    train_result = trainer.train()

    if args.wandb:
        # log training metrics to W&B
        wandb.log({
            "train_loss": train_result.training_loss,
            "epochs": args.num_train_epochs,
            "lr": args.lr,
            "batch_size": args.per_device_train_batch_size,
        })

    # ---------------- Save ----------------
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.wandb:
        wandb.finish()

    print(f"✅ DPO fine-tuning complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()