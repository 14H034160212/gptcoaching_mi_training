#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SFT for MI-style multi-turn coach with TRL + Weights & Biases logging.
"""
import argparse, json, os, random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig

# 可选：只有在显式开启 wandb 时才导入，避免环境无 wandb 报错
try:
    import wandb
except Exception:
    wandb = None

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

    # ===== W&B 相关 =====
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="mi-coach-sft")
    ap.add_argument("--wandb_run_name", type=str, default=None)

    # ===== 验证相关（可选）=====
    ap.add_argument("--eval_file", type=str, default=None, help="Optional eval JSON file")
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_total_limit", type=int, default=2)

    args = ap.parse_args()

    bnb_config = None
    if args.bnb_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # （可选）LoRA：用 TRL 时建议直接用 base model + peft config 传入 SFTTrainer 也可以
    if args.lora:
        from peft import LoraConfig, get_peft_model
        peft_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj","v_proj","k_proj","o_proj"]
        )
        model = get_peft_model(model, peft_cfg)

    # ===== 加载数据 =====
    ds = load_dataset("json", data_files={"train": args.train_file})
    def map_fn(ex):
        return {"text": build_prompt(ex)}
    ds = ds.map(map_fn)

    eval_dataset = None
    evaluation_strategy = "no"
    if args.eval_file:
        ds_eval = load_dataset("json", data_files={"eval": args.eval_file})
        ds_eval = ds_eval.map(map_fn)
        eval_dataset = ds_eval["eval"]
        evaluation_strategy = "steps"

    # ===== 启用/配置 W&B =====
    report_to = ["wandb"] if args.wandb else ["none"]
    if args.wandb:
        if wandb is None:
            raise ImportError("`--wandb` 被启用，但未安装 wandb。请先 `pip install wandb`。")
        # HF Trainer 会自动集成 W&B；为方便记录更多元数据，可手动 init 一下
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_name_or_path": args.model_name_or_path,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "learning_rate": args.lr,
                "lora": args.lora,
                "bnb_4bit": args.bnb_4bit,
            }
        )
        # 可选：记录若干样例 prompt，便于在 W&B 中查看数据
        if "train" in ds and len(ds["train"]) > 0:
            preview = [ds["train"][i]["text"][:400] for i in range(min(3, len(ds["train"])))]
            wandb.config.update({"sample_prompts": preview}, allow_val_change=True)

    # ===== 训练配置（含 W&B & Eval）=====
    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.lr,

        # --- 日志/保存 ---
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # --- 训练细节 ---
        gradient_accumulation_steps=4,
        warmup_ratio=0.03,
        bf16=True,

        # --- 上报到 W&B ---
        report_to=["wandb"] if args.wandb else ["none"],

        # --- 评估（新版字段名）---
        eval_strategy="steps" if args.eval_file else "no",
        eval_steps=args.eval_steps if args.eval_file else None,
        load_best_model_at_end=bool(args.eval_file),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=eval_dataset,
        args=cfg,
    )

    # 可选：梯度/参数监控（较费性能，可按需开启）
    if args.wandb and wandb is not None:
        wandb.watch(model, log="all", log_freq=max(1, args.logging_steps))

    trainer.train()

    # 结束前保存
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.wandb and wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
