# GPTCoach MI Training Starter

This repo scaffolds **data preparation + SFT + DPO** training for a multi-turn *Motivational Interviewing* (MI) style coach model using Hugging Face Transformers + TRL.

## What you get
- `data/` — schema + a tiny synthetic example dataset (JSONL) to verify the pipeline.
- `scripts/data_prep.py` — normalize MI datasets (MI-TAGS / AnnoMI / MI-Dataset) into a common JSONL format.
- `scripts/sft_train.py` — Supervised fine-tuning with TRL SFTTrainer. Supports LoRA + 4-bit.
- `scripts/dpo_train.py` — Preference optimization with TRL DPOTrainer.
- `scripts/infer_demo.py` — Run inference with memory + MI persona prompt.
- `scripts/metrics_mi.py` — Simple MI-style behavioral metrics (coverage of open-question/reflect/affirm, etc.).
- `configs/*.yaml` — Example hyper-parameters.

> ⚠️ You must provide your own dataset paths for MI-TAGS / AnnoMI etc. The included `example_mi_dialogs.jsonl` is **just for smoke tests** (not for real training).

## Install
```bash
pip install -U transformers datasets accelerate trl peft bitsandbytes torch torchvision torchaudio
# If CUDA is not available, install CPU wheels for torch.
```

## Data format (common JSONL)
Each line is a dict:
```json
{
  "dialog_id": "string",
  "turn_id": 7,
  "user_utt": "string",
  "coach_utt": "string",
  "mi_tags": ["open_question","reflection_simple","affirm"],
  "state_before": {},   # optional structured state (goal, barriers, wearable stats, etc.)
  "state_after":  {}    # optional updated state
}
```
You can include additional fields; unknown keys are ignored by the loader.

## Prepare data
Set your dataset file paths and run:
```bash
python scripts/data_prep.py   --mi_tags_csv /path/to/MI-TAGS.csv   --annomi_csv /path/to/AnnoMI.csv   --mi_dataset_json /path/to/Motivational-Interviewing-Dataset.json   --out_jsonl data/mi_unified_train.jsonl
```

## SFT (supervised fine-tuning)
```bash
python scripts/sft_train.py   --model_name_or_path meta-llama/Llama-3-8b-instruct   --train_file data/mi_unified_train.jsonl   --output_dir runs/sft-llama3-mi   --lora --bnb_4bit
```

## DPO (preference optimization)
Prepare a JSONL with pairs of responses (chosen vs rejected) per context/turn.
```bash
python scripts/dpo_train.py   --model_name_or_path runs/sft-llama3-mi   --pref_file data/mi_prefs.jsonl   --output_dir runs/dpo-llama3-mi   --lora --bnb_4bit
```

## Inference demo
```bash
python scripts/infer_demo.py --model_path runs/dpo-llama3-mi
```

## Metrics
```bash
python scripts/metrics_mi.py --pred_file runs/sample_outputs.jsonl
```

---

**Authoring notes**
- Keep persona + MI rules in the *system prompt*.
- Inject structured memory (user profile, wearable summary) as JSON before each generation.
- Track behavioral metrics (open Q, reflections, affirmations) for closed-loop training.


## Label normalization & higher-quality DPO
- Consolidate MI tags:
  ```bash
  python scripts/normalize_labels.py         --in_jsonl data/mi_unified_from_annomi.jsonl         --map_json data/mi_label_map.json         --out_jsonl data/mi_unified_from_annomi.norm.jsonl
  ```
  Then use the normalized file for SFT if you need tag-conditioned training.

- Generate prefs with a **quality margin** (skip near-ties):
  ```bash
  python scripts/make_prefs_from_annomi.py         --model_path runs/sft-llama3-mi-annomi         --input_jsonl data/mi_unified_from_annomi.jsonl         --out_jsonl data/mi_prefs.jsonl         --mode dual-style         --min_margin 0.35         --limit 2000
  ```

## Local demo (FastAPI)
```bash
export MODEL_PATH=runs/dpo-llama3-mi-annomi
uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8000 --reload
# POST http://localhost:8000/chat
# body:
# {
#   "history": [{"user":"I want to be more active.","coach":"What matters most about being active for you?"}],
#   "user_msg":"I'm too busy this week."
# }
```


---
## Model-based MITI/MISC Scoring
Use a HF classifier fine-tuned on MI behaviors:
```bash
python scripts/metrics_mi_model.py       --pred_file runs/sample_outputs.jsonl       --checkpoint /path/to/hf-mi-classifier       --labels open_question,reflection_simple,reflection_complex,affirm,summary,provide_info,ask_permission,directive       --out_csv runs/mi_scores.csv
```

## Front-end (Static HTML)
Start FastAPI (serves / and /chat):
```bash
export MODEL_PATH=runs/dpo-llama3-mi-annomi
uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000/
```

## Rasa Integration (Minimal)
See `integrations/rasa/README.md` for instructions. The custom action calls our FastAPI /chat.

## ConvLab-3 Bridge
See `integrations/convlab/README.md`. The bridge shows how to convert DST state/history to an MI prompt and call the model.


## Train a MI Behavior Classifier (for model-based scoring)
```bash
python scripts/train_mi_classifier.py       --train_file data/mi_unified_from_annomi.norm.jsonl       --labels open_question,reflection_simple,reflection_complex,affirm,summary,provide_info,ask_permission,directive       --model_name_or_path distilbert-base-uncased       --output_dir runs/mi_classifier       --num_train_epochs 3 --per_device_train_batch_size 16
```
Then point `metrics_mi_model.py` and `/score` endpoint to this checkpoint via `CLASSIFIER_PATH`.

## Experiment Loop (ConvLab-style)
```bash
python integrations/convlab/run_experiment.py       --model_path runs/dpo-llama3-mi-annomi       --n_dialogs 20 --max_turns 8       --classifier_path runs/mi_classifier   # optional
```
