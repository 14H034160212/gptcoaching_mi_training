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

## Conda Environment Setup
```
cd /mnt
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /mnt/miniconda3
echo 'export PATH="/mnt/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda create -n gptcoach python=3.10
conda activate gptcoach
pip install -r /mnt/gptcoaching_mi_training/requirements.txt
```

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
### Using AnnoMI-full to generate training set.
```
python scripts/data_prep.py  --annomi_csv ./data/AnnoMI-full.csv --out_jsonl data/mi_unified_from_annomi_full.jsonl
```

### Using AnnoMI-simple to generate validation set.
```bash
python scripts/data_prep.py  --annomi_csv ./data/AnnoMI-simple.csv --out_jsonl data/mi_unified_from_annomi_simple.jsonl
```

## SFT (supervised fine-tuning)
```bash
python scripts/sft_train.py \
  --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
  --train_file data/mi_unified_from_annomi_full.jsonl \
  --eval_file data/mi_unified_from_annomi_simple.jsonl \
  --output_dir outputs/qwen2p5-3b-mi-sft \
  --num_train_epochs 3 \
  --per_device_train_batch_size 20 \
  --lr 2e-5 \
  --eval_steps 200 \
  --save_steps 200 \
  --wandb --wandb_project mi-coach-sft --wandb_run_name qwen2p5_3b_sft \
  --bnb_4bit --lora
```

## DPO (preference optimization) Dataset Generation
This part will generate both positive and negative samples (MI-style coaching response).
```
python scripts/make_dpo_prefs_v2.py \
  --sft_file data/mi_unified_from_annomi_full.jsonl \
  --out_file data/mi_prefs.jsonl \
  --seed 123 \
  --max_samples 5000
```

## DPO (preference optimization)
Prepare a JSONL with pairs of responses (chosen vs rejected) per context/turn.
```bash
python scripts/dpo_train.py \
  --model_name_or_path outputs/qwen2p5-3b-mi-sft/checkpoint-510 \
  --pref_file data/mi_prefs.jsonl \
  --output_dir runs/qwen2p5-3b-mi-dpo \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --lr 2e-5 \
  --logging_steps 10 \
  --save_steps 200 \
  --lora --bnb_4bit \
  --wandb --wandb_project mi-coach-dpo --wandb_run_name qwen2p5_3b_dpo
```

<!-- ## Inference demo
```bash
python scripts/infer_demo.py --model_path runs/qwen2p5-3b-mi-dpo
``` -->

<!-- ## Metrics
```bash
python scripts/metrics_mi.py --pred_file runs/sample_outputs.jsonl
``` -->

<!-- ---

**Authoring notes**
- Keep persona + MI rules in the *system prompt*.
- Inject structured memory (user profile, wearable summary) as JSON before each generation.
- Track behavioral metrics (open Q, reflections, affirmations) for closed-loop training. -->


<!-- ## Label normalization & higher-quality DPO
- Consolidate MI tags:
  ```bash
  python scripts/normalize_labels.py         --in_jsonl data/mi_unified_from_annomi.jsonl         --map_json data/mi_label_map.json         --out_jsonl data/mi_unified_from_annomi.norm.jsonl
  ```
  Then use the normalized file for SFT if you need tag-conditioned training.

- Generate prefs with a **quality margin** (skip near-ties):
  ```bash
  python scripts/make_prefs_from_annomi.py         --model_path runs/sft-llama3-mi-annomi         --input_jsonl data/mi_unified_from_annomi.jsonl         --out_jsonl data/mi_prefs.jsonl         --mode dual-style         --min_margin 0.35         --limit 2000
  ``` -->

## Local demo (FastAPI)
```bash
export HF_HOME=/mnt/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/.cache/huggingface/transformers

# merged model dir from your DPO (or SFT) run
export MODEL_PATH=/mnt/gptcoaching_mi_training/runs/qwen2p5-3b-mi-dpo-merged
uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8000 --reload
# POST http://localhost:8000/chat
# body:
# {
#   "history": [{"user":"I want to be more active.","coach":"What matters most about being active for you?"}],
#   "user_msg":"I'm too busy this week."
# }
```

---

## Kerrio.AI - Digital Cognitive Clinic

Kerrio.AI implements a Mayo Clinic-inspired 7-stage clinical journey for cognitive optimization. This is NOT a chatbot - it's a diagnostic-first digital cognitive clinic.

### Key Concepts (from Mayo Clinic Model)
- **Accurate diagnosis is the foundation of effective treatment**
- **Understanding is a prerequisite for permanent change**
- **Client History and Clinician's Notes are maintained separately**

### The 7-Stage Clinical Journey
1. **Registration** - Client validated as invited guest
2. **History Collection** - Three Pillars (History, Psychology/Philosophy, Physiology)
3. **Consultation** - Clarify ambiguities, uncover blind spots
4. **Diagnosis** - Build Cognitive Wiring Map, explain WHY the problem exists
5. **Proposal** - Personalized treatment plan based on diagnosis
6. **Treatment** - Cognitive Rewiring Maps (Patent Pending)
7. **Monitoring** - Longitudinal progress assessment

### Running Kerrio.AI Demo

```bash
# 1. Set environment variables
export HF_HOME=/mnt/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/.cache/huggingface/transformers
export MODEL_PATH=/mnt/gptcoaching_mi_training/runs/qwen2p5-3b-mi-dpo-merged

# 2. Start the server
uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8000 --reload

# 3. Open browser
# http://localhost:8000/
```

### Kerrio API Endpoints

#### Core Chat
- `POST /api/chat` - Send message and get AI response
  ```json
  {"user_id": "demo_user", "user_msg": "I feel stuck in my career"}
  ```

#### Journey Management
- `GET /api/journey/{user_id}` - Get current journey status and stage
- `POST /api/journey/advance` - Advance to next stage (if requirements met)
- `GET /api/journey/prompts/{user_id}` - Get suggested prompts for current stage

#### Three Pillars History
- `GET /api/journey/history/{user_id}` - Get client's collected history across 3 pillars
- `GET /api/journey/notes/{user_id}` - Get clinician's notes (AI observations)

#### Diagnosis & Treatment
- `GET /api/journey/diagnosis/{user_id}` - Generate/retrieve diagnosis
- `POST /api/journey/diagnosis/confirm/{user_id}` - Confirm understanding of diagnosis
- `GET /api/journey/treatment/{user_id}` - Get treatment proposal with Cognitive Rewiring Map
- `POST /api/journey/treatment/accept/{user_id}` - Accept treatment plan
- `POST /api/journey/treatment/progress/{user_id}` - Update treatment progress

#### Educational Videos
- `GET /api/journey/videos` - Get all educational video library
- `GET /api/journey/videos/{video_id}` - Get specific video details

#### Full Profile
- `GET /api/journey/full-profile/{user_id}` - Complete client profile with all data

#### Cognitive Map
- `POST /api/cogmap` - Build cognitive map from session
- `GET /api/map/{user_id}` - Get cognitive wiring map for user

### Web Interface Features

The web UI (`web/index.html`) includes:

1. **Journey Progress Bar** - Shows current stage (Registration → Monitoring)
2. **Chat Interface** - Conversational interaction with Kerrio
3. **Three Pillars Panel** - View collected history across:
   - History Pillar (life events, patterns)
   - Psychology/Philosophy Pillar (beliefs, values)
   - Physiology Pillar (sleep, stress, health)
4. **Diagnosis Panel** - View:
   - Core Constraints
   - Bottlenecks
   - Root Causes
   - Explanation
   - Recommended Educational Videos
   - "I Understand My Diagnosis" confirmation button
5. **Treatment Panel** - View:
   - Current Wiring patterns
   - Target Wiring (desired state)
   - Rewiring Steps
   - Progress bar
   - "Accept Treatment Plan" button
6. **Cognitive Map Visualization** - Interactive graph with Cytoscape.js

### Testing the Kerrio Journey Module

```bash
# Run the kerrio_journey.py module directly for testing
python scripts/kerrio_journey.py

# This will:
# - Create a test profile
# - Show the stage-specific system prompt
# - Test the Diagnostic Engine
# - Display sample diagnosis output
```

### Key Files

| File | Description |
|------|-------------|
| `scripts/kerrio_journey.py` | Core journey management, data structures, diagnostic & rewiring engines |
| `scripts/app_demo.py` | FastAPI server with all endpoints |
| `scripts/cogmap_utils.py` | Heuristic cognitive map builder |
| `web/index.html` | Full-featured web interface |
| `runs/kerrio_profiles/` | Persistent client profile storage (JSON) |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Interface                            │
│  (Chat, Journey Bar, Three Pillars, Diagnosis, Treatment)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI (app_demo.py)                       │
│  - /api/chat, /api/journey/*, /api/cogmap, /api/map/*           │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ KerriJourney    │ │ DiagnosticEngine│ │ CognitiveRewiring│
│ Manager         │ │                 │ │ Engine           │
│ - 7 stages      │ │ - Root causes   │ │ - Rewiring maps  │
│ - 3 pillars     │ │ - Bottlenecks   │ │ - Treatment steps│
│ - Profile I/O   │ │ - Video recs    │ │                  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          └───────────────────┴───────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM (Qwen/fine-tuned model)                   │
│            Stage-specific system prompts                         │
└─────────────────────────────────────────────────────────────────┘
```


<!-- ---
## Model-based MITI/MISC Scoring
Use a HF classifier fine-tuned on MI behaviors:
```bash
python scripts/metrics_mi_model.py       --pred_file runs/sample_outputs.jsonl       --checkpoint /path/to/hf-mi-classifier       --labels open_question,reflection_simple,reflection_complex,affirm,summary,provide_info,ask_permission,directive       --out_csv runs/mi_scores.csv
``` -->

<!-- ## Front-end (Static HTML)
Start FastAPI (serves / and /chat):
```bash
export MODEL_PATH=runs/dpo-llama3-mi-annomi
uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000/
``` -->

<!-- ## Rasa Integration (Minimal)
See `integrations/rasa/README.md` for instructions. The custom action calls our FastAPI /chat.

## ConvLab-3 Bridge
See `integrations/convlab/README.md`. The bridge shows how to convert DST state/history to an MI prompt and call the model.
 -->

<!-- ## Train a MI Behavior Classifier (for model-based scoring)
```bash
python scripts/train_mi_classifier.py       --train_file data/mi_unified_from_annomi.norm.jsonl       --labels open_question,reflection_simple,reflection_complex,affirm,summary,provide_info,ask_permission,directive       --model_name_or_path distilbert-base-uncased       --output_dir runs/mi_classifier       --num_train_epochs 3 --per_device_train_batch_size 16
```
Then point `metrics_mi_model.py` and `/score` endpoint to this checkpoint via `CLASSIFIER_PATH`.

## Experiment Loop (ConvLab-style)
```bash
python integrations/convlab/run_experiment.py       --model_path runs/dpo-llama3-mi-annomi       --n_dialogs 20 --max_turns 8       --classifier_path runs/mi_classifier   # optional
``` -->
