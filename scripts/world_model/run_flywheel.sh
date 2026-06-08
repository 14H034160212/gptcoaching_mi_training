#!/usr/bin/env bash
# Continuous-improvement flywheel — one cycle. Safe to run on a schedule (cron).
# Harvests real user chat traffic -> pools with ESConv/synth silver -> real-AnnoMI-val-
# gated active loop -> exports the winner to transitions_prod.jsonl (server hot-reloads).
# Idempotent: a cycle that doesn't beat the current model changes nothing.
set -euo pipefail

REPO=/data/qbao775/gptcoaching_mi_training
PY=/data/qbao775/miniconda3/envs/qwen3-rl/bin/python
LOG="$REPO/runs/flywheel.log"

cd "$REPO"
mkdir -p runs
{
  echo "==================== flywheel cycle: $(date -u +%FT%TZ) ===================="
  CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/qbao775/.cache/huggingface PYTHONUNBUFFERED=1 \
    "$PY" -m scripts.world_model.continuous_improve 2>&1 \
    | grep -aE "\[harvest\]|\[continuous\]|loop\] (baseline|round 1:|final|exported)|current_best|jepa_challenger|production_model" || true
  echo "---- cycle done: $(date -u +%FT%TZ) ----"
} >> "$LOG" 2>&1
