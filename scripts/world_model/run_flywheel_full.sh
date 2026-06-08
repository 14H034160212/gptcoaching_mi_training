#!/usr/bin/env bash
# FULL flywheel cycle (weekly) — like run_flywheel.sh but ALSO regenerates in-domain
# synthetic MI data targeting the CURRENT weak class (via the DPO Qwen), accumulating
# it over time. Heavier (runs 3B generation) -> schedule weekly, not daily.
# Uses BASE python (proven for the merged Qwen generation + classifier loading).
set -euo pipefail

REPO=/data/qbao775/gptcoaching_mi_training
PY=/data/qbao775/miniconda3/bin/python          # base env: proven for Qwen-3B generation
LOG="$REPO/runs/flywheel.log"

cd "$REPO"
mkdir -p runs
{
  echo "================ FULL flywheel cycle (regen-synth): $(date -u +%FT%TZ) ================"
  CUDA_VISIBLE_DEVICES=0 HF_HOME=/data/qbao775/.cache/huggingface PYTHONUNBUFFERED=1 \
    "$PY" -m scripts.world_model.continuous_improve --regen-synth --synth-n 500 2>&1 \
    | grep -aE "\[harvest\]|weak class|appended new synth|\[continuous\]|loop\] (baseline|round 1:|final|exported)|current_best|jepa_challenger" || true
  echo "---- full cycle done: $(date -u +%FT%TZ) ----"
} >> "$LOG" 2>&1
