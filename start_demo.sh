#!/bin/bash
# Canonical launcher for the gptcoaching MI demo (uvicorn, port 8081).
# GPU is fixed here (consolidated layout: demo -> GPU 5) so future launches
# never drift onto other people's cards.
#
# Secrets: app_demo.py requires MODEL_PATH and RESEND_API_KEY (and possibly more).
# MODEL_PATH is set below. Put RESEND_API_KEY (and any other secrets) in a
# gitignored .env file next to this script:  RESEND_API_KEY=re_xxx
cd /data/qbao775/gptcoaching_mi_training

export CUDA_VISIBLE_DEVICES=5
export MODEL_PATH=/data/qbao775/gptcoaching_mi_training/runs/qwen2p5-3b-mi-dpo-merged

# Load secrets (RESEND_API_KEY, etc.) if a .env is present.
if [ -r .env ]; then
    set -a
    . ./.env
    set +a
fi

exec /data/qbao775/miniconda3/bin/uvicorn scripts.app_demo:app --host 0.0.0.0 --port 8081
