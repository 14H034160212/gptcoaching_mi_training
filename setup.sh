#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
echo "✅ Environment ready. Activate with: source .venv/bin/activate"
