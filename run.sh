#!/usr/bin/env bash
set -e

# 1) Create & activate venv
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) Login to HF to access LLaMA weights
# hf-cli login

# 4) Launch training via accelerate
accelerate launch --config_file accelerate_config.yaml train.py \
  --data-dir ./data \
  --output-dir ./outputs \
  --epochs 20 \
  --batch-size 1 \
  --lr 2e-5

echo "✅ Training complete. Check ./outputs"
