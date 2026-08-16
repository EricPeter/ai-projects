#!/usr/bin/env bash
set -euo pipefail

echo "Step 1: Prepare dataset"
uv run python data/prepare_dataset.py

echo "Step 2: Train LoRA"
uv run python src/train_lora.py

echo "Step 3: Evaluate (base vs tuned)"
uv run python src/evaluate.py

echo "Pipeline complete. See outputs/eval_report.json"
