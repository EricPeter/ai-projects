#!/usr/bin/env bash
# Full pipeline setup + server smoke-test.
# Run from the project root: bash scripts/run_pipeline.sh
set -euo pipefail

echo "=== Luganda STT — Pipeline Setup ==="

# 1. Build spell dictionary (uses whatever data is already present)
echo ""
echo "Step 1: Build spell checker dictionary"
uv run python scripts/build_spell_dict.py

# 2. Optionally train n-gram LM if kenlm is available
echo ""
echo "Step 2: Train n-gram LM (skipped if kenlm not installed)"
if command -v lmplz &>/dev/null; then
  uv run python scripts/train_ngram_lm.py
else
  echo "  kenlm not found — skipping (install from https://github.com/kpu/kenlm)"
fi

# 3. Start server in background and run a quick health check
echo ""
echo "Step 3: Start API server"
uv run uvicorn src.server:app --host 0.0.0.0 --port 8024 &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null || true" EXIT

echo "  Waiting for server..."
for i in $(seq 30); do
  if curl -sf http://localhost:8024/health > /dev/null 2>&1; then
    echo "  Server ready."
    break
  fi
  sleep 2
done

echo ""
echo "Step 4: Health check"
curl -s http://localhost:8024/health | python3 -m json.tool

echo ""
echo "Step 5: Word boost list"
curl -s http://localhost:8024/word-boost/list | python3 -m json.tool | head -20

echo ""
echo "=== Done ==="
