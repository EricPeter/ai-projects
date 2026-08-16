#!/usr/bin/env bash
set -euo pipefail

echo "=== Running AI Eval Harness ==="
uv run pytest tests/ -v --tb=short

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "REGRESSION DETECTED — blocking deploy" >&2
  exit 1
fi
echo "All evals passed."
