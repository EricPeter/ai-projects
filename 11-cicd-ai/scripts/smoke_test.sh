#!/usr/bin/env bash
set -euo pipefail

TARGET=${TARGET_URL:-http://localhost:8000}
echo "Running smoke tests against $TARGET..."

# Health check
STATUS=$(curl -sf "$TARGET/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'])")
[ "$STATUS" = "ok" ] || { echo "Health check failed"; exit 1; }
echo "Health: OK"

# Flags endpoint
curl -sf "$TARGET/flags" > /dev/null
echo "Flags: OK"

echo "All smoke tests passed."
