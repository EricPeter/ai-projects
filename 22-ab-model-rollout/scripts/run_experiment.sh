#!/usr/bin/env bash
# Sends N requests to the A/B router and polls until a winner is declared.
set -euo pipefail

ROUTER="${ROUTER_URL:-http://localhost:8022}"
N="${1:-500}"

echo "Sending $N requests to $ROUTER..."
for i in $(seq "$N"); do
  curl -s -X POST "$ROUTER/infer" \
    -H "Content-Type: application/json" \
    -d '{"text": "sample request '"$i"'"}' > /dev/null
done

echo ""
echo "Results after $N requests:"
curl -s "$ROUTER/experiment" | python3 -m json.tool
