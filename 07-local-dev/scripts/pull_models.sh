#!/usr/bin/env bash
set -euo pipefail
OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}
echo "Pulling Ollama models from $OLLAMA_HOST..."
ollama pull llama3.2
echo "Done. Models ready."
