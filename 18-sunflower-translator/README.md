# Sunflower Multilingual API

Translation and conversational AI across 31 Ugandan languages, powered by Sunbird Sunflower-14B-W8A8.

## What It Does

- **Translation** between any pair of 31 Ugandan languages and English
- **Auto-detect** source language and translate to any target
- **Chat** with the model in any supported language
- **Batch translation** for multiple texts in one call
- **Streaming** responses for real-time UX

## Supported Languages

Luganda, Runyankole, Acholi, Ateso, Alur, Karamojong, Luo, Lusoga, and 23 more Ugandan languages.

## Setup (GPU required: 12–16GB VRAM)

```bash
cd 18-sunflower-translator
cp .env.example .env
docker-compose up
```

Without GPU — run vLLM locally first:
```bash
pip install vllm
vllm serve Sunbird/Sunflower-14B-W8A8 --port 8080
uv sync && uv run uvicorn src.main:app --reload
```

## Usage

```bash
# Translate to Luganda
curl -X POST http://localhost:8018/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Good morning!", "source_language": "eng", "target_language": "lug"}'

# Chat in Acholi
curl -X POST http://localhost:8018/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is artificial intelligence?", "language": "ach"}'

# Auto-detect language and translate to English
curl -X POST http://localhost:8018/detect-and-translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Oli otya?", "target_language": "eng"}'

# Run full demo
uv run python scripts/demo.py
```

## Tests

```bash
uv run pytest tests/ -v
```
