# Uganda Community Health Bot

A community health information chatbot powered by Sunbird Sunflower-14B — answering health questions in Luganda, Acholi, Ateso, Runyankole, and other Ugandan languages.

## Why This Matters

Millions of Ugandans speak local languages as their primary language. Most health information online is in English. This bot bridges that gap — delivering trusted Ministry of Health guidance on malaria, maternal health, child nutrition, HIV, and more, in the language people actually speak.

## Health Topics Covered

- Malaria prevention, symptoms, and treatment
- Maternal health and antenatal care
- Child health, vaccination, and nutrition
- HIV prevention and treatment
- Family planning
- Water, sanitation, and hygiene

## Setup (GPU required)

```bash
cd 19-uganda-health-bot
cp .env.example .env
uv sync

# Start vLLM with Sunflower model
pip install vllm
vllm serve Sunbird/Sunflower-14B-W8A8 --port 8080

# Start the API
uv run uvicorn src.main:app --reload
```

Open http://localhost:8000 for the chat UI.

## API Usage

```bash
# Ask in Luganda
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Nkwata ddagala ya malaria otya?", "language": "lug"}'

# Ask in Acholi
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Malaria ikwanyo ngo?", "language": "ach"}'

# Emergency detection (immediate response, no LLM call)
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "My child is having convulsions", "language": "eng"}'
```

## WhatsApp Integration

Set Twilio credentials in `.env` and point your WhatsApp webhook to `/whatsapp`. Prefix messages with language code: `[lug] Ekibiina kyange kirina`, `[ach] Ber me kwanyo`, etc.

## Tests

```bash
uv run pytest tests/ -v
```
