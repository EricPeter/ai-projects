# Legal AI Benchmark

Evaluation suite for AI systems on South African legal tasks. Community-driven with a public leaderboard.

## Tasks

| Task | Dataset | Metric |
|------|---------|--------|
| Legal Q&A | 5 statute-based questions | ROUGE-L + Semantic Similarity |
| Contract Analysis | 5 contracts | Key clause extraction accuracy |

## Setup

```bash
cd 17-domain-benchmark
cp .env.example .env   # add ANTHROPIC_API_KEY
uv sync
```

## Run Benchmark

```bash
uv run python scripts/run_benchmark.py --model anthropic/claude-sonnet-4-6
uv run python scripts/run_benchmark.py --model anthropic/claude-haiku-4-5-20251001
```

## Tests

```bash
uv run pytest tests/ -v
```

## Leaderboard

See `leaderboard/leaderboard.md`. Submit results by opening a PR with your `results/*.json` file.

## Adding Tasks

1. Add data to `benchmark/dataset/`
2. Add an evaluator in `benchmark/evaluators/`
3. Register in `benchmark/runner.py`
