# Legal AI Benchmark — Leaderboard

South African legal domain: contract analysis, legal Q&A, case classification, risk assessment.

| Rank | Model | Overall | ROUGE-L | Semantic Sim | Submitted |
|------|-------|---------|---------|--------------|-----------|
| 1 | anthropic/claude-sonnet-4-6 | 0.81 | 0.74 | 0.88 | 2026-08-16 |
| 2 | anthropic/claude-haiku-4-5-20251001 | 0.72 | 0.65 | 0.79 | 2026-08-16 |

## Submit Your Model

1. Run: `uv run python scripts/run_benchmark.py --model provider/model-name`
2. Open a PR with your `results/*.json` file
3. Run `python leaderboard/generate.py` to update this table
