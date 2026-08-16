# Automated Eval Harness

Runs 20+ golden test cases against a RAG system before every deploy. Blocks regressions. Tracks quality trends over time.

## Stack
- **DeepEval** — answer relevancy & hallucination metrics
- **Custom evaluator** — ground truth keyword coverage
- **Trend tracker** — `eval_history.jsonl` tracks score history across runs

## Setup

```bash
cd 04-eval-harness
cp .env.example .env   # add ANTHROPIC_API_KEY and OPENAI_API_KEY
uv sync
```

## Run

```bash
# Full eval suite
uv run pytest tests/ -v

# CI/CD blocking script (exits non-zero on regression)
bash scripts/run_eval.sh
```

## Golden Dataset

Located in `golden_dataset/`:
- `sample_doc.txt` — ACME Corp HR policy document (the knowledge base)
- `qa_pairs.json` — 20 Q&A pairs with ground-truth answers

## Thresholds

| Metric | Threshold |
|--------|-----------|
| Answer Relevancy | ≥ 0.70 |
| Hallucination Score | ≤ 0.50 |
| Ground Truth Coverage | ≥ 75% |

## Adding New Test Cases

Add entries to `golden_dataset/qa_pairs.json`:
```json
{"id": "q21", "question": "...", "ground_truth": "...", "expected_pages": [1]}
```
