# Fine-Tuning Pipeline with LoRA

End-to-end instruction-tuning pipeline: data prep → LoRA fine-tuning → DPO alignment → evaluation with catastrophic forgetting detection.

## Stack
- **PEFT** — LoRA adapter fine-tuning
- **TRL** — SFTTrainer + DPOTrainer
- **facebook/opt-125m** — small base model (runs on CPU for demo)

## Setup

```bash
cd 09-fine-tuning
cp .env.example .env
uv sync
```

## Run Full Pipeline

```bash
bash scripts/run_pipeline.sh
```

Or step-by-step:

```bash
# 1. Generate dataset
uv run python data/prepare_dataset.py

# 2. LoRA fine-tuning
uv run python src/train_lora.py

# 3. Evaluate — base vs fine-tuned + forgetting check
uv run python src/evaluate.py
```

## Outputs

- `outputs/lora-adapter/` — trained LoRA adapter
- `outputs/eval_report.json` — ROUGE-L scores, forgetting detection result

## Scaling Up

Swap `facebook/opt-125m` for a larger model in `src/train_lora.py`. Add `load_in_4bit=True` and `BitsAndBytesConfig` for QLoRA on GPU.
