import json
import os
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

BASE_MODEL = "facebook/opt-125m"
OUTPUT_DIR = "outputs/lora-adapter"


def load_jsonl(path: str) -> Dataset:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    texts = [f"### Instruction:\n{r['instruction']}\n\n### Response:\n{r['output']}" for r in rows]
    return Dataset.from_dict({"text": texts})


def train():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_jsonl("data/train.jsonl")
    eval_dataset = load_jsonl("data/eval.jsonl")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=10,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb" if os.getenv("WANDB_API_KEY") else "none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=256,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    train()
