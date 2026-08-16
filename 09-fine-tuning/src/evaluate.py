import json
from pathlib import Path

import evaluate as hf_evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

BASE_MODEL = "facebook/opt-125m"
LORA_PATH = "outputs/lora-adapter"

TEST_PROMPTS = [
    "### Instruction:\nHow do I reset my password?\n\n### Response:\n",
    "### Instruction:\nWhat is your return policy?\n\n### Response:\n",
    "### Instruction:\nHow do I track my order?\n\n### Response:\n",
]

GENERAL_PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "The sun rises in the",
]

rouge = hf_evaluate.load("rouge")


def generate(pipe, prompt: str) -> str:
    out = pipe(prompt, max_new_tokens=80, do_sample=False, pad_token_id=pipe.tokenizer.eos_token_id)
    return out[0]["generated_text"][len(prompt):].strip()


def score_pair(prediction: str, reference: str) -> float:
    result = rouge.compute(predictions=[prediction], references=[reference])
    return round(result["rougeL"], 4)


def run():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    base_pipe = pipeline("text-generation", model=base_model, tokenizer=tokenizer)

    tuned_model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(BASE_MODEL), LORA_PATH)
    tuned_pipe = pipeline("text-generation", model=tuned_model, tokenizer=tokenizer)

    results = {"task_prompts": [], "general_prompts": [], "forgetting_detected": False}

    print("=== Task Prompts ===")
    for prompt in TEST_PROMPTS:
        base_out = generate(base_pipe, prompt)
        tuned_out = generate(tuned_pipe, prompt)
        improvement = score_pair(tuned_out, base_out)
        entry = {"prompt": prompt[:80], "base": base_out[:100], "tuned": tuned_out[:100]}
        results["task_prompts"].append(entry)
        print(f"Prompt: {prompt[:60]}...")
        print(f"  Base:  {base_out[:80]}")
        print(f"  Tuned: {tuned_out[:80]}\n")

    print("=== Catastrophic Forgetting Check ===")
    base_general_scores, tuned_general_scores = [], []
    for prompt in GENERAL_PROMPTS:
        base_out = generate(base_pipe, prompt)
        tuned_out = generate(tuned_pipe, prompt)
        base_score = score_pair(base_out, prompt)
        tuned_score = score_pair(tuned_out, prompt)
        base_general_scores.append(base_score)
        tuned_general_scores.append(tuned_score)
        results["general_prompts"].append({"prompt": prompt, "base": base_out[:80], "tuned": tuned_out[:80]})

    avg_base = sum(base_general_scores) / len(base_general_scores)
    avg_tuned = sum(tuned_general_scores) / len(tuned_general_scores)
    drop = avg_base - avg_tuned
    if drop > 0.2:
        results["forgetting_detected"] = True
        print(f"WARNING: Potential catastrophic forgetting — general score dropped {drop:.2%}")
    else:
        print(f"General capability preserved (score change: {drop:+.2%})")

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/eval_report.json").write_text(json.dumps(results, indent=2))
    print("Report saved to outputs/eval_report.json")


if __name__ == "__main__":
    run()
