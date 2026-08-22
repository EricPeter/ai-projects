import argparse
import time
from pathlib import Path

import numpy as np


def run(model_path: str, split: str, max_samples: int) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import onnxruntime as ort

    dataset = load_dataset("glue", "sst2", split=split)
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    correct = 0
    latencies = []

    for row in dataset:
        enc = tokenizer(
            row["sentence"],
            return_tensors="np",
            max_length=128,
            padding="max_length",
            truncation=True,
        )
        t0 = time.perf_counter()
        logits = sess.run(None, {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })[0]
        latencies.append((time.perf_counter() - t0) * 1000)
        pred = int(np.argmax(logits))
        correct += pred == row["label"]

    arr = np.array(latencies)
    print(f"Dataset : glue/sst2/{split}  ({len(dataset)} samples)")
    print(f"Accuracy: {correct/len(dataset):.4f}  ({correct}/{len(dataset)})")
    print(f"Latency : p50={np.percentile(arr,50):.2f}ms  p95={np.percentile(arr,95):.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/distilbert.onnx")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    run(args.model, args.split, args.max_samples)
