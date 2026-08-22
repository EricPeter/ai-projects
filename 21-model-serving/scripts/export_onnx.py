import argparse
from pathlib import Path


def export(output: str, seq_len: int = 128) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    model_id = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval()

    dummy = tokenizer(
        "This is a dummy input for export",
        return_tensors="pt",
        max_length=seq_len,
        padding="max_length",
        truncation=True,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(output_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids":      {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits":         {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"Exported: {output_path}  ({size_mb:.1f} MB)")

    # Quick validation
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    out = sess.run(
        None,
        {
            "input_ids":      dummy["input_ids"].numpy(),
            "attention_mask": dummy["attention_mask"].numpy(),
        },
    )
    labels = ["NEGATIVE", "POSITIVE"]
    pred = labels[int(np.argmax(out[0]))]
    print(f"Validation prediction: {pred}  (logits: {out[0]})")
    print("Export successful.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/distilbert.onnx")
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()
    export(args.output, args.seq_len)
