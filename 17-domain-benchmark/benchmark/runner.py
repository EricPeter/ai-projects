import json
from datetime import datetime, timezone
from pathlib import Path

from .evaluators.qa_eval import score
from .models.base_model import BaseModel

QA_PATH = Path("benchmark/dataset/legal_qa.json")


def run(model: BaseModel, model_name: str) -> dict:
    qa_pairs = json.loads(QA_PATH.read_text())
    results = []

    for qa in qa_pairs:
        prediction = model.predict(qa["question"], context=qa.get("context", ""))
        scores = score(prediction, qa["ground_truth"])
        results.append({"id": qa["id"], "question": qa["question"], "prediction": prediction, **scores})

    avg_rouge = sum(r["rouge_l"] for r in results) / len(results)
    avg_sem = sum(r["semantic_similarity"] for r in results) / len(results)
    overall = (avg_rouge + avg_sem) / 2

    output = {
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_score": round(overall, 4),
        "avg_rouge_l": round(avg_rouge, 4),
        "avg_semantic_similarity": round(avg_sem, 4),
        "results": results,
    }

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    (out_dir / f"{model_name.replace('/', '_')}_{ts}.json").write_text(json.dumps(output, indent=2))
    return output
