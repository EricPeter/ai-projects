import json
import os
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase

from src.rag_system import SimpleRAG
from src.trend_tracker import save_run

SAMPLE_DOC = Path("golden_dataset/sample_doc.txt").read_text()
QA_PAIRS = json.loads(Path("golden_dataset/qa_pairs.json").read_text())
ANSWER_RELEVANCY_THRESHOLD = 0.7
FAITHFULNESS_THRESHOLD = 0.7


@pytest.fixture(scope="session")
def rag() -> SimpleRAG:
    system = SimpleRAG()
    system.index(SAMPLE_DOC, page=1)
    return system


@pytest.fixture(scope="session")
def eval_results(rag: SimpleRAG) -> list[dict]:
    results = []
    for qa in QA_PAIRS:
        answer, chunks = rag.answer(qa["question"])
        results.append({
            "id": qa["id"],
            "question": qa["question"],
            "answer": answer,
            "ground_truth": qa["ground_truth"],
            "context": [c.content for c in chunks],
        })
    return results


def test_answer_relevancy(rag, eval_results):
    metric = AnswerRelevancyMetric(threshold=ANSWER_RELEVANCY_THRESHOLD)
    scores = []
    for r in eval_results[:5]:  # test first 5 to keep CI fast
        tc = LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            retrieval_context=r["context"],
        )
        metric.measure(tc)
        scores.append(metric.score)
    avg = sum(scores) / len(scores)
    assert avg >= ANSWER_RELEVANCY_THRESHOLD, f"Answer relevancy {avg:.2f} below threshold {ANSWER_RELEVANCY_THRESHOLD}"


def test_no_hallucination(rag, eval_results):
    metric = HallucinationMetric(threshold=0.5)
    for r in eval_results[:5]:
        tc = LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            context=r["context"],
        )
        metric.measure(tc)
        assert metric.score <= 0.5, f"Hallucination score {metric.score:.2f} too high for: {r['question']}"


def test_ground_truth_coverage(rag, eval_results):
    """Answers must contain key terms from the ground truth."""
    passed = 0
    for r in eval_results:
        gt_keywords = set(r["ground_truth"].lower().split()) - {"the", "a", "an", "is", "are", "of", "to", "and"}
        answer_lower = r["answer"].lower()
        overlap = sum(1 for kw in gt_keywords if kw in answer_lower)
        if overlap / max(len(gt_keywords), 1) >= 0.4:
            passed += 1
    coverage = passed / len(eval_results)
    assert coverage >= 0.75, f"Ground truth coverage {coverage:.0%} below 75%"
    save_run({"ground_truth_coverage": coverage})


def test_all_questions_answered(rag):
    """Every question in the golden set must return a non-empty answer."""
    for qa in QA_PAIRS:
        answer, _ = rag.answer(qa["question"])
        assert answer.strip(), f"Empty answer for question: {qa['question']}"
