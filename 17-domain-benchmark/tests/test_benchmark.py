from unittest.mock import patch, MagicMock
import pytest

from benchmark.evaluators.qa_eval import score
from benchmark.models.anthropic_model import AnthropicModel


class TestQAEval:
    def test_perfect_match(self):
        result = score("6 months from delivery", "6 months from delivery")
        assert result["rouge_l"] == 1.0

    def test_partial_match(self):
        result = score("3 years from when the debt is due", "3 years")
        assert 0 < result["rouge_l"] < 1.0

    def test_no_match(self):
        result = score("This is completely unrelated text", "6 months from delivery")
        assert result["rouge_l"] < 0.5

    def test_semantic_similarity_in_range(self):
        result = score("six months", "6 months")
        assert 0.0 <= result["semantic_similarity"] <= 1.0


class TestAnthropicModel:
    def test_predict_calls_api(self):
        model = AnthropicModel("claude-haiku-4-5-20251001")
        mock_msg = MagicMock()
        mock_msg.content = [MagicMock(text="3 years")]

        with patch.object(model._client.messages, "create", return_value=mock_msg):
            result = model.predict("What is the prescription period?", context="Prescription Act...")
            assert result == "3 years"
