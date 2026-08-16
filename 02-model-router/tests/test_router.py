import pytest
from src.router import classify, estimate_cost, HAIKU, SONNET, OPUS


class TestClassify:
    def test_short_simple_routes_to_haiku(self):
        result = classify("What is the capital of France?")
        assert result.model == HAIKU
        assert result.complexity == "simple"

    def test_code_routes_to_opus(self):
        result = classify("```python\ndef quicksort(arr): pass\n``` Implement this fully.")
        assert result.model == OPUS
        assert result.complexity == "complex"

    def test_analyze_keyword_routes_to_opus(self):
        result = classify("Analyze the trade-offs between microservices and monolithic architecture in detail.")
        assert result.model == OPUS

    def test_medium_query_routes_to_sonnet(self):
        result = classify("Explain how transformers work in machine learning. Include attention mechanisms.")
        assert result.model == SONNET
        assert result.complexity == "medium"

    def test_routing_reason_is_non_empty(self):
        result = classify("Hello, how are you?")
        assert result.reason


class TestCostEstimate:
    def test_haiku_cheaper_than_opus(self):
        haiku_cost = estimate_cost(HAIKU, 1000, 500)
        opus_cost = estimate_cost(OPUS, 1000, 500)
        assert haiku_cost < opus_cost

    def test_zero_tokens_zero_cost(self):
        assert estimate_cost(SONNET, 0, 0) == 0.0

    def test_cost_increases_with_tokens(self):
        low = estimate_cost(SONNET, 100, 50)
        high = estimate_cost(SONNET, 10_000, 5_000)
        assert high > low
