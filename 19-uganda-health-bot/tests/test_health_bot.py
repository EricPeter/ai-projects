import pytest
from src.health_bot import respond, EMERGENCY_KEYWORDS


class TestEmergencyDetection:
    def test_detects_emergency_keyword(self):
        assert EMERGENCY_KEYWORDS.search("I am unconscious and need help")

    def test_no_emergency_for_normal_question(self):
        assert not EMERGENCY_KEYWORDS.search("How do I prevent malaria?")

    def test_emergency_response_contains_number(self):
        result = respond("My child is convulsing and unconscious", "eng")
        assert result["is_emergency"]
        assert "0800-100-066" in result["emergency_number"]

    def test_emergency_response_in_luganda(self):
        result = respond("The baby is not breathing", "lug")
        assert result["is_emergency"]


class TestLanguageSupport:
    def test_english_response(self):
        result = respond("How do I prevent malaria?", "eng")
        assert result["language"] == "English"
        assert not result["is_emergency"]

    def test_unsupported_language_defaults(self):
        result = respond("How do I prevent malaria?", "xyz")
        assert result["language"] == "xyz"
