import pytest
from src.languages import validate_language, language_name, SUPPORTED_LANGUAGES


class TestLanguageValidation:
    def test_valid_english(self):
        assert validate_language("eng")

    def test_valid_luganda(self):
        assert validate_language("lug")

    def test_invalid_code(self):
        assert not validate_language("xyz")

    def test_empty_code(self):
        assert not validate_language("")

    def test_all_supported_languages_validate(self):
        for code in SUPPORTED_LANGUAGES:
            assert validate_language(code), f"Expected {code} to be valid"


class TestLanguageName:
    def test_english_name(self):
        assert language_name("eng") == "English"

    def test_luganda_name(self):
        assert language_name("lug") == "Luganda"

    def test_unknown_returns_code(self):
        assert language_name("xyz") == "xyz"
