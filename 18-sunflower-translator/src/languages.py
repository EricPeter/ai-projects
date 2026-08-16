SUPPORTED_LANGUAGES: dict[str, str] = {
    "eng": "English",
    "lug": "Luganda",
    "nyn": "Runyankole",
    "teo": "Ateso",
    "myx": "Lugishu",
    "ach": "Acholi",
    "alz": "Alur",
    "luo": "Luo",
    "kdj": "Karamojong",
    "xog": "Lusoga",
    "nkl": "Runyakitara",
    "ttj": "Rutooro",
    "cgg": "Rukiga",
    "nyoro": "Runyoro",
    "koo": "Konzo",
    "olu": "Olugwere",
    "sog": "Lusoga",
    "rny": "Ruhima",
}

LANGUAGE_CODES = set(SUPPORTED_LANGUAGES.keys())


def validate_language(code: str) -> bool:
    return code in LANGUAGE_CODES


def language_name(code: str) -> str:
    return SUPPORTED_LANGUAGES.get(code, code)
