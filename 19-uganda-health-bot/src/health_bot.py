import re
from data.health_knowledge import HEALTH_TOPICS, UGANDAN_LANGUAGES
from .llm import ask

EMERGENCY_KEYWORDS = re.compile(
    r"\b(emergency|dying|unconscious|convuls\w*|heavy bleeding|cannot breathe|severe pain|fainted|not breathing)\b",
    re.IGNORECASE,
)

EMERGENCY_RESPONSE = {
    "eng": "THIS IS AN EMERGENCY. Go to the nearest hospital NOW. Call 0800-100-066 (Uganda health emergency line).",
    "lug": "OLI MU KABI. Genda eddwaaliro ly'okumpi KATI. Yita 0800-100-066.",
    "ach": "CITO KI LYEC. Cit i dyeel pa laro mapod cok KOMBEDI. Kwan 0800-100-066.",
}


def _topic_context(question: str) -> str:
    q_lower = question.lower()
    for topic, sections in HEALTH_TOPICS.items():
        if topic in q_lower or any(kw in q_lower for kw in topic.split("_")):
            return "\n".join(f"- {k}: {v}" for k, v in sections.items())
    all_info = []
    for sections in HEALTH_TOPICS.values():
        all_info.extend(sections.values())
    return "\n".join(f"- {info}" for info in all_info[:6])


def respond(question: str, language_code: str = "eng") -> dict:
    if EMERGENCY_KEYWORDS.search(question):
        lang = UGANDAN_LANGUAGES.get(language_code, "English")
        emergency_msg = EMERGENCY_RESPONSE.get(language_code, EMERGENCY_RESPONSE["eng"])
        return {
            "response": emergency_msg,
            "is_emergency": True,
            "language": lang,
            "emergency_number": "0800-100-066",
        }

    lang = UGANDAN_LANGUAGES.get(language_code, "English")
    context = _topic_context(question)

    system = (
        f"You are a trusted community health worker in Uganda. "
        f"Respond ONLY in {lang}. "
        f"Give practical, accurate health advice based on Uganda Ministry of Health guidelines. "
        f"Use simple language. Keep answers under 150 words. "
        f"Always recommend visiting the nearest health centre for serious concerns. "
        f"Do not diagnose conditions. "
        f"Health knowledge context:\n{context}"
    )

    response = ask(system, question, max_tokens=300, temperature=0.3)
    return {"response": response, "is_emergency": False, "language": lang}
