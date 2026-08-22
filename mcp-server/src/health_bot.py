import re
import anthropic

EMERGENCY_KEYWORDS = re.compile(
    r"\b(emergency|dying|unconscious|convuls\w*|heavy bleeding|cannot breathe|severe pain|fainted|not breathing)\b",
    re.IGNORECASE,
)

EMERGENCY_RESPONSE = {
    "eng": "THIS IS AN EMERGENCY. Go to the nearest health centre NOW. Call 0800-100-066 (Uganda health emergency line).",
    "lug": "OLI MU KABI. Genda eddwaaliro ly'okumpi KATI. Yita 0800-100-066.",
    "ach": "CITO KI LYEC. Cit i dyeel pa laro mapod cok KOMBEDI. Kwan 0800-100-066.",
    "teo": "AKILOS AKIDWONG. Cit icen ki hospital epedipedi SAA ENO. Ere 0800-100-066.",
    "nyn": "OLI MU KABI. Genda omu kliniki eyegombeka NOMBA. Ita 0800-100-066.",
}

UGANDAN_LANGUAGES = {
    "eng": "English",
    "lug": "Luganda",
    "ach": "Acholi",
    "teo": "Ateso",
    "nyn": "Runyankole",
}

HEALTH_TOPICS = {
    "malaria": "Malaria is spread by mosquito bites. Symptoms: fever, chills, headache. Prevent with nets and repellent. Seek care within 24 hours of fever.",
    "maternal": "Attend all antenatal visits. Deliver at a health facility. Signs of danger: heavy bleeding, severe headache, fits, no baby movement.",
    "nutrition": "Breastfeed exclusively for 6 months. Introduce soft foods at 6 months. Give MUAC screening for child malnutrition.",
    "hiv": "HIV is preventable. Test regularly. ART keeps you healthy. PMTCT prevents mother-to-child transmission.",
    "child": "Immunise all children. ORS for diarrhoea. Zinc reduces diarrhoea severity. Seek care for fast breathing in children.",
    "covid": "Wash hands frequently. Wear masks in crowds. Ventilate rooms. Get vaccinated.",
    "tb": "Tuberculosis: cough for 2+ weeks, night sweats, weight loss. Seek testing. Treatment is free at health centres.",
    "water": "Boil or treat water before drinking. Use pit latrines. Wash hands after toilet and before eating.",
}


def _topic_context(question: str) -> str:
    q = question.lower()
    for topic, info in HEALTH_TOPICS.items():
        if topic in q:
            return info
    return " ".join(list(HEALTH_TOPICS.values())[:3])


def ask(language_code: str, question: str) -> dict:
    if EMERGENCY_KEYWORDS.search(question):
        return {
            "response": EMERGENCY_RESPONSE.get(language_code, EMERGENCY_RESPONSE["eng"]),
            "is_emergency": True,
            "language": UGANDAN_LANGUAGES.get(language_code, "English"),
            "emergency_number": "0800-100-066",
        }

    lang = UGANDAN_LANGUAGES.get(language_code, "English")
    context = _topic_context(question)

    client = anthropic.Anthropic()
    system = (
        f"You are a trusted community health worker in Uganda. "
        f"Respond ONLY in {lang}. "
        f"Give practical, accurate health advice based on Uganda Ministry of Health guidelines. "
        f"Use simple language. Keep answers under 150 words. "
        f"Always recommend visiting the nearest health centre for serious concerns. "
        f"Do not diagnose conditions.\n"
        f"Relevant health knowledge: {context}"
    )
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return {
        "response": msg.content[0].text,
        "is_emergency": False,
        "language": lang,
    }
