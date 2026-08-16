from dataclasses import dataclass, field

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    PRESIDIO_AVAILABLE = True
except Exception:
    PRESIDIO_AVAILABLE = False


@dataclass
class RedactionResult:
    redacted_text: str
    entities_found: list[str] = field(default_factory=list)
    pii_detected: bool = False


def redact(text: str, language: str = "en") -> RedactionResult:
    if not PRESIDIO_AVAILABLE:
        return RedactionResult(redacted_text=text)

    results = _analyzer.analyze(text=text, language=language)
    if not results:
        return RedactionResult(redacted_text=text)

    anonymized = _anonymizer.anonymize(text=text, analyzer_results=results)
    entity_types = list({r.entity_type for r in results})

    return RedactionResult(
        redacted_text=anonymized.text,
        entities_found=entity_types,
        pii_detected=True,
    )
