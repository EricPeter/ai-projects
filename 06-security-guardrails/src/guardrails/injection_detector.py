import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above|your)\s+",
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if\s+you\s+(are|were)|a|an)",
    r"forget\s+(everything|all|your\s+previous)",
    r"your\s+(new\s+)?instructions?\s+(are|is|will\s+be)",
    r"jailbreak",
    r"DAN\s*mode",
    r"developer\s+mode",
    r"override\s+(safety|guidelines|instructions?|restrictions?)",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass
class InjectionResult:
    is_injection: bool
    confidence: float
    matched_pattern: str | None


def detect(text: str) -> InjectionResult:
    matches = [p.pattern for p in _compiled if p.search(text)]
    confidence = min(1.0, len(matches) * 0.4)
    return InjectionResult(
        is_injection=confidence >= 0.7,
        confidence=confidence,
        matched_pattern=matches[0] if matches else None,
    )
