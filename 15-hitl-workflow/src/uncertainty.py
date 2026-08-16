import re
from dataclasses import dataclass

IRREVERSIBLE_PATTERNS = re.compile(
    r"\b(delete|drop|remove|send email|send message|publish|deploy|purchase|charge|transfer funds)\b",
    re.IGNORECASE,
)
SENSITIVE_KEYWORDS = re.compile(
    r"\b(financial|medical|legal|patient|diagnosis|investment|prescription|lawsuit)\b",
    re.IGNORECASE,
)


@dataclass
class UncertaintyResult:
    should_pause: bool
    reason: str


def assess(task: str, plan: str) -> UncertaintyResult:
    if IRREVERSIBLE_PATTERNS.search(task) or IRREVERSIBLE_PATTERNS.search(plan):
        return UncertaintyResult(True, "Action appears irreversible — human review required")
    if SENSITIVE_KEYWORDS.search(task):
        return UncertaintyResult(True, "Task touches a sensitive domain — human approval required")
    if len(plan.split()) > 200:
        return UncertaintyResult(True, "Complex multi-step plan — human review recommended")
    return UncertaintyResult(False, "")
