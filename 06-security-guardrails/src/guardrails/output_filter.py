import re
from dataclasses import dataclass, field

TOXIC_KEYWORDS = ["hate", "kill", "murder", "terrorist", "bomb", "attack"]
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),          # OpenAI keys
    re.compile(r"AKIA[0-9A-Z]{16}"),              # AWS access keys
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]{20,}"),
    re.compile(r"(?i)password\s*[:=]\s*\S{6,}"),
]


@dataclass
class FilterResult:
    is_safe: bool
    issues: list[str] = field(default_factory=list)
    filtered_text: str = ""


def filter_output(text: str) -> FilterResult:
    issues = []

    lower = text.lower()
    toxic_found = [kw for kw in TOXIC_KEYWORDS if kw in lower]
    if toxic_found:
        issues.append(f"toxic_content:{','.join(toxic_found)}")

    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            issues.append("secret_leakage")
            break

    return FilterResult(
        is_safe=len(issues) == 0,
        issues=issues,
        filtered_text=text if not issues else "[Response filtered due to policy violation]",
    )
