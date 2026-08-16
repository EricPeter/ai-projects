import re
from dataclasses import dataclass

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-7"

# Anthropic pricing per 1M tokens (input/output) as of 2025
PRICING = {
    HAIKU:  {"input": 0.80,  "output": 4.00},
    SONNET: {"input": 3.00,  "output": 15.00},
    OPUS:   {"input": 15.00, "output": 75.00},
}

COMPLEX_KEYWORDS = re.compile(
    r"\b(analyze|implement|debug|refactor|architecture|algorithm|optimize|derive|prove|"
    r"compare|evaluate|synthesize|design|explain in detail|step.by.step)\b",
    re.IGNORECASE,
)
CODE_PATTERN = re.compile(r"```|def |class |function |SELECT |import |#include")


@dataclass
class RoutingDecision:
    model: str
    reason: str
    complexity: str


def classify(query: str) -> RoutingDecision:
    word_count = len(query.split())
    has_code = bool(CODE_PATTERN.search(query))
    has_complex = bool(COMPLEX_KEYWORDS.search(query))
    has_question_mark = "?" in query
    sentence_count = len([s for s in re.split(r"[.!?]", query) if s.strip()])

    # Simple: short, no code, no complex keywords, single sentence
    if word_count < 30 and not has_code and not has_complex and sentence_count <= 1:
        return RoutingDecision(HAIKU, "Short, simple query — no reasoning required", "simple")

    # Complex: code tasks, explicit complex keywords, long multi-step queries
    if has_code or has_complex or word_count > 150 or sentence_count > 4:
        return RoutingDecision(OPUS, "Code/analysis/multi-step — high-capability model required", "complex")

    # Medium: everything else
    return RoutingDecision(SONNET, "Moderate complexity — balanced model", "medium")


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING[model]
    return (input_tokens / 1_000_000) * p["input"] + (output_tokens / 1_000_000) * p["output"]
