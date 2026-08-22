import re
from dataclasses import dataclass

HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
OPUS   = "claude-opus-4-7"

PRICING = {
    HAIKU:  {"input": 0.25,  "output": 1.25},
    SONNET: {"input": 3.0,   "output": 15.0},
    OPUS:   {"input": 15.0,  "output": 75.0},
}

_CODE_PAT    = re.compile(r"```|\bdef \b|\bclass \b|\bfunction\b", re.IGNORECASE)
_COMPLEX_PAT = re.compile(
    r"\b(analyz|compar|evaluat|design|architect|implement|trade.?off|in detail|explain why)\w*\b",
    re.IGNORECASE,
)


@dataclass
class RoutingDecision:
    model: str
    complexity: str
    reason: str
    estimated_cost_per_1k_tokens_usd: float


def classify(query: str) -> RoutingDecision:
    words    = query.split()
    wc       = len(words)
    sents    = len(re.findall(r"[.!?]+", query)) or 1
    has_code = bool(_CODE_PAT.search(query))
    has_cplx = bool(_COMPLEX_PAT.search(query))

    if has_code or has_cplx or wc > 150 or sents > 4:
        model, complexity = OPUS, "complex"
        reason = (
            "contains code" if has_code else
            "contains complex analysis keywords" if has_cplx else
            f"long query ({wc} words, {sents} sentences)"
        )
    elif wc < 30 and not has_code and not has_cplx and sents <= 1:
        model, complexity = HAIKU, "simple"
        reason = f"short, simple query ({wc} words)"
    else:
        model, complexity = SONNET, "medium"
        reason = f"medium query ({wc} words, {sents} sentences)"

    cost = PRICING[model]["input"] / 1_000_000 * 1000
    return RoutingDecision(model=model, complexity=complexity, reason=reason,
                           estimated_cost_per_1k_tokens_usd=round(cost, 6))
