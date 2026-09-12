"""Text metrics for ASR evaluation: WER and CER.

Kept dependency-free (no NeMo/torch) so it can be imported and unit-tested
without the full training stack installed.
"""

from __future__ import annotations


def _edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance between two token sequences."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref):
        curr = [i + 1] + [0] * len(hyp)
        for j, h in enumerate(hyp):
            curr[j + 1] = min(
                prev[j] + (0 if r == h else 1),  # substitution / match
                prev[j + 1] + 1,                 # deletion
                curr[j] + 1,                     # insertion
            )
        prev = curr
    return prev[len(hyp)]


def _rate(ref_tokens: list, hyp_tokens: list) -> float:
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def _words(text: str) -> list[str]:
    return text.lower().split()


def _chars(text: str) -> list[str]:
    # Character sequence including spaces (matches NeMo use_cer semantics).
    return list(text.lower())


# ── Word Error Rate ────────────────────────────────────────────────────────────

def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance normalised by reference word count."""
    return _rate(_words(reference), _words(hypothesis))


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """Aggregate WER: total word edits over total reference words."""
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must be the same length")
    total_err = 0
    total = 0
    for r, h in zip(refs, hyps):
        rw, hw = _words(r), _words(h)
        total += len(rw)
        total_err += _edit_distance(rw, hw) if rw else len(hw)
    return total_err / total if total else 0.0


# ── Character Error Rate ───────────────────────────────────────────────────────

def char_error_rate(reference: str, hypothesis: str) -> float:
    """Character-level edit distance normalised by reference char count."""
    return _rate(_chars(reference), _chars(hypothesis))


def corpus_cer(refs: list[str], hyps: list[str]) -> float:
    """Aggregate CER: total char edits over total reference chars."""
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps must be the same length")
    total_err = 0
    total = 0
    for r, h in zip(refs, hyps):
        rc, hc = _chars(r), _chars(h)
        total += len(rc)
        total_err += _edit_distance(rc, hc) if rc else len(hc)
    return total_err / total if total else 0.0
