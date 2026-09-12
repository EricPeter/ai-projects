"""
End-of-turn (EOU) detection for streaming ASR.

Combines the two signals the NeMo Voice Agent uses:

  1. Inline turn tokens emitted by EOU-capable models (e.g.
     parakeet_realtime_eou_120m-v1): ``<EOU>`` = end of utterance/turn,
     ``<EOB>`` = end of backchannel (a short "mm-hm" that shouldn't steal the
     turn). These are stripped from the surfaced transcript.
  2. VAD trailing-silence endpointing (``stop_secs``) for models that do NOT
     emit turn tokens (e.g. our Luganda cache-aware FastConformer fine-tune).

A hard ``max_buffer_words`` flush prevents an unbounded turn when neither signal
fires (very long monologue / missing silence).

This module is pure-Python and deterministic so it can be unit-tested without
audio or NeMo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EOU_TOKEN = "<EOU>"
EOB_TOKEN = "<EOB>"
_TOKEN_RE = re.compile(r"\s*<(?:EOU|EOB)>\s*", re.IGNORECASE)


def strip_turn_tokens(text: str) -> tuple[str, bool, bool]:
    """Return (clean_text, saw_eou, saw_eob)."""
    saw_eou = EOU_TOKEN.lower() in text.lower()
    saw_eob = EOB_TOKEN.lower() in text.lower()
    clean = _TOKEN_RE.sub(" ", text).strip()
    return clean, saw_eou, saw_eob


@dataclass
class TurnDecision:
    end_of_turn: bool
    text: str                 # clean transcript (turn tokens stripped)
    reason: str | None = None  # "eou" | "eob" | "silence" | "max_words"
    is_backchannel: bool = False


class EndOfTurnDetector:
    def __init__(
        self,
        stop_secs: float = 0.8,
        min_words: int = 1,
        max_buffer_words: int = 60,
        use_eou_tokens: bool = True,
    ):
        self.stop_secs = stop_secs
        self.min_words = min_words
        self.max_buffer_words = max_buffer_words
        self.use_eou_tokens = use_eou_tokens
        self._silence = 0.0

    def reset(self) -> None:
        """Call after a turn has been consumed to start the next turn cleanly."""
        self._silence = 0.0

    def update(self, hypothesis: str, is_speech: bool, dt: float) -> TurnDecision:
        """Advance the detector by one streaming chunk.

        hypothesis : latest (partial) transcript for the current turn, possibly
                     containing inline <EOU>/<EOB> tokens.
        is_speech  : True if this chunk contained speech (from VAD), else silence.
        dt         : duration of this chunk in seconds.
        """
        clean, saw_eou, saw_eob = strip_turn_tokens(hypothesis)
        n_words = len(clean.split())

        # Accumulate trailing silence; any speech resets it.
        if is_speech:
            self._silence = 0.0
        else:
            self._silence += dt

        if self.use_eou_tokens and saw_eou:
            self._silence = 0.0
            return TurnDecision(True, clean, "eou")

        if self.use_eou_tokens and saw_eob and n_words > 0:
            self._silence = 0.0
            return TurnDecision(True, clean, "eob", is_backchannel=True)

        if n_words >= self.max_buffer_words:
            self._silence = 0.0
            return TurnDecision(True, clean, "max_words")

        if n_words >= self.min_words and self._silence >= self.stop_secs:
            self._silence = 0.0
            return TurnDecision(True, clean, "silence")

        return TurnDecision(False, clean, None)
