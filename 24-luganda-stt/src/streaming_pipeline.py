"""
Live streaming pipeline: audio chunks -> interim transcript -> end-of-turn ->
speaker-attributed final turn.

Design
------
* ``StreamingLive`` loads the shared models once (streaming ASR, Sortformer
  diarizer, spell checker) and hands out a lightweight ``StreamingSession`` per
  WebSocket connection.
* ``StreamingSession`` accumulates the current turn's audio, produces interim
  transcripts (cache-aware streaming output equals offline, so re-decoding the
  growing turn buffer is correct), runs ``EndOfTurnDetector`` on the (VAD +
  <EOU>) signals, and on turn end attributes a speaker and emits a final turn.

Speaker attribution and spell correction run only at turn boundaries to keep the
interim path low-latency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .config import Settings
from .spell_checker import LugandaSpellChecker
from .streaming_asr import StreamingASR
from .streaming_diarizer import StreamingDiarizer
from .turn_detection import EndOfTurnDetector

logger = logging.getLogger(__name__)

SR = 16000


@dataclass
class InterimEvent:
    text: str
    type: str = "interim"


@dataclass
class TurnEvent:
    text: str            # spell-corrected final
    raw: str             # model output (turn tokens stripped)
    speaker: str
    reason: str          # eou | eob | silence | max_words
    backchannel: bool = False
    type: str = "turn"


class StreamingLive:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._asr = StreamingASR(settings)
        self._diar = StreamingDiarizer(settings)
        self._spell = LugandaSpellChecker(settings.spell_dict_path)

    def load(self) -> None:
        logger.info("Loading streaming ASR...")
        self._asr.load()
        logger.info("Loading streaming diarizer...")
        try:
            self._diar.load()
        except Exception as e:  # noqa: BLE001 - diarizer is optional
            logger.warning("Diarizer unavailable (%s); turns will be single-speaker.", e)
        self._spell.load()
        logger.info("Streaming pipeline ready.")

    @property
    def diarization_ready(self) -> bool:
        return self._diar.loaded

    def session(self) -> "StreamingSession":
        return StreamingSession(self, self._settings)


class StreamingSession:
    def __init__(self, live: StreamingLive, settings: Settings):
        self._live = live
        self._settings = settings
        self._detector = EndOfTurnDetector(
            stop_secs=settings.turn_stop_secs,
            min_words=settings.turn_min_words,
            max_buffer_words=settings.turn_max_buffer_words,
            use_eou_tokens=settings.use_eou_tokens,
        )
        self._turn_audio: list[np.ndarray] = []
        self._last_interim = ""

    # ── Streaming API ──────────────────────────────────────────────────────────

    def push(self, pcm: np.ndarray, dt: float | None = None) -> list:
        """Feed one audio chunk (float32 mono @16k). Returns interim/turn events."""
        pcm = np.asarray(pcm, dtype=np.float32)
        if dt is None:
            dt = len(pcm) / SR
        self._turn_audio.append(pcm)

        is_speech = _rms(pcm) > 0.01

        # Only re-decode when there's speech; silence just advances endpointing.
        if is_speech:
            self._last_interim = self._live._asr.transcribe_turn(self._turn_buffer())

        decision = self._detector.update(self._last_interim, is_speech, dt)
        events: list = [InterimEvent(text=decision.text)]

        if decision.end_of_turn:
            events.append(self._finalize_turn(decision.text, decision.reason, decision.is_backchannel))
            self._reset_turn()
        return events

    def finalize(self) -> list:
        """Flush any in-progress turn when the stream closes."""
        if not self._turn_audio or not self._last_interim.strip():
            return []
        from .turn_detection import strip_turn_tokens

        clean, _, _ = strip_turn_tokens(self._last_interim)
        if not clean.strip():
            return []
        ev = self._finalize_turn(clean, "close", False)
        self._reset_turn()
        return [ev]

    # ── Internals ──────────────────────────────────────────────────────────────

    def _turn_buffer(self) -> np.ndarray:
        return np.concatenate(self._turn_audio) if self._turn_audio else np.zeros(0, np.float32)

    def _finalize_turn(self, text: str, reason: str, backchannel: bool) -> TurnEvent:
        corrected = self._live._spell.correct(text) if self._live._spell.loaded else text
        speaker = "speaker_0"
        if self._live.diarization_ready:
            try:
                speaker = self._live._diar.speaker_for_turn(self._turn_buffer(), SR)
            except Exception as e:  # noqa: BLE001
                logger.warning("Diarization failed for turn (%s).", e)
        return TurnEvent(text=corrected, raw=text, speaker=speaker,
                         reason=reason, backchannel=backchannel)

    def _reset_turn(self) -> None:
        self._turn_audio = []
        self._last_interim = ""
        self._detector.reset()


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
