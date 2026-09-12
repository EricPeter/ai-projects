"""
End-to-end Luganda STT pipeline.

Stage order:
  1. Preprocessing   — resample, mono, peak-normalise, optional denoise
  2. VAD             — speech vs silence (MarbleNet / energy fallback)
  3. Diarization     — who spoke when (NeuralDiarizer MSDD / VAD fallback)
  4. ASR             — Canary-1B-Flash + NGPU-LM + GPU-PB word boosting
  5. Spell check     — SymSpell on Luganda frequency dictionary
  6. Punctuation     — NeMo punct model / rule-based fallback
  7. ITN             — inverse text normalisation (numbers, abbreviations)
  8. Translation     — Canary S2T (fast) or SALM (rich)

Output: PipelineResult with per-speaker segments + full transcript + translation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .asr import CanaryASR
from .config import Settings
from .diarizer import Diarizer, DiarizedSegment
from .preprocessing import preprocess, to_wav_bytes
from .punctuation import ITNProcessor, PunctuationRestorer
from .spell_checker import LugandaSpellChecker
from .speechlm2 import SALMTranslator
from .vad import VAD

logger = logging.getLogger(__name__)


@dataclass
class WordToken:
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class SegmentResult:
    speaker: str
    start: float
    end: float
    transcript: str              # raw ASR
    transcript_corrected: str    # after spell check
    transcript_final: str        # after punct + ITN
    translation: str
    confidence: float
    words: list[WordToken] = field(default_factory=list)
    boosted_words_hit: int = 0


@dataclass
class PipelineResult:
    segments: list[SegmentResult]
    full_transcript: str         # all segments joined (punct + ITN applied)
    full_translation: str
    num_speakers: int
    duration: float
    language: str = "lug"
    mode: str = "fast"
    latency_ms: float = 0.0

    @property
    def speaker_labels(self) -> list[str]:
        return sorted({s.speaker for s in self.segments})


class STTPipeline:
    def __init__(self, settings: Settings, mode: str = "fast", denoise: bool = False):
        assert mode in ("fast", "rich")
        self._settings = settings
        self._mode = mode
        self._denoise = denoise

        self._vad         = VAD()
        self._diarizer    = Diarizer()
        self._asr         = CanaryASR(settings)
        self._spell       = LugandaSpellChecker(settings.spell_dict_path)
        self._punct       = PunctuationRestorer()
        self._itn         = ITNProcessor()
        self._salm        = SALMTranslator(settings.salm_model) if mode == "rich" else None

    # ── Startup ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        logger.info("Loading VAD...")
        self._vad.load()

        logger.info("Loading Diarizer...")
        self._diarizer.load()

        logger.info("Loading ASR (Canary)...")
        self._asr.load()

        logger.info("Loading Punctuation / ITN...")
        self._punct.load()
        self._itn.load()

        # Seed spell checker with boosted words so they're never corrected away
        for word in self._asr.boosted_words:
            self._spell.add_to_dictionary(word, count=100)

        if self._salm:
            try:
                logger.info("Loading SALM...")
                self._salm.load()
            except RuntimeError as e:
                logger.warning("SALM not available (%s). Falling back to Canary translation.", e)
                self._salm = None

        logger.info("Pipeline ready. Mode=%s  boosted_words=%d  spell=%s",
                    self._mode, len(self._asr.boosted_words), self._spell.loaded)

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(self, audio_bytes: bytes) -> PipelineResult:
        t0 = time.perf_counter()

        # 1. Preprocessing
        samples, sr = preprocess(audio_bytes, denoise=self._denoise)
        duration = len(samples) / sr

        # 2. VAD
        speech_segments = self._vad.get_speech_segments(samples, sr)
        logger.debug("VAD: %d speech segments", len(speech_segments))

        # 3. Diarization
        diarized = self._diarizer.diarize(samples, sr, vad_segments=speech_segments)
        logger.debug("Diarization: %d segments", len(diarized))

        # 4–8. Per-segment ASR → spell → punct → ITN → translation
        results: list[SegmentResult] = []
        for seg in diarized:
            result = self._process_segment(seg)
            results.append(result)

        # Aggregate full transcript and translation
        full_transcript = " ".join(r.transcript_final for r in results if r.transcript_final)
        if self._salm and full_transcript:
            full_translation = self._salm.translate(full_transcript, target_lang="English")
        else:
            full_translation = " ".join(r.translation for r in results if r.translation)

        latency_ms = (time.perf_counter() - t0) * 1000

        return PipelineResult(
            segments=results,
            full_transcript=full_transcript,
            full_translation=full_translation,
            num_speakers=len({r.speaker for r in results}),
            duration=round(duration, 2),
            language="lug",
            mode=self._mode,
            latency_ms=round(latency_ms, 1),
        )

    # ── Per-segment processing ────────────────────────────────────────────────

    def _process_segment(self, seg: DiarizedSegment) -> SegmentResult:
        wav_bytes = to_wav_bytes(seg.samples)

        # ASR
        transcript, words, confidence = self._asr.transcribe_with_details(wav_bytes)

        # Spell check
        transcript_corrected = self._spell.correct(transcript)

        # Punctuation restoration
        transcript_punct = self._punct.restore(transcript_corrected)

        # ITN
        transcript_final = self._itn.normalize(transcript_punct)

        # Translation (fast: Canary S2T; rich: SALM on corrected text)
        if self._salm and transcript_final:
            translation = self._salm.translate(transcript_final, target_lang="English")
        elif transcript_final:
            translation = self._asr.translate(wav_bytes, target_lang=self._settings.target_lang)
        else:
            translation = ""

        # Count boosted words hit
        boosted = self._asr.boosted_words
        hits = sum(1 for w in boosted if w.lower() in transcript_final.lower())

        # Offset word timestamps to absolute file time
        words_offset = [
            WordToken(
                word=w.word,
                start=round(seg.start + w.start, 3),
                end=round(seg.start + w.end, 3),
                confidence=w.confidence,
            )
            for w in words
        ]

        return SegmentResult(
            speaker=seg.speaker,
            start=seg.start,
            end=seg.end,
            transcript=transcript,
            transcript_corrected=transcript_corrected,
            transcript_final=transcript_final,
            translation=translation,
            confidence=round(confidence, 4),
            words=words_offset,
            boosted_words_hit=hits,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def boosted_words(self) -> list[str]:
        return self._asr.boosted_words

    @property
    def spell_checker_active(self) -> bool:
        return self._spell.loaded
