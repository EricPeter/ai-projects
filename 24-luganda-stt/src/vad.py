"""
Voice Activity Detection using NeMo MarbleNet.

MarbleNet is a lightweight (80K parameter) CTC model that classifies each
10 ms frame as speech or non-speech.  We use it to:
  1. Filter silence before feeding segments to the ASR model (saves compute)
  2. Produce speech timestamps for diarization pre-segmentation

If NeMo is unavailable, falls back to a simple energy-based VAD.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .preprocessing import save_wav

logger = logging.getLogger(__name__)

MARBLENET_MODEL = "vad_marblenet"


@dataclass
class SpeechSegment:
    start: float   # seconds
    end: float     # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


class VAD:
    def __init__(self, min_speech_duration: float = 0.3, min_silence_duration: float = 0.3):
        self._model = None
        self._min_speech = min_speech_duration
        self._min_silence = min_silence_duration

    def load(self) -> None:
        try:
            import nemo.collections.asr as nemo_asr
            self._model = nemo_asr.models.EncDecClassificationModel.from_pretrained(MARBLENET_MODEL)
            self._model.eval()
            logger.info("MarbleNet VAD loaded.")
        except Exception as e:
            logger.warning("MarbleNet unavailable (%s). Using energy-based VAD fallback.", e)
            self._model = None

    def get_speech_segments(
        self, samples: np.ndarray, sr: int = 16000
    ) -> list[SpeechSegment]:
        if self._model is not None:
            return self._nemo_vad(samples, sr)
        return self._energy_vad(samples, sr)

    def _nemo_vad(self, samples: np.ndarray, sr: int) -> list[SpeechSegment]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            save_wav(samples, sr, tmp)
            # NeMo VAD returns per-frame probabilities
            probs = self._model.get_vad_probs(tmp)  # shape: (T,)
            frame_duration = 0.01  # 10 ms per frame
            return self._probs_to_segments(probs, frame_duration)
        finally:
            Path(tmp).unlink(missing_ok=True)

    def _probs_to_segments(
        self, probs: np.ndarray, frame_duration: float
    ) -> list[SpeechSegment]:
        threshold = 0.5
        in_speech = False
        seg_start = 0.0
        segments: list[SpeechSegment] = []

        for i, p in enumerate(probs):
            t = i * frame_duration
            if not in_speech and p >= threshold:
                seg_start = t
                in_speech = True
            elif in_speech and p < threshold:
                if t - seg_start >= self._min_speech:
                    segments.append(SpeechSegment(seg_start, t))
                in_speech = False

        if in_speech:
            t = len(probs) * frame_duration
            if t - seg_start >= self._min_speech:
                segments.append(SpeechSegment(seg_start, t))

        return self._merge_close_segments(segments)

    def _merge_close_segments(self, segments: list[SpeechSegment]) -> list[SpeechSegment]:
        if not segments:
            return segments
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg.start - merged[-1].end < self._min_silence:
                merged[-1] = SpeechSegment(merged[-1].start, seg.end)
            else:
                merged.append(seg)
        return merged

    def _energy_vad(self, samples: np.ndarray, sr: int) -> list[SpeechSegment]:
        """Energy-based fallback: split on frames where RMS < threshold."""
        frame_len = int(0.025 * sr)
        hop_len   = int(0.010 * sr)
        frame_duration = hop_len / sr

        energies = []
        for start in range(0, len(samples) - frame_len, hop_len):
            frame = samples[start : start + frame_len]
            energies.append(np.sqrt(np.mean(frame ** 2)))

        if not energies:
            return [SpeechSegment(0.0, len(samples) / sr)]

        energies = np.array(energies)
        # Threshold: 10% of max energy
        threshold = energies.max() * 0.10
        probs = (energies >= threshold).astype(float)
        return self._probs_to_segments(probs, frame_duration)
