"""
Speaker diarization using NeMo NeuralDiarizer (MSDD).

Pipeline:
  1. VAD segments  →  extract speaker embeddings (TitaNet-Large)
  2. Agglomerative clustering  →  coarse speaker labels
  3. MSDD (Multi-Scale Diarization Decoder)  →  refined boundaries

MSDD is NeMo's state-of-the-art diarizer (significantly better than pure
clustering on overlapping speech).  It is speaker-count-agnostic by default.

Outputs a list of DiarizedSegment objects, each carrying:
  - speaker label ("SPEAKER_00", "SPEAKER_01", …)
  - start / end timestamps (seconds)
  - the audio samples for that segment (for per-segment ASR)
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .preprocessing import save_wav
from .vad import SpeechSegment

logger = logging.getLogger(__name__)

SPEAKER_MODEL = "titanet_large"       # speaker embedding model
DIARIZER_CONFIG = "diar_infer_meeting"  # NeMo pretrained MSDD config


@dataclass
class DiarizedSegment:
    speaker: str
    start: float
    end: float
    samples: np.ndarray   # 16 kHz mono float32 audio for this segment

    @property
    def duration(self) -> float:
        return self.end - self.start


class Diarizer:
    def __init__(self, max_speakers: int | None = None):
        self._max_speakers = max_speakers
        self._model = None

    def load(self) -> None:
        try:
            from nemo.collections.asr.models import NeuralDiarizer
            self._model = NeuralDiarizer.from_pretrained(DIARIZER_CONFIG)
            self._model.eval()
            logger.info("NeuralDiarizer (MSDD) loaded.")
        except Exception as e:
            logger.warning("NeuralDiarizer unavailable (%s). Using VAD-only segmentation.", e)
            self._model = None

    def diarize(
        self,
        samples: np.ndarray,
        sr: int = 16000,
        vad_segments: list[SpeechSegment] | None = None,
    ) -> list[DiarizedSegment]:
        if self._model is not None:
            return self._nemo_diarize(samples, sr)
        return self._vad_fallback(samples, sr, vad_segments or [])

    def _nemo_diarize(self, samples: np.ndarray, sr: int) -> list[DiarizedSegment]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            audio_path = tmp_dir / "input.wav"
            save_wav(samples, sr, audio_path)

            # NeMo diarizer expects a manifest JSON
            manifest_path = tmp_dir / "manifest.json"
            import json
            manifest_path.write_text(
                json.dumps({
                    "audio_filepath": str(audio_path),
                    "offset": 0,
                    "duration": len(samples) / sr,
                    "label": "infer",
                    "text": "-",
                    "num_speakers": self._max_speakers,
                    "rttm_filepath": None,
                    "uem_filepath": None,
                }),
                encoding="utf-8",
            )

            self._model.diarize(manifest_filepath=str(manifest_path))

            rttm_path = tmp_dir / "pred_rttms" / "input.rttm"
            if not rttm_path.exists():
                logger.warning("No RTTM output from diarizer. Falling back to single speaker.")
                return self._single_speaker(samples, sr)

            return self._rttm_to_segments(rttm_path, samples, sr)

    def _rttm_to_segments(
        self, rttm_path: Path, samples: np.ndarray, sr: int
    ) -> list[DiarizedSegment]:
        segments: list[DiarizedSegment] = []
        for line in rttm_path.read_text().splitlines():
            # RTTM format: SPEAKER <file> <chn> <start> <dur> <NA> <NA> <spk> <NA> <NA>
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start    = float(parts[3])
            duration = float(parts[4])
            speaker  = parts[7]
            end      = start + duration

            start_sample = int(start * sr)
            end_sample   = min(int(end * sr), len(samples))
            seg_samples  = samples[start_sample:end_sample]

            if len(seg_samples) < int(0.1 * sr):
                continue

            segments.append(DiarizedSegment(
                speaker=speaker,
                start=round(start, 3),
                end=round(end, 3),
                samples=seg_samples,
            ))

        return sorted(segments, key=lambda s: s.start)

    def _vad_fallback(
        self, samples: np.ndarray, sr: int, vad_segments: list[SpeechSegment]
    ) -> list[DiarizedSegment]:
        """Single-speaker mode using VAD segments — used when MSDD is unavailable."""
        if not vad_segments:
            return self._single_speaker(samples, sr)

        result = []
        for seg in vad_segments:
            s = int(seg.start * sr)
            e = min(int(seg.end * sr), len(samples))
            result.append(DiarizedSegment(
                speaker="SPEAKER_00",
                start=seg.start,
                end=seg.end,
                samples=samples[s:e],
            ))
        return result

    def _single_speaker(self, samples: np.ndarray, sr: int) -> list[DiarizedSegment]:
        return [DiarizedSegment(
            speaker="SPEAKER_00",
            start=0.0,
            end=round(len(samples) / sr, 3),
            samples=samples,
        )]
