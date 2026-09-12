"""
Streaming speaker diarization with NeMo Sortformer.

Uses ``SortformerEncLabelModel`` (nvidia/diar_streaming_sortformer_4spk-v2), an
end-to-end streaming diarizer with an Arrival-Order Speaker Cache (AOSC). It
runs online, supports up to 4 speakers, and is the diarizer paired with the
cache-aware streaming ASR in NVIDIA's voice-agent reference.

For the turn-based live pipeline we attribute each finalized turn to its dominant
speaker via ``speaker_for_turn``. ``diarize_segments`` exposes the full segment
list for callers that want per-word attribution.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class DiarSegment:
    start: float
    end: float
    speaker: str


class StreamingDiarizer:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None

    def load(self) -> None:
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as e:  # pragma: no cover - requires NeMo
            raise RuntimeError("NeMo not installed. Run: pip install nemo_toolkit[asr]") from e

        logger.info("Loading Sortformer streaming diarizer: %s", self._settings.diar_streaming_model)
        self._model = SortformerEncLabelModel.from_pretrained(self._settings.diar_streaming_model)
        self._model.eval()

        # Configure the streaming chunk if the model exposes streaming params.
        # (Attribute names vary by NeMo version; applied best-effort.)
        cfg = getattr(self._model, "cfg", None)
        try:
            if cfg is not None and hasattr(cfg, "streaming"):
                # chunk length in mel frames (~10 ms/frame); keep small for latency.
                cfg.streaming.chunk_len = int(self._settings.diar_chunk_secs * 100)
        except Exception as e:  # noqa: BLE001
            logger.debug("Sortformer streaming cfg not adjusted (%s).", e)

        logger.info("Sortformer diarizer ready.")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ── Inference ──────────────────────────────────────────────────────────────

    def diarize_segments(self, samples: np.ndarray, sample_rate: int = 16000) -> list[DiarSegment]:
        if self._model is None:
            raise RuntimeError("Call load() before diarize")

        audio = np.asarray(samples, dtype=np.float32)
        try:
            preds = self._model.diarize(audio=audio, batch_size=1, sample_rate=sample_rate)
        except TypeError:
            # older signature without sample_rate kwarg
            preds = self._model.diarize(audio=audio, batch_size=1)

        return _parse_sortformer_output(preds)

    def speaker_for_turn(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """Dominant speaker (by total voiced time) over the turn's audio."""
        segs = self.diarize_segments(samples, sample_rate)
        if not segs:
            return "speaker_0"
        totals: dict[str, float] = defaultdict(float)
        for s in segs:
            totals[s.speaker] += max(0.0, s.end - s.start)
        return max(totals, key=totals.get)


def _parse_sortformer_output(preds) -> list[DiarSegment]:
    """Normalise Sortformer's diarize() return into DiarSegment list.

    NeMo returns one prediction per input; each is typically a list of RTTM-style
    strings "start end speaker". We defensively handle strings, tuples, and
    objects with start/end/speaker attributes.
    """
    if not preds:
        return []
    first = preds[0] if isinstance(preds, (list, tuple)) else preds

    segments: list[DiarSegment] = []
    items = first if isinstance(first, (list, tuple)) else [first]
    for entry in items:
        seg = _coerce_segment(entry)
        if seg is not None:
            segments.append(seg)
    return segments


def _coerce_segment(entry) -> DiarSegment | None:
    if isinstance(entry, str):
        parts = entry.split()
        # RTTM-ish "start end speaker" or "start end speaker label..."
        try:
            if len(parts) >= 3:
                return DiarSegment(float(parts[0]), float(parts[1]), str(parts[2]))
        except ValueError:
            return None
        return None
    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
        try:
            return DiarSegment(float(entry[0]), float(entry[1]), str(entry[2]))
        except (ValueError, TypeError):
            return None
    if all(hasattr(entry, a) for a in ("start", "end", "speaker")):
        return DiarSegment(float(entry.start), float(entry.end), str(entry.speaker))
    return None
