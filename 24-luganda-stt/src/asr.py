"""
Canary-1B-Flash ASR engine for Luganda.

Handles:
  - Model loading (pre-trained or fine-tuned checkpoint)
  - NGPU-LM fusion (GPU n-gram language model) at decode time
  - GPU-PB phrase boosting at decode time (configured on the beam decoding cfg)
  - Word-level timestamps + confidence scores

Luganda note: Canary-1B-Flash ships prompt/language tokens for en/de/fr/es only.
Luganda is mapped onto the English slot (source_lang/target_lang = "en"); its BPE
tokenizer already covers Latin-script Luganda text and the fine-tuned model learns
the rest. All transcribe calls therefore use "en".
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Settings
from .word_booster import load_word_lists

logger = logging.getLogger(__name__)

# Canary has no Luganda token; we route Luganda through the English prompt slot.
LANG = "en"


@dataclass
class WordResult:
    word: str
    start: float       # seconds, relative to segment start
    end: float
    confidence: float


class CanaryASR:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._boost_words = load_word_lists(settings.word_lists_dir)

    def load(self) -> None:
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as e:
            raise RuntimeError("NeMo not installed. Run: pip install nemo_toolkit[asr]") from e

        checkpoint = self._settings.finetuned_checkpoint
        if checkpoint and Path(checkpoint).exists():
            logger.info("Loading fine-tuned checkpoint: %s", checkpoint)
            self._model = nemo_asr.models.EncDecMultiTaskModel.restore_from(str(checkpoint))
        else:
            logger.info("Downloading pre-trained model: %s", self._settings.canary_model)
            self._model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(
                self._settings.canary_model
            )

        self._model.eval()
        self._configure_decoding()

        logger.info(
            "Canary ASR ready. boosted_words=%d  ngram_lm=%s",
            len(self._boost_words),
            self._settings.ngram_lm_path.exists(),
        )

    def _configure_decoding(self) -> None:
        """Set up beam decoding with GPU-PB phrase boosting and NGPU-LM fusion.

        These features are version-sensitive (GPU-PB requires a recent NeMo).
        We apply them best-effort and log — never crash the server — if the
        installed NeMo lacks the corresponding config fields.
        """
        from omegaconf import open_dict

        try:
            decode_cfg = self._model.cfg.decoding
            with open_dict(decode_cfg):
                decode_cfg.strategy = "beam"
                if decode_cfg.get("beam") is None:
                    decode_cfg.beam = {}
                decode_cfg.beam.beam_size = self._settings.beam_size

                # GPU-PB phrase boosting (NGPU-LM infrastructure). depth_scaling
                # 1.0 is recommended for Canary/AED models.
                if self._boost_words:
                    decode_cfg.beam.boosting_tree = {
                        "key_phrases_list": list(self._boost_words),
                        "context_score": 1.0,
                        "depth_scaling": 1.0,
                    }
                    decode_cfg.beam.boosting_tree_alpha = self._settings.word_boost_factor

                # NGPU-LM n-gram shallow fusion, if a trained ARPA is present.
                lm_path = self._settings.ngram_lm_path
                if lm_path.exists():
                    decode_cfg.beam.ngram_lm_model = str(lm_path)
                    decode_cfg.beam.ngram_lm_alpha = 0.5

            self._model.change_decoding_strategy(decode_cfg)
        except Exception as e:  # noqa: BLE001 — decoding config is best-effort
            logger.warning(
                "Could not apply beam/boosting decoding config (%s). "
                "Falling back to the model's default decoding.", e,
            )

    def _transcribe_files(self, paths: list[str], *, target_lang: str = LANG, pnc: bool = False):
        """Call model.transcribe with kwargs that degrade across NeMo versions."""
        base = dict(audio=paths, source_lang=LANG, target_lang=target_lang,
                    pnc=pnc, return_hypotheses=True)
        try:
            return self._model.transcribe(**base, timestamps=True)
        except TypeError:
            return self._model.transcribe(**base)

    # ── Transcription ─────────────────────────────────────────────────────────

    def transcribe(self, audio_bytes: bytes, source_lang: str = LANG) -> str:
        text, _, _ = self.transcribe_with_details(audio_bytes, source_lang)
        return text

    def transcribe_with_details(
        self, audio_bytes: bytes, source_lang: str = LANG
    ) -> tuple[str, list[WordResult], float]:
        """
        Returns (transcript, word_results, mean_confidence).
        word_results contains per-word start/end timestamps and confidence.
        """
        if self._model is None:
            raise RuntimeError("Call load() before transcribe()")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            f.write(audio_bytes)

        try:
            hypotheses = self._transcribe_files([tmp_path])
            hyp = hypotheses[0] if hypotheses else None
            if hyp is None:
                return "", [], 0.0

            text = hyp.text if hasattr(hyp, "text") else str(hyp)
            words = self._extract_words(hyp)
            confidence = np.mean([w.confidence for w in words]) if words else 0.0

            return text, words, float(confidence)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _extract_words(self, hypothesis) -> list[WordResult]:
        """Extract word-level timestamps and confidence from a NeMo hypothesis."""
        words: list[WordResult] = []

        # Recent NeMo exposes hyp.timestamp = {"word": [...], "segment": [...]}.
        ts = None
        tdict = getattr(hypothesis, "timestamp", None)
        if isinstance(tdict, dict):
            ts = tdict.get("word")
        if ts is None:
            ts = getattr(hypothesis, "word_timestamps", None) or getattr(hypothesis, "timestep", None)
        conf = getattr(hypothesis, "token_confidence", None)

        text = getattr(hypothesis, "text", "") or ""
        if ts is None:
            # No timestamp info — return token-split words with dummy timing.
            for i, tok in enumerate(text.split()):
                words.append(WordResult(word=tok, start=i * 0.3, end=(i + 1) * 0.3, confidence=0.9))
            return words

        frame_duration = 0.04  # 40 ms per output frame for Canary
        for i, entry in enumerate(ts):
            if isinstance(entry, dict):
                word = entry.get("word", entry.get("char", ""))
                if "start" in entry:                      # seconds
                    start = entry["start"]
                    end = entry.get("end", start)
                else:                                     # frame offsets
                    start = entry.get("start_offset", i) * frame_duration
                    end = entry.get("end_offset", i + 1) * frame_duration
                c = conf[i] if conf and i < len(conf) else 0.9
            elif hasattr(entry, "word"):
                word = entry.word
                start = getattr(entry, "start_time", i * frame_duration)
                end = getattr(entry, "end_time", (i + 1) * frame_duration)
                c = getattr(entry, "confidence", 0.9)
            else:
                continue
            words.append(WordResult(word=str(word), start=float(start), end=float(end), confidence=float(c)))

        return words

    # ── Translation ───────────────────────────────────────────────────────────

    def translate(self, audio_bytes: bytes, target_lang: str = "en") -> str:
        """Direct speech-to-translation via Canary's S2T translation task.

        Base Canary translates among en/de/fr/es only; with Luganda mapped to the
        English slot this is effectively a pass-through unless you fine-tune a
        translation task. Kept for API compatibility.
        """
        if self._model is None:
            raise RuntimeError("Call load() before translate()")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            f.write(audio_bytes)

        try:
            hypotheses = self._transcribe_files([tmp_path], target_lang=target_lang, pnc=True)
            hyp = hypotheses[0] if hypotheses else None
            return hyp.text if (hyp and hasattr(hyp, "text")) else str(hyp or "")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @property
    def boosted_words(self) -> list[str]:
        return self._boost_words
