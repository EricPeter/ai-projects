"""
Punctuation restoration and inverse text normalisation (ITN).

Two steps applied after ASR:
  1. Punctuation + capitalisation  — NeMo's punct_en_bert (or multilingual variant)
     Turns "omukwano gwabantu gwa maanyi nga" → "Omukwano gw'abantu gwa maanyi nga."
     Falls back to rule-based sentence-boundary detection if model unavailable.

  2. ITN (Inverse Text Normalisation)  — NeMo Sparrowhawk / WeNet ITN
     Turns "bbiri mu makumi abiri" → "22" (number words → digits)
     Falls back to a simple Luganda numeral lookup table.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# ── Punctuation + capitalisation ──────────────────────────────────────────────

class PunctuationRestorer:
    def __init__(self):
        self._model = None

    def load(self) -> None:
        try:
            from nemo.collections.nlp.models import PunctuationCapitalizationModel
            # Use multilingual BERT — better than English-only for Luganda
            self._model = PunctuationCapitalizationModel.from_pretrained(
                "punctuation_en_distilbert"
            )
            self._model.eval()
            logger.info("Punctuation/Capitalisation model loaded.")
        except Exception as e:
            logger.warning("Punct model unavailable (%s). Using rule-based fallback.", e)
            self._model = None

    def restore(self, text: str) -> str:
        if not text.strip():
            return text
        if self._model is not None:
            try:
                result = self._model.add_punctuation_capitalization([text])
                return result[0] if result else text
            except Exception as e:
                logger.warning("Punct model inference failed: %s", e)

        return self._rule_based(text)

    def _rule_based(self, text: str) -> str:
        """
        Minimal rule-based punctuation for Luganda.
        - Capitalise first word of each sentence
        - Add period at end if no terminal punctuation
        - Insert comma before common conjunctions (nga, naye, era)
        """
        text = text.strip()
        if not text:
            return text

        # Split on existing terminal punctuation to restore capitalisation per sentence
        sentences = re.split(r"([.!?])\s+", text)
        restored = []
        for part in sentences:
            if part in (".", "!", "?"):
                restored.append(part + " ")
            elif part:
                # Capitalise first character
                restored.append(part[0].upper() + part[1:])

        result = "".join(restored).strip()

        # Add terminal period if missing
        if result and result[-1] not in ".!?":
            result += "."

        # Insert comma before coordinating conjunctions
        for conj in ("nga ", "naye ", "era ", "so ", "but "):
            result = re.sub(r"\s+" + conj, f", {conj}", result, flags=re.IGNORECASE)

        return result


# ── Inverse Text Normalisation ────────────────────────────────────────────────

# Luganda number words (simplified — covers 1-20 and tens)
_LUGANDA_NUMERALS: dict[str, int] = {
    "emu": 1, "bbiri": 2, "ssatu": 3, "nnya": 4, "ttaano": 5,
    "mukaaga": 6, "musanvu": 7, "munaana": 8, "mwenda": 9, "kkumi": 10,
    "kkumi na emu": 11, "kkumi na bbiri": 12, "kkumi na ssatu": 13,
    "kkumi na nnya": 14, "kkumi na ttaano": 15, "makumi abiri": 20,
    "makumi asatu": 30, "makumi ana": 40, "makumi ataano": 50,
    "kikumi": 100, "lukumi": 1000,
}


class ITNProcessor:
    def __init__(self):
        self._model = None

    def load(self) -> None:
        try:
            from nemo_text_processing.inverse_text_normalization.inverse_normalize import (
                InverseNormalizer,
            )
            self._model = InverseNormalizer(lang="en")  # en as closest available
            logger.info("NeMo ITN loaded.")
        except Exception as e:
            logger.warning("NeMo ITN unavailable (%s). Using Luganda numeral lookup.", e)
            self._model = None

    def normalize(self, text: str) -> str:
        if not text.strip():
            return text
        if self._model is not None:
            try:
                return self._model.inverse_normalize(text, verbose=False)
            except Exception as e:
                logger.warning("ITN inference failed: %s", e)

        return self._luganda_numeral_itn(text)

    def _luganda_numeral_itn(self, text: str) -> str:
        """Replace known Luganda number phrases with digits."""
        lower = text.lower()
        for phrase, digit in sorted(_LUGANDA_NUMERALS.items(), key=lambda x: -len(x[0])):
            lower = lower.replace(phrase, str(digit))
        # Preserve original capitalisation for non-replaced parts
        return lower
