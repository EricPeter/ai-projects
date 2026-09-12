"""
SpeechLM2 / SALM integration for richer translation.

SALM (Speech-Augmented Language Model) fuses speech embeddings with an LLM,
enabling instruction-following on top of ASR output.  We use it as a
translation + post-processing step: given the Luganda transcript, SALM can
translate, summarise, or answer questions about the content.

SALMAutomodel (NeMo Automodel variant) is used here for simpler loading and
LoRA fine-tuning support.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SALMTranslator:
    """Wraps SALM for Luganda → English translation with LLM capabilities."""

    def __init__(self, model_id: str = "nvidia/salm"):
        self._model_id = model_id
        self._model: Any = None

    def load(self) -> None:
        try:
            from nemo.collections.speechlm2 import SALMAutomodel
        except ImportError as e:
            raise RuntimeError(
                "NeMo SpeechLM2 not available. Ensure nemo_toolkit[speechlm2] is installed."
            ) from e

        logger.info("Loading SALM model: %s", self._model_id)
        self._model = SALMAutomodel.from_pretrained(self._model_id)
        self._model.eval()
        logger.info("SALM ready.")

    def translate(
        self,
        transcript: str,
        target_lang: str = "English",
        instruction: str | None = None,
    ) -> str:
        """
        Translate a Luganda transcript to the target language using SALM.

        Optionally accepts a custom instruction for instruction-following tasks
        (e.g. summarise, answer a clinical question from the transcript).
        """
        if self._model is None:
            raise RuntimeError("Call load() before translate()")

        prompt = instruction or (
            f"Translate the following Luganda text to {target_lang}. "
            "Output only the translation.\n\n"
            f"Luganda: {transcript}\n{target_lang}:"
        )
        result = self._model.generate(text=prompt)
        return result.strip()

    def answer(self, transcript: str, question: str) -> str:
        """Answer a question about the speech content — useful for health-bot use cases."""
        instruction = (
            f"The following is a transcription from a Luganda speaker:\n\n"
            f"{transcript}\n\n"
            f"Question: {question}\n"
            f"Answer in English:"
        )
        return self.translate(transcript, instruction=instruction)
