"""
Word/phrase boosting for GPU-PB (GPU Prefix-Boosting) decoding.

GPU-PB is NeMo's GPU-accelerated phrase boosting built on the NGPU-LM
infrastructure. It is applied at decode time (shallow fusion) for CTC, RNN-T/TDT
and AED (Canary) models and does NOT require retraining. For AED models it is
configured on the beam decoding config via a ``boosting_tree`` block, then
activated with ``model.change_decoding_strategy(...)`` — see ``asr.py``.

This module only loads the phrase list; applying it to a model lives in
``CanaryASR`` because the decoding config is model-specific.

NOTE: Canary is a cased model, so for the *pretrained* checkpoint boost phrases
should be capitalised. Here we fine-tune on lowercase Luganda text, so the
lowercase phrases in ``data/word_lists/`` match the model's output.
"""

from __future__ import annotations

from pathlib import Path


def load_word_lists(word_lists_dir: Path) -> list[str]:
    """Load and de-duplicate boost phrases from every ``*.txt`` in the dir.

    Blank lines and ``#`` comments are ignored. Order is preserved.
    """
    words: list[str] = []
    if not word_lists_dir.exists():
        return words
    for txt_file in sorted(word_lists_dir.glob("*.txt")):
        for line in txt_file.read_text(encoding="utf-8").splitlines():
            word = line.strip()
            if word and not word.startswith("#"):
                words.append(word)

    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            unique.append(w)
    return unique
