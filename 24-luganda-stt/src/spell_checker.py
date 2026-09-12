"""
Luganda spell checker backed by SymSpell.

SymSpell uses a frequency dictionary and edit-distance lookup to correct OOV
words produced by the ASR decoder.  It's language-agnostic — you train it on
any word-frequency list.

Build the frequency dict once with scripts/build_spell_dict.py, then load it
here at startup.
"""

from __future__ import annotations

import re
from pathlib import Path

from symspellpy import SymSpell, Verbosity


class LugandaSpellChecker:
    def __init__(self, dict_path: Path, max_edit_distance: int = 2):
        self._sym = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        if dict_path.exists():
            self._sym.load_dictionary(
                str(dict_path),
                term_index=0,
                count_index=1,
                separator=" ",
            )
            self._loaded = True
        else:
            # No dictionary yet — pass-through mode until built
            self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def correct_word(self, word: str) -> str:
        if not self._loaded:
            return word
        suggestions = self._sym.lookup(
            word.lower(), Verbosity.CLOSEST, max_edit_distance=2
        )
        if not suggestions:
            return word
        corrected = suggestions[0].term
        # Preserve original capitalisation
        if word[0].isupper():
            corrected = corrected.capitalize()
        return corrected

    def correct(self, text: str) -> str:
        """Correct each token in a transcript string."""
        if not self._loaded or not text:
            return text

        tokens = re.split(r"(\s+|[.,!?;:])", text)
        corrected = []
        for tok in tokens:
            if re.match(r"[a-zA-ZÀ-ÿ]{2,}", tok):
                corrected.append(self.correct_word(tok))
            else:
                corrected.append(tok)
        return "".join(corrected)

    def add_to_dictionary(self, word: str, count: int = 1) -> None:
        """Add a domain-specific word at runtime (used when loading word-boost lists)."""
        self._sym.create_dictionary_entry(word.lower(), count)
        self._loaded = True
