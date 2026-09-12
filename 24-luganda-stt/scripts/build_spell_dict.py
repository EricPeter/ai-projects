"""
Build a SymSpell frequency dictionary from Luganda text corpora.

Sources used (in priority order):
  1. Training manifests (transcripts from data/manifests/)
  2. Plain text files in data/ngram/
  3. Word boost lists (data/word_lists/*.txt) — given artificially high frequency

Output: data/spell/luganda_frequency.txt
  Format: <word> <frequency>  (space-separated, one per line)

Usage:
  python scripts/build_spell_dict.py
"""

import json
import re
from collections import Counter
from pathlib import Path


MANIFEST_DIR = Path("data/manifests")
NGRAM_DIR    = Path("data/ngram")
WORD_LIST_DIR = Path("data/word_lists")
OUT_PATH     = Path("data/spell/luganda_frequency.txt")


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower())


def collect_from_manifests(counter: Counter) -> int:
    total = 0
    for manifest in MANIFEST_DIR.glob("*.json"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                text = json.loads(line).get("text", "")
                tokens = tokenise(text)
                counter.update(tokens)
                total += len(tokens)
            except json.JSONDecodeError:
                pass
    return total


def collect_from_text_files(counter: Counter) -> int:
    total = 0
    for txt in NGRAM_DIR.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            tokens = tokenise(line)
            counter.update(tokens)
            total += len(tokens)
    return total


def collect_from_word_lists(counter: Counter) -> int:
    total = 0
    for txt in WORD_LIST_DIR.glob("*.txt"):
        for line in txt.read_text(encoding="utf-8").splitlines():
            word = line.strip().lower()
            if word and not word.startswith("#"):
                # High frequency so spell checker never flags boosted domain words
                counter[word] += 1000
                total += 1
    return total


def main():
    counter: Counter = Counter()

    n_manifest = collect_from_manifests(counter)
    n_text     = collect_from_text_files(counter)
    n_boost    = collect_from_word_lists(counter)

    print(f"Tokens from manifests : {n_manifest:,}")
    print(f"Tokens from text files: {n_text:,}")
    print(f"Words from boost lists: {n_boost:,}")
    print(f"Unique words          : {len(counter):,}")

    # Drop single-character and very rare words
    filtered = {w: c for w, c in counter.items() if len(w) >= 2 and c >= 2}
    print(f"After filtering       : {len(filtered):,}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for word, count in sorted(filtered.items(), key=lambda x: -x[1]):
            f.write(f"{word} {count}\n")

    print(f"Dictionary written → {OUT_PATH}")


if __name__ == "__main__":
    main()
