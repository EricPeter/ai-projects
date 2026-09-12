"""
Train a 5-gram KenLM language model on Luganda text.

The ARPA file produced here is consumed by NeMo's NGPU-LM decoder
(GPU-accelerated n-gram fusion for CTC / RNN-T / AED models).

Prerequisites:
  - KenLM installed: https://github.com/kpu/kenlm
    brew install kenlm   OR   apt-get install kenlm

Usage:
  python scripts/train_ngram_lm.py [--order 5] [--output outputs/ngram_lm]
"""

import argparse
import subprocess
from pathlib import Path


CORPUS_FILES = [
    "data/ngram/luganda_bible.txt",
    "data/ngram/luganda_wikipedia.txt",
    "data/ngram/luganda_transcripts.txt",
]


def extract_transcripts(out_path: Path) -> None:
    """Pull text from training manifests into a flat corpus file."""
    import json
    manifest_dir = Path("data/manifests")
    lines = []
    for manifest in manifest_dir.glob("*.json"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                text = json.loads(line).get("text", "").strip()
                if text:
                    lines.append(text)
            except json.JSONDecodeError:
                pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Extracted {len(lines):,} transcript lines → {out_path}")


def merge_corpora(tmp_path: Path) -> None:
    parts = []
    for p in CORPUS_FILES:
        path = Path(p)
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
        else:
            print(f"  (skipping missing corpus: {p})")
    tmp_path.write_text("\n".join(parts), encoding="utf-8")
    total_lines = sum(t.count("\n") for t in parts)
    print(f"Merged corpus: {total_lines:,} lines → {tmp_path}")


def train_kenlm(corpus: Path, out_dir: Path, order: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    arpa_path = out_dir / f"luganda_{order}gram.arpa"
    binary_path = out_dir / f"luganda_{order}gram.binary"

    print(f"Training {order}-gram LM with KenLM...")
    subprocess.run(
        [
            "lmplz",
            "-o", str(order),
            "--discount_fallback",
            "--prune", "0", "0", "1", "1", "1",
            "--text", str(corpus),
            "--arpa", str(arpa_path),
        ],
        check=True,
    )
    print(f"ARPA → {arpa_path}")

    print("Building binary for faster loading...")
    subprocess.run(
        ["build_binary", str(arpa_path), str(binary_path)],
        check=True,
    )
    print(f"Binary → {binary_path}")
    return arpa_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/ngram_lm"))
    args = parser.parse_args()

    # Ensure transcript corpus is fresh
    extract_transcripts(Path("data/ngram/luganda_transcripts.txt"))

    tmp_corpus = args.output / "merged_corpus.txt"
    merge_corpora(tmp_corpus)
    train_kenlm(tmp_corpus, args.output, args.order)
