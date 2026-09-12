"""
Build a Luganda SentencePiece (BPE/unigram) tokenizer for the streaming
FastConformer fine-tune.

Cache-aware streaming Conformer/Parakeet is a character/sub-word transducer, so
adapting it to Luganda means a Luganda tokenizer (the pretrained English
tokenizer would waste capacity and mis-segment). We train an SPE unigram model
(recommended vocab 1024) from the manifest transcripts.

Usage:
  python scripts/build_tokenizer.py \
      --manifests data/manifests/train.json data/manifests/dev.json \
      --vocab-size 1024 --out data/tokenizers

Then point the streaming config at the printed directory:
  model.tokenizer.dir=<printed dir>  model.tokenizer.type=bpe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_corpus(manifests: list[Path], out_file: Path) -> int:
    lines: list[str] = []
    for m in manifests:
        if not m.exists():
            print(f"  (skipping missing manifest: {m})")
            continue
        for line in m.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                text = json.loads(line).get("text", "").strip()
            except json.JSONDecodeError:
                continue
            if text:
                lines.append(text)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Corpus: {len(lines):,} lines -> {out_file}")
    return len(lines)


def build(corpus: Path, out_dir: Path, vocab_size: int) -> Path:
    try:
        from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model
    except ImportError as e:
        raise SystemExit(
            f"NeMo not installed ({e}). Run inside the NeMo container, or "
            "build the tokenizer with scripts/tokenizers/process_asr_text_tokenizer.py."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    # create_spt_model writes tokenizer.model + vocab into out_dir.
    model_path, vocab_path = create_spt_model(
        data_file=str(corpus),
        vocab_size=vocab_size,
        sample_size=-1,
        do_lower_case=True,
        output_dir=str(out_dir),
        tokenizer_type="unigram",
        character_coverage=1.0,
    )
    print(f"Tokenizer model : {model_path}")
    print(f"Tokenizer vocab : {vocab_path}")
    print(f"\nSet in config:  model.tokenizer.dir={out_dir}  model.tokenizer.type=bpe")
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", type=Path,
                        default=[Path("data/manifests/train.json"),
                                 Path("data/manifests/dev.json")])
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=Path("data/tokenizers/luganda_spe"))
    args = parser.parse_args()

    corpus_file = args.out.parent / "luganda_corpus.txt"
    n = collect_corpus(args.manifests, corpus_file)
    if n < 10:
        raise SystemExit("Not enough transcript text to train a tokenizer (need >=10 lines).")
    build(corpus_file, args.out, args.vocab_size)
