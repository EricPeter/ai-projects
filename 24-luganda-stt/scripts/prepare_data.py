"""
Download and format Luganda speech data into NeMo manifest format.

NeMo manifest format (one JSON object per line):
  {"audio_filepath": "...", "duration": 3.2, "text": "...", "task": "asr",
   "source_lang": "lug", "target_lang": "lug"}

Sources tried in order (automatic):
  1. google/fleurs  — "lg_ug" config, reliable HuggingFace availability
  2. mozilla-foundation/common_voice_16_1 — older CV versions that have lg splits
  3. mozilla-foundation/common_voice_11_0 — earliest version with some Luganda
  4. Manual Common Voice archive (--cv-dir)

Usage:
  python scripts/prepare_data.py                        # auto-download
  python scripts/prepare_data.py --source fleurs        # force FLEURS
  python scripts/prepare_data.py --source commonvoice   # force Common Voice
  python scripts/prepare_data.py --cv-dir /path/to/cv   # local CV archive
  python scripts/prepare_data.py --max-samples 1000
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import wave
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

MANIFEST_DIR = Path("data/manifests")
AUDIO_DIR    = MANIFEST_DIR / "audio"


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_record(wav_path: Path, text: str, duration: float) -> dict:
    return {
        "audio_filepath": str(wav_path.resolve()),
        "duration": round(duration, 2),
        "text": text,
        # Canary-1B-Flash has no Luganda token, so we map Luganda onto the
        # English prompt slot ("en"): its BPE tokenizer already covers
        # Latin-script Luganda text. FLEURS/CV Luganda text is lowercase and
        # unpunctuated, so pnc="no".
        "taskname": "asr",
        "pnc": "no",
        "source_lang": "en",
        "target_lang": "en",
    }


def save_wav(array, sample_rate: int, out_path: Path) -> float:
    sf.write(str(out_path), array, sample_rate)
    return len(array) / sample_rate


def write_manifest(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_records(records: list[dict]) -> dict[str, list[dict]]:
    n = len(records)
    return {
        "train": records[: int(n * 0.8)],
        "dev":   records[int(n * 0.8) : int(n * 0.9)],
        "test":  records[int(n * 0.9) :],
    }


# ── Source: FLEURS (google/fleurs, lg_ug) ────────────────────────────────────

def download_fleurs(max_samples: int) -> bool:
    """
    FLEURS has Luganda under the config name 'lg_ug'.
    It contains ~1 500 utterances — small but clean and well-labelled.
    """
    from datasets import load_dataset

    print("Trying google/fleurs (lg_ug)...")
    try:
        ds = load_dataset("google/fleurs", "lg_ug", trust_remote_code=True)
    except Exception as e:
        print(f"  FLEURS failed: {e}")
        return False

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for split_name in ("train", "validation", "test"):
        split = ds.get(split_name)
        if split is None:
            continue
        for i, row in enumerate(tqdm(split, desc=f"  fleurs/{split_name}")):
            if len(records) >= max_samples:
                break
            audio = row["audio"]
            text  = row["transcription"].strip()
            if not text:
                continue
            wav_path = AUDIO_DIR / f"fleurs_{split_name}_{i:05d}.wav"
            duration = save_wav(audio["array"], audio["sampling_rate"], wav_path)
            if 0.5 <= duration <= 30.0:
                records.append(make_record(wav_path, text, duration))

    if not records:
        print("  FLEURS returned no usable rows.")
        return False

    splits = split_records(records)
    for name, rows in splits.items():
        write_manifest(rows, MANIFEST_DIR / f"{name}.json")
        print(f"  {name}: {len(rows)} examples")
    print(f"Total from FLEURS: {len(records)}")
    return True


# ── Source: Common Voice (HuggingFace) ───────────────────────────────────────

CV_VERSIONS = [
    "mozilla-foundation/common_voice_16_1",
    "mozilla-foundation/common_voice_13_0",
    "mozilla-foundation/common_voice_11_0",
]


def download_commonvoice_hf(max_samples: int) -> bool:
    from datasets import load_dataset

    for version in CV_VERSIONS:
        print(f"Trying {version} (lg)...")
        try:
            ds = load_dataset(version, "lg", split="train+validation+test",
                              trust_remote_code=True)
            print(f"  Loaded {len(ds)} rows from {version}.")
        except Exception as e:
            print(f"  {version}: {e}")
            continue

        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []

        for i, row in enumerate(tqdm(ds, total=min(max_samples, len(ds)), desc="  converting")):
            if len(records) >= max_samples:
                break
            audio = row["audio"]
            text  = row.get("sentence", row.get("transcription", "")).strip()
            if not text:
                continue
            wav_path = AUDIO_DIR / f"cv_{i:06d}.wav"
            duration = save_wav(audio["array"], audio["sampling_rate"], wav_path)
            if 0.5 <= duration <= 30.0:
                records.append(make_record(wav_path, text, duration))

        if not records:
            print(f"  No usable rows from {version}.")
            continue

        splits = split_records(records)
        for name, rows in splits.items():
            write_manifest(rows, MANIFEST_DIR / f"{name}.json")
            print(f"  {name}: {len(rows)} examples")
        print(f"Total from {version}: {len(records)}")
        return True

    return False


# ── Source: local Common Voice archive ───────────────────────────────────────

def convert_local_cv(cv_dir: Path, max_samples: int) -> bool:
    """Convert a manually downloaded Common Voice .tar.gz extraction."""
    clips_dir = cv_dir / "clips"
    if not clips_dir.exists():
        print(f"  {clips_dir} not found")
        return False

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    all_records: dict[str, list[dict]] = {}

    for split in ("train", "dev", "test"):
        tsv = cv_dir / f"{split}.tsv"
        if not tsv.exists():
            print(f"  Skipping missing {tsv}")
            continue

        records: list[dict] = []
        with open(tsv, encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            path_idx     = header.index("path")
            sentence_idx = header.index("sentence")
            for line in tqdm(f, desc=f"  {split}"):
                if len(records) >= max_samples:
                    break
                parts = line.strip().split("\t")
                if len(parts) <= max(path_idx, sentence_idx):
                    continue
                src = clips_dir / parts[path_idx]
                text = parts[sentence_idx].strip()
                if not src.exists() or not text:
                    continue
                try:
                    info = sf.info(str(src))
                    duration = info.duration
                except Exception:
                    continue
                if not (0.5 <= duration <= 30.0):
                    continue
                records.append(make_record(src, text, duration))

        all_records[split] = records
        write_manifest(records, MANIFEST_DIR / f"{split}.json")
        print(f"  {split}: {len(records)} examples")

    return bool(all_records)


# ── Fallback: stub manifests for pipeline smoke-testing ──────────────────────

def write_stub_manifests() -> None:
    """
    Write silent-audio stub manifests so the rest of the pipeline (LM training,
    spell dict, server smoke-test) can run without real speech data.
    """
    print("\nWriting stub manifests for pipeline testing...")

    stub_texts = [
        "omukwano gw'abantu gwa maanyi",
        "abantu ba Uganda balina amagezi mangi",
        "eddwaliro liri mu kibuga kya Kampala",
        "omwana akyawa okumala ekyakula",
        "Katonda awerera abantu obulamu",
    ]

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    sample_rate = 16000

    for i, text in enumerate(stub_texts * 6):  # 30 stubs total
        wav_path = AUDIO_DIR / f"stub_{i:04d}.wav"
        n = sample_rate  # 1 second of silence
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
        records.append(make_record(wav_path, text, 1.0))

    splits = split_records(records)
    for name, rows in splits.items():
        write_manifest(rows, MANIFEST_DIR / f"{name}.json")
        print(f"  stub {name}: {len(rows)} examples")

    print(
        "\nStub manifests written.  To use real data:\n"
        "  Option A: python scripts/prepare_data.py --source fleurs\n"
        "  Option B: download from https://commonvoice.mozilla.org/lg/datasets\n"
        "            then: python scripts/prepare_data.py --cv-dir /path/to/lg"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "fleurs", "commonvoice"],
                        default="auto", help="Data source to use")
    parser.add_argument("--cv-dir", type=Path,
                        help="Path to extracted local Common Voice archive")
    parser.add_argument("--output-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--max-samples", type=int, default=5000)
    args = parser.parse_args()

    # Override global MANIFEST_DIR if output-dir is given
    if args.output_dir != MANIFEST_DIR:
        MANIFEST_DIR = args.output_dir
        AUDIO_DIR    = MANIFEST_DIR / "audio"

    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        print("datasets not installed — run: pip install datasets soundfile")
        write_stub_manifests()
        raise SystemExit(0)

    if args.cv_dir:
        success = convert_local_cv(args.cv_dir, args.max_samples)
    elif args.source == "fleurs":
        success = download_fleurs(args.max_samples)
    elif args.source == "commonvoice":
        success = download_commonvoice_hf(args.max_samples)
    else:
        # auto: try fleurs first (most reliable), then CV
        success = download_fleurs(args.max_samples) or download_commonvoice_hf(args.max_samples)

    if not success:
        write_stub_manifests()
