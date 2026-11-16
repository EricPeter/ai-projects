"""
Download and format speech data for multiple Ugandan languages into the
SpeechLLM manifest format used by scripts/finetune_nemotron.py.

Manifest format (one JSON per line):
  {
    "audio_filepath": "/abs/path/to/audio.wav",
    "duration": 4.2,
    "context": "Transcribe the following Luganda audio verbatim.",
    "answer": "omukwano gw'abantu gwa maanyi",
    "language": "Luganda",
    "language_code": "lug"
  }

Sources (tried in order per language):
  - Sunbird AI SALT  — Luganda, Runyankore, Acholi, Ateso, Lugbara, English
  - google/fleurs    — Luganda (lg_ug)
  - Common Voice     — Luganda (lg)
  - AfriSpeech-200   — pan-African; filtered for Ugandan contributors

Usage:
  python scripts/prepare_data_multilang.py                   # all languages
  python scripts/prepare_data_multilang.py --langs lug nyn   # subset
  python scripts/prepare_data_multilang.py --max-per-lang 2000
  python scripts/prepare_data_multilang.py --output-dir data/manifests/multilang
"""

from __future__ import annotations

import argparse
import json
import struct
import wave
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

# ── Language registry ─────────────────────────────────────────────────────────

LANGUAGES: dict[str, dict] = {
    "lug": {
        "name": "Luganda",
        "fleurs_config": "lg_ug",
        "cv_locale": "lg",
        "salt_name": "luganda",
    },
    "nyn": {
        "name": "Runyankore",
        "fleurs_config": None,
        "cv_locale": None,
        "salt_name": "runyankore",
    },
    "ach": {
        "name": "Acholi",
        "fleurs_config": None,
        "cv_locale": None,
        "salt_name": "acholi",
    },
    "teo": {
        "name": "Ateso",
        "fleurs_config": None,
        "cv_locale": None,
        "salt_name": "ateso",
    },
    "lgg": {
        "name": "Lugbara",
        "fleurs_config": None,
        "cv_locale": None,
        "salt_name": "lugbara",
    },
    "nkl": {
        "name": "Runyoro",
        "fleurs_config": None,
        "cv_locale": None,
        "salt_name": "runyoro",
    },
}

DEFAULT_LANGS = list(LANGUAGES.keys())

# ── Helpers ───────────────────────────────────────────────────────────────────

def instruction(language_name: str) -> str:
    return f"Transcribe the following {language_name} audio verbatim."


def make_record(
    wav_path: Path,
    text: str,
    duration: float,
    lang_code: str,
) -> dict:
    info = LANGUAGES[lang_code]
    return {
        "audio_filepath": str(wav_path.resolve()),
        "duration": round(duration, 2),
        "context": instruction(info["name"]),
        "answer": text,
        "language": info["name"],
        "language_code": lang_code,
    }


def save_wav(array, sample_rate: int, out_path: Path) -> float:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), array, sample_rate)
    return len(array) / sample_rate


def write_manifest(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> {path}  ({len(records)} examples)")


def split_records(records: list[dict]) -> dict[str, list[dict]]:
    n = len(records)
    return {
        "train": records[: int(n * 0.8)],
        "dev":   records[int(n * 0.8) : int(n * 0.9)],
        "test":  records[int(n * 0.9) :],
    }


def acceptable(duration: float) -> bool:
    return 0.5 <= duration <= 30.0


# ── Source: Sunbird AI SALT ───────────────────────────────────────────────────

SALT_REPO = "Sunbird/salt"


def download_salt(lang_code: str, audio_dir: Path, max_samples: int) -> list[dict]:
    """
    SALT (Speech Annotation for Low-resource Transcription) from Sunbird AI /
    Makerere NLP.  Covers: Luganda, Runyankore, Acholi, Ateso, Lugbara (+ English).

    Dataset: https://huggingface.co/datasets/Sunbird/salt
    """
    from datasets import load_dataset

    salt_name = LANGUAGES[lang_code]["salt_name"]
    if not salt_name:
        return []

    print(f"  [SALT] {lang_code} ({salt_name})...")
    try:
        ds = load_dataset(SALT_REPO, salt_name, trust_remote_code=True)
    except Exception as e:
        print(f"    SALT/{salt_name} unavailable: {e}")
        return []

    records: list[dict] = []
    for split_name in ("train", "dev", "test"):
        split = ds.get(split_name)
        if split is None:
            continue
        for i, row in enumerate(tqdm(split, desc=f"    {split_name}")):
            if len(records) >= max_samples:
                break
            audio = row.get("audio") or {}
            text = (row.get("text") or row.get("transcription") or "").strip()
            if not text or not audio:
                continue
            wav_path = audio_dir / f"salt_{lang_code}_{split_name}_{i:06d}.wav"
            duration = save_wav(audio["array"], audio["sampling_rate"], wav_path)
            if acceptable(duration):
                records.append(make_record(wav_path, text, duration, lang_code))

    print(f"    SALT {lang_code}: {len(records)} usable examples")
    return records


# ── Source: FLEURS ────────────────────────────────────────────────────────────

def download_fleurs(lang_code: str, audio_dir: Path, max_samples: int) -> list[dict]:
    from datasets import load_dataset

    fleurs_config = LANGUAGES[lang_code]["fleurs_config"]
    if not fleurs_config:
        return []

    print(f"  [FLEURS] {lang_code} ({fleurs_config})...")
    try:
        ds = load_dataset("google/fleurs", fleurs_config, trust_remote_code=True)
    except Exception as e:
        print(f"    FLEURS/{fleurs_config} unavailable: {e}")
        return []

    records: list[dict] = []
    for split_name in ("train", "validation", "test"):
        split = ds.get(split_name)
        if split is None:
            continue
        for i, row in enumerate(tqdm(split, desc=f"    {split_name}")):
            if len(records) >= max_samples:
                break
            audio = row["audio"]
            text = row.get("transcription", row.get("raw_transcription", "")).strip()
            if not text:
                continue
            wav_path = audio_dir / f"fleurs_{lang_code}_{split_name}_{i:05d}.wav"
            duration = save_wav(audio["array"], audio["sampling_rate"], wav_path)
            if acceptable(duration):
                records.append(make_record(wav_path, text, duration, lang_code))

    print(f"    FLEURS {lang_code}: {len(records)} usable examples")
    return records


# ── Source: Common Voice ──────────────────────────────────────────────────────

CV_VERSIONS = [
    "mozilla-foundation/common_voice_17_0",
    "mozilla-foundation/common_voice_16_1",
    "mozilla-foundation/common_voice_13_0",
    "mozilla-foundation/common_voice_11_0",
]


def download_commonvoice(lang_code: str, audio_dir: Path, max_samples: int) -> list[dict]:
    from datasets import load_dataset

    cv_locale = LANGUAGES[lang_code]["cv_locale"]
    if not cv_locale:
        return []

    for version in CV_VERSIONS:
        print(f"  [CV] {lang_code} from {version}...")
        try:
            ds = load_dataset(
                version, cv_locale,
                split="train+validation+test",
                trust_remote_code=True,
            )
            print(f"    loaded {len(ds)} rows")
        except Exception as e:
            print(f"    {version}/{cv_locale}: {e}")
            continue

        records: list[dict] = []
        for i, row in enumerate(tqdm(ds, total=min(max_samples, len(ds)), desc="    converting")):
            if len(records) >= max_samples:
                break
            audio = row["audio"]
            text = (row.get("sentence") or row.get("transcription") or "").strip()
            if not text:
                continue
            wav_path = audio_dir / f"cv_{lang_code}_{i:06d}.wav"
            duration = save_wav(audio["array"], audio["sampling_rate"], wav_path)
            if acceptable(duration):
                records.append(make_record(wav_path, text, duration, lang_code))

        if records:
            print(f"    CV {lang_code}: {len(records)} usable examples")
            return records

    return []


# ── Source: AfriSpeech-200 ────────────────────────────────────────────────────

# Ugandan English accents in AfriSpeech-200 (accent labels used in the dataset).
_UGANDAN_ACCENTS = {"Uganda", "ugandan", "ug"}


def download_afrispeech(audio_dir: Path, max_samples: int) -> list[dict]:
    """
    AfriSpeech-200 is a 200-hour pan-African clinical + news speech corpus.
    We filter for Ugandan-accented English as a bridge/auxiliary training signal.

    Dataset: https://huggingface.co/datasets/tobiolatunji/AfriSpeech-200
    """
    from datasets import load_dataset

    print("  [AfriSpeech-200] Ugandan English...")
    try:
        ds = load_dataset(
            "tobiolatunji/AfriSpeech-200",
            "16000_pcm_speeches",
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"    AfriSpeech-200 unavailable: {e}")
        return []

    # Inject a pseudo-entry for Ugandan English in the registry (not a Bantu
    # language, but useful as an auxiliary signal if training multilingual models).
    records: list[dict] = []
    for i, row in enumerate(tqdm(ds, desc="    filtering Ugandan")):
        if len(records) >= max_samples:
            break
        accent = row.get("accent", "")
        if accent not in _UGANDAN_ACCENTS:
            continue
        audio = row["audio"]
        text = (row.get("transcript") or row.get("text") or "").strip()
        if not text:
            continue
        wav_path = audio_dir / f"afrispeech_ug_{i:06d}.wav"
        duration = save_wav(audio["array"], audio["sampling_rate"], wav_path)
        if acceptable(duration):
            records.append({
                "audio_filepath": str(wav_path.resolve()),
                "duration": round(duration, 2),
                "context": instruction("Ugandan English"),
                "answer": text,
                "language": "Ugandan English",
                "language_code": "eng_ug",
            })

    print(f"    AfriSpeech Ugandan: {len(records)} usable examples")
    return records


# ── Stub manifests ────────────────────────────────────────────────────────────

def write_stub_manifests(out_dir: Path) -> None:
    print("\nWriting stub multilang manifests for pipeline smoke-testing...")

    stubs_per_lang = {
        "lug": ["omukwano gw'abantu gwa maanyi", "abantu ba Uganda balina amagezi mangi",
                "eddwaliro liri mu kibuga kya Kampala"],
        "nyn": ["turikugyenda omu Kampala", "omuri nyowe nkugyenda"],
        "ach": ["wangi piny Uganda ber tutwal", "maber matek"],
    }

    sample_rate = 16000
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_records: list[dict] = []
    for lang_code, texts in stubs_per_lang.items():
        for i, text in enumerate(texts * 4):  # pad to 12 per lang
            wav_path = audio_dir / f"stub_{lang_code}_{i:04d}.wav"
            n = sample_rate  # 1 second of silence
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
            all_records.append(make_record(wav_path, text, 1.0, lang_code))

    splits = split_records(all_records)
    for name, rows in splits.items():
        write_manifest(rows, out_dir / f"multilang_{name}.json")

    print(
        "\nStub manifests written. For real data:\n"
        "  python scripts/prepare_data_multilang.py --langs lug nyn ach\n"
        "\nSee README for SALT / Common Voice access instructions."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def build_manifests(
    lang_codes: list[str],
    out_dir: Path,
    max_per_lang: int,
    include_afrispeech: bool,
) -> None:
    audio_dir = out_dir / "audio"
    all_records: list[dict] = []

    for lang_code in lang_codes:
        if lang_code not in LANGUAGES:
            print(f"Unknown language code: {lang_code}. Skipping.")
            continue

        name = LANGUAGES[lang_code]["name"]
        print(f"\n=== {name} ({lang_code}) ===")
        lang_records: list[dict] = []

        # Priority: SALT (most Uganda-specific) → FLEURS → Common Voice
        lang_records += download_salt(lang_code, audio_dir, max_per_lang - len(lang_records))
        if len(lang_records) < max_per_lang:
            lang_records += download_fleurs(lang_code, audio_dir, max_per_lang - len(lang_records))
        if len(lang_records) < max_per_lang:
            lang_records += download_commonvoice(lang_code, audio_dir, max_per_lang - len(lang_records))

        if not lang_records:
            print(f"  No data found for {name}. Skipping.")
            continue

        print(f"  Total {name}: {len(lang_records)} examples")
        all_records.extend(lang_records)

    if include_afrispeech:
        print("\n=== Ugandan English (AfriSpeech-200) ===")
        all_records += download_afrispeech(audio_dir, max_per_lang)

    if not all_records:
        write_stub_manifests(out_dir)
        return

    # Write per-language manifests for diagnostics
    by_lang: dict[str, list[dict]] = {}
    for r in all_records:
        by_lang.setdefault(r["language_code"], []).append(r)
    for lc, rows in by_lang.items():
        for split_name, split_rows in split_records(rows).items():
            write_manifest(split_rows, out_dir / "per_language" / lc / f"{split_name}.json")

    # Write combined multilang manifests (what finetune_nemotron.py reads)
    splits = split_records(all_records)
    for split_name, rows in splits.items():
        write_manifest(rows, out_dir / f"multilang_{split_name}.json")

    total = len(all_records)
    print(f"\nDone. {total} total examples across {len(by_lang)} language(s).")
    print(f"  train: {len(splits['train'])}  dev: {len(splits['dev'])}  test: {len(splits['test'])}")
    print("\nManifest fields: audio_filepath, duration, context, answer, language, language_code")
    print("Next step: python scripts/finetune_nemotron.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build multilingual NeMo SpeechLLM manifests for Ugandan languages."
    )
    parser.add_argument(
        "--langs", nargs="+", default=DEFAULT_LANGS,
        choices=list(LANGUAGES.keys()),
        help="Language codes to include (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/manifests/multilang"),
        help="Directory to write manifests and audio files",
    )
    parser.add_argument(
        "--max-per-lang", type=int, default=5000,
        help="Maximum samples per language (default: 5000)",
    )
    parser.add_argument(
        "--no-afrispeech", action="store_true",
        help="Skip AfriSpeech-200 Ugandan English data",
    )
    args = parser.parse_args()

    try:
        from datasets import load_dataset  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError:
        print("Missing dependencies — run: pip install datasets soundfile tqdm")
        write_stub_manifests(args.output_dir)
        raise SystemExit(0)

    build_manifests(
        lang_codes=args.langs,
        out_dir=args.output_dir,
        max_per_lang=args.max_per_lang,
        include_afrispeech=not args.no_afrispeech,
    )
