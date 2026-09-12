"""
Benchmark WER + CER for both serving modes, baseline vs fine-tuned.

  SYNC  — batch Canary (EncDecMultiTaskModel), offline full-context decoding.
  LIVE  — cache-aware streaming FastConformer/Parakeet, chunk-by-chunk decoding
          (plus the same model decoded offline, to show the streaming penalty).

For each mode it decodes the test manifest with the pretrained baseline and with
your fine-tuned checkpoint, then reports corpus WER/CER and the improvement.

Usage
-----
  # both modes, baselines auto (from config defaults):
  python scripts/benchmark.py --mode both \
      --test-manifest data/manifests/test.json \
      --canary-finetuned outputs/canary_luganda/canary_luganda.nemo \
      --streaming-finetuned outputs/streaming_luganda/streaming_luganda.nemo

  # just the batch model:
  python scripts/benchmark.py --mode sync \
      --canary-finetuned outputs/canary_luganda/canary_luganda.nemo

Results are printed as a table and written to outputs/benchmarks/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Local, dependency-free metrics (import works without NeMo).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import corpus_cer, corpus_wer  # noqa: E402

SR = 16000


# ── Data ────────────────────────────────────────────────────────────────────────

def load_manifest(path: Path, max_samples: int) -> tuple[list[str], list[str]]:
    audio, refs = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        audio.append(r["audio_filepath"])
        refs.append(r["text"])
        if len(audio) >= max_samples:
            break
    return audio, refs


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf

    samples, sr = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != SR:
        import librosa

        samples = librosa.resample(samples, orig_sr=sr, target_sr=SR)
    return samples.astype(np.float32)


def evaluate(refs: list[str], hyps: list[str]) -> dict:
    return {
        "wer": round(corpus_wer(refs, hyps) * 100, 2),
        "cer": round(corpus_cer(refs, hyps) * 100, 2),
        "n": len(refs),
    }


# ── SYNC (batch Canary) ──────────────────────────────────────────────────────────

def _load_canary(model_id: str | None, checkpoint: str | None):
    from nemo.collections.asr.models import EncDecMultiTaskModel

    if checkpoint and Path(checkpoint).exists():
        m = EncDecMultiTaskModel.restore_from(checkpoint)
    else:
        m = EncDecMultiTaskModel.from_pretrained(model_id)
    m.eval()
    return m


def sync_transcribe(model_id, checkpoint, audio_paths) -> list[str]:
    m = _load_canary(model_id, checkpoint)
    hyps = m.transcribe(audio=audio_paths, source_lang="en", target_lang="en", pnc=False)
    return [h.text if hasattr(h, "text") else str(h) for h in hyps]


# ── LIVE (cache-aware streaming) ─────────────────────────────────────────────────

def _load_streaming(model_id: str | None, checkpoint: str | None):
    from src.config import Settings
    from src.streaming_asr import StreamingASR

    s = Settings()
    if model_id:
        s.streaming_asr_model = model_id
    s.streaming_asr_checkpoint = Path(checkpoint) if checkpoint else None
    asr = StreamingASR(s)
    asr.load()
    return asr


def live_transcribe(model_id, checkpoint, audio_paths) -> tuple[list[str], list[str]]:
    """Return (streaming_hyps, offline_hyps) for the same model."""
    asr = _load_streaming(model_id, checkpoint)
    streaming, offline = [], []
    for p in audio_paths:
        samples = load_audio(p)
        streaming.append(asr.stream_array(samples)[0])
        # same model decoded offline (full context) for the streaming-penalty row
        out = asr._model.transcribe(audio=[p])
        offline.append(out[0].text if hasattr(out[0], "text") else str(out[0]))
    return streaming, offline


# ── External competitive baselines (Luganda-capable) ─────────────────────────────

def external_transcribe(name: str, audio_paths: list[str]) -> list[str]:
    """Transcribe with a published Luganda-capable model for a fair comparison.

    Supported names:
      mms      - facebook/mms-1b-all with the native "lug" adapter (CTC)
      whisper  - openai/whisper-large-v3 with language="lg"
    Requires `transformers` (and torch). Both decode offline (full context).
    """
    if name == "mms":
        return _mms_transcribe(audio_paths)
    if name == "whisper":
        return _whisper_transcribe(audio_paths)
    raise ValueError(f"Unknown external model '{name}' (use: mms, whisper)")


def _mms_transcribe(audio_paths: list[str]) -> list[str]:
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    model_id = "facebook/mms-1b-all"
    processor = AutoProcessor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    processor.tokenizer.set_target_lang("lug")   # native Luganda adapter
    model.load_adapter("lug")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    hyps = []
    for p in audio_paths:
        samples = load_audio(p)
        inputs = processor(samples, sampling_rate=SR, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        hyps.append(processor.batch_decode(ids)[0])
    return hyps


def _whisper_transcribe(audio_paths: list[str]) -> list[str]:
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device)
    hyps = []
    for p in audio_paths:
        out = pipe(load_audio(p), generate_kwargs={"language": "lg", "task": "transcribe"})
        hyps.append(out["text"].strip())
    return hyps


# ── Reporting ─────────────────────────────────────────────────────────────────

def _delta(base: dict, tuned: dict) -> dict:
    return {"wer": round(base["wer"] - tuned["wer"], 2),
            "cer": round(base["cer"] - tuned["cer"], 2)}


def format_table(results: dict) -> str:
    lines = []
    for mode, rows in results.items():
        lines.append(f"\n=== {mode.upper()} ===")
        lines.append(f"{'model':<28}{'WER%':>8}{'CER%':>8}")
        lines.append("-" * 44)
        for name, r in rows.items():
            if not isinstance(r, dict) or "wer" not in r:
                continue
            lines.append(f"{name:<28}{r['wer']:>8}{r['cer']:>8}")
        lines.append("(lower is better; improvement = baseline - finetuned, positive is better)")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["sync", "live", "both"], default="both")
    p.add_argument("--test-manifest", type=Path, default=Path("data/manifests/test.json"))
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--canary-baseline", default="nvidia/canary-1b-flash")
    p.add_argument("--canary-finetuned", default=None)
    p.add_argument("--streaming-baseline", default="nvidia/parakeet_realtime_eou_120m-v1")
    p.add_argument("--streaming-finetuned", default=None)
    p.add_argument("--external", default="",
                   help="comma list of competitive Luganda baselines: mms,whisper")
    p.add_argument("--out", type=Path, default=Path("outputs/benchmarks"))
    args = p.parse_args()

    audio, refs = load_manifest(args.test_manifest, args.max_samples)
    print(f"Loaded {len(refs)} test utterances from {args.test_manifest}")

    results: dict = {}

    if args.mode in ("sync", "both"):
        print("\n[SYNC] baseline...")
        base = evaluate(refs, sync_transcribe(args.canary_baseline, None, audio))
        rows = {"baseline": base}
        if args.canary_finetuned:
            print("[SYNC] finetuned...")
            tuned = evaluate(refs, sync_transcribe(args.canary_baseline, args.canary_finetuned, audio))
            rows["finetuned"] = tuned
            rows["improvement"] = _delta(base, tuned)
        results["sync"] = rows

    if args.mode in ("live", "both"):
        print("\n[LIVE] baseline (streaming)...")
        b_stream, b_offline = live_transcribe(args.streaming_baseline, None, audio)
        rows = {"baseline": evaluate(refs, b_stream)}
        if args.streaming_finetuned:
            print("[LIVE] finetuned (streaming + offline)...")
            f_stream, f_offline = live_transcribe(args.streaming_baseline, args.streaming_finetuned, audio)
            rows["finetuned"] = evaluate(refs, f_stream)
            rows["finetuned_offline"] = evaluate(refs, f_offline)
            rows["improvement"] = _delta(rows["baseline"], rows["finetuned"])
            # streaming penalty = streaming - offline (finetuned)
            rows["streaming_penalty"] = {
                "wer": round(rows["finetuned"]["wer"] - rows["finetuned_offline"]["wer"], 2),
                "cer": round(rows["finetuned"]["cer"] - rows["finetuned_offline"]["cer"], 2),
            }
        results["live"] = rows

    externals = [e.strip() for e in args.external.split(",") if e.strip()]
    if externals:
        rows = {}
        for name in externals:
            print(f"\n[EXTERNAL] {name} (Luganda-capable, offline)...")
            try:
                rows[name] = evaluate(refs, external_transcribe(name, audio))
            except Exception as e:  # noqa: BLE001
                print(f"  skipped {name}: {e}")
        if rows:
            results["external"] = rows


    table = format_table(results)
    print(table)

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"bench_{int(time.time())}.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults -> {out_file}")


if __name__ == "__main__":
    main()
