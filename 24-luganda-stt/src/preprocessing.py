"""
Audio preprocessing before ASR.

Ensures every audio chunk entering the pipeline is:
  - Mono
  - 16 kHz sample rate (NeMo ASR standard)
  - Float32 in [-1, 1]
  - Peak-normalised to -3 dBFS
  - Optionally noise-reduced (spectral subtraction)
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(data: bytes | str | Path) -> tuple[np.ndarray, int]:
    """Load audio from bytes or file path. Returns (samples, sample_rate)."""
    if isinstance(data, (str, Path)):
        samples, sr = sf.read(str(data), dtype="float32", always_2d=False)
    else:
        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return samples, sr


def to_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        return samples.mean(axis=1)
    return samples


def resample(samples: np.ndarray, src_sr: int, dst_sr: int = 16000) -> np.ndarray:
    if src_sr == dst_sr:
        return samples
    try:
        import librosa
        return librosa.resample(samples, orig_sr=src_sr, target_sr=dst_sr)
    except ImportError:
        # scipy fallback
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_sr, dst_sr)
        return resample_poly(samples, dst_sr // g, src_sr // g).astype(np.float32)


def peak_normalise(samples: np.ndarray, target_dbfs: float = -3.0) -> np.ndarray:
    peak = np.abs(samples).max()
    if peak < 1e-8:
        return samples
    target_amplitude = 10 ** (target_dbfs / 20.0)
    return samples * (target_amplitude / peak)


def spectral_subtraction(samples: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simple spectral subtraction noise reduction.
    Estimates noise floor from the first 0.5 s (assumed to be silence/background).
    """
    frame_len = int(0.025 * sr)  # 25 ms
    hop_len   = int(0.010 * sr)  # 10 ms
    n_noise_frames = max(1, int(0.5 * sr / hop_len))

    frames = []
    for start in range(0, len(samples) - frame_len, hop_len):
        frames.append(samples[start : start + frame_len])

    if not frames:
        return samples

    spectra = np.array([np.abs(np.fft.rfft(f * np.hanning(len(f)))) for f in frames])
    noise_profile = spectra[:n_noise_frames].mean(axis=0)

    cleaned = []
    for mag in spectra:
        cleaned.append(np.maximum(mag - noise_profile, 0.0))

    # Reconstruct (magnitude-only, zero-phase) — good enough for ASR
    output = np.zeros(len(samples), dtype=np.float32)
    window = np.hanning(frame_len)
    for i, mag in enumerate(cleaned):
        frame = np.fft.irfft(mag, n=frame_len).real * window
        start = i * hop_len
        end   = start + frame_len
        if end <= len(output):
            output[start:end] += frame

    # Renormalise after reconstruction
    peak = np.abs(output).max()
    if peak > 1e-8:
        output /= peak
    return output


def preprocess(
    data: bytes | str | Path,
    denoise: bool = False,
    target_sr: int = 16000,
) -> tuple[np.ndarray, int]:
    """Full preprocessing chain. Returns (samples_float32_mono_16kHz, 16000)."""
    samples, sr = load_audio(data)
    samples = to_mono(samples)
    samples = resample(samples, sr, target_sr)
    samples = peak_normalise(samples)
    if denoise:
        samples = spectral_subtraction(samples, target_sr)
    return samples, target_sr


def to_wav_bytes(samples: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def save_wav(samples: np.ndarray, sr: int, path: str | Path) -> None:
    sf.write(str(path), samples, sr, subtype="PCM_16")
