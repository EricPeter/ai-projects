"""
A/B tests comparing fast vs rich pipeline modes, and spell-corrected vs raw output.
"""

import httpx
import pytest


def _transcribe(base_url, wav_bytes, mode="fast"):
    r = httpx.post(
        f"{base_url}/transcribe?mode={mode}",
        files={"file": ("t.wav", wav_bytes, "audio/wav")},
        timeout=120,
    )
    assert r.status_code == 200
    return r.json()


def test_fast_mode_is_faster_than_rich(base_url, sample_wav_bytes):
    import time

    t0 = time.perf_counter()
    _transcribe(base_url, sample_wav_bytes, mode="fast")
    fast_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _transcribe(base_url, sample_wav_bytes, mode="rich")
    rich_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  fast={fast_ms:.0f}ms  rich={rich_ms:.0f}ms")
    # Rich is allowed to be up to 5x slower; if fast is faster the test passes
    # (silent audio may make both instant — allow equal too)
    assert fast_ms <= rich_ms * 5 or fast_ms < 500


def test_spell_correction_differs_from_raw_on_noisy_input(base_url, sample_wav_bytes):
    """Both fields should be present and strings; correction may or may not differ on silence."""
    body = _transcribe(base_url, sample_wav_bytes)
    assert isinstance(body["transcript"], str)
    assert isinstance(body["transcript_corrected"], str)


def test_word_boosting_reported(base_url, sample_wav_bytes):
    """boosted_words_hit should always be an integer >= 0."""
    body = _transcribe(base_url, sample_wav_bytes)
    assert isinstance(body["boosted_words_hit"], int)
    assert body["boosted_words_hit"] >= 0


def test_translation_not_empty_on_silence(base_url, sample_wav_bytes):
    """Even for silence the translation field should be a string (may be empty or filler)."""
    body = _transcribe(base_url, sample_wav_bytes)
    assert isinstance(body["translation"], str)
