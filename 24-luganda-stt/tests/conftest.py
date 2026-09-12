import io
import os
import struct
import wave

import pytest


@pytest.fixture(scope="session")
def base_url():
    return os.getenv("BASE_URL", "http://localhost:8024")


@pytest.fixture(scope="session")
def sample_wav_bytes() -> bytes:
    """1-second silent WAV at 16 kHz mono — enough to exercise the pipeline."""
    buf = io.BytesIO()
    sample_rate = 16000
    n_samples = sample_rate
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return buf.getvalue()
