"""Offline validation that committed manifests match the Canary schema.

Canary/EncDecMultiTaskModel expects per-utterance fields:
  audio_filepath, duration, text, taskname, pnc, source_lang, target_lang
Luganda is mapped onto the English prompt slot, so source_lang == target_lang.
"""

import json
from pathlib import Path

import pytest

MANIFEST_DIR = Path(__file__).parent.parent / "data" / "manifests"
REQUIRED = {"audio_filepath", "duration", "text", "taskname", "pnc",
            "source_lang", "target_lang"}


def _records(name: str):
    p = MANIFEST_DIR / f"{name}.json"
    if not p.exists():
        pytest.skip(f"{p} not present")
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        pytest.skip(f"{p} is empty (stub run)")
    return [json.loads(l) for l in lines]


@pytest.mark.parametrize("name", ["train", "dev", "test"])
def test_manifest_schema(name):
    for r in _records(name):
        missing = REQUIRED - r.keys()
        assert not missing, f"{name}: record missing {missing}"
        assert r["taskname"] == "asr"
        assert r["pnc"] in ("yes", "no")
        assert r["source_lang"] == r["target_lang"], "lang slots must match for ASR"
        assert r["duration"] > 0
        assert isinstance(r["text"], str) and r["text"].strip()
        assert str(r["audio_filepath"]).endswith(".wav")


@pytest.mark.parametrize("name", ["train", "dev", "test"])
def test_no_legacy_task_field(name):
    # The old schema used "task" instead of "taskname"; ensure it's gone.
    for r in _records(name):
        assert "task" not in r, f"{name}: legacy 'task' field still present"
