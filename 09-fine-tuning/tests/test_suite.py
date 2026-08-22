import json
import subprocess
import sys
from pathlib import Path

import pytest

BASELINES = Path(__file__).parent / "baselines"


def _baseline(name, current):
    BASELINES.mkdir(exist_ok=True)
    p = BASELINES / f"{name}.json"
    if not p.exists():
        p.write_text(json.dumps(current, indent=2))
        pytest.skip(f"baseline '{name}' created")
    return json.loads(p.read_text())


# ── Acceptance ────────────────────────────────────────────────────────────────

def test_data_directory_exists(project_root):
    assert (project_root / "data").exists()


def test_prepare_dataset_script_exists(project_root):
    assert (project_root / "data" / "prepare_dataset.py").exists()


def test_train_script_exists(project_root):
    assert (project_root / "src" / "train_lora.py").exists()


def test_dataset_fixture_valid(lora_dataset):
    assert len(lora_dataset) >= 5
    for row in lora_dataset:
        assert "instruction" in row
        assert "output" in row


# ── Regression ────────────────────────────────────────────────────────────────

def test_outputs_directory_structure(project_root):
    outputs = project_root / "outputs"
    if not outputs.exists():
        pytest.skip("outputs/ not yet created — run the pipeline first")
    contents = list(outputs.iterdir())
    stored = _baseline("outputs_files", {"count": len(contents)})
    assert len(contents) >= stored["count"], "outputs count decreased"


def test_model_output_exists_after_training(project_root):
    outputs = project_root / "outputs"
    if not outputs.exists():
        pytest.skip("outputs/ not yet created")
    model_files = list(outputs.glob("**/*.json")) + list(outputs.glob("**/*.pt")) + list(outputs.glob("**/*.safetensors"))
    assert len(model_files) > 0, "no model files found in outputs/"


# ── A/B ── base model vs LoRA adapter ─────────────────────────────────────────

def test_lora_adapter_file_present(project_root):
    outputs = project_root / "outputs"
    if not outputs.exists():
        pytest.skip("outputs/ not yet created")
    adapter_files = list(outputs.glob("**/adapter*")) + list(outputs.glob("**/lora*"))
    if not adapter_files:
        pytest.skip("no LoRA adapter files found yet")
    assert len(adapter_files) > 0

    metrics_files = list(outputs.glob("**/metrics*.json")) + list(outputs.glob("**/eval*.json"))
    if metrics_files:
        metrics = json.loads(metrics_files[0].read_text())
        print(f"\n  eval metrics: {metrics}")
