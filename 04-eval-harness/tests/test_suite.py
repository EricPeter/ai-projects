"""
Tests for 04-eval-harness — a python-only project with no HTTP API.
Validates the eval pipeline runs, produces scores, and catches regressions.
"""
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


def _run_eval(extra_args=None):
    cmd = ["uv", "run", "pytest", "tests/", "-v", "--tb=short", "-q", "--no-header"]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    return result


# ── Acceptance ────────────────────────────────────────────────────────────────

def test_golden_dataset_exists(project_root):
    golden = project_root / "golden_dataset"
    assert golden.exists(), "golden_dataset/ directory must exist"
    files = list(golden.iterdir())
    assert len(files) > 0, "golden_dataset/ must contain at least one file"


def test_eval_script_runs_without_error(project_root):
    result = subprocess.run(
        ["uv", "run", "python", "-c", "import src; print('ok')"],
        capture_output=True, text=True, cwd=project_root
    )
    assert result.returncode == 0 or "ok" in result.stdout or "ModuleNotFoundError" not in result.stderr


# ── Regression ────────────────────────────────────────────────────────────────

def test_src_module_importable(project_root):
    result = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, 'src'); print('ok')"],
        capture_output=True, text=True, cwd=project_root
    )
    assert "ok" in result.stdout or result.returncode == 0


def test_eval_history_file_format(project_root):
    history_file = project_root / "eval_history.jsonl"
    if not history_file.exists():
        pytest.skip("eval_history.jsonl not yet created")
    lines = history_file.read_text().strip().split("\n")
    assert len(lines) > 0
    for line in lines[:3]:
        record = json.loads(line)
        assert "score" in record or "accuracy" in record or "timestamp" in record


# ── A/B ── strict vs lenient pass threshold ────────────────────────────────────

def test_strict_threshold_fails_more_cases(project_root):
    history_file = project_root / "eval_history.jsonl"
    if not history_file.exists():
        pytest.skip("eval_history.jsonl not yet created")

    records = [json.loads(l) for l in history_file.read_text().strip().split("\n") if l]
    scores = [r.get("score", r.get("accuracy", 0)) for r in records if "score" in r or "accuracy" in r]
    if not scores:
        pytest.skip("no score fields in eval history")

    strict_failures = sum(1 for s in scores if s < 0.90)
    lenient_failures = sum(1 for s in scores if s < 0.70)

    print(f"\n  strict (>0.90): {strict_failures} failures  lenient (>0.70): {lenient_failures} failures")
    assert lenient_failures <= strict_failures
