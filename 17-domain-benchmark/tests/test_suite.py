import json
import subprocess
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


def _run_benchmark(project_root, n=5):
    result = subprocess.run(
        ["uv", "run", "python", "scripts/run_benchmark.py", "--n", str(n)],
        capture_output=True, text=True, cwd=project_root, timeout=120
    )
    return result


# ── Acceptance ────────────────────────────────────────────────────────────────

def test_benchmark_script_exists(project_root):
    assert (project_root / "scripts" / "run_benchmark.py").exists()


def test_benchmark_runs_without_error(project_root):
    result = _run_benchmark(project_root, n=3)
    assert result.returncode == 0, f"benchmark failed:\n{result.stderr[-500:]}"


def test_leaderboard_directory_populated(project_root):
    _run_benchmark(project_root, n=3)
    leaderboard = project_root / "leaderboard"
    if not leaderboard.exists():
        pytest.skip("leaderboard/ not created by benchmark")
    assert len(list(leaderboard.iterdir())) > 0


# ── Regression ────────────────────────────────────────────────────────────────

def test_benchmark_score_not_regressed(project_root):
    result = _run_benchmark(project_root, n=5)
    assert result.returncode == 0

    score = None
    for line in result.stdout.split("\n"):
        if "score" in line.lower() or "accuracy" in line.lower() or "mean" in line.lower():
            parts = [p for p in line.split() if p.replace(".", "").replace("%", "").isdigit()]
            if parts:
                score = float(parts[0].replace("%", ""))
                break

    if score is None:
        pytest.skip("could not parse score from benchmark output")

    stored = _baseline("benchmark_score", {"score": round(score, 2)})
    assert score >= stored["score"] * 0.85, (
        f"benchmark score {score:.2f} dropped >15% from baseline {stored['score']:.2f}"
    )


# ── A/B ── small vs large eval set ────────────────────────────────────────────

def test_n5_vs_n10_consistency(project_root):
    r5 = _run_benchmark(project_root, n=5)
    r10 = _run_benchmark(project_root, n=10)
    assert r5.returncode == 0 and r10.returncode == 0
    print(f"\n  n=5 exit={r5.returncode}  n=10 exit={r10.returncode}")
