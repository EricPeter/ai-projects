import json
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

def test_package_json_exists(project_root):
    assert (project_root / "package.json").exists()


def test_next_config_exists(project_root):
    assert (project_root / "next.config.js").exists()


def test_app_directory_exists(project_root):
    has_app = (project_root / "app").exists() or (project_root / "{app").exists()
    assert has_app, "Next.js app directory must exist"


def test_build_output_exists(project_root):
    build_dir = project_root / ".next"
    if not build_dir.exists():
        pytest.skip(".next/ build output not present — run npm run build first")
    assert (build_dir / "BUILD_ID").exists() or any(build_dir.iterdir())


# ── Regression ────────────────────────────────────────────────────────────────

def test_package_json_scripts_stable(project_root):
    pkg = json.loads((project_root / "package.json").read_text())
    scripts = sorted(pkg.get("scripts", {}).keys())
    stored = _baseline("scripts", {"scripts": scripts})
    assert scripts == stored["scripts"]


def test_dependency_count_stable(project_root):
    pkg = json.loads((project_root / "package.json").read_text())
    count = len(pkg.get("dependencies", {})) + len(pkg.get("devDependencies", {}))
    stored = _baseline("dep_count", {"count": count})
    assert count >= stored["count"], "dependencies removed unexpectedly"


# ── A/B ── source directories ─────────────────────────────────────────────────

def test_pages_vs_app_router_not_mixed(project_root):
    has_pages = (project_root / "pages").exists()
    has_app = (project_root / "app").exists()
    has_brace_app = (project_root / "{app").exists()

    print(f"\n  pages/: {has_pages}  app/: {has_app}  {{app/: {has_brace_app}")
    # Project should use one router strategy, not both
    assert not (has_pages and has_app), "mixing pages/ and app/ routers detected"
