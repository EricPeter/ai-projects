# CI/CD for AI Systems

Complete CI/CD pipeline for AI services: automated eval gates, canary deployments, ArgoCD GitOps, feature flags.

## Pipeline Overview

```
PR opened
  → lint (ruff, mypy)
  → unit tests
  → eval harness (golden test suite, blocks merge if score < 90%)
  → security scan (bandit)

Merge to main
  → Docker build & push to GHCR
  → ArgoCD deploys canary (via image tag update)
  → Smoke tests run
  → Promote to 100% OR rollback
```

## File Structure

```
.github/workflows/
  ci.yml          — PR checks + eval gate
  cd.yml          — build → canary → promote/rollback
argocd/
  application.yml — ArgoCD app manifest
k8s/
  deployment.yml  — Kubernetes Deployment
  configmap.yml   — Feature flags
src/
  app.py          — Sample AI service
  feature_flags.py— Read flags from env/ConfigMap
scripts/
  smoke_test.sh   — Post-deploy smoke test
```

## Setup

1. Fork this repo and update `argocd/application.yml` with your repo URL
2. Create GitHub secrets: `ANTHROPIC_API_KEY`, `CANARY_URL`
3. Install ArgoCD in your cluster and apply `argocd/application.yml`
4. Push to main — pipeline runs automatically

## Feature Flags

Set in `k8s/configmap.yml` and read at runtime with no restart required (use watch + reload in production).

```bash
# Locally
ENABLE_NEW_MODEL=true uv run uvicorn src.app:app
```
