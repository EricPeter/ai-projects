# Human-in-the-Loop Workflow

AI task planner that detects uncertainty and pauses for human approval before executing irreversible or sensitive actions.

## How It Works

1. Submit a task → AI creates a plan
2. Uncertainty detector checks for: irreversible actions, sensitive domains, complex multi-step plans
3. If risky: pauses and creates an approval request
4. Human approves/rejects via API or approval UI
5. On approval: plan executes with optional human feedback incorporated
6. Full audit trail on every workflow

## Setup

```bash
cd 15-hitl-workflow
cp .env.example .env
uv sync
uv run uvicorn src.main:app --reload
```

Open `ui/index.html` in a browser for the approval UI.

## Usage

```bash
# Submit a task (irreversible — will pause for approval)
curl -X POST http://localhost:8000/workflows \
  -H "Content-Type: application/json" \
  -d '{"task": "Delete all inactive user accounts older than 2 years"}'

# Returns: {"workflow_id": "...", "status": "awaiting_approval"}

# List pending approvals
curl http://localhost:8000/workflows/pending

# Approve with feedback
curl -X POST http://localhost:8000/workflows/WORKFLOW_ID/approve \
  -H "Content-Type: application/json" \
  -d '{"decided_by": "alice", "feedback": "Only delete accounts with no purchases"}'

# View audit trail
curl http://localhost:8000/workflows/WORKFLOW_ID/audit
```
