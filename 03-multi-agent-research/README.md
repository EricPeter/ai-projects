# Multi-Agent Research System

A CrewAI pipeline where a Researcher, Writer, and Fact-Checker collaborate on research reports, with a human approval gate before finalisation.

## Agents

| Agent | Role |
|-------|------|
| Researcher | Gathers and synthesises information |
| Writer | Drafts a structured markdown report |
| Fact Checker | Marks uncertain claims with [UNCERTAIN]; **pauses for human approval** |

## Setup

```bash
cd 03-multi-agent-research
cp .env.example .env   # add ANTHROPIC_API_KEY
uv sync
```

## Run

```bash
uv run python -m src.main "The future of quantum computing"
```

The pipeline will pause at the Fact Checker stage and prompt you to review the draft in the terminal before continuing.

## Outputs

- `outputs/report.md` — final report
- `outputs/audit_trail.jsonl` — full audit log of every agent action

## Audit Trail Format

```json
{"timestamp": "...", "agent": "system", "action": "start", "input": "...", "output": "..."}
```
