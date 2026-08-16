#!/usr/bin/env python3
"""CLI: python -m src.main "AI in healthcare" """
import sys
from pathlib import Path
from .crew import run_research


def main():
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Large Language Models in 2025"
    print(f"\nResearching: {topic}\n{'='*60}")
    report = run_research(topic)

    out = Path("outputs/report.md")
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    print(f"\nReport saved to {out}")
    print(f"Audit trail: outputs/audit_trail.jsonl")


if __name__ == "__main__":
    main()
