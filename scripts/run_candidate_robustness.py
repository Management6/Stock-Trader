"""Run robustness validation for accepted approval-search candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.operations.candidate_robustness import run_candidate_robustness


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic robustness validation for accepted approval-search candidates.")
    parser.add_argument("--settings", default="multi_agent_trading_lab/config/settings.approval_paper.yaml", help="Path to approval profile settings YAML.")
    parser.add_argument("--input", required=True, help="Path to approval_search_summary.json.")
    parser.add_argument("--output-root", default="multi_agent_trading_lab/validation_runs", help="Directory where robustness artifacts are written.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum accepted candidates to evaluate.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_candidate_robustness(args.settings, args.input, output_root=args.output_root, limit=args.limit)
    print("Candidate robustness validation complete.")
    print(f"Artifacts: {summary.output_dir}")
    print(f"Summary: {summary.report_path}")
    print(f"Candidates evaluated: {summary.evaluated_count}")
    print(f"Status counts: {summary.status_counts}")
    print(f"Top failure reasons: {summary.top_failure_reasons}")
    print(f"Top robust candidate: {summary.top_robust_candidate}")
    print(f"Top fragile candidate: {summary.top_fragile_candidate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
