"""Run expanded deterministic strategy search under approval-paper gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.operations.approval_search import DEFAULT_APPROVAL_SEARCH_FAMILIES, run_approval_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run expanded approval-profile optimizer search without changing gates.")
    parser.add_argument("--settings", default="multi_agent_trading_lab/config/settings.approval_paper.yaml", help="Path to approval profile settings YAML.")
    parser.add_argument("--trials", type=int, default=100, help="Optimizer trials per strategy family.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic optimizer seed.")
    parser.add_argument("--families", default=",".join(DEFAULT_APPROVAL_SEARCH_FAMILIES), help="Comma-separated strategy families or aliases.")
    parser.add_argument("--output-root", default="multi_agent_trading_lab/validation_runs", help="Directory where search artifacts are written.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    summary = run_approval_search(args.settings, trials=args.trials, seed=args.seed, output_root=args.output_root, families=families)
    print("Approval search complete.")
    print(f"Artifacts: {summary.output_dir}")
    print(f"Summary: {summary.output_dir / 'approval_search_summary.json'}")
    print(f"Families: {', '.join(summary.families)}")
    print(f"Trials per family: {summary.trials_per_family}")
    print(f"Total trials: {summary.total_trials}")
    print(f"Accepted: {summary.accepted_count}")
    print(f"Rejected: {summary.rejected_count}")
    print(f"Accepted by family: {summary.accepted_by_family}")
    print(f"Rejected by family: {summary.rejected_by_family}")
    print(f"Top rejection reasons: {summary.rejection_breakdown}")
    print(f"Best candidate by family: {summary.best_candidate_by_family}")
    print(f"Best near miss: {summary.best_near_miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
