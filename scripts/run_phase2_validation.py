"""Run deterministic Phase 2 paper-trading validation scenarios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.operations.validation import VALIDATION_SCENARIOS, Phase2ValidationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 2 paper-trading validation scenarios.")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["all", *VALIDATION_SCENARIOS],
        help="Scenario to run. Defaults to all.",
    )
    parser.add_argument(
        "--output-root",
        default="multi_agent_trading_lab/validation_runs",
        help="Directory where validation run artifacts are written.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = Phase2ValidationRunner(output_root=args.output_root).run(args.scenario)
    print(f"Phase 2 validation result: {'PASS' if summary.passed else 'FAIL'}")
    print(f"Scenarios: {', '.join(summary.scenario_names)}")
    print(f"Artifacts: {summary.output_dir}")
    print(f"JSON summary: {summary.output_dir / 'validation_summary.json'}")
    print(f"Operator report: {summary.output_dir / 'validation_report.md'}")
    if summary.manual_review_recommended:
        print("Manual review recommended:")
        for item in summary.manual_review_recommended:
            print(f"- {item}")
    return 0 if summary.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
