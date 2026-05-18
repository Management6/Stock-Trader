"""Run the strict paper research profile validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.operations.strict_profile_validation import run_strict_profile_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic strict paper profile validation.")
    parser.add_argument("--settings", default="multi_agent_trading_lab/config/settings.strict_paper.yaml", help="Path to strict profile settings YAML.")
    parser.add_argument("--output-root", default="multi_agent_trading_lab/validation_runs", help="Directory where validation artifacts are written.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_strict_profile_validation(settings_path=args.settings, output_root=args.output_root)
    print(f"Strict paper validation result: {'PASS' if result.passed else 'FAIL'}")
    print(f"Artifacts: {result.output_dir}")
    print(f"Summary: {result.output_dir / 'strict_profile_validation_summary.json'}")
    print(f"Daily report: {result.report_path}")
    print(f"Audit log: {result.audit_log_path}")
    if result.accepted_example:
        print(f"Accepted example: {result.accepted_example}")
    if result.rejected_example:
        print(f"Rejected example: {result.rejected_example}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
