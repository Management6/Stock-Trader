"""Run a paper-only cycle scoped to an operator-selected watchlist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.operations.watchlist_paper_cycle import run_watchlist_paper_cycle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one paper-only monitoring cycle scoped to a selected watchlist.")
    parser.add_argument("--settings", default="multi_agent_trading_lab/config/settings.approval_paper.yaml", help="Path to settings YAML.")
    parser.add_argument("--watchlist", required=True, help="Path to paper_watchlist.json from select_paper_watchlist.py.")
    parser.add_argument("--output-root", default="multi_agent_trading_lab/validation_runs", help="Directory where cycle artifacts are written.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_watchlist_paper_cycle(args.settings, args.watchlist, output_root=args.output_root)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print("Watchlist paper cycle complete.")
    print(f"Artifacts: {summary.output_dir}")
    print(f"Candidates considered: {[item['candidate_id'] for item in summary.candidates_considered]}")
    print(f"Skipped candidates: {len(summary.candidates_skipped)}")
    print(f"Signals generated: {len(summary.signals)}")
    print(f"Orders: {len(summary.orders)}")
    print(f"Audit log: {summary.audit_log_path}")
    print(f"Paper report JSON: {summary.report_json_path}")
    print(f"Paper report Markdown: {summary.report_markdown_path}")
    print(f"Live trading enabled: {summary.live_trading_enabled}")
    print(summary.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
