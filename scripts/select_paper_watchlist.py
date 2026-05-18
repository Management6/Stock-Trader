"""Select an operator-reviewed paper monitoring watchlist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.operations.paper_watchlist import select_paper_watchlist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select robust candidates for an operator-reviewed paper watchlist.")
    parser.add_argument("--approval-summary", required=True, help="Path to approval_search_summary.json.")
    parser.add_argument("--robustness-summary", required=True, help="Path to candidate_robustness_summary.json.")
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum watchlist candidates to select.")
    parser.add_argument("--output-root", default="multi_agent_trading_lab/validation_runs", help="Directory where watchlist artifacts are written.")
    parser.add_argument("--no-diversify-by-family", action="store_true", help="Disable round-robin diversification by strategy family.")
    parser.add_argument("--write-approval-queue", action="store_true", help="Write selected candidates to a local approval queue artifact. Defaults to read-only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = select_paper_watchlist(
        approval_summary_path=args.approval_summary,
        robustness_summary_path=args.robustness_summary,
        max_candidates=args.max_candidates,
        output_root=args.output_root,
        diversify_by_family=not args.no_diversify_by_family,
        write_approval_queue=bool(args.write_approval_queue),
    )
    print("Paper watchlist selection complete.")
    print(f"Artifacts: {summary.output_dir}")
    print(f"JSON watchlist: {summary.watchlist_json_path}")
    print(f"Markdown report: {summary.watchlist_markdown_path}")
    print(f"Selected candidates: {summary.selected_count}")
    print(f"Approval queue written: {summary.approval_queue_written}")
    if summary.approval_queue_path is not None:
        print(f"Approval queue: {summary.approval_queue_path}")
    for candidate in summary.selected_candidates:
        print(f"- {candidate['rank']}: {candidate['candidate_id']} ({candidate['strategy_family']})")
    print(summary.warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
