"""Generate a deterministic daily report for local paper trading."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.operations.paper_reporting import DailyPaperReportService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the daily paper-trading report.")
    parser.add_argument("--date", help="Report date in YYYY-MM-DD format. Defaults to today in UTC.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_date = date.fromisoformat(args.date) if args.date else None
    report = DailyPaperReportService().generate(report_date)
    print(report.text_report)
    print()
    print(f"Wrote multi_agent_trading_lab/reports/paper_report_{report.date}.json")
    print(f"Wrote multi_agent_trading_lab/reports/paper_report_{report.date}.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
