"""Review pending strategy promotion requests for paper trading."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.operations.approvals import ApprovalQueue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review pending paper strategy promotions.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List pending promotion requests.")
    group.add_argument("--approve", metavar="STRATEGY_ID", help="Approve a strategy for paper trading.")
    group.add_argument("--reject", metavar="STRATEGY_ID", help="Reject a pending strategy promotion.")
    parser.add_argument("--reason", default="", help="Reason for rejection.")
    parser.add_argument("--reviewer", default="operator", help="Reviewer name recorded in the audit log.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    queue = ApprovalQueue()
    if args.list:
        pending = queue.list_pending()
        if not pending:
            print("No pending strategy promotions.")
            return 0
        for request in pending:
            metrics = request.metrics
            print(
                f"{request.strategy_id} | requested_at={request.requested_at} | "
                f"return={metrics.get('total_return')} drawdown={metrics.get('max_drawdown')} "
                f"sharpe={metrics.get('sharpe_ratio', metrics.get('sharpe'))} | {request.rationale}"
            )
        return 0
    if args.approve:
        try:
            decision = queue.approve(args.approve, reviewer=args.reviewer)
        except KeyError:
            print(
                f"No pending promotion request found for {args.approve!r}. "
                "Run `python3 scripts/review_strategy_promotions.py --list` and approve one of the listed strategy IDs.",
                file=sys.stderr,
            )
            return 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Approved {decision.strategy_id} for paper trading.")
        return 0
    if args.reject:
        if not args.reason.strip():
            print("--reason is required when rejecting a promotion.", file=sys.stderr)
            return 2
        try:
            decision = queue.reject(args.reject, reviewer=args.reviewer, reason=args.reason)
        except KeyError:
            print(
                f"No pending promotion request found for {args.reject!r}. "
                "Run `python3 scripts/review_strategy_promotions.py --list` and reject one of the listed strategy IDs.",
                file=sys.stderr,
            )
            return 1
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(f"Rejected {decision.strategy_id}: {decision.decision_reason}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
