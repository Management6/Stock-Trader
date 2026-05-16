"""Activate the local paper-trading kill switch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.operations.kill_switch import activate_kill_switch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Activate the local paper-trading kill switch.")
    parser.add_argument("--reason", required=True, help="Operator-facing reason for stopping paper trading.")
    parser.add_argument("--operator", default="operator", help="Operator name recorded in the audit log.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = activate_kill_switch(args.reason, operator=args.operator)
    print(f"Kill switch enabled: {state.get('kill_switch_enabled')}")
    print(f"Reason: {state.get('kill_switch_reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
