"""Analyze a moving-average parameter pair from experiment history."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.experiments.analysis import analyze_ma_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an example moving-average strategy from experiment history.")
    parser.add_argument("--short-window", type=int, default=6)
    parser.add_argument("--long-window", type=int, default=35)
    args = parser.parse_args()

    summary = analyze_ma_strategy(short_window=args.short_window, long_window=args.long_window)
    print("Research analysis only. This does not place trades or enable live trading.")
    print(f"Strategy: {summary['strategy_name']}")
    print(f"Params: {summary['params']}")
    if summary["status"] != "found":
        print("No matching experiments found.")
        return

    data_range = summary.get("data_range") or {}
    decision = summary.get("decision") or {}
    print(f"Experiment: {summary['experiment_id']}")
    print(f"Data range: {data_range.get('start_date')} to {data_range.get('end_date')} | symbols={data_range.get('symbols')}")
    print(f"Total return: {summary['total_return']:.2%}")
    print(f"Max drawdown: {summary['max_drawdown']:.2%}")
    print(f"Sharpe ratio: {summary['sharpe_ratio']:.2f}")
    print(f"Score: {summary['score']:.2f}")
    print(f"Risk decision: {decision.get('approved')} ({decision.get('reason')})")


if __name__ == "__main__":
    main()
