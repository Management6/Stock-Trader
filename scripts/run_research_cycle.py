"""Run a safe research and backtesting cycle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator, load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a safe research and backtesting cycle.")
    parser.add_argument("--settings", default="multi_agent_trading_lab/config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--strategy-families", default=None, help="Comma-separated families: ma, ma_rsi, breakout.")
    parser.add_argument("--universe", default=None, help="Named universe from config/settings.yaml, e.g. core_asx, us_tech, full_mix.")
    parser.add_argument("--n-variants", type=int, default=None)
    args = parser.parse_args()
    families = [item.strip() for item in args.strategy_families.split(",") if item.strip()] if args.strategy_families else None
    settings = load_settings(args.settings)
    if args.universe:
        settings.setdefault("data", {})["default_universe"] = args.universe
    result = TradingLabOrchestrator(settings).run_research_cycle(
        n_variants=args.n_variants,
        strategy_families=families,
    )
    objective = result.details.get("objective", {"type": "sharpe"})
    data = result.details.get("data", {})
    ranges = data.get("ranges", {})
    loaded_symbols = ", ".join(data.get("symbols", []))
    if ranges:
        starts = [payload["start_date"] for payload in ranges.values() if payload.get("start_date")]
        ends = [payload["end_date"] for payload in ranges.values() if payload.get("end_date")]
        source = data.get("source", "unknown")
        print(
            f"Loaded {source} daily data for {loaded_symbols} "
            f"from {min(starts) if starts else data.get('configured_start_date')} "
            f"to {max(ends) if ends else data.get('configured_end_date') or 'latest available'}."
        )
    print("Research cycle complete.")
    print(f"Objective: {objective.get('type', 'sharpe')} (risk-adjusted return; change in config/settings.yaml)")
    print(f"Experiments run: {len(result.details['records'])}")
    print("\n".join(result.summaries))
    ranked = sorted(
        result.details["records"],
        key=lambda record: record.get("objective_score", 0.0),
        reverse=True,
    )
    print("\nTop 3 variants:")
    for index, record in enumerate(ranked[:3], start=1):
        params = record["strategy_params"]
        metrics = record["metrics"]
        sharpe = metrics.get("sharpe_ratio", metrics.get("sharpe"))
        print(
            f"{index}) family: {record['strategy_name']}, params: {params}, total_return: {metrics['total_return']:.2%}, "
            f"max_drawdown: {metrics['max_drawdown']:.2%}, sharpe_ratio: {sharpe:.2f}, "
            f"score: {record.get('objective_score', 0.0):.2f}"
        )


if __name__ == "__main__":
    main()
