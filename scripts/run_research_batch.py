"""Run multiple research iterations in one process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator, load_settings
from multi_agent_trading_lab.research.batch import run_research_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated research cycles.")
    parser.add_argument("--strategy-name", default="moving_average_crossover")
    parser.add_argument("--universe", default=None, help="Named universe from config/settings.yaml.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--n-variants", type=int, default=5)
    args = parser.parse_args()

    settings = load_settings("multi_agent_trading_lab/config/settings.yaml")
    if args.universe:
        settings.setdefault("data", {})["default_universe"] = args.universe
    orchestrator = TradingLabOrchestrator(settings)
    data_config = orchestrator._data_config()
    raw_data = orchestrator.data_agent.fetch_data(
        list(data_config["symbols"]),
        str(data_config["start_date"]),
        None if data_config.get("end_date") in {None, "null", ""} else str(data_config.get("end_date")),
    )
    strategy_name = {"ma": "moving_average_crossover", "ma_rsi": "moving_average_rsi_filter", "breakout": "breakout_trend"}.get(args.strategy_name, args.strategy_name)
    strategy_settings = orchestrator.settings.get("strategies", {}).get(strategy_name, {})
    short_bounds = strategy_settings.get("short_window") or strategy_settings.get("exit_window", {"min": 5, "max": 50})
    long_bounds = strategy_settings.get("long_window") or strategy_settings.get("breakout_window", {"min": 20, "max": 200})
    windows = list(
        range(
            int(min(short_bounds["min"], long_bounds["min"])),
            int(max(short_bounds["max"], long_bounds["max"])) + 1,
        )
    )
    rsi_bounds = strategy_settings.get("rsi_period")
    if rsi_bounds:
        windows.extend(range(int(rsi_bounds["min"]), int(rsi_bounds["max"]) + 1))
    featured_data = orchestrator.data_agent.build_features(raw_data, sorted(set(windows)))
    records_by_iteration = run_research_batch(
        strategy_name=strategy_name,
        iterations=args.iterations,
        n_variants_per_iteration=args.n_variants,
        strategy_agent=orchestrator._strategy_agent_for(strategy_name),
        backtest_agent=orchestrator.backtest_agent,
        experiment_logger=orchestrator.logger,
        data=featured_data,
    )
    for index, records in enumerate(records_by_iteration, start=1):
        if not records:
            print(f"Iteration {index}: no experiments")
            continue
        top = max(records, key=lambda record: float(record.metrics.get("sharpe_ratio") or 0.0))
        in_band = [
            record
            for record in records
            if "short_window" in record.strategy_params
            and 45 <= int(record.strategy_params["short_window"]) <= 50
            and 105 <= int(record.strategy_params["long_window"]) <= 120
        ]
        print(
            f"Iteration {index}: top_sharpe={top.metrics.get('sharpe_ratio'):.2f}, "
            f"top_params={top.strategy_params}, champion_band_fraction={len(in_band) / len(records):.0%}"
        )


if __name__ == "__main__":
    main()
