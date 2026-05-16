"""Walk-forward and out-of-sample validation helpers."""

from __future__ import annotations

from typing import Any

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.data.data_sources import MarketData

DateRange = tuple[str, str]


def run_walk_forward_validation(
    strategy_config: dict[str, Any],
    data: MarketData,
    in_sample: DateRange,
    out_of_sample: DateRange,
    backtest_agent: BacktestAgent | None = None,
) -> dict[str, Any]:
    """Evaluate one fixed strategy over separate in-sample and OOS windows."""

    tester = backtest_agent or BacktestAgent()
    in_sample_result = tester.run_backtest(strategy_config, _slice_market_data(data, in_sample))
    out_of_sample_result = tester.run_backtest(strategy_config, _slice_market_data(data, out_of_sample))
    return {
        "strategy": strategy_config,
        "in_sample": _period_result(in_sample_result, in_sample),
        "out_of_sample": _period_result(out_of_sample_result, out_of_sample),
    }


def _period_result(backtest_result: dict[str, Any], date_range: DateRange) -> dict[str, Any]:
    return {
        "date_range": {"start_date": date_range[0], "end_date": date_range[1]},
        "symbols": backtest_result["symbols"],
        "metrics": backtest_result["metrics"],
        "per_symbol_results": backtest_result["per_symbol_results"],
    }


def _slice_market_data(data: MarketData, date_range: DateRange) -> MarketData:
    start_date, end_date = date_range
    sliced: MarketData = {}
    for symbol, bars in data.items():
        period_bars = [
            bar
            for bar in bars
            if start_date <= str(bar.get("date", "")) <= end_date
        ]
        if period_bars:
            sliced[symbol] = period_bars
    return sliced
