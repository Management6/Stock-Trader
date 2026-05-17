"""Portfolio-level backtesting helpers."""

from __future__ import annotations

from typing import Any

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.data.data_sources import MarketData
from multi_agent_trading_lab.metrics import compute_backtest_metrics


def run_portfolio_backtest(
    strategy_config: dict[str, Any],
    data: MarketData,
    backtest_agent: BacktestAgent | None = None,
    data_quality_report: dict[str, Any] | None = None,
    allocation_method: str = "equal_weight",
    max_symbols: int | None = None,
) -> dict[str, Any]:
    """Run a deterministic equal-weight portfolio backtest across symbols."""

    if allocation_method != "equal_weight":
        raise ValueError("Only equal_weight portfolio allocation is supported.")
    base_agent = backtest_agent or BacktestAgent()
    selected_items = list(data.items())[:max_symbols] if max_symbols is not None else list(data.items())
    per_symbol_results: dict[str, Any] = {}
    skipped_symbols: list[dict[str, Any]] = []
    quality_by_symbol = _quality_issues_by_symbol(data_quality_report or {"passed": True, "issues": []})

    runnable_symbols = [symbol for symbol, _bars in selected_items if not _has_quality_errors(quality_by_symbol.get(symbol, []))]
    allocation_count = max(1, len(runnable_symbols))
    symbol_capital = float(base_agent.starting_capital) / allocation_count

    for symbol, bars in selected_items:
        quality_issues = quality_by_symbol.get(symbol, [])
        if _has_quality_errors(quality_issues):
            skipped_symbols.append({"symbol": symbol, "reason": "Data quality errors.", "data_quality_issues": quality_issues})
            continue
        try:
            symbol_agent = BacktestAgent(
                starting_capital=symbol_capital,
                max_capital_per_trade_pct=base_agent.max_capital_per_trade_pct,
                allow_leverage=base_agent.allow_leverage,
                commission_per_trade=base_agent.commission_per_trade,
                slippage_pct=base_agent.slippage_pct,
            )
            result = symbol_agent.run_backtest(strategy_config, {symbol: bars}, data_quality_report=data_quality_report)
            per_symbol_results[symbol] = result["per_symbol_results"][symbol]
        except Exception as exc:
            skipped_symbols.append({"symbol": symbol, "reason": str(exc), "data_quality_issues": quality_issues})

    if not per_symbol_results:
        raise ValueError("Portfolio backtest requires at least one runnable symbol.")

    portfolio_values = _combine_equity_curves([result["portfolio_values"] for result in per_symbol_results.values()])
    metrics = compute_backtest_metrics(portfolio_values)
    per_symbol_returns = {
        symbol: result["metrics"].get("total_return")
        for symbol, result in per_symbol_results.items()
    }
    per_symbol_trade_counts = {
        symbol: int(result.get("trade_count", result["metrics"].get("trade_count", 0)) or 0)
        for symbol, result in per_symbol_results.items()
    }
    allocations = {symbol: 1.0 / len(per_symbol_results) for symbol in per_symbol_results}
    aggregate_exposure = {
        "allocation_method": allocation_method,
        "symbol_count": len(per_symbol_results),
        "max_allocated_pct": max(allocations.values()) if allocations else 0.0,
        "gross_allocated_pct": sum(allocations.values()),
    }
    cost_assumptions = base_agent._cost_assumptions()
    metrics.update(
        {
            "per_symbol_returns": per_symbol_returns,
            "per_symbol_trade_counts": per_symbol_trade_counts,
            "aggregate_exposure": aggregate_exposure,
            "cost_assumptions": cost_assumptions,
            "data_quality": data_quality_report or {"passed": True, "issues": []},
        }
    )
    return {
        "strategy": strategy_config,
        "symbols": list(per_symbol_results.keys()),
        "allocation_method": allocation_method,
        "allocations": allocations,
        "portfolio_values": portfolio_values,
        "metrics": metrics,
        "per_symbol_results": per_symbol_results,
        "skipped_symbols": skipped_symbols,
        "aggregate_exposure": aggregate_exposure,
        "cost_assumptions": cost_assumptions,
        "data_quality": data_quality_report or {"passed": True, "issues": []},
    }


def _combine_equity_curves(curves: list[list[float]]) -> list[float]:
    min_length = min(len(curve) for curve in curves)
    return [sum(curve[index] for curve in curves) for index in range(min_length)]


def _quality_issues_by_symbol(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    issues_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for issue in report.get("issues", []):
        issues_by_symbol.setdefault(str(issue.get("symbol")), []).append(dict(issue))
    return issues_by_symbol


def _has_quality_errors(issues: list[dict[str, Any]]) -> bool:
    return any(issue.get("severity") == "error" for issue in issues)
