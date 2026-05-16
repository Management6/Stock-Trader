"""Backtest metric utilities.

Sharpe ratio definition used by this project:

- Returns are computed from consecutive portfolio values.
- The initial bar size is daily, so the annualization factor defaults to 252.
- Risk-free rate is assumed to be 0 for now.
- Sharpe ratio = mean(daily_returns) / std(daily_returns) * sqrt(252).
"""

from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any


DEFAULT_METRIC_KEYS = ("total_return", "max_drawdown", "sharpe_ratio", "sharpe", "history_points")


def compute_backtest_metrics(
    portfolio_values: list[float],
    annualization_factor: int = 252,
) -> dict[str, float | int | None]:
    """Compute consistent backtest metrics from a portfolio equity curve."""

    if len(portfolio_values) < 2:
        return {
            "total_return": None,
            "max_drawdown": None,
            "sharpe_ratio": None,
            "sharpe": None,
            "history_points": len(portfolio_values),
        }

    start_value = portfolio_values[0]
    end_value = portfolio_values[-1]
    daily_returns = [
        (current - previous) / previous
        for previous, current in zip(portfolio_values, portfolio_values[1:])
        if previous != 0
    ]
    sharpe_ratio = compute_sharpe_ratio(daily_returns, annualization_factor=annualization_factor)
    return {
        "total_return": (end_value - start_value) / start_value if start_value else None,
        "max_drawdown": compute_max_drawdown(portfolio_values),
        "sharpe_ratio": sharpe_ratio,
        "sharpe": sharpe_ratio,
        "history_points": len(daily_returns),
    }


def compute_sharpe_ratio(
    returns: list[float],
    annualization_factor: int = 252,
) -> float | None:
    """Return annualized Sharpe ratio assuming a zero risk-free rate."""

    if len(returns) < 2:
        return None
    volatility = pstdev(returns)
    if volatility == 0:
        return 0.0
    return (mean(returns) / volatility) * sqrt(annualization_factor)


def compute_max_drawdown(portfolio_values: list[float]) -> float | None:
    """Return max drawdown as a negative fraction from peak equity."""

    if not portfolio_values:
        return None
    peak = portfolio_values[0]
    drawdowns: list[float] = []
    for value in portfolio_values:
        peak = max(peak, value)
        drawdowns.append((value - peak) / peak if peak else 0.0)
    return min(drawdowns)


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Ensure experiment metrics include the standard metric keys."""

    normalized = dict(metrics)
    sharpe = normalized.get("sharpe_ratio", normalized.get("sharpe", normalized.get("sharpe_placeholder")))
    normalized.setdefault("total_return", None)
    normalized.setdefault("max_drawdown", None)
    normalized["sharpe_ratio"] = sharpe
    normalized["sharpe"] = sharpe
    normalized.setdefault("history_points", None)
    return normalized
