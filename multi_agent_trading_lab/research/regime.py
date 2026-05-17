"""Deterministic market-regime classification helpers."""

from __future__ import annotations

from math import sqrt
from statistics import mean, pstdev
from typing import Any


def classify_market_regime(
    bars: list[dict[str, Any]],
    *,
    benchmark_symbol: str = "SPY",
    lookback_days: int = 120,
    trend_ma_days: int = 50,
    max_volatility: float = 0.30,
) -> dict[str, Any]:
    """Classify a benchmark as bullish, bearish, high volatility, or insufficient."""

    required = max(lookback_days, trend_ma_days) + 1
    if len(bars) < required:
        return {
            "regime": "insufficient_data",
            "benchmark_symbol": benchmark_symbol,
            "lookback_days": lookback_days,
            "trend_metric": None,
            "volatility_metric": None,
            "reason": f"Insufficient benchmark data: {len(bars)} bars available, {required} required.",
            "details": {"available_bars": len(bars), "required_bars": required, "trend_ma_days": trend_ma_days},
        }

    lookback = bars[-lookback_days:]
    closes = [float(bar["close"]) for bar in lookback if bar.get("close") is not None]
    if len(closes) < required - 1:
        return {
            "regime": "insufficient_data",
            "benchmark_symbol": benchmark_symbol,
            "lookback_days": lookback_days,
            "trend_metric": None,
            "volatility_metric": None,
            "reason": "Insufficient benchmark close data.",
            "details": {"available_closes": len(closes), "required_bars": required},
        }

    latest_close = closes[-1]
    moving_average = mean(closes[-trend_ma_days:])
    trend_metric = (latest_close - moving_average) / moving_average if moving_average else 0.0
    returns = [(current - previous) / previous for previous, current in zip(closes, closes[1:]) if previous]
    volatility = pstdev(returns) * sqrt(252) if len(returns) >= 2 else 0.0

    if volatility > max_volatility:
        regime = "high_volatility"
        reason = f"Benchmark volatility {volatility:.2f} exceeds threshold {max_volatility:.2f}."
    elif latest_close < moving_average:
        regime = "bearish"
        reason = "Benchmark close is below moving average."
    else:
        regime = "bullish"
        reason = "Benchmark close is above moving average with acceptable volatility."

    return {
        "regime": regime,
        "benchmark_symbol": benchmark_symbol,
        "lookback_days": lookback_days,
        "trend_metric": trend_metric,
        "volatility_metric": volatility,
        "reason": reason,
        "details": {
            "latest_close": latest_close,
            "moving_average": moving_average,
            "trend_ma_days": trend_ma_days,
            "max_volatility": max_volatility,
        },
    }
