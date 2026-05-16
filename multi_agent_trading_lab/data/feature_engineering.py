"""Feature builders for market data."""

from __future__ import annotations

from collections import deque
from typing import Iterable

from multi_agent_trading_lab.data.data_sources import MarketData, PriceBar


def add_moving_averages(
    market_data: MarketData,
    windows: Iterable[int] = (5, 20),
) -> MarketData:
    """Return a copy of market data with simple moving average columns."""

    window_list = sorted(set(windows))
    enriched: MarketData = {}
    for symbol, bars in market_data.items():
        queues = {window: deque(maxlen=window) for window in window_list}
        enriched_bars: list[PriceBar] = []
        for bar in bars:
            close = float(bar["close"])
            new_bar: PriceBar = dict(bar)
            for window, queue in queues.items():
                queue.append(close)
                key = f"sma_{window}"
                new_bar[key] = sum(queue) / len(queue)
            enriched_bars.append(new_bar)
        enriched[symbol] = enriched_bars
    return enriched


def add_breakout_channels(
    market_data: MarketData,
    windows: Iterable[int] = (20, 10),
) -> MarketData:
    """Return a copy of market data with prior rolling high/low channels.

    Channel values exclude the current bar so the strategy cannot look ahead.
    """

    window_list = sorted(set(int(window) for window in windows))
    enriched: MarketData = {}
    for symbol, bars in market_data.items():
        high_queues = {window: deque(maxlen=window) for window in window_list}
        low_queues = {window: deque(maxlen=window) for window in window_list}
        enriched_bars: list[PriceBar] = []
        for bar in bars:
            new_bar: PriceBar = dict(bar)
            for window in window_list:
                high_queue = high_queues[window]
                low_queue = low_queues[window]
                if len(high_queue) >= window:
                    new_bar[f"highest_high_{window}"] = max(high_queue)
                if len(low_queue) >= window:
                    new_bar[f"lowest_low_{window}"] = min(low_queue)
                high_queue.append(float(bar["high"]))
                low_queue.append(float(bar["low"]))
            enriched_bars.append(new_bar)
        enriched[symbol] = enriched_bars
    return enriched


def add_rsi(
    market_data: MarketData,
    periods: Iterable[int] = (14,),
) -> MarketData:
    """Return a copy of market data with simple rolling RSI columns."""

    period_list = sorted(set(int(period) for period in periods))
    enriched: MarketData = {}
    for symbol, bars in market_data.items():
        close_queues = {period: deque(maxlen=period + 1) for period in period_list}
        enriched_bars: list[PriceBar] = []
        for bar in bars:
            close = float(bar["close"])
            new_bar: PriceBar = dict(bar)
            for period, queue in close_queues.items():
                queue.append(close)
                if len(queue) < period + 1:
                    continue
                changes = [queue[index] - queue[index - 1] for index in range(1, len(queue))]
                gains = [max(change, 0.0) for change in changes]
                losses = [abs(min(change, 0.0)) for change in changes]
                avg_gain = sum(gains) / period
                avg_loss = sum(losses) / period
                if avg_loss == 0:
                    rsi = 100.0 if avg_gain > 0 else 50.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                new_bar[f"rsi_{period}"] = rsi
            enriched_bars.append(new_bar)
        enriched[symbol] = enriched_bars
    return enriched


def add_strategy_features(
    market_data: MarketData,
    windows: Iterable[int],
) -> MarketData:
    """Apply all currently supported deterministic strategy features."""

    return add_rsi(add_breakout_channels(add_moving_averages(market_data, windows), windows), windows)
