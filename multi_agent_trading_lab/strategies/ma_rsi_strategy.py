"""Moving-average crossover strategy with an RSI entry filter."""

from __future__ import annotations

from multi_agent_trading_lab.data.data_sources import PriceBar
from multi_agent_trading_lab.strategies.base_strategy import BaseStrategy, StrategyConfig


class MovingAverageRsiFilterStrategy(BaseStrategy):
    """Long/flat MA crossover strategy that filters entries by RSI band."""

    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 50,
        rsi_period: int = 14,
        rsi_min: float = 35.0,
        rsi_max: float = 65.0,
        version: str = "0.1.0",
        supported_symbols: list[str] | None = None,
    ) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        if rsi_period < 2:
            raise ValueError("rsi_period must be at least 2")
        if rsi_min >= rsi_max:
            raise ValueError("rsi_min must be less than rsi_max")
        params = {
            "short_window": int(short_window),
            "long_window": int(long_window),
            "rsi_period": int(rsi_period),
            "rsi_min": float(rsi_min),
            "rsi_max": float(rsi_max),
        }
        config = StrategyConfig(
            name="moving_average_rsi_filter",
            version=version,
            parameters=params,
            supported_symbols=supported_symbols or [],
            supported_timeframes=["1d"],
            description="Moving-average crossover with RSI band filter for entries.",
        )
        super().__init__(config)
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.rsi_period = int(rsi_period)
        self.rsi_min = float(rsi_min)
        self.rsi_max = float(rsi_max)
        self._is_long = False

    def generate_signal(self, bar: PriceBar) -> int:
        short_value = float(bar.get(f"sma_{self.short_window}", bar["close"]))
        long_value = float(bar.get(f"sma_{self.long_window}", bar["close"]))
        ma_long = short_value > long_value
        if self._is_long and not ma_long:
            self._is_long = False
            return 0
        if self._is_long:
            return 1
        rsi_value = bar.get(f"rsi_{self.rsi_period}")
        if rsi_value is None:
            return 0
        if ma_long and self.rsi_min <= float(rsi_value) <= self.rsi_max:
            self._is_long = True
        return 1 if self._is_long else 0
