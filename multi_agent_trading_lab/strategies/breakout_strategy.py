"""Simple long-only breakout trend strategy."""

from __future__ import annotations

from multi_agent_trading_lab.data.data_sources import PriceBar
from multi_agent_trading_lab.strategies.base_strategy import BaseStrategy, StrategyConfig


class BreakoutTrendStrategy(BaseStrategy):
    """Long when close breaks above prior channel; flat below exit channel."""

    def __init__(
        self,
        breakout_window: int = 20,
        exit_window: int = 10,
        version: str = "0.1.0",
        supported_symbols: list[str] | None = None,
    ) -> None:
        if breakout_window <= exit_window:
            raise ValueError("breakout_window must be greater than exit_window")
        params = {
            "breakout_window": int(breakout_window),
            "exit_window": int(exit_window),
        }
        config = StrategyConfig(
            name="breakout_trend",
            version=version,
            parameters=params,
            supported_symbols=supported_symbols or [],
            supported_timeframes=["1d"],
            description="Long-only daily channel breakout strategy.",
        )
        super().__init__(config)
        self.breakout_window = int(breakout_window)
        self.exit_window = int(exit_window)
        self._is_long = False

    def generate_signal(self, bar: PriceBar) -> int:
        breakout_level = bar.get(f"highest_high_{self.breakout_window}")
        exit_level = bar.get(f"lowest_low_{self.exit_window}")
        close = float(bar["close"])
        if breakout_level is None or exit_level is None:
            return 0
        if self._is_long and close < float(exit_level):
            self._is_long = False
        elif not self._is_long and close > float(breakout_level):
            self._is_long = True
        return 1 if self._is_long else 0
