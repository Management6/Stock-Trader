"""Example moving-average crossover strategy."""

from __future__ import annotations

from multi_agent_trading_lab.data.data_sources import PriceBar
from multi_agent_trading_lab.strategies.base_strategy import BaseStrategy, StrategyConfig


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Versioned moving-average strategy used as a safe placeholder."""

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        version: str = "0.1.0",
        supported_symbols: list[str] | None = None,
    ) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        params = {
            "short_window": int(short_window),
            "long_window": int(long_window),
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
        }
        config = StrategyConfig(
            name="moving_average_crossover",
            version=version,
            parameters=params,
            supported_symbols=supported_symbols or [],
            supported_timeframes=["1d"],
            description="Simple long/flat moving-average crossover strategy.",
        )
        super().__init__(config)
        self.params = params
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def generate_signal(self, bar: PriceBar) -> int:
        short_value = float(bar.get(f"sma_{self.short_window}", bar["close"]))
        long_value = float(bar.get(f"sma_{self.long_window}", bar["close"]))
        return 1 if short_value > long_value else 0
