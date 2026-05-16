"""A minimal long/flat trading environment."""

from __future__ import annotations

from dataclasses import dataclass

from multi_agent_trading_lab.data.data_sources import PriceBar


@dataclass
class TradingState:
    index: int
    cash: float
    position: int
    portfolio_value: float
    bar: PriceBar


class TradingEnvironment:
    """Simple one-symbol, long/flat simulator for strategy experiments."""

    def __init__(
        self,
        bars: list[PriceBar],
        initial_cash: float = 10_000.0,
        max_capital_per_trade_pct: float = 0.10,
        allow_leverage: bool = False,
        commission_per_trade: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        if len(bars) < 2:
            raise ValueError("TradingEnvironment requires at least two bars")
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not 0 < max_capital_per_trade_pct <= 1:
            raise ValueError("max_capital_per_trade_pct must be in (0, 1]")
        if commission_per_trade < 0:
            raise ValueError("commission_per_trade must be non-negative")
        if slippage_pct < 0:
            raise ValueError("slippage_pct must be non-negative")
        self.bars = bars
        self.initial_cash = initial_cash
        self.max_capital_per_trade_pct = max_capital_per_trade_pct
        self.allow_leverage = allow_leverage
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct
        self.reset()

    def reset(self) -> TradingState:
        self.index = 0
        self.cash = self.initial_cash
        self.position = 0
        return self.state

    @property
    def state(self) -> TradingState:
        price = float(self.bars[self.index]["close"])
        return TradingState(
            index=self.index,
            cash=self.cash,
            position=self.position,
            portfolio_value=self.cash + self.position * price,
            bar=self.bars[self.index],
        )

    def step(self, action: int) -> tuple[TradingState, float, bool]:
        """Apply an action: ``1`` long, ``0`` flat."""

        current_price = float(self.bars[self.index]["close"])
        previous_value = self.cash + self.position * current_price

        if action == 1 and self.position == 0:
            max_trade_notional = previous_value * self.max_capital_per_trade_pct
            available_notional = max_trade_notional if self.allow_leverage else min(self.cash, max_trade_notional)
            fill_price = current_price * (1.0 + self.slippage_pct)
            self.position = int(max(0.0, available_notional - self.commission_per_trade) // fill_price)
            if self.position > 0:
                self.cash -= self.position * fill_price + self.commission_per_trade
        elif action == 0 and self.position > 0:
            fill_price = current_price * (1.0 - self.slippage_pct)
            self.cash += self.position * fill_price - self.commission_per_trade
            self.position = 0

        self.index += 1
        next_price = float(self.bars[self.index]["close"])
        next_value = self.cash + self.position * next_price
        reward = next_value - previous_value
        done = self.index >= len(self.bars) - 1
        return self.state, reward, done
