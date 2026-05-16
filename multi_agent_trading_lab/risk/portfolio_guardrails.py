"""Portfolio-level helper checks built on top of risk policy limits."""

from __future__ import annotations

from multi_agent_trading_lab.brokers.base_broker import Position


def total_position_notional(positions: list[Position]) -> float:
    """Return gross long notional for normalized positions."""

    return sum(position.quantity * position.market_price for position in positions)


def has_symbol_exposure(positions: list[Position], symbol: str) -> bool:
    """Return whether a symbol is currently held."""

    return any(position.symbol == symbol.upper() and position.quantity > 0 for position in positions)

