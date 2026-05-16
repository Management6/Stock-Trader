"""Broker interface for paper and future live execution providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from multi_agent_trading_lab.execution.order_models import OrderRequest, OrderStatus


@dataclass(frozen=True)
class AccountState:
    """Normalized account state returned by brokers."""

    cash: float
    equity: float
    buying_power: float
    mode: str = "paper"


@dataclass(frozen=True)
class Position:
    """Normalized position returned by brokers."""

    symbol: str
    quantity: int
    average_price: float
    market_price: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseBroker(ABC):
    """Provider-neutral broker contract.

    Live integrations must implement this interface without leaking provider
    credentials or SDK-specific payloads into the rest of the trading system.
    """

    name: str

    @abstractmethod
    def get_account_state(self) -> AccountState:
        """Return cash, equity, and buying power."""

    @abstractmethod
    def list_positions(self) -> list[Position]:
        """Return currently open positions."""

    @abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderStatus:
        """Submit an order request to the broker."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> OrderStatus:
        """Cancel an open order."""

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        """Return broker order status by identifier."""

