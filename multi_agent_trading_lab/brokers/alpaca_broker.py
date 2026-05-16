"""Safe stub for a future Alpaca broker integration."""

from __future__ import annotations

from multi_agent_trading_lab.brokers.base_broker import AccountState, BaseBroker, Position
from multi_agent_trading_lab.execution.order_models import OrderRequest, OrderStatus


class AlpacaBroker(BaseBroker):
    """Non-trading placeholder for future live or paper Alpaca integration.

    This class intentionally does not read credentials or place orders. A future
    implementation should load credentials from environment variables or a
    secret manager, use Alpaca's official SDK, and keep the same BaseBroker
    boundary.
    """

    name = "alpaca_stub"

    def _disabled(self) -> RuntimeError:
        return RuntimeError("AlpacaBroker is a safe stub and cannot place live orders yet.")

    def get_account_state(self) -> AccountState:
        raise self._disabled()

    def list_positions(self) -> list[Position]:
        raise self._disabled()

    def submit_order(self, order: OrderRequest) -> OrderStatus:
        raise self._disabled()

    def cancel_order(self, order_id: str) -> OrderStatus:
        raise self._disabled()

    def get_order_status(self, order_id: str) -> OrderStatus:
        raise self._disabled()

