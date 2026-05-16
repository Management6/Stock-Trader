"""Safe Alpaca paper broker adapter with an injectable client boundary."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from multi_agent_trading_lab.brokers.base_broker import AccountState, BaseBroker, Position
from multi_agent_trading_lab.execution.order_models import OrderRequest, OrderStatus


class AlpacaPaperClient(Protocol):
    """Minimal client surface required by the paper adapter."""

    def get_account(self) -> Any:
        ...

    def list_positions(self) -> list[Any]:
        ...

    def submit_order(self, **kwargs: Any) -> Any:
        ...

    def get_order(self, order_id: str) -> Any:
        ...

    def cancel_order(self, order_id: str) -> Any:
        ...


class AlpacaBroker(BaseBroker):
    """Provider-neutral adapter for Alpaca paper trading.

    The adapter intentionally requires an injected client so tests and local
    workflows never need real credentials. It supports paper mode only; live
    execution remains guarded outside this broker by ``ExecutionEngine`` and is
    also rejected here at construction time.
    """

    name = "alpaca_paper"

    def __init__(self, client: AlpacaPaperClient | None = None, paper: bool = True) -> None:
        if not paper:
            raise ValueError("AlpacaBroker supports paper mode only.")
        if client is None:
            raise ValueError("AlpacaBroker requires an injected paper client; credentials are not loaded from the repo.")
        self.client = client
        self.paper = paper

    def get_account_state(self) -> AccountState:
        account = self.client.get_account()
        return AccountState(
            cash=_float(_get(account, "cash")),
            equity=_float(_get(account, "equity")),
            buying_power=_float(_get(account, "buying_power")),
            mode="paper",
        )

    def list_positions(self) -> list[Position]:
        positions: list[Position] = []
        for payload in self.client.list_positions():
            quantity = int(float(_get(payload, "qty", "quantity", default=0)))
            average_price = _float(_get(payload, "avg_entry_price", "average_price"))
            market_price = _position_market_price(payload, quantity, average_price)
            positions.append(
                Position(
                    symbol=str(_get(payload, "symbol")).upper(),
                    quantity=quantity,
                    average_price=average_price,
                    market_price=market_price,
                    metadata=_to_metadata(payload),
                )
            )
        return positions

    def submit_order(self, order: OrderRequest) -> OrderStatus:
        payload = self.client.submit_order(
            symbol=order.symbol,
            qty=order.quantity,
            side=order.side,
            type=order.order_type,
            time_in_force="day",
            limit_price=order.limit_price,
        )
        return _order_status(payload)

    def cancel_order(self, order_id: str) -> OrderStatus:
        return _order_status(self.client.cancel_order(order_id))

    def get_order_status(self, order_id: str) -> OrderStatus:
        return _order_status(self.client.get_order(order_id))

    def reconcile(self) -> dict[str, Any]:
        """Return a normalized paper-account snapshot for reconciliation."""

        orders = []
        raw_orders = getattr(self.client, "orders", {})
        if isinstance(raw_orders, dict):
            orders = [_order_status(payload) for payload in raw_orders.values()]
        return {
            "account": self.get_account_state(),
            "positions": self.list_positions(),
            "orders": orders,
        }


def _order_status(payload: Any) -> OrderStatus:
    return OrderStatus(
        order_id=str(_get(payload, "id", "order_id")),
        symbol=str(_get(payload, "symbol")).upper(),
        side=str(_get(payload, "side")).lower(),
        quantity=int(float(_get(payload, "qty", "quantity", default=0))),
        status=_normalize_status(str(_get(payload, "status", default="unknown"))),
        filled_quantity=int(float(_get(payload, "filled_qty", "filled_quantity", default=0) or 0)),
        average_fill_price=_optional_float(_get(payload, "filled_avg_price", "average_fill_price", default=None)),
        message=f"Alpaca paper order {_normalize_status(str(_get(payload, 'status', default='unknown')))}.",
        timestamp=str(_get(payload, "submitted_at", "timestamp", default=datetime.now(UTC).isoformat())),
    )


def _normalize_status(status: str) -> str:
    return "cancelled" if status == "canceled" else status


def _position_market_price(payload: Any, quantity: int, average_price: float) -> float:
    if _get(payload, "market_price", default=None) is not None:
        return _float(_get(payload, "market_price"))
    market_value = _get(payload, "market_value", default=None)
    if market_value is not None and quantity:
        return _float(market_value) / quantity
    return average_price


def _get(payload: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if isinstance(payload, dict) and key in payload:
            return payload[key]
        if hasattr(payload, key):
            return getattr(payload, key)
    return default


def _float(value: Any) -> float:
    return float(value or 0.0)


def _optional_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _to_metadata(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    return {key: getattr(payload, key) for key in dir(payload) if not key.startswith("_") and not callable(getattr(payload, key))}
