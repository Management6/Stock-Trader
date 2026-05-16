"""File-backed paper broker for safe local execution workflows."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from multi_agent_trading_lab.brokers.base_broker import AccountState, BaseBroker, Position
from multi_agent_trading_lab.execution.order_models import OrderRequest, OrderStatus


class PaperBroker(BaseBroker):
    """Simulates broker order handling with persistent JSON state."""

    name = "paper"

    def __init__(self, state_path: str | Path = "multi_agent_trading_lab/state/paper_broker_state.json") -> None:
        self.state_path = Path(state_path)
        self._state = self._load_state()

    def get_account_state(self) -> AccountState:
        return AccountState(
            cash=float(self._state["cash"]),
            equity=float(self._state["equity"]),
            buying_power=float(self._state["cash"]),
            mode="paper",
        )

    def list_positions(self) -> list[Position]:
        positions: list[Position] = []
        for symbol, payload in self._state["positions"].items():
            positions.append(
                Position(
                    symbol=symbol,
                    quantity=int(payload["quantity"]),
                    average_price=float(payload["average_price"]),
                    market_price=float(payload.get("market_price", payload["average_price"])),
                )
            )
        return positions

    def submit_order(self, order: OrderRequest) -> OrderStatus:
        fill_price = float(order.limit_price or order.estimated_price or 100.0)
        cost = fill_price * order.quantity
        status = "filled"
        message = "Paper order filled."

        if order.side == "buy":
            if cost > float(self._state["cash"]):
                status = "rejected"
                message = "Insufficient paper cash."
            else:
                self._state["cash"] = float(self._state["cash"]) - cost
                self._upsert_position(order.symbol, order.quantity, fill_price)
        elif order.side == "sell":
            existing = self._state["positions"].get(order.symbol, {"quantity": 0, "average_price": fill_price})
            sell_quantity = min(order.quantity, int(existing["quantity"]))
            if sell_quantity <= 0:
                status = "rejected"
                message = "No paper position to sell."
            else:
                existing["quantity"] = int(existing["quantity"]) - sell_quantity
                self._state["cash"] = float(self._state["cash"]) + fill_price * sell_quantity
                if existing["quantity"] <= 0:
                    self._state["positions"].pop(order.symbol, None)
                else:
                    self._state["positions"][order.symbol] = existing
        else:
            status = "rejected"
            message = f"Unsupported side: {order.side}"

        order_id = str(uuid4())
        order_status = OrderStatus(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            status=status,
            filled_quantity=order.quantity if status == "filled" else 0,
            average_fill_price=fill_price if status == "filled" else None,
            message=message,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._state["orders"][order_id] = order_status.to_dict()
        self._mark_to_market()
        self._save_state()
        return order_status

    def cancel_order(self, order_id: str) -> OrderStatus:
        current = self.get_order_status(order_id)
        if current.status == "filled":
            return current
        updated = OrderStatus(**{**current.to_dict(), "status": "cancelled", "message": "Paper order cancelled."})
        self._state["orders"][order_id] = updated.to_dict()
        self._save_state()
        return updated

    def get_order_status(self, order_id: str) -> OrderStatus:
        payload = self._state["orders"].get(order_id)
        if payload is None:
            return OrderStatus(
                order_id=order_id,
                symbol="UNKNOWN",
                side="buy",
                quantity=0,
                status="not_found",
                filled_quantity=0,
                average_fill_price=None,
                message="Order not found.",
                timestamp=datetime.now(UTC).isoformat(),
            )
        return OrderStatus(**payload)

    def _upsert_position(self, symbol: str, quantity: int, fill_price: float) -> None:
        existing = self._state["positions"].get(symbol)
        if existing is None:
            self._state["positions"][symbol] = {
                "quantity": quantity,
                "average_price": fill_price,
                "market_price": fill_price,
            }
            return
        old_quantity = int(existing["quantity"])
        new_quantity = old_quantity + quantity
        weighted_price = ((old_quantity * float(existing["average_price"])) + (quantity * fill_price)) / new_quantity
        existing.update({"quantity": new_quantity, "average_price": weighted_price, "market_price": fill_price})

    def _load_state(self) -> dict[str, object]:
        if not self.state_path.exists():
            return {"cash": 100_000.0, "equity": 100_000.0, "positions": {}, "orders": {}}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _mark_to_market(self) -> None:
        position_value = 0.0
        for position in self._state["positions"].values():
            position_value += int(position["quantity"]) * float(position.get("market_price", position["average_price"]))
        self._state["equity"] = float(self._state["cash"]) + position_value

