import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.brokers.alpaca_broker import AlpacaBroker
from multi_agent_trading_lab.execution.execution_engine import ExecutionEngine
from multi_agent_trading_lab.execution.order_models import OrderRequest
from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy


class FakeAlpacaClient:
    def __init__(self) -> None:
        self.account = {"cash": "100000.00", "equity": "100500.00", "buying_power": "50000.00"}
        self.positions = [{"symbol": "AAPL", "qty": "2", "avg_entry_price": "150.00", "market_value": "320.00"}]
        self.orders = {}
        self.submitted_orders = []

    def get_account(self):
        return self.account

    def list_positions(self):
        return self.positions

    def submit_order(self, **kwargs):
        self.submitted_orders.append(kwargs)
        order = {
            "id": "alpaca-order-1",
            "symbol": kwargs["symbol"],
            "side": kwargs["side"],
            "qty": str(kwargs["qty"]),
            "status": "accepted",
            "filled_qty": "0",
            "filled_avg_price": None,
            "submitted_at": "2026-05-11T09:00:00Z",
        }
        self.orders[order["id"]] = order
        return order

    def get_order(self, order_id):
        return self.orders[order_id]

    def cancel_order(self, order_id):
        order = dict(self.orders[order_id])
        order["status"] = "canceled"
        self.orders[order_id] = order
        return order


class AlpacaBrokerTests(unittest.TestCase):
    def test_paper_adapter_submits_order_through_injected_client(self) -> None:
        client = FakeAlpacaClient()
        broker = AlpacaBroker(client=client, paper=True)

        status = broker.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=3, estimated_price=100.0).normalized())

        self.assertEqual(status.order_id, "alpaca-order-1")
        self.assertEqual(status.status, "accepted")
        self.assertEqual(client.submitted_orders[0]["symbol"], "AAPL")
        self.assertEqual(client.submitted_orders[0]["qty"], 3)

    def test_get_order_status_maps_provider_payload(self) -> None:
        client = FakeAlpacaClient()
        broker = AlpacaBroker(client=client, paper=True)
        submitted = broker.submit_order(OrderRequest(symbol="MSFT", side="sell", quantity=1).normalized())

        status = broker.get_order_status(submitted.order_id)

        self.assertEqual(status.symbol, "MSFT")
        self.assertEqual(status.side, "sell")
        self.assertEqual(status.quantity, 1)

    def test_list_positions_maps_provider_payload(self) -> None:
        broker = AlpacaBroker(client=FakeAlpacaClient(), paper=True)

        positions = broker.list_positions()

        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].symbol, "AAPL")
        self.assertEqual(positions[0].quantity, 2)
        self.assertEqual(positions[0].average_price, 150.0)
        self.assertEqual(positions[0].market_price, 160.0)

    def test_reconcile_returns_account_positions_and_open_orders(self) -> None:
        client = FakeAlpacaClient()
        broker = AlpacaBroker(client=client, paper=True)
        submitted = broker.submit_order(OrderRequest(symbol="AAPL", side="buy", quantity=3).normalized())

        snapshot = broker.reconcile()

        self.assertEqual(snapshot["account"].mode, "paper")
        self.assertEqual(snapshot["positions"][0].symbol, "AAPL")
        self.assertEqual(snapshot["orders"][0].order_id, submitted.order_id)

    def test_live_trading_remains_blocked_without_existing_safeguards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ExecutionEngine(
                broker=AlpacaBroker(client=FakeAlpacaClient(), paper=True),
                risk_policy=RiskPolicy(),
                audit_log=AuditLog(Path(tmpdir) / "audit.jsonl"),
                mode="live",
                live_enabled=False,
            )

            status = engine.execute_order(OrderRequest(symbol="AAPL", side="buy", quantity=1, estimated_price=100.0))

            self.assertEqual(status.status, "rejected")
            self.assertIn("Live mode guard", status.message)


if __name__ == "__main__":
    unittest.main()
