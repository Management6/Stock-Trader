import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.brokers.paper_broker import PaperBroker
from multi_agent_trading_lab.execution.order_models import OrderRequest


class PaperBrokerTests(unittest.TestCase):
    def test_paper_broker_submits_and_lists_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            broker = PaperBroker(Path(tmpdir) / "paper.json")
            status = broker.submit_order(
                OrderRequest(symbol="AAPL", side="buy", quantity=2, estimated_price=100.0).normalized()
            )

            self.assertEqual(status.status, "filled")
            self.assertEqual(len(broker.list_positions()), 1)
            self.assertEqual(broker.list_positions()[0].symbol, "AAPL")


if __name__ == "__main__":
    unittest.main()
