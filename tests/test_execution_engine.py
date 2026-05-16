import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.brokers.paper_broker import PaperBroker
from multi_agent_trading_lab.execution.execution_engine import ExecutionEngine
from multi_agent_trading_lab.execution.order_models import OrderRequest
from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy


class ExecutionEngineTests(unittest.TestCase):
    def test_invalid_order_is_rejected_before_broker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ExecutionEngine(
                broker=PaperBroker(Path(tmpdir) / "paper.json"),
                risk_policy=RiskPolicy(),
                audit_log=AuditLog(Path(tmpdir) / "audit.jsonl"),
                mode="paper",
            )

            status = engine.execute_order(OrderRequest(symbol="", side="buy", quantity=1))

            self.assertEqual(status.status, "rejected")
            self.assertIn("symbol is required", status.message)

    def test_live_guard_blocks_even_with_broker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = ExecutionEngine(
                broker=PaperBroker(Path(tmpdir) / "paper.json"),
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
