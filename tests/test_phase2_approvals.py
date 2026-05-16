import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.operations.approvals import ApprovalQueue
from multi_agent_trading_lab.operations.alerting import AlertManager, StubNotifierSink
from multi_agent_trading_lab.strategies.registry import StrategyRegistry
from scripts.review_strategy_promotions import main as review_main


class ApprovalQueueTests(unittest.TestCase):
    def test_request_stays_pending_until_explicit_approval_promotes_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry = StrategyRegistry(root / "registry.json")
            audit = AuditLog(root / "audit.jsonl")
            stub = StubNotifierSink()
            queue = ApprovalQueue(
                root / "approvals.json",
                registry=registry,
                audit_log=audit,
                alert_manager=AlertManager([stub]),
            )
            strategy = {"id": "s1", "name": "moving_average_crossover", "strategy_params": {"short_window": 5, "long_window": 20}}
            registry.register(strategy, stage="candidate")

            request = queue.request_promotion(
                strategy=strategy,
                rationale="Risk policy passed.",
                metrics={"total_return": 0.1, "max_drawdown": -0.05},
                requester="test",
            )

            self.assertEqual(request.status, "pending")
            self.assertEqual(registry.list_by_stage("active"), [])
            self.assertEqual(stub.sent_alerts[0].event_type, "strategy_promotion_awaiting_approval")

            decision = queue.approve("s1", reviewer="operator")

            self.assertEqual(decision.status, "approved")
            self.assertEqual(registry.list_by_stage("active")[0]["id"], "s1")
            audit_events = [json.loads(line)["event_type"] for line in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("strategy_promotion_approved", audit_events)

    def test_rejection_records_reason_and_does_not_promote(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry = StrategyRegistry(root / "registry.json")
            audit = AuditLog(root / "audit.jsonl")
            stub = StubNotifierSink()
            queue = ApprovalQueue(
                root / "approvals.json",
                registry=registry,
                audit_log=audit,
                alert_manager=AlertManager([stub]),
            )
            strategy = {"id": "s2", "name": "moving_average_crossover", "strategy_params": {"short_window": 8, "long_window": 30}}
            registry.register(strategy, stage="candidate")
            queue.request_promotion(strategy, "Needs review.", {"max_drawdown": -0.19}, requester="test")

            decision = queue.reject("s2", reviewer="operator", reason="Too close to drawdown limit.")

            self.assertEqual(decision.status, "rejected")
            self.assertEqual(registry.list_by_stage("active"), [])
            self.assertEqual(registry.list_by_stage("rejected")[0]["id"], "s2")
            self.assertEqual(stub.sent_alerts[-1].event_type, "strategy_promotion_rejected")

    def test_review_cli_requires_reason_for_rejection(self) -> None:
        self.assertEqual(review_main(["--reject", "s1"]), 2)

    def test_review_cli_lists_pending_requests(self) -> None:
        class FakeQueue:
            def list_pending(self):
                return []

        with patch("scripts.review_strategy_promotions.ApprovalQueue", return_value=FakeQueue()):
            self.assertEqual(review_main(["--list"]), 0)

    def test_review_cli_handles_unknown_approval_id_without_traceback(self) -> None:
        class FakeQueue:
            def approve(self, strategy_id, reviewer="operator"):
                raise KeyError(strategy_id)

        with patch("scripts.review_strategy_promotions.ApprovalQueue", return_value=FakeQueue()):
            self.assertEqual(review_main(["--approve", "STRATEGY_ID"]), 1)


if __name__ == "__main__":
    unittest.main()
