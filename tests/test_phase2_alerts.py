import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from multi_agent_trading_lab.operations.alerting import (
    AlertManager,
    AlertSeverity,
    FileAlertSink,
    StubNotifierSink,
)


class AlertingTests(unittest.TestCase):
    def test_file_and_stub_sinks_receive_structured_alert_once_during_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "alerts.jsonl"
            stub = StubNotifierSink()
            manager = AlertManager(
                sinks=[FileAlertSink(path), stub],
                cooldown=timedelta(minutes=5),
                clock=lambda: datetime(2026, 5, 11, 1, 0, tzinfo=UTC),
            )

            first = manager.emit(
                event_type="risk_breach",
                severity=AlertSeverity.WARNING,
                message="Drawdown exceeded warning level.",
                strategy_id="s1",
                symbol="WES.AX",
                metadata={"drawdown": -0.21},
            )
            second = manager.emit(
                event_type="risk_breach",
                severity=AlertSeverity.WARNING,
                message="Drawdown exceeded warning level.",
                strategy_id="s1",
                symbol="WES.AX",
                metadata={"drawdown": -0.22},
            )

            self.assertIsNotNone(first)
            self.assertIsNone(second)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["event_type"], "risk_breach")
            self.assertEqual(records[0]["severity"], "WARNING")
            self.assertEqual(records[0]["strategy_id"], "s1")
            self.assertEqual(stub.sent_alerts[0].symbol, "WES.AX")

    def test_cooldown_allows_same_alert_after_window(self) -> None:
        ticks = [
            datetime(2026, 5, 11, 1, 0, tzinfo=UTC),
            datetime(2026, 5, 11, 1, 10, tzinfo=UTC),
        ]
        manager = AlertManager(sinks=[], cooldown=timedelta(minutes=5), clock=lambda: ticks.pop(0))

        first = manager.emit("paper_cycle_failure", AlertSeverity.CRITICAL, "Paper cycle failed.")
        second = manager.emit("paper_cycle_failure", AlertSeverity.CRITICAL, "Paper cycle failed.")

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)


if __name__ == "__main__":
    unittest.main()
