import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.operations.alerting import AlertManager, StubNotifierSink
from multi_agent_trading_lab.operations.kill_switch import activate_kill_switch
from multi_agent_trading_lab.operations.paper_reporting import DailyPaperReportService
from multi_agent_trading_lab.state.system_state import SystemStateStore


class DailyPaperReportTests(unittest.TestCase):
    def test_report_summarizes_orders_positions_alerts_and_writes_json_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paper_state = root / "paper.json"
            audit_path = root / "audit.jsonl"
            alerts_path = root / "alerts.jsonl"
            reports_dir = root / "reports"
            paper_state.write_text(
                json.dumps(
                    {
                        "cash": 99_000.0,
                        "equity": 100_250.0,
                        "positions": {"WES.AX": {"quantity": 10, "average_price": 100.0, "market_price": 125.0}},
                        "orders": {
                            "o1": {"status": "filled", "symbol": "WES.AX", "source_strategy": "s1"},
                            "o2": {"status": "rejected", "symbol": "WES.AX", "source_strategy": "s1"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            AuditLog(audit_path).record("risk_rejected", {"strategy_id": "s1", "reason": "max position"})
            alerts_path.write_text(
                json.dumps(
                    {
                        "timestamp": datetime(2026, 5, 11, 2, 0, tzinfo=UTC).isoformat(),
                        "event_type": "risk_breach",
                        "severity": "WARNING",
                        "strategy_id": "s1",
                        "symbol": "WES.AX",
                        "message": "Risk breach.",
                        "metadata": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            service = DailyPaperReportService(
                paper_state_path=paper_state,
                audit_log_path=audit_path,
                alerts_path=alerts_path,
                reports_dir=reports_dir,
                clock=lambda: datetime(2026, 5, 11, 9, 0, tzinfo=UTC),
            )
            report = service.generate()

            self.assertEqual(report.date, "2026-05-11")
            self.assertEqual(report.ending_equity, 100_250.0)
            self.assertEqual(report.order_counts_by_status["filled"], 1)
            self.assertEqual(report.order_counts_by_status["rejected"], 1)
            self.assertEqual(report.unrealized_pnl, 250.0)
            self.assertIn("Daily paper trading report", report.text_report)
            self.assertTrue((reports_dir / "paper_report_2026-05-11.json").exists())
            self.assertTrue((reports_dir / "paper_report_2026-05-11.md").exists())

    def test_report_notes_previous_equity_change_when_previous_report_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reports_dir = root / "reports"
            reports_dir.mkdir()
            (reports_dir / "paper_report_2026-05-10.json").write_text(
                json.dumps({"ending_equity": 99_500.0}),
                encoding="utf-8",
            )
            paper_state = root / "paper.json"
            paper_state.write_text(json.dumps({"cash": 100_000.0, "equity": 100_100.0, "positions": {}, "orders": {}}), encoding="utf-8")

            report = DailyPaperReportService(
                paper_state_path=paper_state,
                audit_log_path=root / "audit.jsonl",
                alerts_path=root / "alerts.jsonl",
                reports_dir=reports_dir,
                clock=lambda: datetime(2026, 5, 11, 9, 0, tzinfo=UTC),
            ).generate()

            self.assertIn("Equity changed by 600.00", report.notable_changes[0])


class KillSwitchTests(unittest.TestCase):
    def test_circuit_breaker_failure_count_persists_and_resets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            state = SystemStateStore(Path(tmpdir) / "system_state.json")

            first = state.record_repeated_failure("risk_rejected")
            second = state.record_repeated_failure("broker_rejected")
            reset = state.reset_repeated_failures()

            self.assertEqual(first["repeated_failures"], 1)
            self.assertEqual(second["repeated_failures"], 2)
            self.assertEqual(second["last_repeated_failure_reason"], "broker_rejected")
            self.assertEqual(reset["repeated_failures"], 0)
            self.assertIsNone(reset.get("last_repeated_failure_reason"))

    def test_activate_kill_switch_sets_state_audits_and_emits_critical_alert(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit = AuditLog(root / "audit.jsonl")
            state = SystemStateStore(root / "system_state.json")
            stub = StubNotifierSink()

            result = activate_kill_switch(
                reason="manual stop after repeated order rejects",
                state_store=state,
                audit_log=audit,
                alert_manager=AlertManager([stub]),
                operator="tester",
            )

            self.assertTrue(result["kill_switch_enabled"])
            self.assertEqual(state.read()["kill_switch_reason"], "manual stop after repeated order rejects")
            self.assertEqual(stub.sent_alerts[0].severity.value, "CRITICAL")
            event = json.loads((root / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(event["event_type"], "kill_switch_activated")


if __name__ == "__main__":
    unittest.main()
