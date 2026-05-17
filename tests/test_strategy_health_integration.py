import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from multi_agent_trading_lab.operations.paper_reporting import DailyPaperReportService
from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator


class StrategyHealthIntegrationTests(unittest.TestCase):
    def test_daily_report_includes_strategy_health_and_audits_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paper_state = root / "paper.json"
            audit_path = root / "audit.jsonl"
            reports_dir = root / "reports"
            paper_state.write_text(
                json.dumps(
                    {
                        "cash": 100_000.0,
                        "equity": 99_900.0,
                        "positions": {},
                        "orders": {
                            f"o{i}": {"status": "filled", "symbol": "AAPL", "source_strategy": "s1", "realized_pnl": -20.0}
                            for i in range(5)
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit_path.write_text(
                json.dumps(
                    {
                        "timestamp": datetime(2026, 5, 11, 1, 0, tzinfo=UTC).isoformat(),
                        "event_type": "strategy_promotion_requested",
                        "payload": {"strategy_id": "s1", "metrics": {"oos_total_return": 0.05}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = DailyPaperReportService(
                paper_state_path=paper_state,
                audit_log_path=audit_path,
                alerts_path=root / "alerts.jsonl",
                reports_dir=reports_dir,
                clock=lambda: datetime(2026, 5, 11, 9, 0, tzinfo=UTC),
                strategy_health_config={"enabled": True, "min_signals_before_quarantine": 5},
            ).generate()

            self.assertEqual(report.strategy_health["decisions"]["s1"]["status"], "quarantined")
            self.assertIn("Strategy health", report.text_report)
            audit_events = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(event["event_type"] == "strategy_health_decision" for event in audit_events))

    def test_quarantined_strategy_is_skipped_in_paper_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            strategy = {
                "id": "s1",
                "name": "moving_average_crossover",
                "version": "0.1.0",
                "short_window": 5,
                "long_window": 20,
                "strategy_params": {"short_window": 5, "long_window": 20},
            }
            settings = {
                "data": {"symbols": ["AAPL"], "start_date": "2023-01-01", "end_date": "2023-01-10"},
                "universes": {},
                "experiment_log_path": str(root / "experiments.jsonl"),
                "audit_log_path": str(root / "audit.jsonl"),
                "alert_log_path": str(root / "alerts.jsonl"),
                "approval_queue_path": str(root / "approvals.json"),
                "strategy_registry_path": str(root / "registry.json"),
                "system_state_path": str(root / "state.json"),
                "broker": {"paper_state_path": str(root / "paper.json"), "live_enabled": False},
                "strategy_health": {"enabled": True},
            }
            orchestrator = TradingLabOrchestrator(settings)
            orchestrator.strategy_registry.register(strategy, stage="active", reason="test")
            orchestrator.system_state.write({"strategy_health": {"s1": {"strategy_id": "s1", "status": "quarantined", "reasons": ["paper attribution"]}}})

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.load_selected_strategy_config", return_value=None):
                result = orchestrator.run_paper_cycle()

            self.assertEqual(result.details["orders"], [])
            self.assertIn("Strategy skipped: quarantined due to paper attribution.", "\n".join(result.summaries))
            audit_events = [json.loads(line) for line in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(event["event_type"] == "strategy_health_skip" for event in audit_events))


if __name__ == "__main__":
    unittest.main()
