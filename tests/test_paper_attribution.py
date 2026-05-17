import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from multi_agent_trading_lab.operations.paper_attribution import build_paper_attribution
from multi_agent_trading_lab.operations.paper_reporting import DailyPaperReportService


class PaperAttributionTests(unittest.TestCase):
    def test_attribution_groups_pnl_orders_rejections_and_expectations(self) -> None:
        report = build_paper_attribution(
            orders=[
                self._order("s1", "AAPL", "filled", realized_pnl=12.5),
                self._order("s1", "MSFT", "filled", realized_pnl=-2.5),
                self._order("s2", "AAPL", "rejected", reason="risk policy"),
                self._order("s2", "NVDA", "skipped", reason="data quality"),
            ],
            audit_records=[
                {"event_type": "rejected_trade", "payload": {"reason": "risk policy", "order": {"source_strategy": "s2", "symbol": "AAPL"}}},
                {"event_type": "rejected_trade", "payload": {"reason": "data quality", "order": {"source_strategy": "s2", "symbol": "NVDA"}}},
            ],
            expected_metrics_by_strategy={"s1": {"total_return": 0.08, "oos_total_return": 0.04}, "s2": {"total_return": 0.03}},
            starting_equity=100_000.0,
            ending_equity=100_010.0,
        )

        self.assertEqual(report["summary"]["total_pnl"], 10.0)
        self.assertEqual(report["summary"]["total_return"], 0.0001)
        self.assertEqual(report["summary"]["signals"], 4)
        self.assertEqual(report["summary"]["accepted_orders"], 2)
        self.assertEqual(report["summary"]["rejected_or_skipped_orders"], 2)
        self.assertEqual(report["by_strategy"]["s1"]["pnl"], 10.0)
        self.assertEqual(report["by_symbol"]["AAPL"]["pnl"], 12.5)
        self.assertEqual(report["rejection_breakdown"]["risk policy"], 2)
        self.assertEqual(report["rejection_breakdown"]["data quality"], 2)
        self.assertEqual(report["expectation_vs_actual"]["s1"]["expected_oos_return"], 0.04)
        self.assertEqual(report["expectation_vs_actual"]["s1"]["actual_pnl"], 10.0)

    def test_no_paper_trades_report_is_explicit(self) -> None:
        report = build_paper_attribution(orders=[], audit_records=[], expected_metrics_by_strategy={})

        self.assertEqual(report["summary"]["signals"], 0)
        self.assertEqual(report["summary"]["total_pnl"], 0.0)
        self.assertIn("No paper trades recorded yet.", report["warnings"])
        self.assertEqual(report["by_strategy"], {})
        self.assertEqual(report["by_symbol"], {})

    def test_mismatch_detection_flags_negative_paper_pnl_against_positive_oos(self) -> None:
        report = build_paper_attribution(
            orders=[self._order("s1", "AAPL", "filled", realized_pnl=-25.0)],
            audit_records=[],
            expected_metrics_by_strategy={"s1": {"oos_total_return": 0.05}},
        )

        self.assertIn("s1 has positive expected OOS return but negative paper PnL.", report["warnings"])
        self.assertIn("expectation_mismatch", report["diagnostic_flags"])

    def test_high_rejection_rate_and_repeated_reasons_are_flagged(self) -> None:
        report = build_paper_attribution(
            orders=[
                self._order("s1", "AAPL", "rejected", reason="risk policy"),
                self._order("s1", "MSFT", "rejected", reason="risk policy"),
                self._order("s1", "NVDA", "rejected", reason="data quality"),
                self._order("s1", "AMD", "filled", realized_pnl=1.0),
            ],
            audit_records=[],
            high_rejection_rate_threshold=0.50,
        )

        self.assertIn("High rejection rate: 75.00%.", report["warnings"])
        self.assertIn("Repeated rejection reason: risk policy.", report["warnings"])
        self.assertIn("high_rejection_rate", report["diagnostic_flags"])
        self.assertIn("repeated_risk_policy_rejections", report["diagnostic_flags"])

    def test_daily_report_includes_paper_attribution(self) -> None:
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
                        "equity": 100_010.0,
                        "positions": {},
                        "orders": {
                            "o1": {"status": "filled", "symbol": "AAPL", "source_strategy": "s1", "realized_pnl": -10.0},
                            "o2": {"status": "rejected", "symbol": "MSFT", "source_strategy": "s1", "reason": "risk policy"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit_path.write_text(
                json.dumps(
                    {
                        "timestamp": datetime(2026, 5, 11, 2, 0, tzinfo=UTC).isoformat(),
                        "event_type": "strategy_promotion_requested",
                        "payload": {"strategy_id": "s1", "metrics": {"oos_total_return": 0.04}},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = DailyPaperReportService(
                paper_state_path=paper_state,
                audit_log_path=audit_path,
                alerts_path=alerts_path,
                reports_dir=reports_dir,
                clock=lambda: datetime(2026, 5, 11, 9, 0, tzinfo=UTC),
            ).generate()

            self.assertEqual(report.paper_attribution["by_strategy"]["s1"]["pnl"], -10.0)
            self.assertIn("expectation_mismatch", report.paper_attribution["diagnostic_flags"])
            self.assertIn("Paper attribution", report.text_report)
            payload = json.loads((reports_dir / "paper_report_2026-05-11.json").read_text(encoding="utf-8"))
            self.assertIn("paper_attribution", payload)

    def _order(self, strategy: str, symbol: str, status: str, realized_pnl: float = 0.0, reason: str | None = None) -> dict:
        order = {"source_strategy": strategy, "symbol": symbol, "status": status, "realized_pnl": realized_pnl}
        if reason:
            order["reason"] = reason
        return order


if __name__ == "__main__":
    unittest.main()
