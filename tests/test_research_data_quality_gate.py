import json
import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.data.data_quality import validate_market_data
from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator


class ResearchDataQualityGateTests(unittest.TestCase):
    def test_backtest_and_research_records_expose_data_quality_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = self._orchestrator(root)
            bars = self._bars()
            bars[1]["volume"] = 0.0
            self._wire_candidate_and_data(orchestrator, bars)

            result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            record = result.details["records"][0]
            self.assertTrue(record["data_quality"]["passed"])
            self.assertEqual(record["data_quality"]["issues"][0]["code"], "zero_volume")
            self.assertEqual(record["data_quality"]["issues"][0]["severity"], "warning")
            self.assertEqual(record["metrics"]["data_quality"]["issues"][0]["code"], "zero_volume")
            self.assertIn("data_quality=passed with 1 warning(s)", result.summaries[0])

    def test_severe_data_quality_failure_blocks_promotion_with_visible_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = self._orchestrator(root)
            bars = self._bars()
            bars[1]["close"] = -1.0
            self._wire_candidate_and_data(orchestrator, bars)

            result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(orchestrator.approval_queue.list_pending(), [])
            record = result.details["records"][0]
            self.assertFalse(record["risk_decision"]["approved"])
            self.assertIn("Data quality gate failed", record["risk_decision"]["reason"])
            self.assertEqual(record["risk_decision"]["details"]["failed_data_quality_codes"], ["invalid_prices"])
            audit_events = [json.loads(line) for line in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
            rejection = next(event for event in audit_events if event["event_type"] == "strategy_rejected")
            self.assertEqual(rejection["payload"]["data_quality"]["issues"][0]["code"], "invalid_prices")

    def test_warning_only_data_quality_issues_do_not_block_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = self._orchestrator(root)
            bars = self._bars()
            bars[2]["volume"] = 0.0
            self._wire_candidate_and_data(orchestrator, bars)

            result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(len(orchestrator.approval_queue.list_pending()), 1)
            self.assertTrue(result.details["records"][0]["risk_decision"]["approved"])
            pending_metrics = orchestrator.approval_queue.list_pending()[0].metrics
            self.assertEqual(pending_metrics["data_quality"]["issues"][0]["severity"], "warning")

    def test_quality_report_covers_required_issue_codes_for_research_gate(self) -> None:
        cases = {
            "missing_columns": [{"date": "2026-01-01", "close": 100.0}],
            "duplicate_dates": [self._bar("2026-01-01"), self._bar("2026-01-01")],
            "stale_data": [self._bar("2026-01-01"), self._bar("2026-01-02")],
            "invalid_prices": [{**self._bar("2026-01-01"), "open": 0.0}],
            "large_gap": [self._bar("2026-01-01"), self._bar("2026-01-10")],
            "zero_volume": [{**self._bar("2026-01-01"), "volume": 0.0}],
        }

        for code, bars in cases.items():
            with self.subTest(code=code):
                report = validate_market_data(
                    {"TEST": bars},
                    as_of_date="2026-01-20",
                    max_staleness_days=3,
                    max_gap_days=3,
                )
                self.assertIn(code, [issue.code for issue in report.issues])

    def _orchestrator(self, root: Path) -> TradingLabOrchestrator:
        return TradingLabOrchestrator(
            {
                "data": {
                    "provider": "synthetic",
                    "symbols": ["TEST"],
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-06",
                    "as_of_date": "2026-01-06",
                },
                "universes": {},
                "data_quality": {
                    "block_on_missing_columns": True,
                    "block_on_duplicate_dates": True,
                    "block_on_invalid_prices": True,
                    "block_on_stale_data": False,
                    "block_on_large_gaps": False,
                    "block_on_zero_volume": False,
                    "expect_volume": True,
                    "max_gap_days": 5,
                    "max_staleness_days": 5,
                },
                "experiment_log_path": str(root / "experiments.jsonl"),
                "audit_log_path": str(root / "audit.jsonl"),
                "alert_log_path": str(root / "alerts.jsonl"),
                "approval_queue_path": str(root / "approvals.json"),
                "strategy_registry_path": str(root / "registry.json"),
                "system_state_path": str(root / "state.json"),
                "broker": {"paper_state_path": str(root / "paper.json"), "live_enabled": False},
                "risk": {"min_backtest_return": -0.05, "research": {"max_drawdown": -0.20}, "paper": {"max_drawdown": -0.20}},
                "oos_validation": {"enabled": False, "split_ratio": 0.7, "min_sharpe": 0.0, "max_drawdown": -0.2, "min_return": -0.05},
            }
        )

    def _wire_candidate_and_data(self, orchestrator: TradingLabOrchestrator, bars: list[dict]) -> None:
        variant = {
            "id": "dq_candidate_s5_l20",
            "name": "moving_average_crossover",
            "version": "0.1.0",
            "short_window": 5,
            "long_window": 20,
            "strategy_params": {"short_window": 5, "long_window": 20},
        }
        report = validate_market_data(
            {"TEST": bars},
            as_of_date="2026-01-06",
            max_staleness_days=5,
            max_gap_days=5,
        )
        orchestrator.strategy_agent.suggest_from_history = lambda *_args, **_kwargs: [variant]
        orchestrator.data_agent.fetch_data = lambda *_args, **_kwargs: {"TEST": bars}
        orchestrator.data_agent.build_features = lambda raw_data, _windows: raw_data
        orchestrator.data_agent.last_data_quality = report
        orchestrator.data_agent.last_load_source = "test_fixture"

    def _bars(self) -> list[dict[str, float | str]]:
        return [
            {"date": f"2026-01-{day:02d}", "open": 100.0 + day, "high": 101.0 + day, "low": 99.0 + day, "close": 100.0 + day, "volume": 1000.0}
            for day in range(1, 7)
        ]

    def _bar(self, date: str) -> dict[str, float | str]:
        return {"date": date, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}


if __name__ == "__main__":
    unittest.main()
