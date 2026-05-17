import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator


class OutOfSamplePromotionGateTests(unittest.TestCase):
    def test_good_in_sample_poor_oos_strategy_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = self._orchestrator(root)
            self._wire_single_candidate(orchestrator)
            walk_forward = self._walk_forward(oos={"total_return": -0.10, "max_drawdown": -0.24, "sharpe_ratio": -0.20})

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.run_walk_forward_validation", return_value=walk_forward):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(orchestrator.approval_queue.list_pending(), [])
            self.assertFalse(result.details["records"][0]["risk_decision"]["approved"])
            self.assertIn("OOS", result.details["records"][0]["risk_decision"]["reason"])
            audit_events = [json.loads(line) for line in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
            rejection = next(event for event in audit_events if event["event_type"] == "strategy_rejected")
            self.assertIn("walk_forward", rejection["payload"])
            self.assertEqual(rejection["payload"]["walk_forward"]["out_of_sample"]["metrics"]["total_return"], -0.10)

    def test_good_in_sample_and_good_oos_strategy_can_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = self._orchestrator(root)
            self._wire_single_candidate(orchestrator)
            walk_forward = self._walk_forward(oos={"total_return": 0.04, "max_drawdown": -0.08, "sharpe_ratio": 0.35})

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.run_walk_forward_validation", return_value=walk_forward):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            pending = orchestrator.approval_queue.list_pending()
            self.assertEqual(len(pending), 1)
            self.assertTrue(result.details["records"][0]["risk_decision"]["approved"])
            self.assertIn("walk_forward", pending[0].metrics)
            self.assertEqual(pending[0].metrics["walk_forward"]["out_of_sample"]["metrics"]["sharpe_ratio"], 0.35)

    def test_oos_rejection_reason_names_failed_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = self._orchestrator(root)
            self._wire_single_candidate(orchestrator)
            walk_forward = self._walk_forward(oos={"total_return": 0.04, "max_drawdown": -0.08, "sharpe_ratio": -0.01})

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.run_walk_forward_validation", return_value=walk_forward):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            decision = result.details["records"][0]["risk_decision"]
            self.assertFalse(decision["approved"])
            self.assertIn("OOS sharpe", decision["reason"])
            self.assertEqual(decision["details"]["failed_threshold"], "min_sharpe")

    def _orchestrator(self, root: Path) -> TradingLabOrchestrator:
        return TradingLabOrchestrator(
            {
                "data": {"symbols": ["TEST"], "start_date": "2023-01-01", "end_date": "2023-01-10"},
                "universes": {},
                "experiment_log_path": str(root / "experiments.jsonl"),
                "audit_log_path": str(root / "audit.jsonl"),
                "alert_log_path": str(root / "alerts.jsonl"),
                "approval_queue_path": str(root / "approvals.json"),
                "strategy_registry_path": str(root / "registry.json"),
                "system_state_path": str(root / "state.json"),
                "broker": {"paper_state_path": str(root / "paper.json")},
                "oos_validation": {"enabled": True, "split_ratio": 0.5, "min_sharpe": 0.10, "max_drawdown": -0.20, "min_return": -0.02},
            }
        )

    def _wire_single_candidate(self, orchestrator: TradingLabOrchestrator) -> None:
        variant = {
            "id": "oos_candidate_s5_l20",
            "name": "moving_average_crossover",
            "version": "0.1.0",
            "short_window": 5,
            "long_window": 20,
            "strategy_params": {"short_window": 5, "long_window": 20},
        }
        bars = [
            {"date": f"2023-01-{day:02d}", "close": 100.0 + day, "sma_5": 101.0 + day, "sma_20": 100.0 + day}
            for day in range(1, 11)
        ]
        orchestrator.strategy_agent.suggest_from_history = lambda *_args, **_kwargs: [variant]
        orchestrator.data_agent.fetch_data = lambda *_args, **_kwargs: {"TEST": bars}
        orchestrator.data_agent.build_features = lambda raw_data, _windows: raw_data
        orchestrator.backtest_agent.run_backtest = lambda *_args, **_kwargs: {
            "symbols": ["TEST"],
            "strategy": variant,
            "per_symbol_results": {},
            "metrics": {"total_return": 0.12, "max_drawdown": -0.05, "sharpe_ratio": 0.70, "sharpe": 0.70, "history_points": 10},
        }

    def _walk_forward(self, oos: dict[str, float]) -> dict:
        return {
            "strategy": {"id": "oos_candidate_s5_l20"},
            "in_sample": {
                "date_range": {"start_date": "2023-01-01", "end_date": "2023-01-05"},
                "symbols": ["TEST"],
                "metrics": {"total_return": 0.12, "max_drawdown": -0.05, "sharpe_ratio": 0.70, "sharpe": 0.70, "history_points": 5},
            },
            "out_of_sample": {
                "date_range": {"start_date": "2023-01-06", "end_date": "2023-01-10"},
                "symbols": ["TEST"],
                "metrics": {**oos, "sharpe": oos["sharpe_ratio"], "history_points": 5},
            },
        }


if __name__ == "__main__":
    unittest.main()
