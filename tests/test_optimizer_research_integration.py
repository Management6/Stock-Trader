import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from multi_agent_trading_lab.agents.strategy_agent import StrategyAgent
from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator


class OptimizerResearchIntegrationTests(unittest.TestCase):
    def test_heuristic_search_is_used_when_optimizer_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = StrategyAgent(
                experiment_logger=None,
                search_config={"optimizer": {"enabled": False}},
                config_path=Path(tmpdir) / "variants.json",
            )

            with patch("multi_agent_trading_lab.agents.strategy_agent.propose_candidates") as optimizer:
                suggestions = agent.suggest_from_history("moving_average_crossover", 2)

            optimizer.assert_not_called()
            self.assertEqual(len(suggestions), 2)

    def test_optimizer_candidates_still_pass_through_promotion_gates_and_record_trials(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = TradingLabOrchestrator(
                {
                    "data": {"symbols": ["TEST"], "start_date": "2023-01-01", "end_date": "2023-01-10"},
                    "universes": {},
                    "experiment_log_path": str(root / "experiments.jsonl"),
                    "audit_log_path": str(root / "audit.jsonl"),
                    "alert_log_path": str(root / "alerts.jsonl"),
                    "approval_queue_path": str(root / "approvals.json"),
                    "strategy_registry_path": str(root / "registry.json"),
                    "system_state_path": str(root / "state.json"),
                    "broker": {"paper_state_path": str(root / "paper.json"), "live_enabled": False},
                    "oos_validation": {"enabled": False},
                    "portfolio": {"enabled": False},
                    "regime": {"enabled": False},
                    "optimizer": {"enabled": True, "method": "deterministic_random", "seed": 5, "n_trials": 2},
                }
            )
            bars = [
                {"date": f"2023-01-{day:02d}", "open": 100.0 + day, "high": 101.0 + day, "low": 99.0 + day, "close": 100.0 + day, "volume": 1000.0, "sma_5": 101.0 + day, "sma_20": 100.0 + day}
                for day in range(1, 11)
            ]
            orchestrator.data_agent.fetch_data = lambda *_args, **_kwargs: {"TEST": bars}
            orchestrator.data_agent.build_features = lambda raw_data, _windows: raw_data

            result = orchestrator.run_research_cycle(n_variants=2, strategy_families=["ma"])

            self.assertEqual(len(result.details["records"]), 2)
            self.assertIn("optimizer_trial", result.details["records"][0])
            self.assertIn(result.details["records"][0]["optimizer_trial"]["gate_outcome"], {"accepted", "rejected"})
            persisted = [json.loads(line) for line in (root / "experiments.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all("optimizer_trial" in record["metrics"] for record in persisted))
            self.assertTrue(all(record["metrics"]["optimizer_trial"]["gate_outcome"] in {"accepted", "rejected"} for record in persisted))
            pending = orchestrator.approval_queue.list_pending()
            self.assertTrue(all("optimizer_trial" in request.metrics for request in pending))

    def test_failed_optimizer_trials_are_recorded_without_crashing_research(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = TradingLabOrchestrator(
                {
                    "data": {"symbols": ["TEST"], "start_date": "2023-01-01", "end_date": "2023-01-10"},
                    "universes": {},
                    "experiment_log_path": str(root / "experiments.jsonl"),
                    "audit_log_path": str(root / "audit.jsonl"),
                    "alert_log_path": str(root / "alerts.jsonl"),
                    "approval_queue_path": str(root / "approvals.json"),
                    "strategy_registry_path": str(root / "registry.json"),
                    "system_state_path": str(root / "state.json"),
                    "broker": {"paper_state_path": str(root / "paper.json"), "live_enabled": False},
                    "oos_validation": {"enabled": False},
                    "portfolio": {"enabled": False},
                    "regime": {"enabled": False},
                    "optimizer": {"enabled": True, "method": "deterministic_random", "seed": 9, "n_trials": 2},
                }
            )
            bars = [
                {"date": f"2023-01-{day:02d}", "open": 100.0 + day, "high": 101.0 + day, "low": 99.0 + day, "close": 100.0 + day, "volume": 1000.0, "sma_5": 101.0 + day, "sma_20": 100.0 + day}
                for day in range(1, 11)
            ]
            orchestrator.data_agent.fetch_data = lambda *_args, **_kwargs: {"TEST": bars}
            orchestrator.data_agent.build_features = lambda raw_data, _windows: raw_data
            original = orchestrator.backtest_agent.run_backtest
            calls = {"count": 0}

            def flaky_backtest(*args, **kwargs):
                calls["count"] += 1
                if calls["count"] == 1:
                    raise ValueError("synthetic optimizer trial failure")
                return original(*args, **kwargs)

            orchestrator.backtest_agent.run_backtest = flaky_backtest
            result = orchestrator.run_research_cycle(n_variants=2, strategy_families=["ma"])

            self.assertEqual(len(result.details["records"]), 2)
            self.assertEqual(result.details["records"][0]["optimizer_trial"]["gate_outcome"], "failed")
            self.assertIn("synthetic optimizer trial failure", result.details["records"][0]["optimizer_trial"]["rejection_reason"])
            persisted = [json.loads(line) for line in (root / "experiments.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(persisted[0]["metrics"]["optimizer_trial"]["gate_outcome"], "failed")


if __name__ == "__main__":
    unittest.main()
