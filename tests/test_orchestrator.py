import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator


class OrchestratorTests(unittest.TestCase):
    def test_paper_cycle_reports_configured_paper_drawdown_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = TradingLabOrchestrator(
                {
                    "experiment_log_path": str(root / "experiments.jsonl"),
                    "audit_log_path": str(root / "audit.jsonl"),
                    "strategy_registry_path": str(root / "registry.json"),
                    "system_state_path": str(root / "state.json"),
                    "broker": {"paper_state_path": str(root / "paper.json")},
                    "risk": {"kill_switch_enabled": True, "paper": {"max_drawdown": -0.25}},
                }
            )

            result = orchestrator.run_paper_cycle()

            self.assertIn("Paper mode risk limits: max_drawdown=-0.25", result.summaries[0])

    def test_paper_cycle_runs_in_safe_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = TradingLabOrchestrator(
                {
                    "symbols": ["AAPL"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-03-31",
                    "experiment_log_path": str(root / "experiments.jsonl"),
                    "audit_log_path": str(root / "audit.jsonl"),
                    "strategy_registry_path": str(root / "registry.json"),
                    "system_state_path": str(root / "state.json"),
                    "broker": {"paper_state_path": str(root / "paper.json")},
                    "risk": {"allowed_symbols": ["AAPL"], "max_capital_per_trade": 10000.0},
                }
            )

            result = orchestrator.run_paper_cycle()

            self.assertEqual(result.stage, "paper")
            self.assertTrue(result.summaries)
            self.assertNotIn("Live", result.summaries[0])


if __name__ == "__main__":
    unittest.main()
