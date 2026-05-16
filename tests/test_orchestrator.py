import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_paper_order_rejection_audit_includes_allowed_symbol_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_path = root / "audit.jsonl"
            orchestrator = TradingLabOrchestrator(
                {
                    "data": {"symbols": ["WES.AX"], "start_date": "2023-01-01", "end_date": "2023-03-31"},
                    "experiment_log_path": str(root / "experiments.jsonl"),
                    "audit_log_path": str(audit_path),
                    "strategy_registry_path": str(root / "registry.json"),
                    "system_state_path": str(root / "state.json"),
                    "broker": {"paper_state_path": str(root / "paper.json")},
                    "risk": {"allowed_symbols": ["AAPL"], "max_capital_per_trade": 10000.0},
                }
            )
            orchestrator.strategy_registry.register(
                {
                    "id": "audit_details_s5_l20",
                    "name": "moving_average_crossover",
                    "version": "0.1.0",
                    "short_window": 5,
                    "long_window": 20,
                    "strategy_params": {"short_window": 5, "long_window": 20},
                },
                stage="active",
            )
            bars = {"WES.AX": [{"date": "2023-03-31", "close": 100.0, "sma_5": 105.0, "sma_20": 95.0}]}
            orchestrator.data_agent.fetch_data = lambda *_args, **_kwargs: bars
            orchestrator.data_agent.build_features = lambda raw_data, _windows: raw_data

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.load_selected_strategy_config", return_value=None):
                result = orchestrator.run_paper_cycle()

            self.assertIn("WES.AX is not in the allowed universe", "\n".join(result.summaries))
            records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
            rejection = next(record for record in records if record["event_type"] == "rejected_trade")
            self.assertEqual(rejection["payload"]["details"]["symbol"], "WES.AX")
            self.assertEqual(rejection["payload"]["details"]["allowed_symbols"], ["AAPL"])

    def test_repeated_risk_failures_persist_and_activate_circuit_breaker(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audit_path = root / "audit.jsonl"
            state_path = root / "state.json"
            orchestrator = TradingLabOrchestrator(
                {
                    "data": {"symbols": ["WES.AX"], "start_date": "2023-01-01", "end_date": "2023-03-31"},
                    "experiment_log_path": str(root / "experiments.jsonl"),
                    "audit_log_path": str(audit_path),
                    "alert_log_path": str(root / "alerts.jsonl"),
                    "strategy_registry_path": str(root / "registry.json"),
                    "system_state_path": str(state_path),
                    "broker": {"paper_state_path": str(root / "paper.json")},
                    "risk": {
                        "allowed_symbols": ["WES.AX"],
                        "max_capital_per_trade": 1.0,
                        "repeated_failure_limit": 2,
                    },
                }
            )
            orchestrator.strategy_registry.register(
                {
                    "id": "circuit_s5_l20",
                    "name": "moving_average_crossover",
                    "version": "0.1.0",
                    "short_window": 5,
                    "long_window": 20,
                    "strategy_params": {"short_window": 5, "long_window": 20},
                },
                stage="active",
            )
            bars = {"WES.AX": [{"date": "2023-03-31", "close": 100.0, "sma_5": 105.0, "sma_20": 95.0}]}
            orchestrator.data_agent.fetch_data = lambda *_args, **_kwargs: bars
            orchestrator.data_agent.build_features = lambda raw_data, _windows: raw_data

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.load_selected_strategy_config", return_value=None):
                first = orchestrator.run_paper_cycle()
                second = orchestrator.run_paper_cycle()
                third = orchestrator.run_paper_cycle()

            self.assertIn("Order notional exceeds max capital per trade", "\n".join(first.summaries))
            self.assertIn("Order notional exceeds max capital per trade", "\n".join(second.summaries))
            self.assertIn("Circuit breaker triggered after repeated failures", "\n".join(third.summaries))
            self.assertEqual(orchestrator.system_state.read()["repeated_failures"], 3)
            records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(record["event_type"] == "circuit_breaker_activated" for record in records))

    def test_risk_policy_repeated_failures_are_loaded_from_system_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "state.json"
            orchestrator = TradingLabOrchestrator(
                {
                    "experiment_log_path": str(root / "experiments.jsonl"),
                    "audit_log_path": str(root / "audit.jsonl"),
                    "strategy_registry_path": str(root / "registry.json"),
                    "system_state_path": str(state_path),
                    "broker": {"paper_state_path": str(root / "paper.json")},
                    "risk": {"repeated_failure_limit": 2, "repeated_failures": 0},
                }
            )
            orchestrator.system_state.record_repeated_failure("risk_rejected")
            orchestrator.system_state.record_repeated_failure("broker_rejected")

            restarted = TradingLabOrchestrator(
                {
                    "experiment_log_path": str(root / "experiments.jsonl"),
                    "audit_log_path": str(root / "audit.jsonl"),
                    "strategy_registry_path": str(root / "registry.json"),
                    "system_state_path": str(state_path),
                    "broker": {"paper_state_path": str(root / "paper.json")},
                    "risk": {"repeated_failure_limit": 2, "repeated_failures": 0},
                }
            )

            self.assertEqual(restarted.risk_policy.repeated_failures, 2)


if __name__ == "__main__":
    unittest.main()
