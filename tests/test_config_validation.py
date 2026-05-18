import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.config.validation import normalize_trading_config, validate_trading_config
from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator, load_settings


class ConfigValidationTests(unittest.TestCase):
    def test_default_config_has_explicit_core_trading_assumptions(self) -> None:
        settings = load_settings("multi_agent_trading_lab/config/settings.yaml")
        report = validate_trading_config(settings)

        self.assertEqual(report.errors, [])
        self.assertEqual(report.active_universe, "full_mix")
        self.assertGreater(len(report.active_symbols), 0)
        self.assertEqual(set(report.allowed_symbols), set(report.active_symbols))
        self.assertIn("min_sharpe", settings["oos_validation"])
        self.assertIn("max_drawdown", settings["oos_validation"])
        self.assertIn("min_return", settings["oos_validation"])
        self.assertGreater(settings["oos_validation"]["split_ratio"], 0.0)
        self.assertGreaterEqual(settings["backtest"]["commission_per_trade"], 0.0)
        self.assertGreaterEqual(settings["backtest"]["slippage_pct"], 0.0)
        self.assertIn("portfolio", settings)
        self.assertFalse(settings["portfolio"]["enabled"])
        self.assertEqual(settings["portfolio"]["allocation_method"], "equal_weight")
        self.assertEqual(settings["optimizer"]["enabled"], False)
        self.assertEqual(settings["optimizer"]["method"], "deterministic_random")
        self.assertIsInstance(settings["optimizer"]["seed"], int)
        self.assertGreater(settings["optimizer"]["n_trials"], 0)
        self.assertNotIn("objective", settings["optimizer"])

    def test_allowed_symbol_drift_from_active_universe_is_visible(self) -> None:
        report = validate_trading_config(
            {
                "data": {"default_universe": "full_mix"},
                "universes": {"full_mix": ["AAPL", "MSFT", "NVDA"]},
                "risk": {"allowed_symbols": ["AAPL"]},
                "oos_validation": {"split_ratio": 0.7, "min_sharpe": 0.0, "max_drawdown": -0.2, "min_return": -0.05},
                "backtest": {"commission_per_trade": 0.0, "slippage_pct": 0.0005},
            }
        )

        self.assertEqual(report.errors, [])
        self.assertIn("risk.allowed_symbols does not match active universe", "\n".join(report.warnings))
        self.assertEqual(report.resolved["risk"]["allowed_symbols"], ["AAPL"])

    def test_missing_allowed_symbols_are_derived_from_active_universe(self) -> None:
        resolved, report = normalize_trading_config(
            {
                "data": {"default_universe": "full_mix"},
                "universes": {"full_mix": ["AAPL", "MSFT"]},
                "risk": {},
                "oos_validation": {"split_ratio": 0.7, "min_sharpe": 0.0, "max_drawdown": -0.2, "min_return": -0.05},
                "backtest": {"commission_per_trade": 0.0, "slippage_pct": 0.0005},
            }
        )

        self.assertEqual(report.errors, [])
        self.assertEqual(resolved["data"]["symbols"], ["AAPL", "MSFT"])
        self.assertEqual(resolved["risk"]["allowed_symbols"], ["AAPL", "MSFT"])
        self.assertIn("derived from active universe", "\n".join(report.warnings))

    def test_stale_top_level_keys_are_ignored_when_data_section_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = {
                "symbols": ["LEGACY"],
                "start_date": "1999-01-01",
                "end_date": "1999-12-31",
                "data": {"symbols": ["ACTIVE"], "start_date": "2023-01-01", "end_date": "2023-01-31"},
                "universes": {},
                "risk": {"allowed_symbols": ["ACTIVE"]},
                "broker": {"paper_state_path": str(root / "paper.json"), "live_enabled": False},
                "experiment_log_path": str(root / "experiments.jsonl"),
                "audit_log_path": str(root / "audit.jsonl"),
                "alert_log_path": str(root / "alerts.jsonl"),
                "approval_queue_path": str(root / "approvals.json"),
                "strategy_registry_path": str(root / "registry.json"),
                "system_state_path": str(root / "state.json"),
                "oos_validation": {"split_ratio": 0.7, "min_sharpe": 0.0, "max_drawdown": -0.2, "min_return": -0.05},
                "backtest": {"commission_per_trade": 0.0, "slippage_pct": 0.0005},
            }

            report = validate_trading_config(settings)
            orchestrator = TradingLabOrchestrator(settings)
            data_config = orchestrator._data_config()

            self.assertIn("Ignoring legacy top-level symbols", "\n".join(report.warnings))
            self.assertEqual(data_config["symbols"], ["ACTIVE"])
            self.assertEqual(data_config["start_date"], "2023-01-01")
            self.assertEqual(data_config["end_date"], "2023-01-31")

    def test_settings_risk_section_is_runtime_source_over_risk_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = TradingLabOrchestrator(
                {
                    "data": {"symbols": ["ACTIVE"], "start_date": "2023-01-01"},
                    "universes": {},
                    "risk": {"allowed_symbols": ["ACTIVE"]},
                    "broker": {"paper_state_path": str(root / "paper.json"), "live_enabled": False},
                    "experiment_log_path": str(root / "experiments.jsonl"),
                    "audit_log_path": str(root / "audit.jsonl"),
                    "alert_log_path": str(root / "alerts.jsonl"),
                    "approval_queue_path": str(root / "approvals.json"),
                    "strategy_registry_path": str(root / "registry.json"),
                    "system_state_path": str(root / "state.json"),
                }
            )

            self.assertEqual(orchestrator.risk_policy.allowed_symbols, {"ACTIVE"})

    def test_invalid_oos_and_cost_settings_are_errors(self) -> None:
        report = validate_trading_config(
            {
                "data": {"symbols": ["AAPL"]},
                "risk": {"allowed_symbols": ["AAPL"]},
                "oos_validation": {"split_ratio": 1.7, "min_sharpe": 0.0, "max_drawdown": 0.2, "min_return": -0.05},
                "backtest": {"commission_per_trade": -1.0, "slippage_pct": -0.01},
            }
        )

        self.assertIn("oos_validation.split_ratio must be between 0 and 1.", report.errors)
        self.assertIn("oos_validation.max_drawdown must be zero or negative.", report.errors)
        self.assertIn("backtest.commission_per_trade must be non-negative.", report.errors)
        self.assertIn("backtest.slippage_pct must be non-negative.", report.errors)

    def test_invalid_optimizer_settings_are_errors(self) -> None:
        report = validate_trading_config(
            {
                "data": {"symbols": ["AAPL"]},
                "risk": {"allowed_symbols": ["AAPL"]},
                "oos_validation": {"split_ratio": 0.7, "min_sharpe": 0.0, "max_drawdown": -0.2, "min_return": -0.05},
                "backtest": {"commission_per_trade": 0.0, "slippage_pct": 0.0005},
                "optimizer": {"enabled": "yes", "method": "optuna", "seed": "42", "n_trials": 0},
            }
        )

        self.assertIn("optimizer.enabled must be a boolean.", report.errors)
        self.assertIn("optimizer.method must be deterministic_random.", report.errors)
        self.assertIn("optimizer.seed must be an integer.", report.errors)
        self.assertIn("optimizer.n_trials must be a positive integer.", report.errors)

    def test_unknown_optimizer_settings_are_warnings(self) -> None:
        report = validate_trading_config(
            {
                "data": {"symbols": ["AAPL"]},
                "risk": {"allowed_symbols": ["AAPL"]},
                "oos_validation": {"split_ratio": 0.7, "min_sharpe": 0.0, "max_drawdown": -0.2, "min_return": -0.05},
                "backtest": {"commission_per_trade": 0.0, "slippage_pct": 0.0005},
                "optimizer": {
                    "enabled": False,
                    "method": "deterministic_random",
                    "seed": 42,
                    "n_trials": 25,
                    "objective": "risk_adjusted_return",
                },
            }
        )

        self.assertEqual(report.errors, [])
        self.assertIn("optimizer.objective is not used and should be removed.", report.warnings)


if __name__ == "__main__":
    unittest.main()
