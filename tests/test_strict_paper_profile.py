import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.config.validation import validate_trading_config
from multi_agent_trading_lab.operations.strict_profile_validation import run_approval_profile_validation, run_strict_profile_validation
from multi_agent_trading_lab.orchestrator.orchestrator import load_settings


STRICT_PROFILE = "multi_agent_trading_lab/config/settings.strict_paper.yaml"
APPROVAL_PROFILE = "multi_agent_trading_lab/config/settings.approval_paper.yaml"


class StrictPaperProfileTests(unittest.TestCase):
    def test_strict_profile_loads_with_safety_gates_enabled(self) -> None:
        settings = load_settings(STRICT_PROFILE)
        report = validate_trading_config(settings)

        self.assertEqual(report.errors, [])
        self.assertEqual(settings["operating_mode"], "paper")
        self.assertFalse(settings["broker"]["live_enabled"])
        self.assertEqual(settings["execution"]["mode"], "paper")
        self.assertTrue(settings["oos_validation"]["enabled"])
        self.assertGreater(settings["backtest"]["slippage_pct"], 0.0)
        self.assertTrue(settings["data_quality"]["block_on_missing_columns"])
        self.assertTrue(settings["data_quality"]["block_on_duplicate_dates"])
        self.assertTrue(settings["data_quality"]["block_on_invalid_prices"])
        self.assertTrue(settings["portfolio"]["enabled"])
        self.assertGreaterEqual(settings["portfolio"]["min_valid_symbols"], 2)
        self.assertTrue(settings["regime"]["enabled"])
        self.assertTrue(settings["strategy_health"]["enabled"])
        self.assertTrue(settings["optimizer"]["enabled"])
        self.assertEqual(settings["optimizer"]["method"], "deterministic_random")

    def test_strict_profile_validation_runs_research_paper_and_report_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_strict_profile_validation(STRICT_PROFILE, output_root=Path(tmpdir))

            self.assertTrue(result.passed, result.checks)
            self.assertTrue(result.checks["config_validation_passed"])
            self.assertTrue(result.checks["all_optional_gates_enabled"])
            self.assertTrue(result.checks["accepted_has_required_gate_metadata"])
            self.assertTrue(result.checks["rejection_reason_visible"])
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.audit_log_path.exists())
            self.assertIsNotNone(result.accepted_example)
            self.assertIsNotNone(result.rejected_example)

    def test_approval_profile_loads_with_stricter_thresholds_than_strict_profile(self) -> None:
        strict = load_settings(STRICT_PROFILE)
        approval = load_settings(APPROVAL_PROFILE)
        report = validate_trading_config(approval)

        self.assertEqual(report.errors, [])
        self.assertEqual(approval["operating_mode"], "paper")
        self.assertEqual(approval["execution"]["mode"], "paper")
        self.assertFalse(approval["broker"]["live_enabled"])
        self.assertTrue(approval["oos_validation"]["enabled"])
        self.assertTrue(approval["portfolio"]["enabled"])
        self.assertTrue(approval["regime"]["enabled"])
        self.assertTrue(approval["optimizer"]["enabled"])
        self.assertGreaterEqual(approval["backtest"]["slippage_pct"], strict["backtest"]["slippage_pct"])
        self.assertGreaterEqual(approval["oos_validation"]["min_return"], strict["oos_validation"]["min_return"])
        self.assertGreaterEqual(approval["oos_validation"]["min_sharpe"], strict["oos_validation"]["min_sharpe"])
        self.assertGreaterEqual(approval["oos_validation"]["max_drawdown"], strict["oos_validation"]["max_drawdown"])
        self.assertGreaterEqual(approval["portfolio"]["min_total_return"], strict["portfolio"]["min_total_return"])
        self.assertGreaterEqual(approval["portfolio"]["min_sharpe"], strict["portfolio"]["min_sharpe"])
        self.assertGreaterEqual(approval["portfolio"]["max_drawdown"], strict["portfolio"]["max_drawdown"])

    def test_approval_profile_validation_can_pass_with_all_candidates_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_approval_profile_validation(APPROVAL_PROFILE, output_root=Path(tmpdir))

            self.assertTrue(result.passed, result.checks)
            self.assertEqual(result.accepted_count, 0)
            self.assertGreater(result.rejected_count, 0)
            self.assertTrue(result.checks["config_validation_passed"])
            self.assertTrue(result.checks["all_optional_gates_enabled"])
            self.assertTrue(result.checks["optimizer_metadata_persisted"])
            self.assertTrue(result.checks["rejections_have_reasons"])
            self.assertFalse(result.checks["approval_request_created"])
            self.assertTrue(result.report_path.exists())
            self.assertTrue(result.audit_log_path.exists())


if __name__ == "__main__":
    unittest.main()
