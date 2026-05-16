import json
import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.operations.validation import (
    Phase2ValidationRunner,
    validate_runbooks,
)


class Phase2ValidationTests(unittest.TestCase):
    def test_runner_executes_all_scenarios_and_writes_operator_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = Phase2ValidationRunner(output_root=Path(tmpdir) / "validation")

            summary = runner.run("all")

            self.assertTrue(summary.passed)
            self.assertIn("happy_path_daily_cycle", summary.scenario_names)
            self.assertIn("runbook_integrity_check", summary.scenario_names)
            self.assertTrue((summary.output_dir / "validation_summary.json").exists())
            self.assertTrue((summary.output_dir / "validation_report.md").exists())
            payload = json.loads((summary.output_dir / "validation_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"])
            self.assertGreaterEqual(len(payload["scenarios"]), 9)

    def test_approval_required_candidate_does_not_activate_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Phase2ValidationRunner(output_root=Path(tmpdir)).run("approval_required_candidate")
            result = summary.scenarios[0]

            self.assertTrue(result.passed)
            self.assertTrue(result.checks["pending_request_created"])
            self.assertTrue(result.checks["strategy_not_active_without_approval"])
            self.assertTrue((result.artifacts["approval_queue"]).exists())

    def test_repeated_order_rejections_alert_once_due_to_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Phase2ValidationRunner(output_root=Path(tmpdir)).run("repeated_order_rejections")
            result = summary.scenarios[0]

            self.assertTrue(result.passed)
            self.assertEqual(result.checks["rejected_trade_audit_count"], 3)
            self.assertEqual(result.checks["repeated_rejection_alert_count"], 1)

    def test_kill_switch_persists_across_context_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = Phase2ValidationRunner(output_root=Path(tmpdir)).run("kill_switch_persistence")
            result = summary.scenarios[0]

            self.assertTrue(result.passed)
            self.assertTrue(result.checks["kill_switch_enabled_after_restart"])
            self.assertTrue(result.checks["future_cycle_noop_recorded"])

    def test_runbook_validator_fails_when_required_sections_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runbooks = Path(tmpdir) / "runbooks"
            runbooks.mkdir()
            (runbooks / "kill_switch_procedure.md").write_text("# Kill Switch Procedure\n\n## Purpose\n\nStop.", encoding="utf-8")

            result = validate_runbooks(runbooks)

            self.assertFalse(result.passed)
            self.assertIn("missing_files", result.details)


if __name__ == "__main__":
    unittest.main()
