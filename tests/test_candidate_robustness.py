import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.data.data_sources import generate_synthetic_data
from multi_agent_trading_lab.data.feature_engineering import add_strategy_features
from multi_agent_trading_lab.operations.approval_search import run_approval_search
from multi_agent_trading_lab.operations.candidate_robustness import evaluate_candidate_robustness, run_candidate_robustness
from multi_agent_trading_lab.orchestrator.orchestrator import load_settings


APPROVAL_PROFILE = "multi_agent_trading_lab/config/settings.approval_paper.yaml"


class CandidateRobustnessTests(unittest.TestCase):
    def test_robustness_checks_are_deterministic(self) -> None:
        settings = self._settings(min_sharpe=-10.0, min_return=-1.0)
        records = [self._record("robust_ma", 6, 30, approved=True)]
        data = self._featured_data()

        first = evaluate_candidate_robustness(records, settings, data)
        second = evaluate_candidate_robustness(records, settings, data)

        self.assertEqual(first, second)

    def test_candidate_that_passes_all_stress_checks_is_robust(self) -> None:
        settings = self._settings(min_sharpe=-10.0, min_return=-1.0)
        result = evaluate_candidate_robustness([self._record("robust_ma", 6, 30, approved=True)], settings, self._featured_data())

        self.assertEqual(result["status_counts"], {"robust": 1})
        candidate = result["candidates"][0]
        self.assertEqual(candidate["status"], "robust")
        self.assertEqual(candidate["checks_failed"], 0)
        self.assertTrue(any(check["name"].startswith("parameter_perturbation") for check in candidate["checks"]))

    def test_candidate_that_fails_stress_checks_is_fragile_with_failures_recorded(self) -> None:
        settings = self._settings(min_sharpe=50.0, min_return=0.50)
        result = evaluate_candidate_robustness([self._record("fragile_ma", 6, 30, approved=True)], settings, self._featured_data())

        candidate = result["candidates"][0]
        self.assertEqual(candidate["status"], "fragile")
        self.assertGreater(candidate["checks_failed"], 0)
        self.assertIsNotNone(candidate["top_failure_reason"])
        self.assertTrue(any(check["name"].startswith("slippage_") and not check["passed"] for check in candidate["checks"]))
        self.assertTrue(any(check["name"].startswith("alternate_split_") and not check["passed"] for check in candidate["checks"]))
        self.assertIsNotNone(candidate["worst_cost_stress_result"])

    def test_robustness_command_preserves_approval_profile_thresholds(self) -> None:
        before = load_settings(APPROVAL_PROFILE)
        with tempfile.TemporaryDirectory() as search_dir, tempfile.TemporaryDirectory() as robustness_dir:
            search = run_approval_search(APPROVAL_PROFILE, trials=2, seed=5, output_root=Path(search_dir), families=["ma"])
            summary = run_candidate_robustness(APPROVAL_PROFILE, search.output_dir / "approval_search_summary.json", output_root=Path(robustness_dir))
            self.assertTrue(summary.report_path.exists())
        after = load_settings(APPROVAL_PROFILE)

        self.assertEqual(after["oos_validation"], before["oos_validation"])
        self.assertEqual(after["portfolio"], before["portfolio"])
        self.assertEqual(summary.evaluated_count, summary.status_counts.get("robust", 0) + summary.status_counts.get("fragile", 0))

    def _settings(self, min_sharpe: float, min_return: float) -> dict:
        settings = load_settings(APPROVAL_PROFILE)
        settings["oos_validation"] = {**settings["oos_validation"], "min_sharpe": min_sharpe, "min_return": min_return}
        settings["portfolio"] = {**settings["portfolio"], "min_sharpe": min_sharpe, "min_total_return": min_return}
        return settings

    def _featured_data(self) -> dict:
        return add_strategy_features(generate_synthetic_data(["ALPHA", "BETA", "SPY"], "2023-01-03", "2023-08-31"), [5, 6, 7, 25, 30, 35])

    def _record(self, strategy_id: str, short_window: int, long_window: int, approved: bool) -> dict:
        strategy = {
            "id": strategy_id,
            "name": "moving_average_crossover",
            "version": "0.1.0",
            "short_window": short_window,
            "long_window": long_window,
            "strategy_params": {"short_window": short_window, "long_window": long_window},
        }
        return {
            "strategy_name": "moving_average_crossover",
            "config": {"strategy": strategy},
            "metrics": {
                "total_return": 0.01,
                "max_drawdown": -0.01,
                "sharpe_ratio": 1.0,
                "optimizer_trial": {"objective_score": 1.0, "gate_outcome": "accepted", "trial_number": 0},
            },
            "risk_decision": {"approved": approved, "reason": "accepted"},
        }


if __name__ == "__main__":
    unittest.main()
