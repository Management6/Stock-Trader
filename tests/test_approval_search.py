import json
import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.operations.approval_search import run_approval_search
from multi_agent_trading_lab.orchestrator.orchestrator import load_settings


APPROVAL_PROFILE = "multi_agent_trading_lab/config/settings.approval_paper.yaml"


class ApprovalSearchTests(unittest.TestCase):
    def test_expanded_search_covers_multiple_existing_families_and_records_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_approval_search(
                APPROVAL_PROFILE,
                trials=2,
                seed=7,
                output_root=Path(tmpdir),
                families=["ma", "ma_rsi", "breakout"],
            )

            self.assertEqual(
                summary.families,
                ["moving_average_crossover", "moving_average_rsi_filter", "breakout_trend"],
            )
            self.assertEqual(summary.total_trials, 6)
            self.assertEqual(summary.trials_by_family["moving_average_crossover"], 2)
            self.assertEqual(summary.trials_by_family["moving_average_rsi_filter"], 2)
            self.assertEqual(summary.trials_by_family["breakout_trend"], 2)
            for family, candidate in summary.best_candidate_by_family.items():
                self.assertIsNotNone(candidate, family)
                self.assertEqual(candidate["family"], family)
                self.assertIn(candidate["gate_outcome"], {"accepted", "rejected", "failed"})
            self.assertEqual(summary.accepted_count + summary.rejected_count, summary.total_trials)
            self.assertTrue(summary.rejection_breakdown_by_family)

    def test_approval_search_is_deterministic_with_seed(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = run_approval_search(APPROVAL_PROFILE, trials=2, seed=11, output_root=Path(first_dir))
            second = run_approval_search(APPROVAL_PROFILE, trials=2, seed=11, output_root=Path(second_dir))

            self.assertEqual(first.trials_by_family, second.trials_by_family)
            self.assertEqual(first.accepted_by_family, second.accepted_by_family)
            self.assertEqual(first.rejected_by_family, second.rejected_by_family)
            self.assertEqual(first.rejection_breakdown, second.rejection_breakdown)
            self.assertEqual(first.best_candidate_by_family, second.best_candidate_by_family)
            self.assertEqual(first.best_near_miss, second.best_near_miss)

    def test_approval_search_does_not_loosen_approval_profile_thresholds(self) -> None:
        before = load_settings(APPROVAL_PROFILE)
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_approval_search(APPROVAL_PROFILE, trials=1, seed=3, output_root=Path(tmpdir))
            summary_path = summary.output_dir / "approval_search_summary.json"
            self.assertTrue(summary_path.exists())
            persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            records = [
                json.loads(line)
                for line in summary.experiment_log_path.read_text(encoding="utf-8").splitlines()
            ]
        after = load_settings(APPROVAL_PROFILE)

        self.assertEqual(persisted_summary["total_trials"], 3)
        self.assertTrue(records)
        self.assertIsInstance(records[0]["data_range"]["start_date"], str)
        self.assertIsInstance(records[0]["data_range"]["end_date"], str)
        self.assertEqual(after["oos_validation"], before["oos_validation"])
        self.assertEqual(after["portfolio"], before["portfolio"])
        self.assertEqual(after["risk"], before["risk"])
        self.assertGreaterEqual(after["backtest"]["slippage_pct"], 0.0005)
        self.assertEqual(summary.trials_per_family, 1)


if __name__ == "__main__":
    unittest.main()
