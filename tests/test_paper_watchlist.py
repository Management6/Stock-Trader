import json
import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.operations.paper_watchlist import PAPER_MONITORING_WARNING, select_paper_watchlist


class PaperWatchlistTests(unittest.TestCase):
    def test_only_robust_accepted_candidates_are_selected_with_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            approval_summary, robustness_summary = self._write_inputs(Path(tmpdir))

            summary = select_paper_watchlist(approval_summary, robustness_summary, max_candidates=5, output_root=Path(tmpdir) / "out")
            selected_ids = [candidate["candidate_id"] for candidate in summary.selected_candidates]
            persisted = json.loads(summary.watchlist_json_path.read_text(encoding="utf-8"))
            markdown = summary.watchlist_markdown_path.read_text(encoding="utf-8")

        self.assertEqual(selected_ids, ["ma_good", "breakout_good"])
        self.assertEqual(summary.excluded_counts["not_accepted"], 1)
        self.assertEqual(summary.excluded_counts["not_robust"], 1)
        self.assertIn("oos_metrics", summary.selected_candidates[0])
        self.assertIn("portfolio_metrics", summary.selected_candidates[0])
        self.assertIn("robustness", summary.selected_candidates[0])
        self.assertIn("cost_assumptions", summary.selected_candidates[0])
        self.assertIn("regime_result", summary.selected_candidates[0])
        self.assertIn(PAPER_MONITORING_WARNING, markdown)
        self.assertEqual(persisted["warning"], PAPER_MONITORING_WARNING)
        self.assertFalse(summary.approval_queue_written)
        self.assertIsNone(summary.approval_queue_path)

    def test_max_candidates_is_respected_and_ranking_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_approval, first_robustness = self._write_inputs(Path(first_dir), include_extra=True)
            second_approval, second_robustness = self._write_inputs(Path(second_dir), include_extra=True)

            first = select_paper_watchlist(first_approval, first_robustness, max_candidates=2, output_root=Path(first_dir) / "out", diversify_by_family=False)
            second = select_paper_watchlist(second_approval, second_robustness, max_candidates=2, output_root=Path(second_dir) / "out", diversify_by_family=False)

        self.assertEqual([item["candidate_id"] for item in first.selected_candidates], ["ma_good", "ma_extra"])
        self.assertEqual([item["candidate_id"] for item in first.selected_candidates], [item["candidate_id"] for item in second.selected_candidates])
        self.assertEqual(first.selected_count, 2)

    def test_diversification_by_family_can_select_next_family_before_second_best_same_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            approval_summary, robustness_summary = self._write_inputs(Path(tmpdir), include_extra=True)

            diversified = select_paper_watchlist(approval_summary, robustness_summary, max_candidates=3, output_root=Path(tmpdir) / "diverse", diversify_by_family=True)
            concentrated = select_paper_watchlist(approval_summary, robustness_summary, max_candidates=3, output_root=Path(tmpdir) / "concentrated", diversify_by_family=False)

        self.assertEqual([item["candidate_id"] for item in diversified.selected_candidates], ["ma_good", "breakout_good", "ma_extra"])
        self.assertEqual([item["candidate_id"] for item in concentrated.selected_candidates], ["ma_good", "ma_extra", "breakout_good"])

    def test_write_approval_queue_is_explicit_and_off_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            approval_summary, robustness_summary = self._write_inputs(Path(tmpdir))

            read_only = select_paper_watchlist(approval_summary, robustness_summary, max_candidates=1, output_root=Path(tmpdir) / "readonly")
            writable = select_paper_watchlist(approval_summary, robustness_summary, max_candidates=1, output_root=Path(tmpdir) / "writable", write_approval_queue=True)
            self.assertTrue(writable.approval_queue_path.exists())

        self.assertFalse(read_only.approval_queue_written)
        self.assertIsNone(read_only.approval_queue_path)
        self.assertTrue(writable.approval_queue_written)
        self.assertIsNotNone(writable.approval_queue_path)

    def _write_inputs(self, root: Path, include_extra: bool = False) -> tuple[Path, Path]:
        root.mkdir(parents=True, exist_ok=True)
        experiment_log = root / "experiments.jsonl"
        records = [
            self._record("ma_good", "moving_average_crossover", accepted=True, oos_sharpe=2.0, oos_return=0.03, oos_drawdown=-0.02, portfolio_return=0.04, portfolio_drawdown=-0.03),
            self._record("breakout_good", "breakout_trend", accepted=True, oos_sharpe=1.2, oos_return=0.02, oos_drawdown=-0.01, portfolio_return=0.03, portfolio_drawdown=-0.02),
            self._record("ma_fragile", "moving_average_crossover", accepted=True, oos_sharpe=3.0, oos_return=0.05, oos_drawdown=-0.01, portfolio_return=0.05, portfolio_drawdown=-0.01),
            self._record("breakout_rejected", "breakout_trend", accepted=False, oos_sharpe=4.0, oos_return=0.07, oos_drawdown=-0.01, portfolio_return=0.06, portfolio_drawdown=-0.01),
        ]
        robustness_candidates = [
            self._robustness("ma_good", "robust"),
            self._robustness("breakout_good", "robust"),
            self._robustness("ma_fragile", "fragile"),
            self._robustness("breakout_rejected", "robust"),
        ]
        if include_extra:
            records.append(self._record("ma_extra", "moving_average_crossover", accepted=True, oos_sharpe=1.9, oos_return=0.025, oos_drawdown=-0.02, portfolio_return=0.035, portfolio_drawdown=-0.025))
            robustness_candidates.append(self._robustness("ma_extra", "robust"))
        experiment_log.write_text("\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n", encoding="utf-8")
        approval_summary = root / "approval_search_summary.json"
        approval_summary.write_text(json.dumps({"experiment_log_path": str(experiment_log), "accepted_count": 3, "rejected_count": 1, "total_trials": len(records)}) + "\n", encoding="utf-8")
        robustness_summary = root / "candidate_robustness_summary.json"
        robustness_summary.write_text(json.dumps({"candidates": robustness_candidates, "status_counts": {"robust": 3, "fragile": 1}}) + "\n", encoding="utf-8")
        return approval_summary, robustness_summary

    def _record(
        self,
        candidate_id: str,
        family: str,
        accepted: bool,
        oos_sharpe: float,
        oos_return: float,
        oos_drawdown: float,
        portfolio_return: float,
        portfolio_drawdown: float,
    ) -> dict:
        strategy = {
            "id": candidate_id,
            "name": family,
            "version": "0.1.0",
            "strategy_params": {"short_window": 5, "long_window": 30},
        }
        return {
            "id": candidate_id,
            "strategy_name": family,
            "strategy_params": strategy["strategy_params"],
            "config": {"strategy": strategy},
            "metrics": {
                "total_return": oos_return,
                "max_drawdown": oos_drawdown,
                "sharpe_ratio": oos_sharpe,
                "cost_assumptions": {"commission_per_trade": 0.0, "slippage_pct": 0.0005},
                "optimizer_trial": {"gate_outcome": "accepted" if accepted else "rejected", "objective_score": oos_sharpe, "trial_number": 1},
                "walk_forward": {"out_of_sample": {"metrics": {"sharpe_ratio": oos_sharpe, "total_return": oos_return, "max_drawdown": oos_drawdown}}},
                "portfolio_backtest": {"metrics": {"total_return": portfolio_return, "max_drawdown": portfolio_drawdown, "sharpe_ratio": oos_sharpe + 0.5}},
                "market_regime": {"regime": "bullish"},
            },
            "risk_decision": {"approved": accepted, "reason": "accepted" if accepted else "rejected"},
        }

    def _robustness(self, candidate_id: str, status: str) -> dict:
        return {
            "candidate_id": candidate_id,
            "status": status,
            "checks_passed": 12 if status == "robust" else 10,
            "checks_failed": 0 if status == "robust" else 2,
            "worst_oos_metric": 0.5,
            "worst_portfolio_metric": 1.0,
            "worst_cost_stress_result": {"name": "slippage_0.0025", "passed": status == "robust"},
        }


if __name__ == "__main__":
    unittest.main()
