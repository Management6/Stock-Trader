import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from multi_agent_trading_lab.experiments.auto_research import (
    AutoResearchConfig,
    GoodStrategyThresholds,
    collect_good_strategies,
    run_auto_research,
)
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentRecord
from scripts.run_auto_research_until_good_strategies import main as cli_main


class AutoResearchTests(unittest.TestCase):
    def test_counts_distinct_good_strategies_and_stops_at_target(self) -> None:
        batches = [
            [
                self._record("good-1", 5, 20, sharpe=0.71, drawdown=-0.10),
                self._record("bad-sharpe", 6, 21, sharpe=0.69, drawdown=-0.10),
                self._record("bad-drawdown", 7, 22, sharpe=0.80, drawdown=-0.151),
            ],
            [
                self._record("good-2", 8, 23, sharpe=0.90, drawdown=-0.12),
                self._record("duplicate-good-1", 5, 20, sharpe=0.72, drawdown=-0.09),
            ],
        ]

        result = run_auto_research(
            AutoResearchConfig(max_iterations=10, target_good_strategies=2),
            run_iteration=lambda iteration: batches[iteration - 1],
        )

        self.assertEqual(result.iterations_run, 2)
        self.assertEqual(result.stop_reason, "target_reached")
        self.assertEqual(len(result.good_strategies), 2)
        self.assertEqual(result.best_strategy.strategy_params["short_window"], 8)

    def test_stops_on_iteration_cap_and_returns_useful_summary(self) -> None:
        result = run_auto_research(
            AutoResearchConfig(max_iterations=3, target_good_strategies=2),
            run_iteration=lambda iteration: [self._record(f"bad-{iteration}", iteration, iteration + 20, sharpe=0.2, drawdown=-0.05)],
        )

        self.assertEqual(result.iterations_run, 3)
        self.assertEqual(result.stop_reason, "iteration_cap")
        self.assertEqual(result.good_strategies, [])
        self.assertIsNotNone(result.best_strategy)

    def test_collect_good_strategies_excludes_duplicates_and_threshold_misses(self) -> None:
        records = [
            self._record("good-1", 5, 20, sharpe=0.70, drawdown=-0.15),
            self._record("duplicate", 5, 20, sharpe=0.80, drawdown=-0.10),
            self._record("low-sharpe", 6, 21, sharpe=0.69, drawdown=-0.10),
            self._record("too-deep", 7, 22, sharpe=0.71, drawdown=-0.151),
        ]

        good = collect_good_strategies(records, GoodStrategyThresholds(min_sharpe=0.70, min_drawdown=-0.15))

        self.assertEqual(len(good), 1)
        self.assertEqual(good[0].strategy_params, {"short_window": 5, "long_window": 20})

    def test_cli_smoke_prints_summary_with_mocked_runner(self) -> None:
        fake_result = run_auto_research(
            AutoResearchConfig(max_iterations=1, target_good_strategies=1),
            run_iteration=lambda _: [self._record("good", 10, 30, sharpe=0.75, drawdown=-0.10)],
        )

        with patch("scripts.run_auto_research_until_good_strategies.run_auto_research", return_value=fake_result):
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = cli_main(["--max-iterations", "1", "--target-good-strategies", "1"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Auto-research complete", output.getvalue())
        self.assertIn("good strategies found: 1", output.getvalue())

    def test_cli_can_run_small_synthetic_loop_without_real_research(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = cli_main(["--max-iterations", "2", "--target-good-strategies", "1", "--synthetic-test-mode"])

        self.assertEqual(exit_code, 0)
        self.assertIn("Auto-research complete", output.getvalue())

    def _record(
        self,
        experiment_id: str,
        short_window: int,
        long_window: int,
        sharpe: float,
        drawdown: float,
        total_return: float = 0.20,
    ) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=experiment_id,
            timestamp="2026-05-11T00:00:00+00:00",
            strategy_name="moving_average_crossover",
            strategy_version="0.1.0",
            strategy_params={"short_window": short_window, "long_window": long_window},
            data_range={"symbols": ["TEST"], "start_date": "2020-01-01", "end_date": None},
            metrics={
                "total_return": total_return,
                "max_drawdown": drawdown,
                "sharpe_ratio": sharpe,
                "history_points": 252,
            },
        )


if __name__ == "__main__":
    unittest.main()
