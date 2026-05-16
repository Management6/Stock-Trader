import unittest
from unittest.mock import patch

from multi_agent_trading_lab.experiments.experiment_logger import ExperimentRecord
from multi_agent_trading_lab.experiments.analysis import (
    find_experiments_by_params,
    get_top_experiments_for_strategy,
)


class ExperimentAnalysisTests(unittest.TestCase):
    def test_top_experiments_are_sorted_limited_and_filtered(self) -> None:
        records = [
            self._record("a", 6, 35, 0.30, -0.10, 0.90),
            self._record("b", 7, 40, 0.20, -0.12, 1.40),
            self._record("c", 8, 45, 0.50, -0.30, 1.80),
            self._record("d", 9, 50, 0.10, -0.05, 0.50),
        ]

        with patch("multi_agent_trading_lab.experiments.analysis.load_experiments_for_strategy", return_value=records):
            top = get_top_experiments_for_strategy(
                "moving_average_crossover",
                limit=2,
                min_sharpe=0.8,
                max_drawdown=-0.20,
            )

        self.assertEqual([record.experiment_id for record in top], ["b", "a"])

    def test_find_experiments_by_params_returns_matching_records(self) -> None:
        records = [
            self._record("a", 6, 35, 0.30, -0.10, 0.90),
            self._record("b", 6, 35, 0.35, -0.12, 1.10),
            self._record("c", 7, 35, 0.20, -0.08, 0.80),
        ]

        with patch("multi_agent_trading_lab.experiments.analysis.load_experiments_for_strategy", return_value=records):
            matches = find_experiments_by_params("moving_average_crossover", short_window=6, long_window=35)
            missing = find_experiments_by_params("moving_average_crossover", short_window=8, long_window=35)

        self.assertEqual([record.experiment_id for record in matches], ["a", "b"])
        self.assertEqual(missing, [])

    def _record(
        self,
        experiment_id: str,
        short_window: int,
        long_window: int,
        total_return: float,
        max_drawdown: float,
        sharpe_ratio: float,
    ) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=experiment_id,
            timestamp="2026-01-01T00:00:00+00:00",
            strategy_name="moving_average_crossover",
            strategy_version="0.1.0",
            strategy_params={"short_window": short_window, "long_window": long_window},
            data_range={"symbols": ["WES.AX"], "start_date": "2010-01-01", "end_date": "2026-01-01"},
            metrics={"total_return": total_return, "max_drawdown": max_drawdown, "sharpe_ratio": sharpe_ratio, "history_points": 100},
        )


if __name__ == "__main__":
    unittest.main()
