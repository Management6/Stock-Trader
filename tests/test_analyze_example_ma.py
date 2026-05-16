import unittest
from unittest.mock import patch

from multi_agent_trading_lab.experiments.analysis import analyze_ma_strategy
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentRecord


class AnalyzeExampleMATests(unittest.TestCase):
    def test_analyze_ma_strategy_returns_best_matching_record_summary(self) -> None:
        records = [
            self._record("low", 6, 35, 0.30, -0.12, 0.80),
            self._record("best", 6, 35, 0.42, -0.20, 1.20),
        ]

        with patch("multi_agent_trading_lab.experiments.analysis.find_experiments_by_params", return_value=records):
            summary = analyze_ma_strategy(short_window=6, long_window=35)

        self.assertEqual(summary["status"], "found")
        self.assertEqual(summary["strategy_name"], "moving_average_crossover")
        self.assertEqual(summary["params"], {"short_window": 6, "long_window": 35})
        self.assertEqual(summary["experiment_id"], "best")
        self.assertEqual(summary["total_return"], 0.42)
        self.assertEqual(summary["max_drawdown"], -0.20)
        self.assertEqual(summary["sharpe_ratio"], 1.20)

    def test_analyze_ma_strategy_reports_no_match(self) -> None:
        with patch("multi_agent_trading_lab.experiments.analysis.find_experiments_by_params", return_value=[]):
            summary = analyze_ma_strategy(short_window=6, long_window=35)

        self.assertEqual(summary["status"], "no_match")
        self.assertEqual(summary["params"], {"short_window": 6, "long_window": 35})
        self.assertIsNone(summary["total_return"])

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
            risk_decision={"approved": True, "reason": "ok"},
        )


if __name__ == "__main__":
    unittest.main()
