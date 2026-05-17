import unittest

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.research.walk_forward import run_walk_forward_validation


class WalkForwardValidationTests(unittest.TestCase):
    def test_candidate_is_evaluated_on_in_sample_and_out_of_sample_ranges(self) -> None:
        strategy = {
            "name": "moving_average_crossover",
            "version": "0.1.0",
            "strategy_params": {"short_window": 5, "long_window": 20},
        }
        data = {"TEST": self._bars()}

        result = run_walk_forward_validation(
            strategy,
            data,
            in_sample=("2023-01-01", "2023-01-05"),
            out_of_sample=("2023-01-06", "2023-01-10"),
            backtest_agent=BacktestAgent(starting_capital=100_000.0, max_capital_per_trade_pct=0.10),
        )

        self.assertEqual(result["strategy"], strategy)
        self.assertEqual(result["in_sample"]["date_range"], {"start_date": "2023-01-01", "end_date": "2023-01-05"})
        self.assertEqual(result["out_of_sample"]["date_range"], {"start_date": "2023-01-06", "end_date": "2023-01-10"})
        self.assertEqual(result["in_sample"]["symbols"], ["TEST"])
        self.assertEqual(result["out_of_sample"]["symbols"], ["TEST"])

    def test_result_exposes_separate_metrics_for_each_period(self) -> None:
        strategy = {
            "name": "moving_average_crossover",
            "version": "0.1.0",
            "strategy_params": {"short_window": 5, "long_window": 20},
        }

        result = run_walk_forward_validation(
            strategy,
            {"TEST": self._bars()},
            in_sample=("2023-01-01", "2023-01-05"),
            out_of_sample=("2023-01-06", "2023-01-10"),
        )

        self.assertIn("metrics", result["in_sample"])
        self.assertIn("metrics", result["out_of_sample"])
        self.assertEqual(result["in_sample"]["metrics"]["history_points"], 4)
        self.assertEqual(result["out_of_sample"]["metrics"]["history_points"], 4)
        self.assertNotEqual(result["in_sample"]["metrics"]["total_return"], result["out_of_sample"]["metrics"]["total_return"])

    def test_result_exposes_cost_assumptions_for_each_period(self) -> None:
        strategy = {
            "name": "moving_average_crossover",
            "version": "0.1.0",
            "strategy_params": {"short_window": 5, "long_window": 20},
        }

        result = run_walk_forward_validation(
            strategy,
            {"TEST": self._bars()},
            in_sample=("2023-01-01", "2023-01-05"),
            out_of_sample=("2023-01-06", "2023-01-10"),
            backtest_agent=BacktestAgent(commission_per_trade=1.0, slippage_pct=0.001),
        )

        self.assertEqual(result["in_sample"]["cost_assumptions"], {"commission_per_trade": 1.0, "slippage_pct": 0.001})
        self.assertEqual(result["out_of_sample"]["cost_assumptions"], {"commission_per_trade": 1.0, "slippage_pct": 0.001})

    def test_existing_backtest_result_shape_still_works(self) -> None:
        strategy = {
            "name": "moving_average_crossover",
            "version": "0.1.0",
            "strategy_params": {"short_window": 5, "long_window": 20},
        }

        result = BacktestAgent().run_backtest(strategy, {"TEST": self._bars()})

        self.assertIn("metrics", result)
        self.assertIn("per_symbol_results", result)
        self.assertNotIn("in_sample", result)
        self.assertNotIn("out_of_sample", result)

    def _bars(self) -> list[dict[str, float | str]]:
        closes = [100, 101, 102, 103, 104, 104, 103, 102, 101, 100]
        return [
            {
                "date": f"2023-01-{index + 1:02d}",
                "close": float(close),
                "sma_5": float(close) * 1.01,
                "sma_20": float(close),
            }
            for index, close in enumerate(closes)
        ]


if __name__ == "__main__":
    unittest.main()
