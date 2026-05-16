import math
import unittest

from multi_agent_trading_lab.metrics import compute_backtest_metrics, compute_sharpe_ratio


class MetricsTests(unittest.TestCase):
    def test_compute_sharpe_ratio_annualizes_daily_returns(self) -> None:
        returns = [0.01, -0.005, 0.015, 0.0]
        expected = (sum(returns) / len(returns)) / math.sqrt(sum((value - sum(returns) / len(returns)) ** 2 for value in returns) / len(returns)) * math.sqrt(252)

        self.assertAlmostEqual(compute_sharpe_ratio(returns), expected)

    def test_compute_backtest_metrics_has_first_class_sharpe_ratio(self) -> None:
        metrics = compute_backtest_metrics([100.0, 101.0, 100.5, 103.0])

        self.assertIn("total_return", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertIn("history_points", metrics)


if __name__ == "__main__":
    unittest.main()
