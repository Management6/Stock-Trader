import math
import unittest

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.orchestrator.orchestrator import load_settings
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy


class BacktestSanityTests(unittest.TestCase):
    def test_small_backtest_metrics_are_finite_and_risk_consistent(self) -> None:
        bars = []
        price = 100.0
        for index in range(80):
            price *= 1.005 if index < 40 else 0.997
            bars.append(
                {
                    "date": f"2023-01-{(index % 28) + 1:02d}",
                    "close": price,
                    "sma_5": price * 1.01,
                    "sma_20": price,
                }
            )
        agent = BacktestAgent(starting_capital=100_000.0, max_capital_per_trade_pct=0.10, allow_leverage=False)
        result = agent.run_backtest(
            {
                "name": "moving_average_crossover",
                "version": "0.1.0",
                "strategy_params": {"short_window": 5, "long_window": 20},
            },
            {"TEST": bars},
        )
        metrics = result["metrics"]
        policy = RiskPolicy.from_config(load_settings("multi_agent_trading_lab/config/risk.yaml"))
        decision = policy.evaluate_strategy(metrics)

        self.assertTrue(math.isfinite(metrics["total_return"]))
        self.assertTrue(math.isfinite(metrics["max_drawdown"]))
        self.assertTrue(-1.0 <= metrics["max_drawdown"] <= 0.0)
        self.assertLess(abs(metrics["total_return"]), 10.0)
        if metrics["max_drawdown"] < policy.max_drawdown_threshold:
            self.assertFalse(decision.approved)


if __name__ == "__main__":
    unittest.main()
