import unittest

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.environment.trading_env import TradingEnvironment


class BacktestSizingTests(unittest.TestCase):
    def test_environment_uses_starting_capital_and_trade_fraction(self) -> None:
        bars = [
            {"date": "2023-01-01", "close": 100.0},
            {"date": "2023-01-02", "close": 100.0},
            {"date": "2023-01-03", "close": 110.0},
        ]
        env = TradingEnvironment(
            bars,
            initial_cash=100_000.0,
            max_capital_per_trade_pct=0.10,
            allow_leverage=False,
        )

        state, _, _ = env.step(1)

        self.assertEqual(env.initial_cash, 100_000.0)
        self.assertEqual(state.position, 100)
        self.assertAlmostEqual(state.cash, 90_000.0)

    def test_backtest_agent_respects_trade_fraction(self) -> None:
        data = {
            "TEST": [
                {"date": "2023-01-01", "close": 100.0, "sma_5": 101.0, "sma_20": 100.0},
                {"date": "2023-01-02", "close": 100.0, "sma_5": 101.0, "sma_20": 100.0},
                {"date": "2023-01-03", "close": 120.0, "sma_5": 101.0, "sma_20": 100.0},
            ]
        }
        agent = BacktestAgent(starting_capital=100_000.0, max_capital_per_trade_pct=0.10, allow_leverage=False)

        result = agent.run_backtest(
            {
                "name": "moving_average_crossover",
                "version": "0.1.0",
                "strategy_params": {"short_window": 5, "long_window": 20},
            },
            data,
        )
        values = result["per_symbol_results"]["TEST"]["portfolio_values"]

        self.assertLessEqual(max(values), 102_000.0)
        self.assertAlmostEqual(result["metrics"]["total_return"], 0.02)


if __name__ == "__main__":
    unittest.main()
