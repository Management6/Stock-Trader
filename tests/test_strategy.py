import unittest

from multi_agent_trading_lab.strategies.base_strategy import BaseStrategy, StrategyConfig
from multi_agent_trading_lab.strategies.example_strategy import MovingAverageCrossoverStrategy


class StrategyTests(unittest.TestCase):
    def test_base_strategy_requires_generate_signal(self) -> None:
        with self.assertRaises(TypeError):
            BaseStrategy(StrategyConfig(name="abstract"))

    def test_moving_average_crossover_signal(self) -> None:
        strategy = MovingAverageCrossoverStrategy(short_window=3, long_window=5)

        self.assertEqual(
            strategy.generate_signal({"date": "2023-01-01", "close": 10.0, "sma_3": 11.0, "sma_5": 10.0}),
            1,
        )
        self.assertEqual(
            strategy.generate_signal({"date": "2023-01-02", "close": 10.0, "sma_3": 9.0, "sma_5": 10.0}),
            0,
        )


if __name__ == "__main__":
    unittest.main()
