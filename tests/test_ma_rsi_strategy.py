import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.data.feature_engineering import add_moving_averages, add_rsi
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentLogger
from multi_agent_trading_lab.strategies.ma_rsi_strategy import MovingAverageRsiFilterStrategy


class MovingAverageRsiFilterStrategyTests(unittest.TestCase):
    def test_rsi_values_are_computed_from_close_prices(self) -> None:
        market_data = {
            "TEST": [
                {"date": f"2026-01-0{index}", "open": close, "high": close, "low": close, "close": close, "volume": 1}
                for index, close in enumerate([1, 2, 3, 2, 3], start=1)
            ]
        }

        bars = add_rsi(market_data, periods=[3])["TEST"]

        self.assertNotIn("rsi_3", bars[2])
        self.assertAlmostEqual(float(bars[3]["rsi_3"]), 66.6667, places=3)
        self.assertAlmostEqual(float(bars[4]["rsi_3"]), 66.6667, places=3)

    def test_no_entry_when_rsi_outside_band(self) -> None:
        strategy = MovingAverageRsiFilterStrategy(short_window=2, long_window=3, rsi_period=3, rsi_min=35, rsi_max=65)

        signal = strategy.generate_signal({"close": 10, "sma_2": 11, "sma_3": 10, "rsi_3": 80})

        self.assertEqual(signal, 0)

    def test_entry_allowed_when_ma_condition_and_rsi_inside_band(self) -> None:
        strategy = MovingAverageRsiFilterStrategy(short_window=2, long_window=3, rsi_period=3, rsi_min=35, rsi_max=65)

        signal = strategy.generate_signal({"close": 10, "sma_2": 11, "sma_3": 10, "rsi_3": 50})

        self.assertEqual(signal, 1)

    def test_exit_uses_ma_condition_without_rsi_filter(self) -> None:
        strategy = MovingAverageRsiFilterStrategy(short_window=2, long_window=3, rsi_period=3, rsi_min=35, rsi_max=65)
        self.assertEqual(strategy.generate_signal({"close": 10, "sma_2": 11, "sma_3": 10, "rsi_3": 50}), 1)

        exit_signal = strategy.generate_signal({"close": 10, "sma_2": 9, "sma_3": 10, "rsi_3": 90})

        self.assertEqual(exit_signal, 0)

    def test_missing_rsi_warmup_does_not_crash_or_enter(self) -> None:
        strategy = MovingAverageRsiFilterStrategy(short_window=2, long_window=3, rsi_period=3)

        self.assertEqual(strategy.generate_signal({"close": 10, "sma_2": 11, "sma_3": 10}), 0)

    def test_invalid_rsi_band_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MovingAverageRsiFilterStrategy(short_window=2, long_window=3, rsi_period=3, rsi_min=70, rsi_max=60)

    def test_strategy_serialization_and_identifier(self) -> None:
        strategy = MovingAverageRsiFilterStrategy(short_window=5, long_window=30, rsi_period=14, rsi_min=35, rsi_max=65)

        config = strategy.to_config()

        self.assertEqual(config["name"], "moving_average_rsi_filter")
        self.assertEqual(
            config["strategy_params"],
            {"short_window": 5, "long_window": 30, "rsi_period": 14, "rsi_min": 35.0, "rsi_max": 65.0},
        )
        self.assertIn("ma_rsi", strategy.identifier)
        self.assertIn("r14_35_65", strategy.identifier)

    def test_backtest_agent_runs_ma_rsi_strategy(self) -> None:
        raw = {
            "TEST": [
                {"date": f"2026-01-{day:02d}", "open": 10 + day, "high": 11 + day, "low": 9 + day, "close": 10 + day, "volume": 1}
                for day in range(1, 20)
            ]
        }
        data = add_rsi(add_moving_averages(raw, windows=[2, 5]), periods=[3])
        result = BacktestAgent().run_backtest(
            {
                "name": "moving_average_rsi_filter",
                "version": "0.1.0",
                "strategy_params": {"short_window": 2, "long_window": 5, "rsi_period": 3, "rsi_min": 30, "rsi_max": 100},
            },
            data,
        )

        self.assertEqual(result["strategy"]["name"], "moving_average_rsi_filter")
        self.assertIn("sharpe_ratio", result["metrics"])

    def test_experiment_logging_preserves_ma_rsi_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(path=Path(tmpdir) / "experiments.jsonl")
            record = logger.log_experiment(
                config={
                    "strategy": {
                        "name": "moving_average_rsi_filter",
                        "version": "0.1.0",
                        "strategy_params": {"short_window": 5, "long_window": 30, "rsi_period": 14, "rsi_min": 35, "rsi_max": 65},
                    }
                },
                metrics={"total_return": 0.1, "max_drawdown": -0.05, "sharpe_ratio": 0.8},
            )

        self.assertEqual(record["strategy_name"], "moving_average_rsi_filter")
        self.assertEqual(record["strategy_params"]["rsi_period"], 14)


if __name__ == "__main__":
    unittest.main()
