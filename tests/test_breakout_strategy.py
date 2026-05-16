import unittest
import tempfile
from pathlib import Path

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.data.feature_engineering import add_breakout_channels
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentLogger
from multi_agent_trading_lab.strategies.breakout_strategy import BreakoutTrendStrategy


class BreakoutStrategyTests(unittest.TestCase):
    def test_breakout_signal_enters_holds_and_exits(self) -> None:
        bars = add_breakout_channels(
            {
                "TEST": [
                    {"date": "2026-01-01", "open": 9, "high": 10, "low": 8, "close": 9, "volume": 1},
                    {"date": "2026-01-02", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 1},
                    {"date": "2026-01-03", "open": 11, "high": 12, "low": 10, "close": 13, "volume": 1},
                    {"date": "2026-01-04", "open": 12, "high": 14, "low": 11, "close": 12, "volume": 1},
                    {"date": "2026-01-05", "open": 11, "high": 12, "low": 8, "close": 8, "volume": 1},
                ]
            },
            windows=[2, 1],
        )["TEST"]
        strategy = BreakoutTrendStrategy(breakout_window=2, exit_window=1)

        signals = [strategy.generate_signal(bar) for bar in bars]

        self.assertEqual(signals, [0, 0, 1, 1, 0])

    def test_strategy_serialization_and_identifier(self) -> None:
        strategy = BreakoutTrendStrategy(breakout_window=20, exit_window=10)
        config = strategy.to_config()

        self.assertEqual(config["name"], "breakout_trend")
        self.assertEqual(config["strategy_params"], {"breakout_window": 20, "exit_window": 10})
        self.assertIn("breakout_trend", strategy.identifier)
        self.assertIn("b20_e10", strategy.identifier)

    def test_invalid_windows_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BreakoutTrendStrategy(breakout_window=10, exit_window=10)

    def test_insufficient_warmup_bars_generate_no_signal(self) -> None:
        strategy = BreakoutTrendStrategy(breakout_window=3, exit_window=2)

        self.assertEqual(strategy.generate_signal({"close": 10, "high": 10, "low": 9}), 0)

    def test_backtest_agent_runs_breakout_strategy(self) -> None:
        raw = {
            "TEST": [
                {"date": f"2026-01-{day:02d}", "open": 10 + day, "high": 11 + day, "low": 9 + day, "close": 10 + day, "volume": 1}
                for day in range(1, 12)
            ]
        }
        data = add_breakout_channels(raw, windows=[3, 2])
        result = BacktestAgent().run_backtest(
            {
                "name": "breakout_trend",
                "version": "0.1.0",
                "strategy_params": {"breakout_window": 3, "exit_window": 2},
            },
            data,
        )

        self.assertEqual(result["strategy"]["name"], "breakout_trend")
        self.assertIn("sharpe_ratio", result["metrics"])

    def test_experiment_logging_preserves_breakout_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(path=Path(tmpdir) / "experiments.jsonl")
            record = logger.log_experiment(
                config={
                    "strategy": {
                        "name": "breakout_trend",
                        "version": "0.1.0",
                        "strategy_params": {"breakout_window": 20, "exit_window": 10},
                    }
                },
                metrics={"total_return": 0.1, "max_drawdown": -0.05, "sharpe_ratio": 0.8},
            )

            self.assertEqual(record["strategy_name"], "breakout_trend")
            self.assertEqual(record["strategy_params"], {"breakout_window": 20, "exit_window": 10})


if __name__ == "__main__":
    unittest.main()
