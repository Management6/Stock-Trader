import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.experiments.experiment_logger import ExperimentLogger, load_experiments, load_experiments_for_strategy


class ExperimentLoggerTests(unittest.TestCase):
    def test_experiment_logger_writes_and_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(Path(tmpdir) / "memory.jsonl")
            record = logger.log_experiment(
                config={"strategy": {"id": "example"}},
                metrics={"total_return": 0.1, "max_drawdown": -0.05, "sharpe_placeholder": 1.2},
                risk_decision={"approved": True, "reason": "ok"},
            )

            recent = logger.list_recent_experiments()

        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0]["id"], record["id"])
        self.assertEqual(recent[0]["metrics"]["total_return"], 0.1)
        self.assertEqual(recent[0]["strategy_params"], {})

    def test_experiment_schema_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "memory.jsonl"
            logger = ExperimentLogger(path)
            logger.log_experiment(
                config={
                    "strategy": {
                        "id": "example_ma_v0_1_0_s5_l20",
                        "name": "moving_average_crossover",
                        "version": "0.1.0",
                        "strategy_params": {"short_window": 5, "long_window": 20},
                    },
                    "symbols": ["AAPL"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-02-01",
                },
                metrics={"total_return": 0.2, "max_drawdown": -0.03, "sharpe_ratio": 1.5, "history_points": 40},
                mode="backtest",
            )

            records = load_experiments(path=path)
            strategy_records = load_experiments_for_strategy("moving_average_crossover", path=path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].strategy_params["short_window"], 5)
        self.assertEqual(records[0].metrics["sharpe_ratio"], 1.5)
        self.assertEqual(records[0].data_range["symbols"], ["AAPL"])
        self.assertEqual(len(strategy_records), 1)


if __name__ == "__main__":
    unittest.main()
