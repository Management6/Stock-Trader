import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.agents.strategy_agent import StrategyAgent
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentLogger


class StrategyAgentTests(unittest.TestCase):
    def test_suggest_from_history_falls_back_without_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = StrategyAgent(
                experiment_logger=ExperimentLogger(Path(tmpdir) / "memory.jsonl"),
                parameter_bounds={
                    "short_window": {"min": 5, "max": 10},
                    "long_window": {"min": 20, "max": 40},
                    "stop_loss_pct": {"min": 0.01, "max": 0.10},
                    "take_profit_pct": {"min": 0.02, "max": 0.20},
                },
            )

            suggestions = agent.suggest_from_history("moving_average_crossover", 3)

        self.assertEqual(len(suggestions), 3)
        self.assertTrue(all("strategy_params" in suggestion for suggestion in suggestions))

    def test_suggest_from_history_uses_best_experiments_and_avoids_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(Path(tmpdir) / "memory.jsonl")
            self._log(logger, 5, 20, 0.01, -0.01, 0.2)
            self._log(logger, 8, 32, 0.30, -0.02, 2.0)
            self._log(logger, 12, 48, -0.10, -0.08, -0.5)
            agent = StrategyAgent(
                experiment_logger=logger,
                parameter_bounds={
                    "short_window": {"min": 5, "max": 15},
                    "long_window": {"min": 20, "max": 60},
                    "stop_loss_pct": {"min": 0.01, "max": 0.10},
                    "take_profit_pct": {"min": 0.02, "max": 0.20},
                },
                objective_config={"type": "sharpe", "sharpe": {"min_history_points": 1}},
            )

            suggestions = agent.suggest_from_history("moving_average_crossover", 4)
            params = [suggestion["strategy_params"] for suggestion in suggestions]
            keys = {(p["short_window"], p["long_window"]) for p in params}

        self.assertTrue(suggestions)
        self.assertEqual(len(keys), len(params))
        self.assertNotIn((8, 32), keys)
        self.assertTrue(any(abs(p["short_window"] - 8) <= 3 for p in params))

    def test_suggest_from_history_respects_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(Path(tmpdir) / "memory.jsonl")
            self._log(logger, 5, 20, 0.20, -0.01, 3.0)
            agent = StrategyAgent(
                experiment_logger=logger,
                parameter_bounds={
                    "short_window": {"min": 5, "max": 6},
                    "long_window": {"min": 20, "max": 21},
                    "stop_loss_pct": {"min": 0.01, "max": 0.10},
                    "take_profit_pct": {"min": 0.02, "max": 0.20},
                },
            )

            suggestions = agent.suggest_from_history("moving_average_crossover", 2)

        for suggestion in suggestions:
            params = suggestion["strategy_params"]
            self.assertGreaterEqual(params["short_window"], 5)
            self.assertLessEqual(params["short_window"], 6)
            self.assertGreaterEqual(params["long_window"], 20)
            self.assertLessEqual(params["long_window"], 21)
            self.assertLess(params["short_window"], params["long_window"])

    def test_suggest_from_history_uses_configured_objective(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ExperimentLogger(Path(tmpdir) / "memory.jsonl")
            self._log(logger, 6, 24, 0.05, -0.01, 3.0)
            self._log(logger, 14, 56, 0.30, -0.10, 0.2)
            agent = StrategyAgent(
                experiment_logger=logger,
                parameter_bounds={
                    "short_window": {"min": 5, "max": 20},
                    "long_window": {"min": 20, "max": 80},
                    "stop_loss_pct": {"min": 0.01, "max": 0.10},
                    "take_profit_pct": {"min": 0.02, "max": 0.20},
                },
                objective_config={"type": "return"},
            )

            suggestions = agent.suggest_from_history("moving_average_crossover", 2)

        self.assertTrue(any(abs(suggestion["strategy_params"]["short_window"] - 14) <= 3 for suggestion in suggestions))

    def _log(
        self,
        logger: ExperimentLogger,
        short_window: int,
        long_window: int,
        total_return: float,
        max_drawdown: float,
        sharpe: float,
    ) -> None:
        logger.log_experiment(
            config={
                "strategy": {
                    "id": f"s{short_window}_l{long_window}",
                    "name": "moving_average_crossover",
                    "version": "0.1.0",
                    "strategy_params": {"short_window": short_window, "long_window": long_window},
                },
                "symbols": ["AAPL"],
                "start_date": "2023-01-01",
                "end_date": "2023-02-01",
            },
            metrics={"total_return": total_return, "max_drawdown": max_drawdown, "sharpe_ratio": sharpe, "history_points": 40},
            mode="backtest",
        )


if __name__ == "__main__":
    unittest.main()
