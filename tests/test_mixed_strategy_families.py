import unittest

from multi_agent_trading_lab.agents.strategy_agent import StrategyAgent
from multi_agent_trading_lab.experiments.auto_research import GoodStrategyThresholds, collect_good_strategies
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentRecord
from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator


class MixedStrategyFamilyTests(unittest.TestCase):
    def test_strategy_agent_suggests_breakout_params(self) -> None:
        agent = StrategyAgent(
            parameter_bounds={
                "breakout_window": {"min": 20, "max": 60},
                "exit_window": {"min": 5, "max": 30},
            }
        )

        candidates = agent.suggest_parameter_grid("breakout_trend", 3)

        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(candidate["name"] == "breakout_trend" for candidate in candidates))
        self.assertTrue(all(candidate["strategy_params"]["breakout_window"] > candidate["strategy_params"]["exit_window"] for candidate in candidates))

    def test_auto_research_uniqueness_includes_strategy_family(self) -> None:
        records = [
            self._record("moving_average_crossover", {"short_window": 5, "long_window": 20}),
            self._record("breakout_trend", {"breakout_window": 20, "exit_window": 5}),
        ]

        good = collect_good_strategies(records, GoodStrategyThresholds(min_sharpe=0.7, min_drawdown=-0.15))

        self.assertEqual(len(good), 2)
        self.assertEqual({item.strategy_name for item in good}, {"moving_average_crossover", "breakout_trend"})

    def test_orchestrator_resolves_strategy_families_without_breaking_ma_alias(self) -> None:
        orchestrator = TradingLabOrchestrator(
            {
                "strategy_families": ["ma", "ma_rsi", "breakout"],
                "strategies": {
                    "moving_average_crossover": {"short_window": {"min": 5, "max": 10}, "long_window": {"min": 20, "max": 30}},
                    "moving_average_rsi_filter": {
                        "short_window": {"min": 5, "max": 10},
                        "long_window": {"min": 30, "max": 60},
                        "rsi_period": {"min": 10, "max": 20},
                        "rsi_min": {"min": 30, "max": 40},
                        "rsi_max": {"min": 60, "max": 70},
                    },
                    "breakout_trend": {"breakout_window": {"min": 20, "max": 30}, "exit_window": {"min": 5, "max": 10}},
                },
            }
        )

        self.assertEqual(
            orchestrator._resolve_strategy_families(["ma", "ma_rsi", "breakout"]),
            ["moving_average_crossover", "moving_average_rsi_filter", "breakout_trend"],
        )

    def test_strategy_agent_suggests_ma_rsi_params(self) -> None:
        agent = StrategyAgent(
            parameter_bounds={
                "short_window": {"min": 5, "max": 10},
                "long_window": {"min": 30, "max": 60},
                "rsi_period": {"min": 10, "max": 20},
                "rsi_min": {"min": 30, "max": 40},
                "rsi_max": {"min": 60, "max": 70},
            }
        )

        candidates = agent.suggest_parameter_grid("moving_average_rsi_filter", 2)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate["name"] == "moving_average_rsi_filter" for candidate in candidates))
        self.assertTrue(all("rsi_period" in candidate["strategy_params"] for candidate in candidates))

    def _record(self, strategy_name: str, params: dict) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=f"{strategy_name}-1",
            timestamp="2026-05-11T00:00:00+00:00",
            strategy_name=strategy_name,
            strategy_version="0.1.0",
            strategy_params=params,
            data_range={"symbols": ["TEST"]},
            metrics={"total_return": 0.2, "max_drawdown": -0.10, "sharpe_ratio": 0.8, "history_points": 252},
        )


if __name__ == "__main__":
    unittest.main()
