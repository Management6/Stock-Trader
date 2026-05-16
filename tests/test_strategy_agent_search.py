import unittest

from multi_agent_trading_lab.agents.strategy_agent import StrategyAgent
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentRecord
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy


class FakeExperimentLogger:
    def __init__(self, records: list[ExperimentRecord]) -> None:
        self.records = records

    def load_experiments_for_strategy(self, strategy_name: str, limit: int | None = None) -> list[ExperimentRecord]:
        records = [record for record in self.records if record.strategy_name == strategy_name]
        return records[-limit:] if limit is not None else records


class StrategyAgentRiskAwareSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = {
            "short_window": {"min": 5, "max": 50},
            "long_window": {"min": 20, "max": 200},
            "stop_loss_pct": {"min": 0.01, "max": 0.10},
            "take_profit_pct": {"min": 0.02, "max": 0.20},
        }
        self.risk_policy = RiskPolicy(max_drawdown_threshold=-0.20, min_backtest_return=-0.05)
        self.objective = {"type": "sharpe", "sharpe": {"min_history_points": 30}}
        self.search_config = {
            "focus_on_risk_passing_pct": 0.70,
            "explore_risky_pct": 0.20,
            "adaptive_narrowing_enabled": True,
            "top_good_experiment_count": 5,
        }

    def test_mixed_history_focuses_more_near_risk_passing_region(self) -> None:
        records = [
            self._record("good-1", short_window=6, long_window=105, total_return=0.34, max_drawdown=-0.16, sharpe=1.10),
            self._record("good-2", short_window=7, long_window=110, total_return=0.36, max_drawdown=-0.18, sharpe=1.20),
            self._record("bad-1", short_window=8, long_window=185, total_return=1.20, max_drawdown=-0.55, sharpe=0.40),
            self._record("bad-2", short_window=9, long_window=195, total_return=1.40, max_drawdown=-0.60, sharpe=0.35),
        ]
        agent = self._agent(records)

        suggestions = agent.suggest_from_history("moving_average_crossover", 10)
        long_windows = [candidate["strategy_params"]["long_window"] for candidate in suggestions]
        near_good = [value for value in long_windows if 85 <= value <= 130]
        risky_region = [value for value in long_windows if value >= 170]

        self.assertGreaterEqual(len(near_good), 6)
        self.assertLessEqual(len(risky_region), 2)

    def test_all_history_too_risky_still_explores_but_avoids_extreme_clustering(self) -> None:
        records = [
            self._record("bad-1", short_window=6, long_window=175, total_return=1.00, max_drawdown=-0.50, sharpe=0.30),
            self._record("bad-2", short_window=8, long_window=190, total_return=1.30, max_drawdown=-0.65, sharpe=0.25),
            self._record("bad-3", short_window=10, long_window=200, total_return=1.50, max_drawdown=-0.70, sharpe=0.20),
        ]
        agent = self._agent(records)

        suggestions = agent.suggest_from_history("moving_average_crossover", 8)
        long_windows = [candidate["strategy_params"]["long_window"] for candidate in suggestions]

        self.assertTrue(suggestions)
        self.assertLess(sum(1 for value in long_windows if value >= 170), len(long_windows) // 2)
        self.assertGreaterEqual(len({value // 25 for value in long_windows}), 3)

    def test_multiple_iterations_shift_toward_safe_high_score_region(self) -> None:
        history = [
            self._record("safe-1", short_window=6, long_window=100, total_return=0.30, max_drawdown=-0.14, sharpe=1.00),
            self._record("safe-2", short_window=7, long_window=115, total_return=0.38, max_drawdown=-0.17, sharpe=1.25),
            self._record("unsafe-1", short_window=8, long_window=185, total_return=1.10, max_drawdown=-0.50, sharpe=0.30),
        ]
        near_safe_counts: list[int] = []
        risky_counts: list[int] = []

        for iteration in range(4):
            agent = self._agent(history)
            suggestions = agent.suggest_from_history("moving_average_crossover", 8)
            long_windows = [candidate["strategy_params"]["long_window"] for candidate in suggestions]
            near_safe_counts.append(sum(1 for value in long_windows if 90 <= value <= 125))
            risky_counts.append(sum(1 for value in long_windows if value >= 170))
            for index, candidate in enumerate(suggestions[:3]):
                params = candidate["strategy_params"]
                safe = 90 <= params["long_window"] <= 125
                history.append(
                    self._record(
                        f"iter-{iteration}-{index}",
                        short_window=params["short_window"],
                        long_window=params["long_window"],
                        total_return=0.35 if safe else 0.8,
                        max_drawdown=-0.16 if safe else -0.45,
                        sharpe=1.15 if safe else 0.25,
                    )
                )

        self.assertGreaterEqual(near_safe_counts[-1], near_safe_counts[0])
        self.assertLessEqual(risky_counts[-1], risky_counts[0])

    def _agent(self, records: list[ExperimentRecord]) -> StrategyAgent:
        return StrategyAgent(
            experiment_logger=FakeExperimentLogger(records),  # type: ignore[arg-type]
            parameter_bounds=self.bounds,
            objective_config=self.objective,
            risk_policy=self.risk_policy,
            search_config=self.search_config,
        )

    def _record(
        self,
        experiment_id: str,
        short_window: int,
        long_window: int,
        total_return: float,
        max_drawdown: float,
        sharpe: float,
    ) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id=experiment_id,
            timestamp="2026-01-01T00:00:00+00:00",
            strategy_name="moving_average_crossover",
            strategy_version="0.1.0",
            strategy_params={"short_window": short_window, "long_window": long_window},
            data_range={"symbols": ["WES.AX"], "start_date": "2010-01-01", "end_date": "2026-01-01"},
            metrics={
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe,
                "history_points": 252,
            },
        )


if __name__ == "__main__":
    unittest.main()
