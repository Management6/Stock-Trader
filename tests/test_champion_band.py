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


class ChampionBandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = {
            "short_window": {"min": 5, "max": 50},
            "long_window": {"min": 20, "max": 200},
            "stop_loss_pct": {"min": 0.01, "max": 0.10},
            "take_profit_pct": {"min": 0.02, "max": 0.20},
        }
        self.champion_band = {
            "example_ma": {
                "short_window": {"min": 45, "max": 50},
                "long_window": {"min": 105, "max": 120},
            }
        }

    def test_champion_band_enabled_biases_majority_inside_band(self) -> None:
        agent = self._agent(champion_enabled=True)

        suggestions = agent.suggest_from_history("example_ma", 10)
        inside = [candidate for candidate in suggestions if self._inside_band(candidate)]

        self.assertGreaterEqual(len(inside), 7)
        self.assertLess(len(inside), len(suggestions))

    def test_champion_band_disabled_uses_existing_adaptive_search(self) -> None:
        agent = self._agent(champion_enabled=False)

        suggestions = agent.suggest_from_history("example_ma", 10)
        inside = [candidate for candidate in suggestions if self._inside_band(candidate)]

        self.assertLess(len(inside), 7)

    def _agent(self, champion_enabled: bool) -> StrategyAgent:
        records = [
            self._record("champion-1", 46, 110, 0.55, -0.17, 0.45),
            self._record("champion-2", 49, 115, 0.60, -0.18, 0.43),
            self._record("champion-3", 50, 120, 0.52, -0.16, 0.41),
            self._record("risky-1", 42, 180, 1.20, -0.55, 0.20),
            self._record("risky-2", 30, 170, 0.90, -0.40, 0.25),
            self._record("other-safe", 10, 60, 0.20, -0.12, 0.30),
        ]
        return StrategyAgent(
            experiment_logger=FakeExperimentLogger(records),  # type: ignore[arg-type]
            parameter_bounds=self.bounds,
            objective_config={"type": "sharpe", "sharpe": {"min_history_points": 30}},
            risk_policy=RiskPolicy(max_drawdown_threshold=-0.20, min_backtest_return=-0.05),
            search_config={
                "focus_on_risk_passing_pct": 0.70,
                "explore_risky_pct": 0.20,
                "adaptive_narrowing_enabled": True,
                "top_good_experiment_count": 5,
                "champion_band_enabled": champion_enabled,
                "champion_band": self.champion_band,
                "champion_band_focus_pct": 0.70,
            },
        )

    def _inside_band(self, candidate: dict) -> bool:
        params = candidate["strategy_params"]
        return 45 <= params["short_window"] <= 50 and 105 <= params["long_window"] <= 120

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
            strategy_name="example_ma",
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
