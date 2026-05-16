import unittest

from multi_agent_trading_lab.agents.strategy_agent import StrategyAgent
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentRecord
from multi_agent_trading_lab.research.batch import run_research_batch
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy


class InMemoryExperimentLogger:
    def __init__(self, records: list[ExperimentRecord]) -> None:
        self.records = records

    def load_experiments_for_strategy(self, strategy_name: str, limit: int | None = None) -> list[ExperimentRecord]:
        records = [record for record in self.records if record.strategy_name == strategy_name]
        return records[-limit:] if limit is not None else records

    def log_experiment(self, config: dict, metrics: dict, **_: object) -> dict:
        strategy = config["strategy"]
        record = ExperimentRecord(
            experiment_id=f"record-{len(self.records)}",
            timestamp="2026-01-01T00:00:00+00:00",
            strategy_name=strategy["name"],
            strategy_version=strategy.get("version", "0.1.0"),
            strategy_params=dict(strategy["strategy_params"]),
            data_range={"symbols": ["TEST"], "start_date": "2020-01-01", "end_date": "2021-01-01"},
            metrics=metrics,
        )
        self.records.append(record)
        return record.to_dict()


class DummyBacktestAgent:
    def run_backtest(self, strategy_config: dict, data: object) -> dict:
        params = strategy_config["strategy_params"]
        in_band = 45 <= params["short_window"] <= 50 and 105 <= params["long_window"] <= 120
        return {
            "metrics": {
                "total_return": 0.55 if in_band else 0.15,
                "max_drawdown": -0.17 if in_band else -0.28,
                "sharpe_ratio": 0.50 if in_band else 0.10,
                "history_points": 252,
            }
        }


class ResearchBatchTests(unittest.TestCase):
    def test_batch_returns_iterations_and_concentrates_in_champion_band(self) -> None:
        logger = InMemoryExperimentLogger(
            [
                self._record("seed-safe", 46, 110, 0.50, -0.17, 0.45),
                self._record("seed-risky", 20, 180, 1.00, -0.50, 0.20),
            ]
        )
        agent = StrategyAgent(
            experiment_logger=logger,  # type: ignore[arg-type]
            parameter_bounds={
                "short_window": {"min": 5, "max": 50},
                "long_window": {"min": 20, "max": 200},
                "stop_loss_pct": {"min": 0.01, "max": 0.10},
                "take_profit_pct": {"min": 0.02, "max": 0.20},
            },
            objective_config={"type": "sharpe", "sharpe": {"min_history_points": 30}},
            risk_policy=RiskPolicy(max_drawdown_threshold=-0.20, min_backtest_return=-0.05),
            search_config={
                "champion_band_enabled": True,
                "champion_band": {
                    "example_ma": {
                        "short_window": {"min": 45, "max": 50},
                        "long_window": {"min": 105, "max": 120},
                    }
                },
                "champion_band_focus_pct": 0.70,
                "focus_on_risk_passing_pct": 0.70,
                "explore_risky_pct": 0.20,
            },
        )

        iterations = run_research_batch(
            strategy_name="example_ma",
            iterations=3,
            n_variants_per_iteration=6,
            strategy_agent=agent,
            backtest_agent=DummyBacktestAgent(),
            experiment_logger=logger,  # type: ignore[arg-type]
            data={},
        )

        self.assertEqual(len(iterations), 3)
        first_fraction = self._band_fraction(iterations[0])
        last_fraction = self._band_fraction(iterations[-1])
        self.assertGreaterEqual(last_fraction, first_fraction)
        self.assertGreaterEqual(last_fraction, 0.70)

    def _band_fraction(self, records: list[ExperimentRecord]) -> float:
        inside = [
            record
            for record in records
            if 45 <= record.strategy_params["short_window"] <= 50 and 105 <= record.strategy_params["long_window"] <= 120
        ]
        return len(inside) / len(records)

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
            data_range={"symbols": ["TEST"], "start_date": "2020-01-01", "end_date": "2021-01-01"},
            metrics={
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe,
                "history_points": 252,
            },
        )


if __name__ == "__main__":
    unittest.main()
