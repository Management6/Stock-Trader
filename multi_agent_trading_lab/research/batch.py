"""In-process multi-iteration research helpers."""

from __future__ import annotations

from typing import Any

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.agents.strategy_agent import StrategyAgent
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentLogger, ExperimentRecord


def run_research_batch(
    strategy_name: str,
    iterations: int,
    n_variants_per_iteration: int,
    strategy_agent: StrategyAgent | None = None,
    backtest_agent: Any | None = None,
    experiment_logger: Any | None = None,
    data: Any | None = None,
) -> list[list[ExperimentRecord]]:
    """Run repeated research iterations in-process and return records by iteration.

    Optional agent/logger/backtester arguments make this function easy to test
    with in-memory fakes. Real usage can pass the project agents and market
    data loaded elsewhere.
    """

    logger = experiment_logger or ExperimentLogger()
    agent = strategy_agent or StrategyAgent(experiment_logger=logger)
    tester = backtest_agent or BacktestAgent()
    all_iterations: list[list[ExperimentRecord]] = []
    for _ in range(iterations):
        iteration_records: list[ExperimentRecord] = []
        candidates = agent.suggest_from_history(strategy_name, n_variants_per_iteration)
        for candidate in candidates:
            result = tester.run_backtest(candidate, data or {})
            payload = logger.log_experiment(
                config={"strategy": candidate, "symbols": [], "start_date": None, "end_date": None},
                metrics=result["metrics"],
                mode="backtest",
            )
            iteration_records.append(_record_from_payload(payload))
        all_iterations.append(iteration_records)
    return all_iterations


def _record_from_payload(payload: dict[str, Any]) -> ExperimentRecord:
    return ExperimentRecord(
        experiment_id=str(payload.get("experiment_id") or payload.get("id")),
        timestamp=str(payload.get("timestamp")),
        strategy_name=str(payload["strategy_name"]),
        strategy_version=str(payload.get("strategy_version", "0.1.0")),
        strategy_params=dict(payload["strategy_params"]),
        data_range=dict(payload.get("data_range", {})),
        metrics=dict(payload["metrics"]),
        mode=str(payload.get("mode", "backtest")),
        risk_decision=dict(payload.get("risk_decision", {})),
        approval_status=str(payload.get("approval_status", "pending")),
        deployment_stage=str(payload.get("deployment_stage", "research")),
        config=dict(payload.get("config", {})),
    )
