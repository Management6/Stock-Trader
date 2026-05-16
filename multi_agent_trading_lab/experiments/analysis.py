"""Helpers for inspecting experiment history."""

from __future__ import annotations

from typing import Any

from multi_agent_trading_lab.experiments.experiment_logger import ExperimentRecord, load_experiments_for_strategy
from multi_agent_trading_lab.research.scoring import DEFAULT_OBJECTIVE_CONFIG, score_metrics


def get_top_experiments_for_strategy(
    strategy_name: str,
    limit: int = 10,
    min_sharpe: float | None = None,
    max_drawdown: float | None = None,
) -> list[ExperimentRecord]:
    """Return top experiments sorted by configured score descending.

    ``max_drawdown`` is expressed as a negative threshold. For example,
    ``max_drawdown=-0.20`` filters out records worse than -20%.
    """

    records = load_experiments_for_strategy(strategy_name)
    filtered: list[ExperimentRecord] = []
    for record in records:
        sharpe = record.metrics.get("sharpe_ratio", record.metrics.get("sharpe"))
        drawdown = record.metrics.get("max_drawdown")
        if min_sharpe is not None and (sharpe is None or float(sharpe) < min_sharpe):
            continue
        if max_drawdown is not None and (drawdown is None or float(drawdown) < max_drawdown):
            continue
        filtered.append(record)
    return sorted(filtered, key=_record_score, reverse=True)[:limit]


def find_experiments_by_params(
    strategy_name: str,
    short_window: int,
    long_window: int,
) -> list[ExperimentRecord]:
    """Return experiments whose moving-average windows match exactly."""

    return [
        record
        for record in load_experiments_for_strategy(strategy_name)
        if int(record.strategy_params.get("short_window", -1)) == short_window
        and int(record.strategy_params.get("long_window", -1)) == long_window
    ]


def analyze_ma_strategy(short_window: int, long_window: int) -> dict[str, Any]:
    """Summarize the best matching moving-average experiment by score."""

    strategy_name = "moving_average_crossover"
    matches = find_experiments_by_params(strategy_name, short_window, long_window)
    if not matches:
        return {
            "status": "no_match",
            "strategy_name": strategy_name,
            "params": {"short_window": short_window, "long_window": long_window},
            "total_return": None,
            "max_drawdown": None,
            "sharpe_ratio": None,
            "score": None,
            "decision": None,
            "data_range": None,
            "experiment_id": None,
        }
    best = max(matches, key=_record_score)
    metrics = best.metrics
    return {
        "status": "found",
        "strategy_name": best.strategy_name,
        "strategy_version": best.strategy_version,
        "params": {"short_window": short_window, "long_window": long_window},
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe_ratio": metrics.get("sharpe_ratio", metrics.get("sharpe")),
        "score": _record_score(best),
        "decision": best.risk_decision,
        "data_range": best.data_range,
        "experiment_id": best.experiment_id,
    }


def _record_score(record: ExperimentRecord) -> float:
    return score_metrics(record.metrics, DEFAULT_OBJECTIVE_CONFIG)
