"""JSONL experiment memory with an explicit research schema.

ExperimentRecord schema:

- experiment_id: stable unique identifier for the run.
- timestamp: ISO-8601 UTC timestamp.
- strategy_name: canonical strategy family name.
- strategy_version: strategy version string.
- strategy_params: machine-readable hyperparameter dictionary.
- data_range: symbols, start date, and end date used for the run.
- metrics: numeric result metrics such as total_return, max_drawdown, sharpe.
- mode: research mode such as backtest or paper.
- risk_decision: optional approval/rejection payload.
- approval_status: approved, rejected, or pending.
- deployment_stage: research, paper, rejected, archived, etc.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from multi_agent_trading_lab.metrics import normalize_metrics


@dataclass(frozen=True)
class ExperimentRecord:
    """Structured experiment record returned by experiment memory helpers."""

    experiment_id: str
    timestamp: str
    strategy_name: str
    strategy_version: str
    strategy_params: dict[str, Any]
    data_range: dict[str, Any]
    metrics: dict[str, Any]
    mode: str = "backtest"
    risk_decision: dict[str, Any] = field(default_factory=dict)
    approval_status: str = "pending"
    deployment_stage: str = "research"
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Compatibility alias for earlier report code."""

        return self.experiment_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["id"] = self.experiment_id
        payload["parameters"] = dict(self.strategy_params)
        return payload


class ExperimentLogger:
    """Append-only JSONL store for backtest and paper-trading experiment memory."""

    def __init__(self, path: str | Path = "multi_agent_trading_lab/experiments/experiment_memory.jsonl") -> None:
        self.path = Path(path)

    def log_experiment(
        self,
        config: dict[str, Any],
        metrics: dict[str, float],
        risk_decision: dict[str, Any] | None = None,
        approval_status: str | None = None,
        deployment_stage: str | None = None,
        mode: str = "backtest",
    ) -> dict[str, Any]:
        """Write an experiment and return its serialized schema."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        strategy = dict(config.get("strategy", {}))
        params = _strategy_params(strategy)
        record = ExperimentRecord(
            experiment_id=str(uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
            strategy_name=str(strategy.get("name", "moving_average_crossover")),
            strategy_version=str(strategy.get("version", "0.1.0")),
            strategy_params=params,
            data_range={
                "symbols": config.get("symbols", []),
                "start_date": config.get("start_date"),
                "end_date": config.get("end_date"),
            },
            metrics=normalize_metrics(metrics),
            mode=mode,
            risk_decision=risk_decision or {},
            approval_status=approval_status or ("approved" if (risk_decision or {}).get("approved") else "pending"),
            deployment_stage=deployment_stage or config.get("deployment_stage", "research"),
            config=config,
        )
        payload = record.to_dict()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")
        return payload

    def list_recent_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent experiments as dictionaries for legacy callers."""

        return [record.to_dict() for record in self.load_experiments(limit=limit)]

    def load_experiments(self, limit: int | None = None) -> list[ExperimentRecord]:
        """Load experiments from JSONL as ExperimentRecord objects."""

        return load_experiments(limit=limit, path=self.path)

    def load_experiments_for_strategy(
        self,
        strategy_name: str,
        limit: int | None = None,
    ) -> list[ExperimentRecord]:
        """Load recent experiments for one strategy family."""

        return load_experiments_for_strategy(strategy_name, limit=limit, path=self.path)


def load_experiments(
    limit: int | None = None,
    path: str | Path = "multi_agent_trading_lab/experiments/experiment_memory.jsonl",
) -> list[ExperimentRecord]:
    """Load experiment records from a JSONL file.

    Older records are normalized into the current ExperimentRecord schema.
    """

    store_path = Path(path)
    if not store_path.exists():
        return []
    lines = store_path.read_text(encoding="utf-8").splitlines()
    records = [_normalize_record(json.loads(line)) for line in lines if line.strip()]
    if limit is None:
        return records
    return records[-limit:]


def load_experiments_for_strategy(
    strategy_name: str,
    limit: int | None = None,
    path: str | Path = "multi_agent_trading_lab/experiments/experiment_memory.jsonl",
) -> list[ExperimentRecord]:
    """Load experiments for a strategy name, newest limited last."""

    records = [record for record in load_experiments(path=path) if record.strategy_name == strategy_name]
    if limit is None:
        return records
    return records[-limit:]


def _normalize_record(payload: dict[str, Any]) -> ExperimentRecord:
    config = dict(payload.get("config", {}))
    strategy = dict(config.get("strategy", {}))
    strategy_name = str(payload.get("strategy_name") or strategy.get("name", "moving_average_crossover"))
    strategy_version = str(payload.get("strategy_version") or strategy.get("version", "0.1.0"))
    strategy_params = dict(payload.get("strategy_params") or payload.get("parameters") or _strategy_params(strategy))
    data_range = dict(
        payload.get("data_range")
        or {
            "symbols": config.get("symbols", []),
            "start_date": config.get("start_date"),
            "end_date": config.get("end_date"),
        }
    )
    return ExperimentRecord(
        experiment_id=str(payload.get("experiment_id") or payload.get("id") or uuid4()),
        timestamp=str(payload.get("timestamp") or datetime.now(UTC).isoformat()),
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        strategy_params=strategy_params,
        data_range=data_range,
        metrics=normalize_metrics(dict(payload.get("metrics", {}))),
        mode=str(payload.get("mode", "backtest")),
        risk_decision=dict(payload.get("risk_decision", {})),
        approval_status=str(payload.get("approval_status", "pending")),
        deployment_stage=str(payload.get("deployment_stage", "research")),
        config=config,
    )


def _strategy_params(strategy: dict[str, Any]) -> dict[str, Any]:
    if "strategy_params" in strategy:
        return dict(strategy["strategy_params"])
    return {
        key: value
        for key, value in strategy.items()
        if key in {"short_window", "long_window", "stop_loss_pct", "take_profit_pct"}
    }
