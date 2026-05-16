"""Controlled auto-research loop for finding good backtested stock strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from multi_agent_trading_lab.experiments.experiment_logger import ExperimentRecord
from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator, load_settings
from multi_agent_trading_lab.strategies.base_strategy import canonical_strategy_id


@dataclass(frozen=True)
class GoodStrategyThresholds:
    min_sharpe: float = 0.70
    min_drawdown: float = -0.15


@dataclass(frozen=True)
class AutoResearchConfig:
    max_iterations: int = 200
    target_good_strategies: int = 10
    thresholds: GoodStrategyThresholds = GoodStrategyThresholds()
    n_variants_per_iteration: int | None = None
    strategy_families: list[str] | None = None
    universe: str | None = None
    settings_path: str = "multi_agent_trading_lab/config/settings.yaml"


@dataclass(frozen=True)
class GoodStrategy:
    strategy_id: str
    strategy_name: str
    strategy_version: str
    strategy_params: dict[str, Any]
    metrics: dict[str, Any]
    experiment_id: str


@dataclass(frozen=True)
class AutoResearchResult:
    iterations_run: int
    target_good_strategies: int
    max_iterations: int
    thresholds: GoodStrategyThresholds
    stop_reason: str
    good_strategies: list[GoodStrategy]
    best_strategy: GoodStrategy | None


ProgressCallback = Callable[[int, list[GoodStrategy], GoodStrategy | None], None]
RunIteration = Callable[[int], Iterable[ExperimentRecord | dict[str, Any]]]


def run_auto_research(
    config: AutoResearchConfig | None = None,
    run_iteration: RunIteration | None = None,
    on_progress: ProgressCallback | None = None,
) -> AutoResearchResult:
    """Run research iterations until enough distinct good strategies are found."""

    config = config or AutoResearchConfig()
    if config.max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")
    if config.target_good_strategies <= 0:
        raise ValueError("target_good_strategies must be positive.")
    runner = run_iteration or _default_iteration_runner(config)
    good_by_key: dict[str, GoodStrategy] = {}
    best: GoodStrategy | None = None
    iterations_run = 0

    for iteration in range(1, config.max_iterations + 1):
        iterations_run = iteration
        records = [_coerce_record(record) for record in runner(iteration)]
        for candidate in collect_good_strategies(records, config.thresholds):
            existing = good_by_key.get(candidate.strategy_id)
            if existing is None or _sharpe(candidate) > _sharpe(existing):
                good_by_key[candidate.strategy_id] = candidate
        best = _best_strategy([*records], current_best=best)
        if on_progress is not None:
            on_progress(iteration, sorted_good_strategies(good_by_key.values()), best)
        if len(good_by_key) >= config.target_good_strategies:
            return AutoResearchResult(
                iterations_run=iterations_run,
                target_good_strategies=config.target_good_strategies,
                max_iterations=config.max_iterations,
                thresholds=config.thresholds,
                stop_reason="target_reached",
                good_strategies=sorted_good_strategies(good_by_key.values()),
                best_strategy=best,
            )

    return AutoResearchResult(
        iterations_run=iterations_run,
        target_good_strategies=config.target_good_strategies,
        max_iterations=config.max_iterations,
        thresholds=config.thresholds,
        stop_reason="iteration_cap",
        good_strategies=sorted_good_strategies(good_by_key.values()),
        best_strategy=best,
    )


def collect_good_strategies(
    records: Iterable[ExperimentRecord | dict[str, Any]],
    thresholds: GoodStrategyThresholds | None = None,
) -> list[GoodStrategy]:
    """Return distinct records meeting Sharpe and drawdown thresholds."""

    thresholds = thresholds or GoodStrategyThresholds()
    good_by_id: dict[str, GoodStrategy] = {}
    for raw in records:
        record = _coerce_record(raw)
        sharpe = _metric(record, "sharpe_ratio", fallback="sharpe")
        drawdown = _metric(record, "max_drawdown")
        if sharpe is None or drawdown is None:
            continue
        if sharpe < thresholds.min_sharpe:
            continue
        if drawdown < thresholds.min_drawdown:
            continue
        candidate = _good_strategy_from_record(record)
        existing = good_by_id.get(candidate.strategy_id)
        if existing is None or _sharpe(candidate) > _sharpe(existing):
            good_by_id[candidate.strategy_id] = candidate
    return sorted_good_strategies(good_by_id.values())


def sorted_good_strategies(strategies: Iterable[GoodStrategy]) -> list[GoodStrategy]:
    return sorted(strategies, key=lambda strategy: (_sharpe(strategy), _total_return(strategy)), reverse=True)


def _default_iteration_runner(config: AutoResearchConfig) -> RunIteration:
    settings = load_settings(config.settings_path)
    if config.universe:
        settings.setdefault("data", {})["default_universe"] = config.universe
    orchestrator = TradingLabOrchestrator(settings)

    def run_iteration(_: int) -> Iterable[ExperimentRecord | dict[str, Any]]:
        result = orchestrator.run_research_cycle(n_variants=config.n_variants_per_iteration, strategy_families=config.strategy_families)
        return result.details.get("records", [])

    return run_iteration


def _coerce_record(record: ExperimentRecord | dict[str, Any]) -> ExperimentRecord:
    if isinstance(record, ExperimentRecord):
        return record
    payload = dict(record)
    config = dict(payload.get("config", {}))
    strategy = dict(config.get("strategy", {}))
    return ExperimentRecord(
        experiment_id=str(payload.get("experiment_id") or payload.get("id")),
        timestamp=str(payload.get("timestamp", "")),
        strategy_name=str(payload.get("strategy_name") or strategy.get("name", "moving_average_crossover")),
        strategy_version=str(payload.get("strategy_version") or strategy.get("version", "0.1.0")),
        strategy_params=dict(payload.get("strategy_params") or payload.get("parameters") or strategy.get("strategy_params", {})),
        data_range=dict(payload.get("data_range", {})),
        metrics=dict(payload.get("metrics", {})),
        mode=str(payload.get("mode", "backtest")),
        risk_decision=dict(payload.get("risk_decision", {})),
        approval_status=str(payload.get("approval_status", "pending")),
        deployment_stage=str(payload.get("deployment_stage", "research")),
        config=config,
    )


def _good_strategy_from_record(record: ExperimentRecord) -> GoodStrategy:
    strategy_id = _strategy_id(record)
    return GoodStrategy(
        strategy_id=strategy_id,
        strategy_name=record.strategy_name,
        strategy_version=record.strategy_version,
        strategy_params=dict(record.strategy_params),
        metrics=dict(record.metrics),
        experiment_id=record.experiment_id,
    )


def _strategy_id(record: ExperimentRecord) -> str:
    strategy = dict(record.config.get("strategy", {})) if record.config else {}
    if strategy.get("id"):
        return str(strategy["id"])
    return canonical_strategy_id(record.strategy_name, record.strategy_version, record.strategy_params)


def _best_strategy(records: list[ExperimentRecord], current_best: GoodStrategy | None) -> GoodStrategy | None:
    candidates = [_good_strategy_from_record(record) for record in records]
    if current_best is not None:
        candidates.append(current_best)
    if not candidates:
        return None
    return max(candidates, key=lambda strategy: (_sharpe(strategy), _total_return(strategy)))


def _metric(record: ExperimentRecord, key: str, fallback: str | None = None) -> float | None:
    value = record.metrics.get(key)
    if value is None and fallback is not None:
        value = record.metrics.get(fallback)
    if value is None:
        return None
    return float(value)


def _sharpe(strategy: GoodStrategy) -> float:
    return float(strategy.metrics.get("sharpe_ratio", strategy.metrics.get("sharpe", 0.0)) or 0.0)


def _total_return(strategy: GoodStrategy) -> float:
    return float(strategy.metrics.get("total_return", 0.0) or 0.0)
