"""Expanded deterministic approval-profile search reporting."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multi_agent_trading_lab.operations.strict_profile_validation import _validation_settings
from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator, load_settings


DEFAULT_APPROVAL_SEARCH_FAMILIES = ["moving_average_crossover", "moving_average_rsi_filter", "breakout_trend"]


@dataclass(frozen=True)
class ApprovalSearchSummary:
    passed: bool
    output_dir: Path
    families: list[str]
    trials_per_family: int
    seed: int
    total_trials: int
    accepted_count: int
    rejected_count: int
    trials_by_family: dict[str, int]
    accepted_by_family: dict[str, int]
    rejected_by_family: dict[str, int]
    rejection_breakdown: dict[str, int]
    rejection_breakdown_by_family: dict[str, dict[str, int]]
    best_candidate_by_family: dict[str, dict[str, Any] | None]
    best_near_miss: dict[str, Any] | None
    approval_queue_path: Path
    experiment_log_path: Path
    audit_log_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("output_dir", "approval_queue_path", "experiment_log_path", "audit_log_path"):
            payload[key] = str(payload[key])
        return payload


def run_approval_search(
    settings_path: str | Path = "multi_agent_trading_lab/config/settings.approval_paper.yaml",
    trials: int = 100,
    seed: int = 42,
    output_root: str | Path = "multi_agent_trading_lab/validation_runs",
    families: list[str] | None = None,
) -> ApprovalSearchSummary:
    """Run an expanded approval-profile search and write an operator summary."""

    selected_families = _canonical_families(families or list(DEFAULT_APPROVAL_SEARCH_FAMILIES))
    output_dir = Path(output_root) / f"approval_search_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    settings = _approval_search_settings(load_settings(settings_path), output_dir, trials, seed, selected_families)
    orchestrator = TradingLabOrchestrator(settings)
    result = orchestrator.run_research_cycle(n_variants=trials, strategy_families=selected_families)
    records = list(result.details.get("records", []))
    summary = _build_summary(records, output_dir, selected_families, trials, seed, settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "approval_search_summary.json").write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _approval_search_settings(
    settings: dict[str, Any],
    output_dir: Path,
    trials: int,
    seed: int,
    families: list[str],
) -> dict[str, Any]:
    payload = _validation_settings(settings, output_dir)
    payload["strategy_families"] = list(families)
    payload["optimizer"] = {**dict(payload.get("optimizer", {})), "enabled": True, "method": "deterministic_random", "seed": int(seed), "n_trials": int(trials)}
    payload["research"] = {
        **dict(payload.get("research", {})),
        "variants_per_cycle": int(trials),
        "max_experiments_per_run": int(trials) * max(1, len(families)),
    }
    payload.setdefault("broker", {})
    payload["broker"]["live_enabled"] = False
    payload["operating_mode"] = "paper"
    payload["execution"] = {**dict(payload.get("execution", {})), "mode": "paper"}
    return payload


def _canonical_families(families: list[str]) -> list[str]:
    aliases = {
        "ma": "moving_average_crossover",
        "moving_average": "moving_average_crossover",
        "ma_rsi": "moving_average_rsi_filter",
        "moving_average_rsi": "moving_average_rsi_filter",
        "breakout": "breakout_trend",
    }
    resolved: list[str] = []
    for family in families:
        canonical = aliases.get(str(family), str(family))
        if canonical not in resolved:
            resolved.append(canonical)
    return resolved


def _build_summary(
    records: list[dict[str, Any]],
    output_dir: Path,
    families: list[str],
    trials: int,
    seed: int,
    settings: dict[str, Any],
) -> ApprovalSearchSummary:
    trials_by_family: Counter[str] = Counter()
    accepted_by_family: Counter[str] = Counter()
    rejected_by_family: Counter[str] = Counter()
    rejection_breakdown: Counter[str] = Counter()
    rejection_by_family: dict[str, Counter[str]] = defaultdict(Counter)
    best_by_family: dict[str, dict[str, Any] | None] = {family: None for family in families}
    near_miss: dict[str, Any] | None = None
    for record in records:
        family = str(record.get("strategy_name", "unknown"))
        trials_by_family[family] += 1
        approved = bool(record.get("risk_decision", {}).get("approved"))
        candidate = _candidate_summary(record)
        if approved:
            accepted_by_family[family] += 1
        else:
            rejected_by_family[family] += 1
            reason = str(record.get("risk_decision", {}).get("reason", "unknown"))
            rejection_breakdown[reason] += 1
            rejection_by_family[family][reason] += 1
            if near_miss is None or _score(candidate) > _score(near_miss):
                near_miss = candidate
        current = best_by_family.get(family)
        if current is None or _score(candidate) > _score(current):
            best_by_family[family] = candidate
    return ApprovalSearchSummary(
        passed=True,
        output_dir=output_dir,
        families=families,
        trials_per_family=trials,
        seed=seed,
        total_trials=len(records),
        accepted_count=sum(accepted_by_family.values()),
        rejected_count=sum(rejected_by_family.values()),
        trials_by_family=dict(trials_by_family),
        accepted_by_family={family: accepted_by_family.get(family, 0) for family in families},
        rejected_by_family={family: rejected_by_family.get(family, 0) for family in families},
        rejection_breakdown=dict(rejection_breakdown),
        rejection_breakdown_by_family={family: dict(rejection_by_family.get(family, Counter())) for family in families},
        best_candidate_by_family=best_by_family,
        best_near_miss=near_miss,
        approval_queue_path=Path(settings["approval_queue_path"]),
        experiment_log_path=Path(settings["experiment_log_path"]),
        audit_log_path=Path(settings["audit_log_path"]),
    )


def _candidate_summary(record: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(record.get("metrics", {}))
    risk = dict(record.get("risk_decision", {}))
    optimizer_trial = dict(metrics.get("optimizer_trial", {}))
    return {
        "strategy_id": record.get("config", {}).get("strategy", {}).get("id"),
        "family": record.get("strategy_name"),
        "trial_number": optimizer_trial.get("trial_number"),
        "gate_outcome": optimizer_trial.get("gate_outcome"),
        "approved": bool(risk.get("approved")),
        "reason": risk.get("reason"),
        "objective_score": optimizer_trial.get("objective_score", record.get("objective_score")),
        "total_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "oos": _nested_metrics(metrics.get("walk_forward"), "out_of_sample"),
        "portfolio": dict(metrics.get("portfolio_backtest", {})).get("metrics"),
        "regime": dict(metrics.get("market_regime", {})).get("regime"),
    }


def _nested_metrics(payload: Any, key: str) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    nested = payload.get(key)
    if not isinstance(nested, dict):
        return None
    metrics = nested.get("metrics")
    return dict(metrics) if isinstance(metrics, dict) else None


def _score(candidate: dict[str, Any]) -> float:
    value = candidate.get("objective_score")
    if value is None:
        value = candidate.get("sharpe_ratio")
    return float(value) if value is not None else float("-inf")
