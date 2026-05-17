"""Strategy health and quarantine decisions from paper attribution."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable

from multi_agent_trading_lab.state.system_state import SystemStateStore


DEFAULT_STRATEGY_HEALTH_CONFIG = {
    "enabled": False,
    "quarantine_on_expectation_mismatch": True,
    "max_rejection_rate": 0.50,
    "repeated_rejection_threshold": 3,
    "max_paper_drawdown": -0.10,
    "min_signals_before_quarantine": 5,
    "allow_manual_reactivation": True,
}


def evaluate_strategy_health(
    attribution: dict[str, Any],
    config: dict[str, Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Evaluate paper-trading strategy health from attribution diagnostics."""

    settings = {**DEFAULT_STRATEGY_HEALTH_CONFIG, **(config or {})}
    now = (clock or (lambda: datetime.now(UTC)))().isoformat()
    decisions = {
        strategy_id: _decision_for_strategy(strategy_id, payload, attribution, settings, now)
        for strategy_id, payload in dict(attribution.get("by_strategy", {})).items()
    }
    return {"enabled": bool(settings.get("enabled", False)), "decisions": decisions, "timestamp": now}


def reactivate_strategy_health(
    state_store: SystemStateStore,
    strategy_id: str,
    operator: str,
    reason: str,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Manually restore a quarantined strategy to active paper health."""

    now = (clock or (lambda: datetime.now(UTC)))().isoformat()
    state = state_store.read()
    health = dict(state.get("strategy_health", {}))
    existing = dict(health.get(strategy_id, {"strategy_id": strategy_id}))
    existing.update(
        {
            "strategy_id": strategy_id,
            "status": "active",
            "reasons": ["Manually reactivated after operator review."],
            "timestamp": now,
            "recommended_action": "Resume paper eligibility and continue monitoring.",
            "manual_override": {"operator": operator, "reason": reason, "timestamp": now},
        }
    )
    health[strategy_id] = existing
    state["strategy_health"] = health
    return state_store.write(state)


def _decision_for_strategy(
    strategy_id: str,
    payload: dict[str, Any],
    attribution: dict[str, Any],
    config: dict[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    signals = int(payload.get("signals", 0) or 0)
    accepted = int(payload.get("accepted_orders", 0) or 0)
    rejected = int(payload.get("rejected_or_skipped_orders", 0) or 0)
    pnl = float(payload.get("pnl", 0.0) or 0.0)
    rejection_rate = rejected / signals if signals else 0.0
    reasons: list[str] = []
    expected = dict(dict(attribution.get("expectation_vs_actual", {})).get(strategy_id, {}))
    expected_oos = _maybe_float(expected.get("expected_oos_return"))
    if bool(config.get("quarantine_on_expectation_mismatch", True)) and expected_oos is not None and expected_oos > 0 and pnl < 0:
        reasons.append("Strategy has positive expected OOS return but negative paper PnL.")
    max_rejection_rate = float(config.get("max_rejection_rate", 0.50))
    if rejection_rate > max_rejection_rate:
        reasons.append(f"Strategy rejection rate {rejection_rate:.2%} exceeds threshold {max_rejection_rate:.2%}.")
    repeated_threshold = int(config.get("repeated_rejection_threshold", 3))
    for reason, count in dict(payload.get("rejection_reasons", {})).items():
        if int(count) >= repeated_threshold:
            reasons.append(f"Strategy rejection reason '{reason}' repeated {int(count)} times.")
    paper_drawdown = _maybe_float(payload.get("max_drawdown"))
    max_drawdown = float(config.get("max_paper_drawdown", -0.10))
    if paper_drawdown is not None and paper_drawdown < max_drawdown:
        reasons.append(f"Strategy paper drawdown {paper_drawdown:.2f} exceeds threshold {max_drawdown:.2f}.")
    if signals >= int(config.get("min_signals_before_quarantine", 5)) and accepted == 0:
        reasons.append(f"Strategy has no accepted orders across {signals} signal(s).")

    if not reasons:
        status = "active"
        reasons = ["Paper attribution is within configured health limits."]
        action = "Continue paper trading and monitoring."
    elif signals >= int(config.get("min_signals_before_quarantine", 5)):
        status = "quarantined"
        action = "Pause paper trading for this strategy until operator review."
    else:
        status = "needs_review"
        action = "Collect more paper observations before quarantine; review current diagnostics."

    return {
        "strategy_id": strategy_id,
        "status": status,
        "reasons": reasons,
        "evidence": {
            "signals": signals,
            "accepted_orders": accepted,
            "rejected_or_skipped_orders": rejected,
            "rejection_rate": rejection_rate,
            "actual_pnl": pnl,
            "expected_oos_return": expected_oos,
            "max_paper_drawdown": paper_drawdown,
        },
        "timestamp": timestamp,
        "recommended_action": action,
    }


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
