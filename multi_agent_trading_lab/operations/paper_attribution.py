"""Paper-trading performance attribution helpers."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def build_paper_attribution(
    orders: list[dict[str, Any]] | dict[str, dict[str, Any]],
    audit_records: list[dict[str, Any]],
    expected_metrics_by_strategy: dict[str, dict[str, Any]] | None = None,
    starting_equity: float | None = None,
    ending_equity: float | None = None,
    high_rejection_rate_threshold: float = 0.50,
) -> dict[str, Any]:
    """Build deterministic strategy/symbol attribution from paper state and audit logs."""

    order_list = list(orders.values()) if isinstance(orders, dict) else list(orders)
    expected = expected_metrics_by_strategy or {}
    by_strategy: dict[str, dict[str, Any]] = defaultdict(_bucket)
    by_symbol: dict[str, dict[str, Any]] = defaultdict(_bucket)
    rejection_breakdown: Counter[str] = Counter()
    warnings: list[str] = []
    flags: set[str] = set()

    for order in order_list:
        strategy = str(order.get("source_strategy") or order.get("strategy_id") or "unknown")
        symbol = str(order.get("symbol") or "unknown")
        status = str(order.get("status", "unknown"))
        pnl = _number(order.get("realized_pnl", order.get("pnl", 0.0)))
        _add_order(by_strategy[strategy], status, pnl)
        _add_order(by_symbol[symbol], status, pnl)
        if status in {"rejected", "skipped"}:
            reason = str(order.get("reason") or order.get("rejection_reason") or "unknown")
            rejection_breakdown[reason] += 1
            reasons = by_strategy[strategy]["rejection_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1

    for record in audit_records:
        if record.get("event_type") not in {"rejected_trade", "risk_rejected", "validation_rejected"}:
            continue
        payload = dict(record.get("payload", {}))
        reason = str(payload.get("reason") or record.get("reason") or "unknown")
        rejection_breakdown[reason] += 1
        order = dict(payload.get("order", {}))
        strategy = order.get("source_strategy") or payload.get("strategy_id")
        if strategy:
            bucket = by_strategy[str(strategy)]
            reasons = bucket["rejection_reasons"]
            reasons[reason] = reasons.get(reason, 0) + 1

    total_pnl = sum(bucket["pnl"] for bucket in by_strategy.values())
    accepted_orders = sum(1 for order in order_list if str(order.get("status")) in {"filled", "accepted"})
    rejected_or_skipped = sum(1 for order in order_list if str(order.get("status")) in {"rejected", "skipped"})
    signal_count = len(order_list)
    rejection_rate = rejected_or_skipped / signal_count if signal_count else 0.0
    if signal_count == 0:
        warnings.append("No paper trades recorded yet.")
    if rejection_rate > high_rejection_rate_threshold:
        warnings.append(f"High rejection rate: {rejection_rate:.2%}.")
        flags.add("high_rejection_rate")
    for reason, count in rejection_breakdown.items():
        if count >= 2:
            warnings.append(f"Repeated rejection reason: {reason}.")
            if "risk" in reason.lower():
                flags.add("repeated_risk_policy_rejections")
            if "data" in reason.lower():
                flags.add("repeated_data_quality_rejections")

    expectation_vs_actual = _expectation_vs_actual(expected, by_strategy, warnings, flags)
    return {
        "summary": {
            "total_pnl": total_pnl,
            "total_return": _total_return(starting_equity, ending_equity, total_pnl),
            "signals": signal_count,
            "accepted_orders": accepted_orders,
            "rejected_or_skipped_orders": rejected_or_skipped,
            "rejection_rate": rejection_rate,
        },
        "by_strategy": dict(by_strategy),
        "by_symbol": dict(by_symbol),
        "rejection_breakdown": dict(rejection_breakdown),
        "expectation_vs_actual": expectation_vs_actual,
        "warnings": warnings,
        "recommendations": _recommendations(flags),
        "diagnostic_flags": sorted(flags),
    }


def expected_metrics_from_audit(audit_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Extract promotion metrics keyed by strategy id from audit records."""

    expected: dict[str, dict[str, Any]] = {}
    for record in audit_records:
        if record.get("event_type") != "strategy_promotion_requested":
            continue
        payload = dict(record.get("payload", {}))
        strategy_id = payload.get("strategy_id") or dict(payload.get("strategy", {})).get("id")
        if strategy_id:
            expected[str(strategy_id)] = dict(payload.get("metrics", {}))
    return expected


def _bucket() -> dict[str, Any]:
    return {"pnl": 0.0, "signals": 0, "accepted_orders": 0, "rejected_or_skipped_orders": 0, "rejection_reasons": {}}


def _add_order(bucket: dict[str, Any], status: str, pnl: float) -> None:
    bucket["signals"] += 1
    bucket["pnl"] += pnl
    if status in {"filled", "accepted"}:
        bucket["accepted_orders"] += 1
    if status in {"rejected", "skipped"}:
        bucket["rejected_or_skipped_orders"] += 1


def _expectation_vs_actual(
    expected: dict[str, dict[str, Any]],
    by_strategy: dict[str, dict[str, Any]],
    warnings: list[str],
    flags: set[str],
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    for strategy, metrics in expected.items():
        actual = by_strategy.get(strategy, _bucket())
        expected_oos = _expected_oos_return(metrics)
        actual_pnl = float(actual["pnl"])
        comparison[strategy] = {
            "expected_total_return": _maybe_number(metrics.get("total_return")),
            "expected_oos_return": expected_oos,
            "actual_pnl": actual_pnl,
            "actual_signals": int(actual["signals"]),
        }
        if expected_oos is not None and expected_oos > 0 and actual_pnl < 0:
            warnings.append(f"{strategy} has positive expected OOS return but negative paper PnL.")
            flags.add("expectation_mismatch")
    return comparison


def _expected_oos_return(metrics: dict[str, Any]) -> float | None:
    if "oos_total_return" in metrics:
        return _maybe_number(metrics.get("oos_total_return"))
    walk_forward = metrics.get("walk_forward")
    if isinstance(walk_forward, dict):
        return _maybe_number(dict(dict(walk_forward.get("out_of_sample", {})).get("metrics", {})).get("total_return"))
    return None


def _recommendations(flags: set[str]) -> list[str]:
    recommendations = []
    if "expectation_mismatch" in flags:
        recommendations.append("Review paper fills, slippage, and current market regime before trusting expected research edge.")
    if "high_rejection_rate" in flags:
        recommendations.append("Inspect risk-policy and data-quality rejection reasons before adding capital or symbols.")
    return recommendations or ["Continue collecting paper-trading observations."]


def _total_return(starting_equity: float | None, ending_equity: float | None, total_pnl: float) -> float | None:
    if starting_equity:
        if ending_equity is not None:
            return (ending_equity - starting_equity) / starting_equity
        return total_pnl / starting_equity
    return None


def _number(value: Any) -> float:
    parsed = _maybe_number(value)
    return 0.0 if parsed is None else parsed


def _maybe_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
