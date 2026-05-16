"""Configurable objective scoring for strategy research."""

from __future__ import annotations

from typing import Any


DEFAULT_OBJECTIVE_CONFIG: dict[str, Any] = {
    "type": "sharpe",
    "sharpe": {
        "annualization_factor": 252,
        "min_history_points": 30,
        "fallback_return_weight": 1.0,
        "fallback_drawdown_penalty": 150.0,
    },
    "custom": {"return_weight": 1.0, "drawdown_penalty": 1.0},
}


def score_metrics(metrics: dict[str, Any], objective_config: dict[str, Any] | None = None) -> float:
    """Return one score for ranking experiments.

    Supported objective types:

    - ``sharpe``: use ``metrics["sharpe_ratio"]`` when enough history exists,
      otherwise fall back to a return/drawdown proxy.
    - ``return``: use total return only.
    - ``custom``: return-weighted total return minus an absolute drawdown
      penalty.

    Max drawdown is stored as a negative fraction, so drawdown penalty uses its
    absolute magnitude to avoid rewarding larger losses.
    """

    config = _merge_objective(objective_config or {})
    objective_type = str(config.get("type", "sharpe"))
    if objective_type == "return":
        return _number(metrics.get("total_return"))
    if objective_type == "custom":
        custom = dict(config.get("custom", {}))
        return _return_drawdown_score(
            metrics,
            return_weight=float(custom.get("return_weight", 1.0)),
            drawdown_penalty=float(custom.get("drawdown_penalty", 1.0)),
        )
    sharpe_config = dict(config.get("sharpe", {}))
    history_points = metrics.get("history_points")
    min_history_points = int(sharpe_config.get("min_history_points", 30))
    sharpe = metrics.get("sharpe_ratio", metrics.get("sharpe"))
    if _is_number(sharpe) and _is_number(history_points) and int(history_points) >= min_history_points:
        return float(sharpe)
    return _return_drawdown_score(
        metrics,
        return_weight=float(sharpe_config.get("fallback_return_weight", 1.0)),
        drawdown_penalty=float(sharpe_config.get("fallback_drawdown_penalty", 1.0)),
    )


def _return_drawdown_score(metrics: dict[str, Any], return_weight: float, drawdown_penalty: float) -> float:
    return return_weight * _number(metrics.get("total_return")) - drawdown_penalty * abs(_number(metrics.get("max_drawdown")))


def _merge_objective(config: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "type": DEFAULT_OBJECTIVE_CONFIG["type"],
        "sharpe": dict(DEFAULT_OBJECTIVE_CONFIG["sharpe"]),
        "custom": dict(DEFAULT_OBJECTIVE_CONFIG["custom"]),
    }
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def _number(value: Any) -> float:
    return float(value) if _is_number(value) else 0.0


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
