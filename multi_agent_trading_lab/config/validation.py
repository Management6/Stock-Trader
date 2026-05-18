"""Validation and resolution helpers for trading configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from multi_agent_trading_lab.data.universes import resolve_universe_symbols


@dataclass(frozen=True)
class ConfigValidationReport:
    """Structured config validation result."""

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    active_universe: str | None = None
    active_symbols: list[str] = field(default_factory=list)
    allowed_symbols: list[str] = field(default_factory=list)
    resolved: dict[str, Any] = field(default_factory=dict)


def normalize_trading_config(settings: dict[str, Any]) -> tuple[dict[str, Any], ConfigValidationReport]:
    """Resolve core research/paper assumptions into one explicit config shape.

    Precedence:
    1. ``data.default_universe`` selects ``universes.<name>`` when present.
    2. Otherwise ``data.symbols`` is the active universe.
    3. Legacy top-level ``symbols/start_date/end_date`` are only fallbacks when
       ``data`` is absent; if ``data`` exists, they are ignored with a warning.
    4. ``risk.allowed_symbols`` is the paper execution allowlist. If missing,
       it is derived from the active symbols. If present and different, it is
       preserved and reported as intentional allowlist precedence.
    """

    resolved = deepcopy(settings)
    warnings: list[str] = []
    errors: list[str] = []

    data_exists = bool(resolved.get("data"))
    if data_exists:
        for key in ("symbols", "start_date", "end_date"):
            if key in resolved:
                warnings.append(f"Ignoring legacy top-level {key}; data.{key} has precedence.")
    else:
        resolved["data"] = {
            "symbols": resolved.get("symbols", ["AAPL"]),
            "start_date": resolved.get("start_date", "2023-01-01"),
            "end_date": resolved.get("end_date"),
        }

    data_config = resolved.setdefault("data", {})
    active_universe = data_config.get("default_universe")
    try:
        active_symbols = resolve_universe_symbols(resolved)
    except KeyError as exc:
        active_symbols = []
        errors.append(str(exc))
    if not active_symbols:
        errors.append("Active universe must contain at least one symbol.")
    data_config["symbols"] = active_symbols
    data_config.setdefault("start_date", "2023-01-01")
    data_config.setdefault("end_date", None)

    risk_config = resolved.setdefault("risk", {})
    configured_allowed = _clean_symbols(risk_config.get("allowed_symbols"))
    if configured_allowed:
        allowed_symbols = configured_allowed
        active_set = {symbol.upper() for symbol in active_symbols}
        allowed_set = {symbol.upper() for symbol in allowed_symbols}
        if allowed_set != active_set:
            missing = sorted(active_set - allowed_set)
            extra = sorted(allowed_set - active_set)
            warnings.append(
                "risk.allowed_symbols does not match active universe; risk.allowed_symbols takes precedence for paper execution "
                f"(missing_from_allowlist={missing}, outside_active_universe={extra})."
            )
    else:
        allowed_symbols = list(active_symbols)
        risk_config["allowed_symbols"] = allowed_symbols
        warnings.append("risk.allowed_symbols was empty and was derived from active universe.")

    _validate_oos(resolved.get("oos_validation", {}), errors)
    _validate_backtest_costs(resolved.get("backtest", {}), errors)
    _validate_optimizer(resolved.get("optimizer", {}), errors, warnings)

    return resolved, ConfigValidationReport(
        warnings=warnings,
        errors=errors,
        active_universe=str(active_universe) if active_universe else None,
        active_symbols=active_symbols,
        allowed_symbols=allowed_symbols,
        resolved=resolved,
    )


def validate_trading_config(settings: dict[str, Any]) -> ConfigValidationReport:
    """Validate trading config without requiring callers to use the resolved copy."""

    _, report = normalize_trading_config(settings)
    return report


def _validate_oos(config: dict[str, Any], errors: list[str]) -> None:
    required = ("split_ratio", "min_sharpe", "max_drawdown", "min_return")
    for key in required:
        if key not in config:
            errors.append(f"oos_validation.{key} is required.")
    if "split_ratio" in config:
        split_ratio = float(config.get("split_ratio", 0.0))
        if split_ratio <= 0.0 or split_ratio >= 1.0:
            errors.append("oos_validation.split_ratio must be between 0 and 1.")
    if "max_drawdown" in config and float(config.get("max_drawdown", 0.0)) > 0.0:
        errors.append("oos_validation.max_drawdown must be zero or negative.")


def _validate_backtest_costs(config: dict[str, Any], errors: list[str]) -> None:
    if "commission_per_trade" not in config:
        errors.append("backtest.commission_per_trade is required.")
    elif float(config.get("commission_per_trade", 0.0)) < 0.0:
        errors.append("backtest.commission_per_trade must be non-negative.")
    if "slippage_pct" not in config:
        errors.append("backtest.slippage_pct is required.")
    elif float(config.get("slippage_pct", 0.0)) < 0.0:
        errors.append("backtest.slippage_pct must be non-negative.")


def _validate_optimizer(config: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    if not config:
        return

    expected_keys = {"enabled", "method", "seed", "n_trials"}
    for key in sorted(set(config) - expected_keys):
        warnings.append(f"optimizer.{key} is not used and should be removed.")

    if "enabled" in config and not isinstance(config["enabled"], bool):
        errors.append("optimizer.enabled must be a boolean.")
    if "method" in config and config["method"] != "deterministic_random":
        errors.append("optimizer.method must be deterministic_random.")
    if "seed" in config and (not isinstance(config["seed"], int) or isinstance(config["seed"], bool)):
        errors.append("optimizer.seed must be an integer.")
    if "n_trials" in config:
        n_trials = config["n_trials"]
        if not isinstance(n_trials, int) or isinstance(n_trials, bool) or n_trials <= 0:
            errors.append("optimizer.n_trials must be a positive integer.")


def _clean_symbols(symbols: Any) -> list[str]:
    if not symbols:
        return []
    cleaned: list[str] = []
    for symbol in symbols:
        text = str(symbol).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned
