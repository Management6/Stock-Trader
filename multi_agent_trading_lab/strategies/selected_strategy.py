"""Selected strategy config and paper-mode validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multi_agent_trading_lab.experiments.analysis import find_experiments_by_params
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy
from multi_agent_trading_lab.strategies.base_strategy import canonical_strategy_id


@dataclass(frozen=True)
class SelectedStrategy:
    name: str
    version: str
    params: dict[str, Any]

    def to_strategy_config(self) -> dict[str, Any]:
        strategy_name = _normalize_strategy_name(self.name)
        return {
            "id": canonical_strategy_id(strategy_name, self.version, self.params),
            "name": strategy_name,
            "version": self.version,
            "strategy_params": dict(self.params),
            "short_window": int(self.params["short_window"]),
            "long_window": int(self.params["long_window"]),
            "deployment_stage": "paper_candidate",
        }


def load_selected_strategy_config(
    path: str | Path = "multi_agent_trading_lab/config/selected_strategy.yaml",
) -> SelectedStrategy | None:
    """Load selected strategy config, returning None when no file exists."""

    config_path = Path(path)
    if not config_path.exists():
        return None
    payload = _load_selected_yaml(config_path)
    selected = payload.get("selected_strategy", payload)
    if not selected:
        return None
    return SelectedStrategy(
        name=str(selected.get("name", "moving_average_crossover")),
        version=str(selected.get("version", "0.1.0")),
        params=dict(selected.get("params", {})),
    )


def evaluate_selected_strategy_for_paper(
    selected: SelectedStrategy | dict[str, Any] | None,
    risk_policy: RiskPolicy,
) -> dict[str, Any]:
    """Validate a selected strategy against recent experiment metrics and risk policy."""

    risk_limits = {"paper_max_drawdown": float(risk_policy.paper_max_drawdown)}
    selected_strategy = _coerce_selected_strategy(selected)
    if selected_strategy is None:
        return {"approved": False, "reason": "No selected strategy configured.", "strategy_config": None, "risk_limits": risk_limits}
    params = selected_strategy.params
    matches = find_experiments_by_params(
        _normalize_strategy_name(selected_strategy.name),
        short_window=int(params["short_window"]),
        long_window=int(params["long_window"]),
    )
    if not matches:
        return {
            "approved": False,
            "reason": "No matching experiment history found for selected strategy.",
            "strategy_config": selected_strategy.to_strategy_config(),
            "risk_limits": risk_limits,
        }
    best = max(matches, key=lambda record: float(record.metrics.get("sharpe_ratio", record.metrics.get("sharpe", 0.0)) or 0.0))
    decision = risk_policy.evaluate_strategy_for_paper(best.metrics)
    return {
        "approved": decision.approved,
        "reason": decision.reason,
        "strategy_config": selected_strategy.to_strategy_config(),
        "experiment_id": best.experiment_id,
        "metrics": best.metrics,
        "risk_limits": risk_limits,
    }


def _coerce_selected_strategy(selected: SelectedStrategy | dict[str, Any] | None) -> SelectedStrategy | None:
    if selected is None:
        return None
    if isinstance(selected, SelectedStrategy):
        return selected
    return SelectedStrategy(
        name=str(selected.get("name", "moving_average_crossover")),
        version=str(selected.get("version", "0.1.0")),
        params=dict(selected.get("params", {})),
    )


def _normalize_strategy_name(name: str) -> str:
    return "moving_average_crossover" if name in {"example_ma", "ma", "moving_average_crossover"} else name


def _load_selected_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        return _parse_simple_selected_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_simple_selected_yaml(path: Path) -> dict[str, Any]:
    """Parse the small selected_strategy.yaml shape without external deps."""

    selected: dict[str, Any] = {"params": {}}
    in_selected = False
    in_params = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped == "selected_strategy:":
            in_selected = True
            continue
        if not in_selected:
            continue
        if stripped == "params:":
            in_params = True
            continue
        key, _, value = stripped.partition(":")
        if not key or not value:
            continue
        parsed = _parse_scalar(value.strip())
        if in_params:
            selected["params"][key.strip()] = parsed
        else:
            selected[key.strip()] = parsed
    return {"selected_strategy": selected}


def _parse_scalar(value: str) -> Any:
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value
