"""Base strategy contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from multi_agent_trading_lab.data.data_sources import PriceBar


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    version: str = "0.1.0"
    parameters: dict[str, Any] = field(default_factory=dict)
    supported_symbols: list[str] = field(default_factory=list)
    supported_timeframes: list[str] = field(default_factory=lambda: ["1d"])
    description: str = ""


class BaseStrategy(ABC):
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    @property
    def metadata(self) -> dict[str, Any]:
        """Return strategy metadata for registry and audit records."""

        return {
            "name": self.config.name,
            "version": self.config.version,
            "parameters": self.config.parameters,
            "supported_symbols": self.config.supported_symbols,
            "supported_timeframes": self.config.supported_timeframes,
            "description": self.config.description,
        }

    @property
    def identifier(self) -> str:
        return canonical_strategy_id(self.config.name, self.config.version, self.config.parameters)

    def get_parameters(self) -> dict[str, Any]:
        """Return machine-readable strategy hyperparameters."""

        return dict(self.config.parameters)

    def to_config(self) -> dict[str, Any]:
        """Return a serializable strategy config for backtests and logging."""

        return {
            "id": self.identifier,
            "name": self.config.name,
            "version": self.config.version,
            "strategy_params": self.get_parameters(),
            "supported_symbols": list(self.config.supported_symbols),
            "supported_timeframes": list(self.config.supported_timeframes),
            "description": self.config.description,
        }

    @abstractmethod
    def generate_signal(self, bar: PriceBar) -> int:
        """Return ``1`` for long exposure or ``0`` for flat."""


def canonical_strategy_id(name: str, version: str, parameters: dict[str, Any] | None = None) -> str:
    """Build a stable identifier from strategy name, version, and key parameters."""

    slug = name.replace("moving_average_crossover", "example_ma").replace(":", "_").replace(" ", "_")
    slug = slug.replace("moving_average_rsi_filter", "ma_rsi")
    version_slug = version.replace(".", "_")
    params = parameters or {}
    breakout_window = params.get("breakout_window")
    exit_window = params.get("exit_window")
    if breakout_window is not None and exit_window is not None:
        return f"{slug}_v{version_slug}_b{breakout_window}_e{exit_window}"
    short_window = params.get("short_window")
    long_window = params.get("long_window")
    if short_window is not None and long_window is not None:
        rsi_period = params.get("rsi_period")
        rsi_min = params.get("rsi_min")
        rsi_max = params.get("rsi_max")
        if rsi_period is not None and rsi_min is not None and rsi_max is not None:
            return f"{slug}_v{version_slug}_s{short_window}_l{long_window}_r{rsi_period}_{_format_param(rsi_min)}_{_format_param(rsi_max)}"
        return f"{slug}_v{version_slug}_s{short_window}_l{long_window}"
    return f"{slug}_v{version_slug}"


def _format_param(value: Any) -> str:
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number).replace(".", "p")
