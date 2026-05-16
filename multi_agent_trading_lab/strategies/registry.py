"""Strategy lifecycle registry."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multi_agent_trading_lab.strategies.base_strategy import canonical_strategy_id


VALID_STAGES = {"candidate", "active", "rejected", "archived"}


class StrategyRegistry:
    """Persists strategy lifecycle state for staged autonomy."""

    def __init__(self, path: str | Path = "multi_agent_trading_lab/strategies/strategy_registry.json") -> None:
        self.path = Path(path)

    def register(self, strategy: dict[str, Any], stage: str = "candidate", reason: str = "registered") -> dict[str, Any]:
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid strategy stage: {stage}")
        state = self._read()
        identifier = str(
            strategy.get("id")
            or canonical_strategy_id(
                str(strategy.get("name", "moving_average_crossover")),
                str(strategy.get("version", "0.1.0")),
                dict(strategy.get("strategy_params") or _extract_params(strategy)),
            )
        )
        strategy = {**strategy, "id": identifier}
        record = {
            "id": identifier,
            "stage": stage,
            "strategy": strategy,
            "reason": reason,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        state[identifier] = record
        self._write(state)
        return record

    def promote(self, strategy_id: str, stage: str, reason: str) -> dict[str, Any]:
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid strategy stage: {stage}")
        state = self._read()
        if strategy_id not in state:
            raise KeyError(f"Unknown strategy id: {strategy_id}")
        state[strategy_id]["stage"] = stage
        state[strategy_id]["reason"] = reason
        state[strategy_id]["updated_at"] = datetime.now(UTC).isoformat()
        self._write(state)
        return state[strategy_id]

    def list_by_stage(self, stage: str) -> list[dict[str, Any]]:
        return [record for record in self._read().values() if record["stage"] == stage]

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._read().values())

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        current = {}
        if self.path.exists():
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current = {}
        current.update(state)
        self.path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _extract_params(strategy: dict[str, Any]) -> dict[str, Any]:
    return {
        key: strategy[key]
        for key in ("short_window", "long_window", "stop_loss_pct", "take_profit_pct")
        if key in strategy
    }
