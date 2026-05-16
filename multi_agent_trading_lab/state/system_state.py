"""Readable JSON-backed state for operating mode and safety controls."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SystemStateStore:
    """Stores durable bot state such as kill switch and active stage."""

    def __init__(self, path: str | Path = "multi_agent_trading_lab/state/system_state.json") -> None:
        self.path = Path(path)

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "mode": "paper",
                "kill_switch_enabled": False,
                "circuit_breaker_enabled": False,
                "repeated_failures": 0,
                "last_updated": datetime.now(UTC).isoformat(),
                "notes": "Default safe state.",
            }
        return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = {**state, "last_updated": datetime.now(UTC).isoformat()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    def set_kill_switch(self, enabled: bool, reason: str) -> dict[str, Any]:
        state = self.read()
        state["kill_switch_enabled"] = enabled
        state["kill_switch_reason"] = reason
        return self.write(state)

    def record_repeated_failure(self, reason: str, limit: int | None = None) -> dict[str, Any]:
        state = self.read()
        failures = int(state.get("repeated_failures", 0)) + 1
        state["repeated_failures"] = failures
        state["last_repeated_failure_reason"] = reason
        if limit is not None and failures >= limit:
            state["circuit_breaker_enabled"] = True
            state["circuit_breaker_reason"] = f"Repeated failure limit reached: {failures}/{limit}"
        return self.write(state)

    def reset_repeated_failures(self) -> dict[str, Any]:
        state = self.read()
        state["repeated_failures"] = 0
        state["circuit_breaker_enabled"] = False
        state["last_repeated_failure_reason"] = None
        state["circuit_breaker_reason"] = None
        return self.write(state)
