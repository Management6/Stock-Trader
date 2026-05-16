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

