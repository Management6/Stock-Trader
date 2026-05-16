"""Append-only JSONL audit log for safety-relevant events."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4


class AuditLog:
    """Records strategy promotions, order decisions, and broker actions."""

    def __init__(
        self,
        path: str | Path = "multi_agent_trading_lab/experiments/audit_log.jsonl",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path)
        self.clock = clock or (lambda: datetime.now(UTC))

    def record(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "id": str(uuid4()),
            "timestamp": self.clock().isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return records[-limit:]
