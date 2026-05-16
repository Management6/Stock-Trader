"""Lightweight alert routing for paper-trading operations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol
from uuid import uuid4


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Alert:
    id: str
    timestamp: str
    event_type: str
    severity: AlertSeverity
    message: str
    strategy_id: str | None = None
    symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.value
        return payload


class AlertSink(Protocol):
    def send(self, alert: Alert) -> None:
        ...


class ConsoleAlertSink:
    """Prints concise alerts for terminal-driven operations."""

    def send(self, alert: Alert) -> None:
        context = " ".join(part for part in [alert.strategy_id, alert.symbol] if part)
        suffix = f" [{context}]" if context else ""
        print(f"{alert.timestamp} {alert.severity.value} {alert.event_type}{suffix}: {alert.message}")


class FileAlertSink:
    """Appends structured JSONL alerts for later reports and audits."""

    def __init__(self, path: str | Path = "multi_agent_trading_lab/experiments/alerts.jsonl") -> None:
        self.path = Path(path)

    def send(self, alert: Alert) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert.to_dict(), sort_keys=True) + "\n")


class StubNotifierSink:
    """In-memory placeholder for future email, Slack, or webhook delivery."""

    def __init__(self) -> None:
        self.sent_alerts: list[Alert] = []

    def send(self, alert: Alert) -> None:
        self.sent_alerts.append(alert)


class AlertManager:
    """Routes alerts to sinks while suppressing duplicate noise."""

    def __init__(
        self,
        sinks: list[AlertSink] | None = None,
        cooldown: timedelta = timedelta(minutes=15),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.sinks = sinks if sinks is not None else [ConsoleAlertSink(), FileAlertSink()]
        self.cooldown = cooldown
        self.clock = clock or (lambda: datetime.now(UTC))
        self._last_sent: dict[tuple[str, str, str | None, str | None], datetime] = {}

    def emit(
        self,
        event_type: str,
        severity: AlertSeverity,
        message: str,
        strategy_id: str | None = None,
        symbol: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Alert | None:
        now = self.clock()
        key = (event_type, severity.value, strategy_id, symbol)
        previous = self._last_sent.get(key)
        if previous is not None and now - previous < self.cooldown:
            return None
        alert = Alert(
            id=str(uuid4()),
            timestamp=now.isoformat(),
            event_type=event_type,
            severity=severity,
            strategy_id=strategy_id,
            symbol=symbol,
            message=message,
            metadata=metadata or {},
        )
        for sink in self.sinks:
            sink.send(alert)
        self._last_sent[key] = now
        return alert


def default_alert_manager() -> AlertManager:
    return AlertManager()
