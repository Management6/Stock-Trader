"""Operator kill-switch workflow for paper trading."""

from __future__ import annotations

from typing import Any

from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.operations.alerting import AlertManager, AlertSeverity, default_alert_manager
from multi_agent_trading_lab.state.system_state import SystemStateStore


def activate_kill_switch(
    reason: str,
    state_store: SystemStateStore | None = None,
    audit_log: AuditLog | None = None,
    alert_manager: AlertManager | None = None,
    operator: str = "operator",
) -> dict[str, Any]:
    """Enable the local kill switch and emit audit/alert evidence."""

    if not reason.strip():
        raise ValueError("A kill switch reason is required.")
    state_store = state_store or SystemStateStore()
    audit_log = audit_log or AuditLog()
    alert_manager = alert_manager or default_alert_manager()
    state = state_store.set_kill_switch(True, reason.strip())
    payload = {"operator": operator, "reason": reason.strip(), "state": state}
    audit_log.record("kill_switch_activated", payload)
    alert_manager.emit(
        "kill_switch_activated",
        AlertSeverity.CRITICAL,
        f"Kill switch activated: {reason.strip()}",
        metadata={"operator": operator},
    )
    return state
