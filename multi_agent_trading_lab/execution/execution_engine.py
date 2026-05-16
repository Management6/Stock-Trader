"""Execution engine that separates signals from broker order placement."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from multi_agent_trading_lab.brokers.base_broker import BaseBroker
from multi_agent_trading_lab.execution.order_models import OrderRequest, OrderStatus
from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy
from multi_agent_trading_lab.state.system_state import SystemStateStore


class ExecutionEngine:
    """Validates, risk-checks, logs, and routes approved orders."""

    def __init__(
        self,
        broker: BaseBroker,
        risk_policy: RiskPolicy,
        audit_log: AuditLog | None = None,
        system_state: SystemStateStore | None = None,
        mode: str = "paper",
        dry_run: bool = False,
        live_enabled: bool = False,
        live_confirmation: str = "",
    ) -> None:
        self.broker = broker
        self.risk_policy = risk_policy
        self.audit_log = audit_log or AuditLog()
        self.system_state = system_state or SystemStateStore()
        self.mode = mode
        self.dry_run = dry_run
        self.live_enabled = live_enabled
        self.live_confirmation = live_confirmation

    def execute_order(self, order: OrderRequest) -> OrderStatus:
        normalized = order.normalized()
        validation_errors = self.validate_order(normalized)
        if validation_errors:
            return self._reject(normalized, "; ".join(validation_errors), "validation_rejected")

        if self.mode == "live" and not self._live_guard_passed():
            return self._reject(normalized, "Live mode guard is not satisfied.", "live_guard_rejected")

        account = self.broker.get_account_state()
        positions = self.broker.list_positions()
        decision = self.risk_policy.evaluate_order(normalized, account, positions, mode=self.mode)
        if not decision.approved:
            return self._reject(normalized, decision.reason, "risk_rejected")

        if self.dry_run:
            status = OrderStatus(
                order_id=f"dry-run-{uuid4()}",
                symbol=normalized.symbol,
                side=normalized.side,
                quantity=normalized.quantity,
                status="dry_run",
                filled_quantity=0,
                average_fill_price=None,
                message="Dry-run mode: order was not sent to broker.",
                timestamp=datetime.now(UTC).isoformat(),
            )
            self.audit_log.record("approved_trade", {"order": normalized.to_dict(), "status": status.to_dict()})
            return status

        status = self.broker.submit_order(normalized)
        self.audit_log.record("broker_action", {"order": normalized.to_dict(), "status": status.to_dict()})
        return status

    def validate_order(self, order: OrderRequest) -> list[str]:
        errors: list[str] = []
        if not order.symbol:
            errors.append("symbol is required")
        if order.side not in {"buy", "sell"}:
            errors.append("side must be buy or sell")
        if order.quantity <= 0:
            errors.append("quantity must be positive")
        if order.order_type not in {"market", "limit"}:
            errors.append("order_type must be market or limit")
        if order.order_type == "limit" and order.limit_price is None:
            errors.append("limit orders require limit_price")
        if order.estimated_price is not None and order.estimated_price <= 0:
            errors.append("estimated_price must be positive")
        return errors

    def _live_guard_passed(self) -> bool:
        return self.live_enabled and self.live_confirmation == "I_UNDERSTAND_LIVE_TRADING_RISK"

    def _reject(self, order: OrderRequest, reason: str, event_type: str) -> OrderStatus:
        status = OrderStatus(
            order_id=f"rejected-{uuid4()}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            status="rejected",
            filled_quantity=0,
            average_fill_price=None,
            message=reason,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self.audit_log.record(event_type, {"order": order.to_dict(), "reason": reason, "status": status.to_dict()})
        return status

