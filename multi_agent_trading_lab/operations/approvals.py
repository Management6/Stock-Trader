"""Local approval queue for paper strategy promotion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.operations.alerting import AlertManager, AlertSeverity, default_alert_manager
from multi_agent_trading_lab.strategies.registry import StrategyRegistry


@dataclass(frozen=True)
class PromotionRequest:
    strategy_id: str
    strategy: dict[str, Any]
    rationale: str
    metrics: dict[str, Any]
    requested_at: str
    requester: str
    status: str = "pending"
    reviewed_at: str | None = None
    reviewer: str | None = None
    decision_reason: str | None = None


class ApprovalQueue:
    """Persists pending strategy promotions until an operator decides."""

    def __init__(
        self,
        path: str | Path = "multi_agent_trading_lab/state/strategy_approvals.json",
        registry: StrategyRegistry | None = None,
        audit_log: AuditLog | None = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self.path = Path(path)
        self.registry = registry or StrategyRegistry()
        self.audit_log = audit_log or AuditLog()
        self.alert_manager = alert_manager or default_alert_manager()

    def request_promotion(
        self,
        strategy: dict[str, Any],
        rationale: str,
        metrics: dict[str, Any],
        requester: str,
    ) -> PromotionRequest:
        strategy_id = str(strategy["id"])
        state = self._read()
        existing = state.get(strategy_id)
        if existing and existing.get("status") == "pending":
            return PromotionRequest(**existing)
        request = PromotionRequest(
            strategy_id=strategy_id,
            strategy=strategy,
            rationale=rationale,
            metrics=metrics,
            requested_at=datetime.now(UTC).isoformat(),
            requester=requester,
        )
        state[strategy_id] = asdict(request)
        self._write(state)
        self.audit_log.record("strategy_promotion_requested", asdict(request))
        self.alert_manager.emit(
            "strategy_promotion_awaiting_approval",
            AlertSeverity.INFO,
            f"Strategy {strategy_id} is awaiting paper promotion approval.",
            strategy_id=strategy_id,
            metadata={"metrics": metrics, "rationale": rationale},
        )
        return request

    def list_pending(self) -> list[PromotionRequest]:
        return [PromotionRequest(**record) for record in self._read().values() if record.get("status") == "pending"]

    def approve(self, strategy_id: str, reviewer: str = "operator") -> PromotionRequest:
        request = self._review(strategy_id, "approved", reviewer, "Approved for paper trading.")
        self.registry.promote(strategy_id, "active", "Approved for paper trading by human operator.")
        self.audit_log.record("strategy_promotion_approved", asdict(request))
        self.alert_manager.emit(
            "strategy_promotion_approved",
            AlertSeverity.INFO,
            f"Strategy {strategy_id} was approved for paper trading.",
            strategy_id=strategy_id,
        )
        return request

    def reject(self, strategy_id: str, reviewer: str = "operator", reason: str = "Rejected by operator.") -> PromotionRequest:
        request = self._review(strategy_id, "rejected", reviewer, reason)
        try:
            self.registry.promote(strategy_id, "rejected", reason)
        except KeyError:
            pass
        self.audit_log.record("strategy_promotion_rejected", asdict(request))
        self.alert_manager.emit(
            "strategy_promotion_rejected",
            AlertSeverity.WARNING,
            f"Strategy {strategy_id} promotion rejected: {reason}",
            strategy_id=strategy_id,
            metadata={"reason": reason},
        )
        return request

    def _review(self, strategy_id: str, status: str, reviewer: str, reason: str) -> PromotionRequest:
        state = self._read()
        if strategy_id not in state:
            raise KeyError(f"No promotion request found for strategy id: {strategy_id}")
        current = dict(state[strategy_id])
        if current.get("status") != "pending":
            raise ValueError(f"Promotion request for {strategy_id} is already {current.get('status')}.")
        current.update(
            {
                "status": status,
                "reviewed_at": datetime.now(UTC).isoformat(),
                "reviewer": reviewer,
                "decision_reason": reason,
            }
        )
        state[strategy_id] = current
        self._write(state)
        return PromotionRequest(**current)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
