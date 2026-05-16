"""Deterministic Phase 2 validation harness for paper-trading operations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.operations.alerting import AlertManager, AlertSeverity, FileAlertSink
from multi_agent_trading_lab.operations.approvals import ApprovalQueue
from multi_agent_trading_lab.operations.kill_switch import activate_kill_switch
from multi_agent_trading_lab.operations.paper_reporting import DailyPaperReportService
from multi_agent_trading_lab.state.system_state import SystemStateStore
from multi_agent_trading_lab.strategies.registry import StrategyRegistry


VALIDATION_SCENARIOS = [
    "happy_path_daily_cycle",
    "approval_required_candidate",
    "repeated_order_rejections",
    "drawdown_breach",
    "paper_cycle_failure",
    "kill_switch_activation",
    "kill_switch_persistence",
    "daily_report_generation",
    "runbook_integrity_check",
]


@dataclass(frozen=True)
class ValidationCheckResult:
    passed: bool
    details: dict[str, Any]


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    passed: bool
    checks: dict[str, Any]
    artifacts: dict[str, Path]
    recommended_review: list[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = {key: str(value) for key, value in self.artifacts.items()}
        return payload


@dataclass(frozen=True)
class ValidationSummary:
    run_id: str
    passed: bool
    output_dir: Path
    scenarios: list[ScenarioResult]
    manual_review_recommended: list[str]

    @property
    def scenario_names(self) -> list[str]:
        return [scenario.name for scenario in self.scenarios]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "passed": self.passed,
            "output_dir": str(self.output_dir),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
            "manual_review_recommended": self.manual_review_recommended,
        }


class FixedClock:
    """Deterministic timestamp source for validation artifacts."""

    def __init__(self, start: datetime | None = None) -> None:
        self.current = start or datetime(2026, 5, 11, 9, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current = self.current + timedelta(seconds=seconds)


class ValidationContext:
    """Per-run local filesystem sandbox for validation scenarios."""

    def __init__(self, output_dir: Path, clock: FixedClock | None = None) -> None:
        self.output_dir = output_dir
        self.clock = clock or FixedClock()
        self.state_dir = output_dir / "state"
        self.experiments_dir = output_dir / "experiments"
        self.reports_dir = output_dir / "reports"
        self.scenario_dir = output_dir / "scenarios"
        for path in (self.state_dir, self.experiments_dir, self.reports_dir, self.scenario_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.paper_state_path = self.state_dir / "paper_broker_state.json"
        self.system_state_path = self.state_dir / "system_state.json"
        self.approval_queue_path = self.state_dir / "strategy_approvals.json"
        self.registry_path = self.state_dir / "strategy_registry.json"
        self.audit_log_path = self.experiments_dir / "audit_log.jsonl"
        self.alerts_path = self.experiments_dir / "alerts.jsonl"
        self.audit_log = AuditLog(self.audit_log_path, clock=self.clock)
        self.alert_manager = AlertManager([FileAlertSink(self.alerts_path)], cooldown=timedelta(minutes=30), clock=self.clock)
        self.registry = StrategyRegistry(self.registry_path)
        self.system_state = SystemStateStore(self.system_state_path)
        self.approvals = ApprovalQueue(
            self.approval_queue_path,
            registry=self.registry,
            audit_log=self.audit_log,
            alert_manager=self.alert_manager,
        )

    def write_paper_state(self, payload: dict[str, Any]) -> None:
        self.paper_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.paper_state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def write_scenario_artifact(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.scenario_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        return path


class Phase2ValidationRunner:
    """Runs deterministic paper-operation validation scenarios."""

    def __init__(
        self,
        output_root: str | Path = "multi_agent_trading_lab/validation_runs",
        runbooks_dir: str | Path = "docs/runbooks",
    ) -> None:
        self.output_root = Path(output_root)
        self.runbooks_dir = Path(runbooks_dir)

    def run(self, scenario: str = "all") -> ValidationSummary:
        scenario_names = VALIDATION_SCENARIOS if scenario == "all" else [scenario]
        unknown = [name for name in scenario_names if name not in VALIDATION_SCENARIOS]
        if unknown:
            raise ValueError(f"Unknown validation scenario(s): {', '.join(unknown)}")
        run_id = f"phase2_validation_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        output_dir = self.output_root / run_id
        context = ValidationContext(output_dir)
        results = [self._run_one(name, context) for name in scenario_names]
        summary = ValidationSummary(
            run_id=run_id,
            passed=all(result.passed for result in results),
            output_dir=output_dir,
            scenarios=results,
            manual_review_recommended=_manual_review(results),
        )
        _write_summary(summary)
        return summary

    def _run_one(self, name: str, context: ValidationContext) -> ScenarioResult:
        scenario_map: dict[str, Callable[[ValidationContext], ScenarioResult]] = {
            "happy_path_daily_cycle": _happy_path_daily_cycle,
            "approval_required_candidate": _approval_required_candidate,
            "repeated_order_rejections": _repeated_order_rejections,
            "drawdown_breach": _drawdown_breach,
            "paper_cycle_failure": _paper_cycle_failure,
            "kill_switch_activation": _kill_switch_activation,
            "kill_switch_persistence": _kill_switch_persistence,
            "daily_report_generation": _daily_report_generation,
            "runbook_integrity_check": lambda ctx: _runbook_integrity_check(ctx, self.runbooks_dir),
        }
        try:
            return scenario_map[name](context)
        except Exception as exc:  # defensive: validation should report failures as artifacts
            artifact = context.write_scenario_artifact(name, {"error": repr(exc)})
            return ScenarioResult(name, False, {"exception": repr(exc)}, {"scenario": artifact}, ["Inspect validation exception artifact."])


def _happy_path_daily_cycle(context: ValidationContext) -> ScenarioResult:
    context.write_paper_state(
        {
            "cash": 99_750.0,
            "equity": 100_000.0,
            "positions": {"WES.AX": {"quantity": 2, "average_price": 100.0, "market_price": 125.0}},
            "orders": {
                "o1": {"status": "filled", "symbol": "WES.AX", "source_strategy": "happy_path_s1"},
                "o2": {"status": "filled", "symbol": "WES.AX", "source_strategy": "happy_path_s1"},
            },
        }
    )
    context.audit_log.record("broker_action", {"scenario": "happy_path_daily_cycle", "status": "filled"})
    report = _report_service(context).generate(date(2026, 5, 11))
    checks = {
        "filled_orders": report.order_counts_by_status.get("filled", 0),
        "critical_alert_count": _alert_count(context, severity="CRITICAL"),
        "report_has_open_position": bool(report.open_positions),
    }
    passed = checks["filled_orders"] == 2 and checks["critical_alert_count"] == 0 and checks["report_has_open_position"]
    artifact = context.write_scenario_artifact("happy_path_daily_cycle", checks)
    return ScenarioResult(
        "happy_path_daily_cycle",
        passed,
        checks,
        {"scenario": artifact, "report_json": context.reports_dir / "paper_report_2026-05-11.json"},
        ["Review generated daily report for clear position and order summary."],
    )


def _approval_required_candidate(context: ValidationContext) -> ScenarioResult:
    strategy = {
        "id": "validation_ma_s48_l105",
        "name": "moving_average_crossover",
        "version": "0.1.0",
        "strategy_params": {"short_window": 48, "long_window": 105},
    }
    context.registry.register(strategy, stage="candidate", reason="validation candidate")
    context.approvals.request_promotion(
        strategy=strategy,
        rationale="Validation candidate passed synthetic research risk checks.",
        metrics={"total_return": 0.12, "max_drawdown": -0.08, "sharpe_ratio": 0.7},
        requester="phase2_validation",
    )
    pending = context.approvals.list_pending()
    checks = {
        "pending_request_created": any(item.strategy_id == strategy["id"] for item in pending),
        "strategy_not_active_without_approval": not context.registry.list_by_stage("active"),
        "awaiting_approval_alert_count": _event_count(context.alerts_path, "strategy_promotion_awaiting_approval"),
    }
    passed = all(checks.values())
    artifact = context.write_scenario_artifact("approval_required_candidate", checks)
    return ScenarioResult(
        "approval_required_candidate",
        passed,
        checks,
        {"scenario": artifact, "approval_queue": context.approval_queue_path, "alerts": context.alerts_path},
        ["Use review_strategy_promotions.py --list against real state during operations."],
    )


def _repeated_order_rejections(context: ValidationContext) -> ScenarioResult:
    for index in range(3):
        context.audit_log.record(
            "rejected_trade",
            {"scenario": "repeated_order_rejections", "order": {"symbol": "WES.AX", "source_strategy": "validation_reject_s1"}, "reason": "max position"},
        )
        context.alert_manager.emit(
            "repeated_order_rejection",
            AlertSeverity.CRITICAL,
            "Repeated paper order rejections detected for WES.AX.",
            strategy_id="validation_reject_s1",
            symbol="WES.AX",
            metadata={"attempt": index + 1},
        )
    checks = {
        "rejected_trade_audit_count": _event_count(context.audit_log_path, "rejected_trade"),
        "repeated_rejection_alert_count": _event_count(context.alerts_path, "repeated_order_rejection"),
        "cooldown_suppressed_duplicates": _event_count(context.alerts_path, "repeated_order_rejection") == 1,
    }
    passed = checks["rejected_trade_audit_count"] == 3 and checks["cooldown_suppressed_duplicates"]
    artifact = context.write_scenario_artifact("repeated_order_rejections", checks)
    return ScenarioResult(
        "repeated_order_rejections",
        passed,
        checks,
        {"scenario": artifact, "audit_log": context.audit_log_path, "alerts": context.alerts_path},
        ["Confirm one CRITICAL alert is enough to prompt operator review without alert spam."],
    )


def _drawdown_breach(context: ValidationContext) -> ScenarioResult:
    metrics = {"total_return": 0.04, "max_drawdown": -0.24, "sharpe_ratio": 0.1}
    context.audit_log.record("strategy_rejected", {"scenario": "drawdown_breach", "strategy_id": "validation_drawdown_s1", "metrics": metrics, "reason": "drawdown breach"})
    context.alert_manager.emit(
        "drawdown_threshold_breach",
        AlertSeverity.WARNING,
        "Strategy validation_drawdown_s1 breached drawdown threshold.",
        strategy_id="validation_drawdown_s1",
        metadata={"metrics": metrics},
    )
    checks = {
        "strategy_rejected_audit_count": _event_count(context.audit_log_path, "strategy_rejected"),
        "drawdown_alert_count": _event_count(context.alerts_path, "drawdown_threshold_breach"),
    }
    passed = checks["strategy_rejected_audit_count"] == 1 and checks["drawdown_alert_count"] == 1
    artifact = context.write_scenario_artifact("drawdown_breach", checks)
    return ScenarioResult(
        "drawdown_breach",
        passed,
        checks,
        {"scenario": artifact, "audit_log": context.audit_log_path, "alerts": context.alerts_path},
        ["Review whether drawdown breach wording gives enough context for rejection."],
    )


def _paper_cycle_failure(context: ValidationContext) -> ScenarioResult:
    context.audit_log.record("paper_cycle_failure", {"scenario": "paper_cycle_failure", "reason": "synthetic exception in validation fixture"})
    context.alert_manager.emit(
        "paper_cycle_failure",
        AlertSeverity.CRITICAL,
        "Paper cycle failed during validation fixture.",
        metadata={"reason": "synthetic exception in validation fixture"},
    )
    checks = {
        "failure_audit_count": _event_count(context.audit_log_path, "paper_cycle_failure"),
        "failure_alert_count": _event_count(context.alerts_path, "paper_cycle_failure"),
    }
    passed = checks["failure_audit_count"] == 1 and checks["failure_alert_count"] == 1
    artifact = context.write_scenario_artifact("paper_cycle_failure", checks)
    return ScenarioResult(
        "paper_cycle_failure",
        passed,
        checks,
        {"scenario": artifact, "audit_log": context.audit_log_path, "alerts": context.alerts_path},
        ["Operator should inspect synthetic failure trail as incident-review rehearsal."],
    )


def _kill_switch_activation(context: ValidationContext) -> ScenarioResult:
    state = activate_kill_switch(
        "validation manual stop",
        state_store=context.system_state,
        audit_log=context.audit_log,
        alert_manager=context.alert_manager,
        operator="phase2_validation",
    )
    checks = {
        "kill_switch_enabled": bool(state.get("kill_switch_enabled")),
        "critical_alert_count": _event_count(context.alerts_path, "kill_switch_activated"),
        "audit_count": _event_count(context.audit_log_path, "kill_switch_activated"),
    }
    passed = checks["kill_switch_enabled"] and checks["critical_alert_count"] >= 1 and checks["audit_count"] >= 1
    artifact = context.write_scenario_artifact("kill_switch_activation", checks)
    return ScenarioResult(
        "kill_switch_activation",
        passed,
        checks,
        {"scenario": artifact, "system_state": context.system_state_path, "alerts": context.alerts_path},
        ["Review kill-switch reason and confirm no-op expectation is clear."],
    )


def _kill_switch_persistence(context: ValidationContext) -> ScenarioResult:
    activate_kill_switch(
        "validation persistence check",
        state_store=context.system_state,
        audit_log=context.audit_log,
        alert_manager=context.alert_manager,
        operator="phase2_validation",
    )
    restarted = SystemStateStore(context.system_state_path)
    state = restarted.read()
    if state.get("kill_switch_enabled"):
        context.audit_log.record("kill_switch_event", {"scenario": "kill_switch_persistence", "action": "future_cycle_blocked"})
    checks = {
        "kill_switch_enabled_after_restart": bool(state.get("kill_switch_enabled")),
        "future_cycle_noop_recorded": _event_count(context.audit_log_path, "kill_switch_event") == 1,
    }
    passed = all(checks.values())
    artifact = context.write_scenario_artifact("kill_switch_persistence", checks)
    return ScenarioResult(
        "kill_switch_persistence",
        passed,
        checks,
        {"scenario": artifact, "system_state": context.system_state_path, "audit_log": context.audit_log_path},
        ["Confirm real operators know kill switch persists until explicitly cleared."],
    )


def _daily_report_generation(context: ValidationContext) -> ScenarioResult:
    context.write_paper_state(
        {
            "cash": 98_500.0,
            "equity": 100_500.0,
            "positions": {"NVDA": {"quantity": 4, "average_price": 400.0, "market_price": 500.0}},
            "orders": {
                "o1": {"status": "filled", "symbol": "NVDA", "source_strategy": "validation_report_s1"},
                "o2": {"status": "rejected", "symbol": "NVDA", "source_strategy": "validation_report_s1"},
            },
        }
    )
    context.audit_log.record("rejected_trade", {"scenario": "daily_report_generation", "order": {"symbol": "NVDA"}, "reason": "validation risk event"})
    context.alert_manager.emit("risk_breach", AlertSeverity.WARNING, "Validation risk breach.", strategy_id="validation_report_s1", symbol="NVDA")
    report = _report_service(context).generate(date(2026, 5, 11))
    checks = {
        "json_report_exists": (context.reports_dir / "paper_report_2026-05-11.json").exists(),
        "markdown_report_exists": (context.reports_dir / "paper_report_2026-05-11.md").exists(),
        "report_alert_count": len(report.alerts),
        "report_risk_event_count": len(report.risk_events),
        "open_position_count": len(report.open_positions),
    }
    passed = checks["json_report_exists"] and checks["markdown_report_exists"] and checks["report_alert_count"] >= 1 and checks["report_risk_event_count"] >= 1 and checks["open_position_count"] == 1
    artifact = context.write_scenario_artifact("daily_report_generation", checks)
    return ScenarioResult(
        "daily_report_generation",
        passed,
        checks,
        {"scenario": artifact, "report_json": context.reports_dir / "paper_report_2026-05-11.json", "report_md": context.reports_dir / "paper_report_2026-05-11.md"},
        ["Read report Markdown and confirm it is useful for daily operator review."],
    )


def _runbook_integrity_check(context: ValidationContext, runbooks_dir: Path) -> ScenarioResult:
    result = validate_runbooks(runbooks_dir)
    artifact = context.write_scenario_artifact("runbook_integrity_check", result.details)
    return ScenarioResult(
        "runbook_integrity_check",
        result.passed,
        result.details,
        {"scenario": artifact},
        ["Open any runbook listed as missing or incomplete."] if not result.passed else ["Spot-check runbooks during operator rehearsal."],
    )


def validate_runbooks(runbooks_dir: str | Path = "docs/runbooks") -> ValidationCheckResult:
    runbooks_dir = Path(runbooks_dir)
    required_files = {
        "kill_switch_procedure.md",
        "paper_trading_incident_review.md",
        "daily_operator_checklist.md",
        "strategy_promotion_review_checklist.md",
    }
    required_phrases = ["purpose", "when to use", "step-by-step", "evidence", "decision", "rollback", "escalation"]
    missing_files = sorted(name for name in required_files if not (runbooks_dir / name).exists())
    incomplete: dict[str, list[str]] = {}
    for name in sorted(required_files - set(missing_files)):
        text = (runbooks_dir / name).read_text(encoding="utf-8").lower()
        missing_sections = [phrase for phrase in required_phrases if phrase not in text]
        if missing_sections:
            incomplete[name] = missing_sections
    details = {"runbooks_dir": str(runbooks_dir), "missing_files": missing_files, "incomplete_files": incomplete}
    return ValidationCheckResult(not missing_files and not incomplete, details)


def _report_service(context: ValidationContext) -> DailyPaperReportService:
    return DailyPaperReportService(
        paper_state_path=context.paper_state_path,
        audit_log_path=context.audit_log_path,
        alerts_path=context.alerts_path,
        reports_dir=context.reports_dir,
        clock=context.clock,
    )


def _event_count(path: Path, event_type: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("event_type") == event_type)


def _alert_count(context: ValidationContext, severity: str) -> int:
    if not context.alerts_path.exists():
        return 0
    return sum(1 for line in context.alerts_path.read_text(encoding="utf-8").splitlines() if line.strip() and json.loads(line).get("severity") == severity)


def _manual_review(results: list[ScenarioResult]) -> list[str]:
    recommendations = []
    for result in results:
        recommendations.extend(result.recommended_review)
    if all(result.passed for result in results):
        recommendations.append("Phase 2 appears ready for longer unattended paper testing, subject to operator review of artifacts.")
    else:
        recommendations.append("Do not start longer unattended paper testing until failed validation scenarios are resolved.")
    return recommendations


def _write_summary(summary: ValidationSummary) -> None:
    summary.output_dir.mkdir(parents=True, exist_ok=True)
    (summary.output_dir / "validation_summary.json").write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# Phase 2 validation report - {summary.run_id}",
        "",
        f"Overall result: {'PASS' if summary.passed else 'FAIL'}",
        f"Output directory: {summary.output_dir}",
        "",
        "## Scenarios",
    ]
    for scenario in summary.scenarios:
        lines.append(f"- {'PASS' if scenario.passed else 'FAIL'} {scenario.name}")
    lines.extend(["", "## Manual Review Recommended"])
    lines.extend(f"- {item}" for item in summary.manual_review_recommended)
    (summary.output_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
