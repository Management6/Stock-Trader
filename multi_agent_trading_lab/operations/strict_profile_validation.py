"""Deterministic validation for the strict paper research profile."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from multi_agent_trading_lab.operations.paper_reporting import DailyPaperReportService
from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator, load_settings


@dataclass(frozen=True)
class StrictProfileValidationResult:
    passed: bool
    output_dir: Path
    checks: dict[str, Any]
    accepted_count: int
    rejected_count: int
    rejection_breakdown: dict[str, int]
    accepted_example: dict[str, Any] | None
    rejected_example: dict[str, Any] | None
    report_path: Path
    audit_log_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["report_path"] = str(self.report_path)
        payload["audit_log_path"] = str(self.audit_log_path)
        return payload


def run_strict_profile_validation(
    settings_path: str | Path = "multi_agent_trading_lab/config/settings.strict_paper.yaml",
    output_root: str | Path = "multi_agent_trading_lab/validation_runs",
) -> StrictProfileValidationResult:
    """Run a small strict-profile research, approval, paper, and report flow."""

    return _run_profile_validation(
        settings_path=settings_path,
        output_root=output_root,
        profile_name="strict_paper",
        require_accepted_candidate=True,
        approve_first_pending=True,
        run_paper_cycle=True,
        summary_filename="strict_profile_validation_summary.json",
    )


def run_approval_profile_validation(
    settings_path: str | Path = "multi_agent_trading_lab/config/settings.approval_paper.yaml",
    output_root: str | Path = "multi_agent_trading_lab/validation_runs",
) -> StrictProfileValidationResult:
    """Run approval-grade research validation without requiring any candidate to pass."""

    return _run_profile_validation(
        settings_path=settings_path,
        output_root=output_root,
        profile_name="approval_paper",
        require_accepted_candidate=False,
        approve_first_pending=False,
        run_paper_cycle=False,
        summary_filename="approval_profile_validation_summary.json",
    )


def _run_profile_validation(
    settings_path: str | Path,
    output_root: str | Path,
    profile_name: str,
    require_accepted_candidate: bool,
    approve_first_pending: bool,
    run_paper_cycle: bool,
    summary_filename: str,
) -> StrictProfileValidationResult:
    run_id = f"{profile_name}_validation_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = Path(output_root) / run_id
    settings = _validation_settings(load_settings(settings_path), output_dir)
    orchestrator = TradingLabOrchestrator(settings)

    research = orchestrator.run_research_cycle()
    pending = orchestrator.approval_queue.list_pending()
    approved_strategy_id = None
    if approve_first_pending and pending:
        approved_strategy_id = pending[0].strategy_id
        orchestrator.approval_queue.approve(approved_strategy_id, reviewer=f"{profile_name}_validation")
    paper = orchestrator.run_paper_cycle() if run_paper_cycle else None
    report = DailyPaperReportService(
        paper_state_path=settings["broker"]["paper_state_path"],
        audit_log_path=settings["audit_log_path"],
        alerts_path=settings["alert_log_path"],
        reports_dir=output_dir / "reports",
        strategy_health_config=settings.get("strategy_health", {}),
    ).generate(date(2023, 8, 31))

    records = list(research.details.get("records", []))
    accepted = _accepted_record(records, pending)
    rejected = _rejected_record(records)
    accepted_count = sum(1 for record in records if record.get("risk_decision", {}).get("approved"))
    rejected_count = sum(1 for record in records if not record.get("risk_decision", {}).get("approved"))
    rejection_breakdown = _rejection_breakdown(records)
    audit_records = _read_jsonl(Path(settings["audit_log_path"]))
    checks = {
        "live_trading_disabled": settings.get("broker", {}).get("live_enabled") is False,
        "paper_mode_active": settings.get("execution", {}).get("mode") == "paper" and settings.get("operating_mode") == "paper",
        "config_validation_passed": orchestrator.config_validation.errors == [],
        "all_optional_gates_enabled": _all_optional_gates_enabled(settings),
        "research_records": len(records),
        "approval_request_created": bool(pending),
        "approved_strategy_id": approved_strategy_id,
        "accepted_has_required_gate_metadata": _has_required_gate_metadata(accepted),
        "optimizer_metadata_persisted": _all_optimizer_metadata_persisted(records),
        "rejections_have_reasons": rejected_count == 0 or all(record.get("risk_decision", {}).get("reason") for record in records if not record.get("risk_decision", {}).get("approved")),
        "rejection_reason_visible": bool(rejected and rejected.get("risk_decision", {}).get("reason")),
        "audit_records_written": len(audit_records) > 0,
        "paper_cycle_completed": paper.stage == "paper" if paper is not None else True,
        "report_generated": (output_dir / "reports" / "paper_report_2023-08-31.json").exists(),
    }
    required_checks = dict(checks)
    if not require_accepted_candidate:
        required_checks.pop("approval_request_created")
        required_checks.pop("accepted_has_required_gate_metadata")
    required_checks.pop("approved_strategy_id")
    result = StrictProfileValidationResult(
        passed=all(bool(value) for value in required_checks.values()),
        output_dir=output_dir,
        checks=checks,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        rejection_breakdown=rejection_breakdown,
        accepted_example=_example_from_record(accepted),
        rejected_example=_example_from_record(rejected),
        report_path=output_dir / "reports" / f"paper_report_{report.date}.json",
        audit_log_path=Path(settings["audit_log_path"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / summary_filename).write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _validation_settings(settings: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    payload = dict(settings)
    payload.setdefault("broker", {})
    payload["broker"] = {**dict(payload["broker"]), "paper_state_path": str(output_dir / "state" / "paper_broker_state.json"), "live_enabled": False, "live_confirmation": ""}
    payload["operating_mode"] = "paper"
    payload["execution"] = {**dict(payload.get("execution", {})), "mode": "paper"}
    payload["experiment_log_path"] = str(output_dir / "experiments" / "experiment_memory.jsonl")
    payload["audit_log_path"] = str(output_dir / "experiments" / "audit_log.jsonl")
    payload["alert_log_path"] = str(output_dir / "experiments" / "alerts.jsonl")
    payload["approval_queue_path"] = str(output_dir / "state" / "strategy_approvals.json")
    payload["strategy_registry_path"] = str(output_dir / "state" / "strategy_registry.json")
    payload["system_state_path"] = str(output_dir / "state" / "system_state.json")
    return payload


def _all_optional_gates_enabled(settings: dict[str, Any]) -> bool:
    return all(
        [
            bool(settings.get("oos_validation", {}).get("enabled")),
            bool(settings.get("portfolio", {}).get("enabled")),
            bool(settings.get("regime", {}).get("enabled")),
            bool(settings.get("strategy_health", {}).get("enabled")),
            bool(settings.get("optimizer", {}).get("enabled")),
            float(settings.get("backtest", {}).get("slippage_pct", 0.0)) > 0.0,
            bool(settings.get("data_quality", {}).get("block_on_missing_columns")),
        ]
    )


def _accepted_record(records: list[dict[str, Any]], pending: list[Any]) -> dict[str, Any] | None:
    pending_ids = {item.strategy_id for item in pending}
    for record in records:
        strategy = dict(record.get("config", {}).get("strategy", {}))
        if record.get("risk_decision", {}).get("approved") and strategy.get("id") in pending_ids:
            return record
    return None


def _rejected_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in records:
        if not record.get("risk_decision", {}).get("approved"):
            return record
    return None


def _has_required_gate_metadata(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    metrics = dict(record.get("metrics", {}))
    return all(
        [
            "optimizer_trial" in metrics,
            "cost_assumptions" in metrics,
            "data_quality" in metrics,
            "walk_forward" in metrics,
            "portfolio_backtest" in metrics,
            "market_regime" in metrics,
        ]
    )


def _all_optimizer_metadata_persisted(records: list[dict[str, Any]]) -> bool:
    return bool(records) and all("optimizer_trial" in dict(record.get("metrics", {})) for record in records)


def _rejection_breakdown(records: list[dict[str, Any]]) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for record in records:
        if record.get("risk_decision", {}).get("approved"):
            continue
        reason = str(record.get("risk_decision", {}).get("reason", "unknown"))
        breakdown[reason] = breakdown.get(reason, 0) + 1
    return breakdown


def _example_from_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    return {
        "strategy": record.get("config", {}).get("strategy", {}).get("id"),
        "gate_outcome": record.get("metrics", {}).get("optimizer_trial", {}).get("gate_outcome"),
        "reason": record.get("risk_decision", {}).get("reason"),
        "metrics": {
            "total_return": record.get("metrics", {}).get("total_return"),
            "max_drawdown": record.get("metrics", {}).get("max_drawdown"),
            "sharpe_ratio": record.get("metrics", {}).get("sharpe_ratio"),
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
