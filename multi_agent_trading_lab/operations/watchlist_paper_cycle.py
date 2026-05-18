"""Watchlist-scoped paper monitoring cycle."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from multi_agent_trading_lab.agents.backtest_agent import _build_strategy
from multi_agent_trading_lab.agents.data_agent import DataAgent
from multi_agent_trading_lab.brokers.paper_broker import PaperBroker
from multi_agent_trading_lab.data.feature_engineering import add_strategy_features
from multi_agent_trading_lab.execution.execution_engine import ExecutionEngine
from multi_agent_trading_lab.execution.order_models import OrderRequest
from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.operations.paper_reporting import DailyPaperReportService
from multi_agent_trading_lab.orchestrator.orchestrator import load_settings
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy
from multi_agent_trading_lab.state.system_state import SystemStateStore


@dataclass(frozen=True)
class WatchlistPaperCycleSummary:
    passed: bool
    output_dir: Path
    watchlist_path: Path
    candidates_considered: list[dict[str, Any]]
    candidates_skipped: list[dict[str, Any]]
    signals: list[dict[str, Any]]
    orders: list[dict[str, Any]]
    attribution_summary: dict[str, Any]
    strategy_health: dict[str, Any]
    audit_log_path: Path
    report_json_path: Path
    report_markdown_path: Path
    paper_state_path: Path
    live_trading_enabled: bool
    execution_mode: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("output_dir", "watchlist_path", "audit_log_path", "report_json_path", "report_markdown_path", "paper_state_path"):
            payload[key] = str(payload[key])
        return payload


def run_watchlist_paper_cycle(
    settings_path: str | Path,
    watchlist_path: str | Path,
    output_root: str | Path = "multi_agent_trading_lab/validation_runs",
) -> WatchlistPaperCycleSummary:
    """Run a paper-only cycle scoped to selected watchlist candidates."""

    settings = load_settings(settings_path)
    watchlist_path = Path(watchlist_path)
    watchlist = load_paper_watchlist(watchlist_path)
    output_dir = Path(output_root) / f"watchlist_paper_cycle_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    paths = _sandbox_paths(settings, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_log = AuditLog(paths["audit_log_path"])
    system_state = SystemStateStore(paths["system_state_path"])
    broker = PaperBroker(paths["paper_state_path"])
    risk_policy = RiskPolicy.from_config(settings.get("risk", {}))
    execution_settings = dict(settings.get("execution", {}))
    engine = ExecutionEngine(
        broker=broker,
        risk_policy=risk_policy,
        audit_log=audit_log,
        system_state=system_state,
        mode="paper",
        dry_run=bool(execution_settings.get("dry_run", False)),
        live_enabled=False,
        live_confirmation="",
    )
    selected = list(watchlist.get("selected_candidates", []))
    selected_ids = {str(candidate.get("candidate_id")) for candidate in selected}
    skipped = _audit_non_selected_candidates(watchlist, selected_ids, audit_log)
    considered: list[dict[str, Any]] = []
    signals: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    if not selected:
        message = "Watchlist paper cycle no-op: no selected candidates."
    else:
        message = "Watchlist paper cycle completed."
        data_config = dict(settings.get("data", {}))
        symbols = list(data_config.get("symbols", []))
        raw_data = DataAgent(settings).fetch_data(
            symbols or None,
            str(data_config.get("start_date", "2023-01-01")),
            None if data_config.get("end_date") in {None, "null", ""} else str(data_config.get("end_date")),
        )
        featured_data = add_strategy_features(raw_data, _feature_windows(selected))
        symbol = list(featured_data.keys())[0]
        latest_bar = featured_data[symbol][-1]
        for candidate in selected:
            candidate_id = str(candidate.get("candidate_id"))
            strategy_config = _strategy_config(candidate)
            health_decision = _paper_health_decision(settings, system_state, strategy_config)
            considered.append({"candidate_id": candidate_id, "strategy_family": candidate.get("strategy_family"), "strategy_health": health_decision})
            if health_decision is not None:
                reason = "Strategy skipped: quarantined due to paper attribution."
                skipped.append({"candidate_id": candidate_id, "reason": reason, "strategy_health": health_decision})
                audit_log.record("strategy_health_skip", {"strategy": strategy_config, "decision": health_decision, "reason": reason})
                continue
            strategy = _build_strategy(strategy_config, dict(strategy_config.get("strategy_params", strategy_config)))
            signal = int(strategy.generate_signal(latest_bar))
            signal_payload = {"candidate_id": candidate_id, "strategy_family": strategy_config.get("name"), "symbol": symbol, "signal": signal}
            signals.append(signal_payload)
            audit_log.record("watchlist_signal_generated", signal_payload)
            if signal != 1:
                reason = "Signal was flat; no paper order submitted."
                skipped.append({"candidate_id": candidate_id, "reason": reason})
                audit_log.record("approved_trade", {"signal": signal, "action": "no_order", "strategy": strategy_config, "reason": reason})
                continue
            order = OrderRequest(
                symbol=symbol,
                side="buy",
                quantity=int(execution_settings.get("default_quantity", 1)),
                order_type="market",
                reason="Watchlist paper cycle strategy signal.",
                source_strategy=candidate_id,
                estimated_price=float(latest_bar["close"]),
            )
            status = engine.execute_order(order)
            orders.append({"candidate_id": candidate_id, "order": order.to_dict(), "status": status.to_dict()})
    report = DailyPaperReportService(
        paper_state_path=paths["paper_state_path"],
        audit_log_path=paths["audit_log_path"],
        alerts_path=paths["alerts_path"],
        reports_dir=paths["reports_dir"],
        strategy_health_config=settings.get("strategy_health", {}),
    ).generate(_report_date(settings))
    summary = WatchlistPaperCycleSummary(
        passed=True,
        output_dir=output_dir,
        watchlist_path=watchlist_path,
        candidates_considered=considered,
        candidates_skipped=skipped,
        signals=signals,
        orders=orders,
        attribution_summary=dict(report.paper_attribution.get("summary", {})),
        strategy_health=report.strategy_health,
        audit_log_path=paths["audit_log_path"],
        report_json_path=paths["reports_dir"] / f"paper_report_{report.date}.json",
        report_markdown_path=paths["reports_dir"] / f"paper_report_{report.date}.md",
        paper_state_path=paths["paper_state_path"],
        live_trading_enabled=False,
        execution_mode="paper",
        message=message,
    )
    (output_dir / "watchlist_paper_cycle_summary.json").write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def load_paper_watchlist(path: str | Path) -> dict[str, Any]:
    """Load and validate a paper watchlist JSON artifact."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Watchlist validation error: malformed JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Watchlist validation error: top-level JSON must be an object.")
    selected = payload.get("selected_candidates")
    if not isinstance(selected, list):
        raise ValueError("Watchlist validation error: selected_candidates must be a list.")
    for index, candidate in enumerate(selected):
        if not isinstance(candidate, dict):
            raise ValueError(f"Watchlist validation error: selected_candidates[{index}] must be an object.")
        if not candidate.get("candidate_id"):
            raise ValueError(f"Watchlist validation error: selected_candidates[{index}].candidate_id is required.")
    return payload


def _sandbox_paths(settings: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    state_dir = output_dir / "state"
    experiments_dir = output_dir / "experiments"
    reports_dir = output_dir / "reports"
    state_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    source_value = settings.get("system_state_path")
    source_system_state = Path(str(source_value)) if source_value else None
    system_state_path = state_dir / "system_state.json"
    if source_system_state is not None and source_system_state.is_file():
        shutil.copyfile(source_system_state, system_state_path)
    return {
        "paper_state_path": state_dir / "paper_broker_state.json",
        "system_state_path": system_state_path,
        "audit_log_path": experiments_dir / "audit_log.jsonl",
        "alerts_path": experiments_dir / "alerts.jsonl",
        "reports_dir": reports_dir,
    }


def _audit_non_selected_candidates(watchlist: dict[str, Any], selected_ids: set[str], audit_log: AuditLog) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    excluded = dict(watchlist.get("excluded_candidate_details", {}))
    for bucket, candidates in excluded.items():
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("candidate_id"))
            if candidate_id in selected_ids:
                continue
            payload = {
                "candidate_id": candidate_id,
                "strategy_family": candidate.get("strategy_family"),
                "reason": f"Not selected for watchlist paper cycle: {candidate.get('reason', bucket)}",
                "bucket": bucket,
            }
            skipped.append(payload)
            audit_log.record("watchlist_candidate_skipped", payload)
    return skipped


def _strategy_config(candidate: dict[str, Any]) -> dict[str, Any]:
    strategy = dict(candidate.get("strategy", {}))
    if not strategy:
        strategy = {
            "id": candidate.get("candidate_id"),
            "name": candidate.get("strategy_family", "moving_average_crossover"),
            "version": "0.1.0",
            "strategy_params": dict(candidate.get("parameters", {})),
        }
    strategy.setdefault("id", candidate.get("candidate_id"))
    strategy.setdefault("name", candidate.get("strategy_family", "moving_average_crossover"))
    strategy.setdefault("strategy_params", dict(candidate.get("parameters", {})))
    return strategy


def _paper_health_decision(settings: dict[str, Any], system_state: SystemStateStore, strategy_config: dict[str, Any]) -> dict[str, Any] | None:
    if not bool(settings.get("strategy_health", {}).get("enabled", False)):
        return None
    strategy_id = str(strategy_config.get("id") or strategy_config.get("name"))
    decision = dict(system_state.read().get("strategy_health", {}).get(strategy_id, {}))
    if decision.get("status") == "quarantined":
        return decision
    return None


def _feature_windows(candidates: list[dict[str, Any]]) -> list[int]:
    windows: set[int] = set()
    for candidate in candidates:
        params = dict(_strategy_config(candidate).get("strategy_params", {}))
        for key, value in params.items():
            if any(token in str(key) for token in ("window", "period")):
                windows.add(max(2, int(value)))
    return sorted(windows or {5, 20})


def _report_date(settings: dict[str, Any]) -> date:
    as_of = dict(settings.get("data", {})).get("as_of_date")
    if as_of:
        return date.fromisoformat(str(as_of))
    return datetime.now(UTC).date()
