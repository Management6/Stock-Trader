"""Daily paper-trading report generation from local state and logs."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Callable

from multi_agent_trading_lab.operations.paper_attribution import build_paper_attribution, expected_metrics_from_audit
from multi_agent_trading_lab.operations.strategy_health import DEFAULT_STRATEGY_HEALTH_CONFIG, evaluate_strategy_health


@dataclass(frozen=True)
class DailyPaperReport:
    date: str
    window_start: str
    window_end: str
    starting_equity: float | None
    ending_equity: float | None
    realized_pnl: float | None
    unrealized_pnl: float | None
    open_positions: list[dict[str, Any]]
    order_counts_by_status: dict[str, int]
    strategy_summary: dict[str, dict[str, Any]]
    risk_events: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    paper_attribution: dict[str, Any]
    strategy_health: dict[str, Any]
    notable_changes: list[str]
    text_report: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DailyPaperReportService:
    """Builds deterministic local paper reports and writes JSON/Markdown artifacts."""

    def __init__(
        self,
        paper_state_path: str | Path = "multi_agent_trading_lab/state/paper_broker_state.json",
        audit_log_path: str | Path = "multi_agent_trading_lab/experiments/audit_log.jsonl",
        alerts_path: str | Path = "multi_agent_trading_lab/experiments/alerts.jsonl",
        reports_dir: str | Path = "multi_agent_trading_lab/reports",
        clock: Callable[[], datetime] | None = None,
        strategy_health_config: dict[str, Any] | None = None,
    ) -> None:
        self.paper_state_path = Path(paper_state_path)
        self.audit_log_path = Path(audit_log_path)
        self.alerts_path = Path(alerts_path)
        self.reports_dir = Path(reports_dir)
        self.clock = clock or (lambda: datetime.now(UTC))
        self.strategy_health_config = {**DEFAULT_STRATEGY_HEALTH_CONFIG, **(strategy_health_config or {})}

    def generate(self, report_date: date | None = None) -> DailyPaperReport:
        now = self.clock()
        current_date = report_date or now.date()
        window_start = datetime.combine(current_date, time.min, tzinfo=UTC)
        window_end = datetime.combine(current_date, time.max, tzinfo=UTC)
        state = self._read_json(self.paper_state_path, default={"cash": 0.0, "equity": 0.0, "positions": {}, "orders": {}})
        orders = dict(state.get("orders", {}))
        positions = dict(state.get("positions", {}))
        ending_equity = _float_or_none(state.get("equity"))
        previous = self._previous_report(current_date)
        starting_equity = _float_or_none(previous.get("ending_equity")) if previous else ending_equity
        open_positions = [_position_summary(symbol, payload) for symbol, payload in positions.items()]
        order_counts = Counter(str(order.get("status", "unknown")) for order in orders.values())
        strategy_summary = _strategy_summary(orders)
        unrealized = sum(float(position.get("quantity", 0)) * (float(position.get("market_price", 0.0)) - float(position.get("average_price", 0.0))) for position in positions.values())
        audit_records = self._read_jsonl(self.audit_log_path)
        alerts = self._records_in_window(self._read_jsonl(self.alerts_path), window_start, window_end)
        risk_events = [
            record
            for record in self._records_in_window(audit_records, window_start, window_end)
            if record.get("event_type") in {"risk_rejected", "rejected_trade", "kill_switch_activated", "live_guard_rejected"}
        ]
        attribution = build_paper_attribution(
            orders,
            audit_records,
            expected_metrics_by_strategy=expected_metrics_from_audit(audit_records),
            starting_equity=starting_equity,
            ending_equity=ending_equity,
        )
        strategy_health = evaluate_strategy_health(attribution, self.strategy_health_config, clock=self.clock)
        if bool(self.strategy_health_config.get("enabled", False)):
            for decision in strategy_health["decisions"].values():
                self._append_audit_record("strategy_health_decision", decision)
        notable_changes = _notable_changes(previous, ending_equity, len(alerts), len(risk_events))
        text_report = _render_text_report(
            current_date.isoformat(),
            starting_equity,
            ending_equity,
            unrealized,
            open_positions,
            dict(order_counts),
            risk_events,
            alerts,
            attribution,
            strategy_health,
            notable_changes,
        )
        report = DailyPaperReport(
            date=current_date.isoformat(),
            window_start=window_start.isoformat(),
            window_end=window_end.isoformat(),
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            realized_pnl=None,
            unrealized_pnl=unrealized,
            open_positions=open_positions,
            order_counts_by_status=dict(order_counts),
            strategy_summary=strategy_summary,
            risk_events=risk_events,
            alerts=alerts,
            paper_attribution=attribution,
            strategy_health=strategy_health,
            notable_changes=notable_changes,
            text_report=text_report,
        )
        self._write_artifacts(report)
        return report

    def _write_artifacts(self, report: DailyPaperReport) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        stem = f"paper_report_{report.date}"
        (self.reports_dir / f"{stem}.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (self.reports_dir / f"{stem}.md").write_text(report.text_report + "\n", encoding="utf-8")

    def _append_audit_record(self, event_type: str, payload: dict[str, Any]) -> None:
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event_type": event_type, "payload": payload, "timestamp": self.clock().isoformat()}
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _previous_report(self, current_date: date) -> dict[str, Any]:
        if not self.reports_dir.exists():
            return {}
        candidates = sorted(path for path in self.reports_dir.glob("paper_report_*.json") if path.name < f"paper_report_{current_date.isoformat()}.json")
        if not candidates:
            return {}
        return self._read_json(candidates[-1], default={})

    @staticmethod
    def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @staticmethod
    def _records_in_window(records: list[dict[str, Any]], start: datetime, end: datetime) -> list[dict[str, Any]]:
        filtered = []
        for record in records:
            timestamp = record.get("timestamp")
            if not timestamp:
                continue
            try:
                parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            except ValueError:
                continue
            if start <= parsed <= end:
                filtered.append(record)
        return filtered


def _position_summary(symbol: str, payload: dict[str, Any]) -> dict[str, Any]:
    quantity = int(payload.get("quantity", 0))
    average_price = float(payload.get("average_price", 0.0))
    market_price = float(payload.get("market_price", average_price))
    return {
        "symbol": symbol,
        "quantity": quantity,
        "average_price": average_price,
        "market_price": market_price,
        "unrealized_pnl": quantity * (market_price - average_price),
    }


def _strategy_summary(orders: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = defaultdict(lambda: {"order_count": 0, "statuses": {}})
    for order in orders.values():
        strategy_id = str(order.get("source_strategy") or "unknown")
        status = str(order.get("status", "unknown"))
        summary[strategy_id]["order_count"] += 1
        statuses = summary[strategy_id]["statuses"]
        statuses[status] = statuses.get(status, 0) + 1
    return dict(summary)


def _notable_changes(previous: dict[str, Any], ending_equity: float | None, alert_count: int, risk_event_count: int) -> list[str]:
    changes = []
    previous_equity = _float_or_none(previous.get("ending_equity")) if previous else None
    if previous_equity is not None and ending_equity is not None:
        changes.append(f"Equity changed by {ending_equity - previous_equity:.2f} since the previous report.")
    if alert_count:
        changes.append(f"{alert_count} alert(s) were triggered in the reporting window.")
    if risk_event_count:
        changes.append(f"{risk_event_count} risk event(s) were recorded in the reporting window.")
    return changes or ["No notable changes from previous report."]


def _render_text_report(
    report_date: str,
    starting_equity: float | None,
    ending_equity: float | None,
    unrealized_pnl: float | None,
    positions: list[dict[str, Any]],
    order_counts: dict[str, int],
    risk_events: list[dict[str, Any]],
    alerts: list[dict[str, Any]],
    paper_attribution: dict[str, Any],
    strategy_health: dict[str, Any],
    notable_changes: list[str],
) -> str:
    lines = [
        f"# Daily paper trading report - {report_date}",
        "",
        f"Starting equity: {_money(starting_equity)}",
        f"Ending equity: {_money(ending_equity)}",
        f"Unrealized PnL: {_money(unrealized_pnl)}",
        f"Open positions: {len(positions)}",
        f"Orders by status: {order_counts or {}}",
        f"Risk events: {len(risk_events)}",
        f"Alerts: {len(alerts)}",
        "",
        "Paper attribution:",
        f"- Total paper PnL: {_money(paper_attribution.get('summary', {}).get('total_pnl'))}",
        f"- Signals: {paper_attribution.get('summary', {}).get('signals', 0)}",
        f"- Accepted orders: {paper_attribution.get('summary', {}).get('accepted_orders', 0)}",
        f"- Rejected/skipped orders: {paper_attribution.get('summary', {}).get('rejected_or_skipped_orders', 0)}",
        "",
        "Strategy health:",
        f"- Enabled: {strategy_health.get('enabled', False)}",
        f"- Decisions: {len(strategy_health.get('decisions', {}))}",
        "",
        "Notable changes:",
    ]
    lines.extend(f"- {change}" for change in notable_changes)
    if positions:
        lines.extend(["", "Open positions:"])
        lines.extend(f"- {p['symbol']}: qty={p['quantity']} avg={p['average_price']:.2f} market={p['market_price']:.2f} unrealized={p['unrealized_pnl']:.2f}" for p in positions)
    return "\n".join(lines)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"
