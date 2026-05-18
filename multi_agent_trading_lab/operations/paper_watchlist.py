"""Operator-reviewed paper watchlist selection from robust candidates."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.operations.alerting import AlertManager, FileAlertSink
from multi_agent_trading_lab.operations.approvals import ApprovalQueue
from multi_agent_trading_lab.strategies.registry import StrategyRegistry


PAPER_MONITORING_WARNING = "Paper monitoring only. This watchlist does not enable live trading or authorize live execution."


@dataclass(frozen=True)
class PaperWatchlistSummary:
    passed: bool
    output_dir: Path
    selected_count: int
    max_candidates: int
    diversify_by_family: bool
    selected_candidates: list[dict[str, Any]]
    excluded_counts: dict[str, int]
    ranking_criteria: list[str]
    warning: str
    watchlist_json_path: Path
    watchlist_markdown_path: Path
    approval_queue_written: bool = False
    approval_queue_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["watchlist_json_path"] = str(self.watchlist_json_path)
        payload["watchlist_markdown_path"] = str(self.watchlist_markdown_path)
        payload["approval_queue_path"] = str(self.approval_queue_path) if self.approval_queue_path is not None else None
        return payload


def select_paper_watchlist(
    approval_summary_path: str | Path,
    robustness_summary_path: str | Path,
    max_candidates: int = 5,
    output_root: str | Path = "multi_agent_trading_lab/validation_runs",
    diversify_by_family: bool = True,
    write_approval_queue: bool = False,
) -> PaperWatchlistSummary:
    """Select a small paper-monitoring watchlist from robust approval candidates."""

    approval_summary_path = Path(approval_summary_path)
    robustness_summary_path = Path(robustness_summary_path)
    max_candidates = max(1, int(max_candidates))
    output_dir = Path(output_root) / f"paper_watchlist_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    approval_summary = json.loads(approval_summary_path.read_text(encoding="utf-8"))
    robustness_summary = json.loads(robustness_summary_path.read_text(encoding="utf-8"))
    approval_records = _read_jsonl(Path(str(approval_summary["experiment_log_path"])))
    robustness_by_id = {
        str(candidate.get("candidate_id")): candidate
        for candidate in robustness_summary.get("candidates", [])
        if candidate.get("candidate_id")
    }
    eligible, excluded_counts = _eligible_candidates(approval_records, robustness_by_id)
    ranked = sorted(eligible, key=_ranking_key)
    selected = _diversified_selection(ranked, max_candidates) if diversify_by_family else ranked[:max_candidates]
    selected = [_with_rank(candidate, rank) for rank, candidate in enumerate(selected, start=1)]
    output_dir.mkdir(parents=True, exist_ok=True)
    watchlist_json_path = output_dir / "paper_watchlist.json"
    watchlist_markdown_path = output_dir / "paper_watchlist.md"
    approval_queue_path: Path | None = None
    if write_approval_queue:
        approval_queue_path = output_dir / "state" / "strategy_approvals.json"
        _write_approval_queue(selected, approval_queue_path, output_dir)
    summary = PaperWatchlistSummary(
        passed=True,
        output_dir=output_dir,
        selected_count=len(selected),
        max_candidates=max_candidates,
        diversify_by_family=diversify_by_family,
        selected_candidates=selected,
        excluded_counts=excluded_counts,
        ranking_criteria=_ranking_criteria(),
        warning=PAPER_MONITORING_WARNING,
        watchlist_json_path=watchlist_json_path,
        watchlist_markdown_path=watchlist_markdown_path,
        approval_queue_written=write_approval_queue,
        approval_queue_path=approval_queue_path,
    )
    watchlist_json_path.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    watchlist_markdown_path.write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def _eligible_candidates(
    approval_records: list[dict[str, Any]],
    robustness_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eligible: list[dict[str, Any]] = []
    excluded = {"not_accepted": 0, "not_robust": 0, "missing_robustness": 0}
    for record in approval_records:
        strategy = dict(record.get("config", {}).get("strategy", {}))
        candidate_id = str(strategy.get("id") or record.get("id"))
        optimizer_trial = dict(record.get("metrics", {}).get("optimizer_trial", {}))
        accepted = bool(record.get("risk_decision", {}).get("approved")) and optimizer_trial.get("gate_outcome") == "accepted"
        if not accepted:
            excluded["not_accepted"] += 1
            continue
        robustness = robustness_by_id.get(candidate_id)
        if robustness is None:
            excluded["missing_robustness"] += 1
            continue
        if robustness.get("status") != "robust":
            excluded["not_robust"] += 1
            continue
        eligible.append(_candidate_payload(record, robustness))
    return eligible, excluded


def _candidate_payload(record: dict[str, Any], robustness: dict[str, Any]) -> dict[str, Any]:
    strategy = dict(record.get("config", {}).get("strategy", {}))
    metrics = dict(record.get("metrics", {}))
    walk_forward = dict(metrics.get("walk_forward", {}))
    oos = dict(dict(walk_forward.get("out_of_sample", {})).get("metrics", {}))
    portfolio = dict(dict(metrics.get("portfolio_backtest", {})).get("metrics", {}))
    cost_assumptions = dict(metrics.get("cost_assumptions") or portfolio.get("cost_assumptions") or {})
    regime = dict(metrics.get("market_regime", {}))
    robustness_summary = {
        "status": robustness.get("status"),
        "checks_passed": robustness.get("checks_passed"),
        "checks_failed": robustness.get("checks_failed"),
        "worst_oos_metric": robustness.get("worst_oos_metric"),
        "worst_portfolio_metric": robustness.get("worst_portfolio_metric"),
        "worst_cost_stress_result": robustness.get("worst_cost_stress_result"),
    }
    return {
        "candidate_id": strategy.get("id"),
        "strategy_family": record.get("strategy_name") or strategy.get("name"),
        "parameters": dict(strategy.get("strategy_params") or record.get("strategy_params", {})),
        "oos_metrics": oos,
        "portfolio_metrics": portfolio,
        "base_metrics": {
            "total_return": metrics.get("total_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
        },
        "objective_score": dict(metrics.get("optimizer_trial", {})).get("objective_score"),
        "trial_number": dict(metrics.get("optimizer_trial", {})).get("trial_number"),
        "cost_assumptions": cost_assumptions,
        "regime_result": regime,
        "robustness": robustness_summary,
        "selection_reasons": _selection_reasons(oos, portfolio, robustness_summary),
        "warning": PAPER_MONITORING_WARNING,
        "strategy": strategy,
    }


def _selection_reasons(oos: dict[str, Any], portfolio: dict[str, Any], robustness: dict[str, Any]) -> list[str]:
    reasons = ["Robustness status is robust.", "Approval gate outcome was accepted."]
    if _float(oos.get("sharpe_ratio")) is not None:
        reasons.append(f"OOS Sharpe {_float(oos.get('sharpe_ratio')):.4f}.")
    if _float(oos.get("total_return")) is not None:
        reasons.append(f"OOS return {_float(oos.get('total_return')):.4f}.")
    if _float(portfolio.get("total_return")) is not None:
        reasons.append(f"Portfolio return {_float(portfolio.get('total_return')):.4f}.")
    if robustness.get("checks_passed") is not None:
        reasons.append(f"Robustness checks passed: {robustness.get('checks_passed')}.")
    return reasons


def _ranking_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    oos = dict(candidate.get("oos_metrics", {}))
    portfolio = dict(candidate.get("portfolio_metrics", {}))
    return (
        -_score(oos.get("sharpe_ratio")),
        -_positive(oos.get("total_return")),
        -_score(oos.get("total_return")),
        -_score(oos.get("max_drawdown")),
        -_positive(portfolio.get("total_return")),
        -_score(portfolio.get("total_return")),
        -_score(portfolio.get("max_drawdown")),
        -_score(candidate.get("objective_score")),
        str(candidate.get("strategy_family", "")),
        str(candidate.get("candidate_id", "")),
    )


def _diversified_selection(ranked: list[dict[str, Any]], max_candidates: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(ranked)
    used_families: set[str] = set()
    while remaining and len(selected) < max_candidates:
        next_index = 0
        for index, candidate in enumerate(remaining):
            family = str(candidate.get("strategy_family"))
            if family not in used_families:
                next_index = index
                break
        candidate = remaining.pop(next_index)
        selected.append(candidate)
        used_families.add(str(candidate.get("strategy_family")))
        if len(used_families) >= len({str(item.get("strategy_family")) for item in remaining + selected}):
            used_families.clear()
    return selected


def _with_rank(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    payload = dict(candidate)
    payload["rank"] = rank
    return payload


def _write_approval_queue(selected: list[dict[str, Any]], approval_queue_path: Path, output_dir: Path) -> None:
    queue = ApprovalQueue(
        approval_queue_path,
        registry=StrategyRegistry(output_dir / "state" / "strategy_registry.json"),
        audit_log=AuditLog(output_dir / "experiments" / "paper_watchlist_audit.jsonl"),
        alert_manager=AlertManager([FileAlertSink(output_dir / "experiments" / "paper_watchlist_alerts.jsonl")]),
    )
    for candidate in selected:
        queue.request_promotion(
            strategy=dict(candidate["strategy"]),
            rationale="Selected for operator-reviewed paper watchlist. Paper monitoring only, not live trading.",
            metrics={
                "oos": candidate.get("oos_metrics", {}),
                "portfolio": candidate.get("portfolio_metrics", {}),
                "robustness": candidate.get("robustness", {}),
            },
            requester="select_paper_watchlist",
        )


def _markdown_report(summary: PaperWatchlistSummary) -> str:
    lines = [
        "# Paper Watchlist",
        "",
        f"Warning: {summary.warning}",
        "",
        f"Selected candidates: {summary.selected_count} of max {summary.max_candidates}",
        f"Diversify by family: {summary.diversify_by_family}",
        f"Approval queue written: {summary.approval_queue_written}",
        "",
        "## Ranking Criteria",
    ]
    lines.extend(f"- {item}" for item in summary.ranking_criteria)
    lines.extend(["", "## Candidates"])
    for candidate in summary.selected_candidates:
        lines.extend(
            [
                "",
                f"### {candidate['rank']}. {candidate['candidate_id']}",
                "",
                f"- Strategy family: {candidate.get('strategy_family')}",
                f"- Parameters: `{json.dumps(candidate.get('parameters', {}), sort_keys=True)}`",
                f"- OOS metrics: `{json.dumps(candidate.get('oos_metrics', {}), sort_keys=True)}`",
                f"- Portfolio metrics: `{json.dumps(candidate.get('portfolio_metrics', {}), sort_keys=True)}`",
                f"- Robustness: `{json.dumps(candidate.get('robustness', {}), sort_keys=True)}`",
                f"- Cost assumptions: `{json.dumps(candidate.get('cost_assumptions', {}), sort_keys=True)}`",
                f"- Regime result: `{json.dumps(candidate.get('regime_result', {}), sort_keys=True)}`",
                "- Reasons:",
            ]
        )
        lines.extend(f"  - {reason}" for reason in candidate.get("selection_reasons", []))
    lines.append("")
    return "\n".join(lines)


def _ranking_criteria() -> list[str]:
    return [
        "Robustness status must be robust.",
        "Approval gate outcome must be accepted.",
        "Prefer higher OOS Sharpe.",
        "Prefer positive and higher OOS return.",
        "Prefer lower OOS drawdown.",
        "Prefer positive and higher portfolio return.",
        "Prefer lower portfolio drawdown.",
        "Prefer higher objective score as a tie-breaker.",
        "Use strategy family and candidate id as deterministic final tie-breakers.",
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _score(value: Any) -> float:
    parsed = _float(value)
    return parsed if parsed is not None else float("-inf")


def _positive(value: Any) -> int:
    parsed = _float(value)
    return 1 if parsed is not None and parsed > 0 else 0
