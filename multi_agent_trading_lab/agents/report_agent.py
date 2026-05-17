"""Reporting agent."""

from __future__ import annotations

from typing import Any

from multi_agent_trading_lab.agents.base_agent import AgentResult, BaseAgent


class ReportAgent(BaseAgent):
    """Summarizes bot activity, experiments, and incidents.

    Inputs: experiment or audit records.
    Outputs: human-readable text reports.
    Extension point: LLM-authored reports with citations to audit records.
    """

    def __init__(self) -> None:
        super().__init__("report_agent")

    def summarize_experiment(self, experiment_record: dict[str, Any]) -> str:
        metrics = experiment_record["metrics"]
        decision = experiment_record.get("risk_decision", {})
        strategy = experiment_record["config"]["strategy"]
        sharpe = metrics.get("sharpe_ratio", metrics.get("sharpe", 0.0)) or 0.0
        cost_assumptions = experiment_record.get("cost_assumptions", metrics.get("cost_assumptions"))
        costs = ""
        if cost_assumptions:
            costs = (
                ", costs="
                f"commission_per_trade={float(cost_assumptions.get('commission_per_trade', 0.0)):.6g}, "
                f"slippage_pct={float(cost_assumptions.get('slippage_pct', 0.0)):.6g}"
            )
        data_quality = experiment_record.get("data_quality", metrics.get("data_quality"))
        quality = ""
        if data_quality:
            issues = data_quality.get("issues", [])
            warning_count = len([issue for issue in issues if issue.get("severity") == "warning"])
            error_count = len([issue for issue in issues if issue.get("severity") == "error"])
            status = "passed" if data_quality.get("passed") else "failed"
            quality = f", data_quality={status} with {warning_count} warning(s), {error_count} error(s)"
        portfolio = ""
        if experiment_record.get("portfolio_backtest"):
            portfolio_metrics = experiment_record["portfolio_backtest"].get("metrics", {})
            portfolio = (
                ", portfolio="
                f"return={float(portfolio_metrics.get('total_return', 0.0) or 0.0):.2%}, "
                f"max_drawdown={float(portfolio_metrics.get('max_drawdown', 0.0) or 0.0):.2%}"
            )
        regime = ""
        if experiment_record.get("market_regime"):
            regime = f", regime={experiment_record['market_regime'].get('regime')}"
        return (
            f"Experiment {experiment_record['id']} | {strategy['id']}: "
            f"return={metrics['total_return']:.2%}, "
            f"max_drawdown={metrics['max_drawdown']:.2%}, "
            f"sharpe_ratio={sharpe:.2f}, "
            f"decision={decision.get('approved')} ({decision.get('reason')})"
            f"{costs}"
            f"{quality}"
            f"{portfolio}"
            f"{regime}"
        )

    def weekly_report(self, experiments: list[dict[str, Any]]) -> str:
        if not experiments:
            return "No experiments logged yet."
        lines = ["Weekly experiment report:"]
        lines.extend(self.summarize_experiment(record) for record in experiments)
        return "\n".join(lines)

    def run(self, experiment_record: dict[str, Any]) -> AgentResult:
        return AgentResult(self.name, self.summarize_experiment(experiment_record))
