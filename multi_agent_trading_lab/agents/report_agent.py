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
        return (
            f"Experiment {experiment_record['id']} | {strategy['id']}: "
            f"return={metrics['total_return']:.2%}, "
            f"max_drawdown={metrics['max_drawdown']:.2%}, "
            f"sharpe_ratio={sharpe:.2f}, "
            f"decision={decision.get('approved')} ({decision.get('reason')})"
        )

    def weekly_report(self, experiments: list[dict[str, Any]]) -> str:
        if not experiments:
            return "No experiments logged yet."
        lines = ["Weekly experiment report:"]
        lines.extend(self.summarize_experiment(record) for record in experiments)
        return "\n".join(lines)

    def run(self, experiment_record: dict[str, Any]) -> AgentResult:
        return AgentResult(self.name, self.summarize_experiment(experiment_record))
