"""Agent wrapper around staged orchestration workflows."""

from __future__ import annotations

from typing import Any

from multi_agent_trading_lab.agents.base_agent import AgentResult, BaseAgent


class OrchestratorAgent(BaseAgent):
    """Coordinates named workflows across research, paper, and live stages."""

    def __init__(self, orchestrator: Any) -> None:
        super().__init__("orchestrator_agent")
        self.orchestrator = orchestrator

    def run(self, workflow: str = "research") -> AgentResult:
        if workflow == "paper":
            return AgentResult(self.name, self.orchestrator.run_paper_cycle())
        if workflow == "live":
            return AgentResult(self.name, self.orchestrator.run_live_cycle())
        return AgentResult(self.name, self.orchestrator.run_research_cycle())
