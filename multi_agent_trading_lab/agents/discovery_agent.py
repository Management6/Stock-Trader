"""Discovery agent for user intent and operating preferences."""

from __future__ import annotations

from typing import Any

from multi_agent_trading_lab.agents.base_agent import AgentResult, BaseAgent


class DiscoveryAgent(BaseAgent):
    """Captures the system profile that constrains autonomous behavior.

    Inputs: optional user preference overrides.
    Outputs: a profile describing risk posture, universe, and autonomy phase.
    Extension point: guided interview or LLM-assisted profile synthesis.
    """

    def __init__(self) -> None:
        super().__init__("discovery_agent")

    def build_profile(self, preferences: dict[str, Any] | None = None) -> dict[str, Any]:
        profile = {
            "autonomy_phase": "phase_1_research",
            "default_mode": "paper",
            "human_review_required_for_live": True,
            "risk_posture": "conservative",
            "notes": "Default profile favors research, paper trading, and explicit human review.",
        }
        profile.update(preferences or {})
        return profile

    def run(self, preferences: dict[str, Any] | None = None) -> AgentResult:
        return AgentResult(self.name, self.build_profile(preferences))

