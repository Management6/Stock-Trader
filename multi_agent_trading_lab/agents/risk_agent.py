"""Risk review agent."""

from __future__ import annotations

from multi_agent_trading_lab.brokers.base_broker import AccountState, Position
from multi_agent_trading_lab.execution.order_models import OrderRequest
from multi_agent_trading_lab.agents.base_agent import AgentResult, BaseAgent
from multi_agent_trading_lab.risk.risk_policy import RiskDecision, RiskPolicy


class RiskAgent(BaseAgent):
    """Checks strategy-level and trade-level risk constraints.

    Inputs: backtest metrics or normalized order requests.
    Outputs: RiskDecision payloads.
    Extension point: add portfolio analytics, scenario tests, or human review.
    """

    def __init__(
        self,
        min_total_return: float = -0.10,
        max_drawdown_limit: float = -0.25,
        policy: RiskPolicy | None = None,
    ) -> None:
        super().__init__("risk_agent")
        self.policy = policy or RiskPolicy(
            min_backtest_return=min_total_return,
            max_drawdown_threshold=max_drawdown_limit,
        )

    def assess_risk(self, metrics: dict[str, float]) -> dict[str, str | bool]:
        decision = self.policy.evaluate_strategy(metrics)
        return {"approved": decision.approved, "reason": decision.reason}

    def approve_or_reject(self, metrics: dict[str, float]) -> str:
        assessment = self.assess_risk(metrics)
        return "approved" if assessment["approved"] else "rejected"

    def assess_trade(
        self,
        order: OrderRequest,
        account: AccountState,
        positions: list[Position],
        mode: str,
    ) -> RiskDecision:
        return self.policy.evaluate_order(order.normalized(), account, positions, mode)

    def run(self, metrics: dict[str, float]) -> AgentResult:
        return AgentResult(self.name, self.assess_risk(metrics))
