"""Execution agent that submits approved orders through the execution engine."""

from __future__ import annotations

from multi_agent_trading_lab.agents.base_agent import AgentResult, BaseAgent
from multi_agent_trading_lab.execution.execution_engine import ExecutionEngine
from multi_agent_trading_lab.execution.order_models import OrderRequest, OrderStatus


class ExecutionAgent(BaseAgent):
    """Routes approved trade requests to the execution engine.

    Inputs: normalized OrderRequest objects.
    Outputs: OrderStatus from dry-run, paper broker, or future live broker.
    Extension point: insert human approval, queueing, or retry policies.
    """

    def __init__(self, engine: ExecutionEngine) -> None:
        super().__init__("execution_agent")
        self.engine = engine

    def submit_order(self, order: OrderRequest) -> OrderStatus:
        return self.engine.execute_order(order)

    def run(self, order: OrderRequest) -> AgentResult:
        return AgentResult(self.name, self.submit_order(order))

