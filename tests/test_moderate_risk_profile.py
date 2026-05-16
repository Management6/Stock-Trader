import unittest

from multi_agent_trading_lab.agents.risk_agent import RiskAgent
from multi_agent_trading_lab.orchestrator.orchestrator import load_settings
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy


class ModerateRiskProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.risk_config = load_settings("multi_agent_trading_lab/config/risk.yaml")
        self.policy = RiskPolicy.from_config(self.risk_config)

    def test_high_return_extreme_drawdown_strategy_is_rejected(self) -> None:
        metrics = {"total_return": 100.0, "max_drawdown": -0.9, "sharpe_ratio": 0.3}

        decision = self.policy.evaluate_strategy(metrics)

        self.assertFalse(decision.approved)
        self.assertIn("drawdown", decision.reason.lower())

    def test_moderate_return_acceptable_drawdown_strategy_is_approved(self) -> None:
        threshold = float(self.risk_config["max_drawdown_threshold"])
        metrics = {"total_return": 0.2, "max_drawdown": max(-0.15, threshold + 0.01), "sharpe_ratio": 1.2}

        decision = RiskAgent(policy=self.policy).assess_risk(metrics)

        self.assertTrue(decision["approved"])


if __name__ == "__main__":
    unittest.main()
