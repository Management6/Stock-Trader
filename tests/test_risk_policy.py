import unittest

from multi_agent_trading_lab.brokers.base_broker import AccountState
from multi_agent_trading_lab.execution.order_models import OrderRequest
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy


class RiskPolicyTests(unittest.TestCase):
    def test_paper_drawdown_inside_configured_threshold_is_accepted(self) -> None:
        policy = RiskPolicy.from_config({"paper": {"max_drawdown": -0.20}})

        decision = policy.evaluate_strategy_for_paper({"total_return": 0.2613, "max_drawdown": -0.12708})

        self.assertTrue(decision.approved)
        self.assertEqual(decision.details["max_drawdown_threshold"], -0.20)

    def test_paper_drawdown_outside_configured_threshold_is_rejected(self) -> None:
        policy = RiskPolicy.from_config({"paper": {"max_drawdown": -0.20}})

        decision = policy.evaluate_strategy_for_paper({"total_return": 0.2613, "max_drawdown": -0.25})

        self.assertFalse(decision.approved)
        self.assertIn("max drawdown exceeds threshold", decision.reason.lower())
        self.assertEqual(decision.details["max_drawdown"], -0.25)
        self.assertEqual(decision.details["max_drawdown_threshold"], -0.20)

    def test_paper_drawdown_threshold_is_config_driven(self) -> None:
        strict_policy = RiskPolicy.from_config({"paper": {"max_drawdown": -0.10}})
        relaxed_policy = RiskPolicy.from_config({"paper": {"max_drawdown": -0.20}})
        metrics = {"total_return": 0.2613, "max_drawdown": -0.12708}

        self.assertFalse(strict_policy.evaluate_strategy_for_paper(metrics).approved)
        self.assertTrue(relaxed_policy.evaluate_strategy_for_paper(metrics).approved)

    def test_missing_nested_drawdown_config_uses_legacy_default(self) -> None:
        policy = RiskPolicy.from_config({})

        self.assertEqual(policy.research_max_drawdown, -0.20)
        self.assertEqual(policy.paper_max_drawdown, -0.20)
        self.assertFalse(policy.evaluate_strategy_for_paper({"total_return": 0.1, "max_drawdown": -0.25}).approved)

    def test_legacy_drawdown_key_still_configures_research_and_paper(self) -> None:
        policy = RiskPolicy.from_config({"max_drawdown_threshold": -0.15})

        self.assertEqual(policy.research_max_drawdown, -0.15)
        self.assertEqual(policy.paper_max_drawdown, -0.15)

    def test_kill_switch_blocks_order(self) -> None:
        policy = RiskPolicy(kill_switch_enabled=True)
        decision = policy.evaluate_order(
            OrderRequest(symbol="AAPL", side="buy", quantity=1, estimated_price=100.0).normalized(),
            AccountState(cash=1000.0, equity=1000.0, buying_power=1000.0),
            [],
            mode="paper",
        )

        self.assertFalse(decision.approved)
        self.assertIn("Kill switch", decision.reason)

    def test_allowed_symbols_are_enforced(self) -> None:
        policy = RiskPolicy(allowed_symbols={"AAPL"})
        decision = policy.evaluate_order(
            OrderRequest(symbol="TSLA", side="buy", quantity=1, estimated_price=100.0).normalized(),
            AccountState(cash=1000.0, equity=1000.0, buying_power=1000.0),
            [],
            mode="paper",
        )

        self.assertFalse(decision.approved)
        self.assertIn("allowed universe", decision.reason)

    def test_allowed_symbol_rejection_includes_diagnostic_details(self) -> None:
        policy = RiskPolicy(allowed_symbols={"AAPL", "MSFT"})

        decision = policy.evaluate_order(
            OrderRequest(symbol="WES.AX", side="buy", quantity=1, estimated_price=100.0).normalized(),
            AccountState(cash=1000.0, equity=1000.0, buying_power=1000.0),
            [],
            mode="paper",
        )

        self.assertFalse(decision.approved)
        self.assertEqual(decision.details["symbol"], "WES.AX")
        self.assertEqual(decision.details["allowed_symbols"], ["AAPL", "MSFT"])


if __name__ == "__main__":
    unittest.main()
