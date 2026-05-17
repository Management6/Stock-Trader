import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from multi_agent_trading_lab.operations.strategy_health import evaluate_strategy_health, reactivate_strategy_health
from multi_agent_trading_lab.state.system_state import SystemStateStore


class StrategyHealthTests(unittest.TestCase):
    def test_expectation_mismatch_quarantines_strategy_after_enough_signals(self) -> None:
        report = self._attribution(pnl=-75.0, signals=6, accepted=6, expected_oos=0.05)

        decision = evaluate_strategy_health(report, self._config(), clock=self._clock)["decisions"]["s1"]

        self.assertEqual(decision["strategy_id"], "s1")
        self.assertEqual(decision["status"], "quarantined")
        self.assertIn("positive expected OOS return but negative paper PnL", decision["reasons"][0])
        self.assertEqual(decision["timestamp"], "2026-05-11T09:00:00+00:00")
        self.assertIn("Pause paper trading", decision["recommended_action"])
        self.assertLess(decision["evidence"]["actual_pnl"], 0.0)

    def test_high_rejection_rate_repeated_reason_and_no_fills_quarantine(self) -> None:
        report = self._attribution(pnl=0.0, signals=6, accepted=0, rejected=6)
        report["by_strategy"]["s1"]["rejection_reasons"] = {"risk policy": 4}

        decision = evaluate_strategy_health(report, self._config(), clock=self._clock)["decisions"]["s1"]

        self.assertEqual(decision["status"], "quarantined")
        self.assertIn("rejection rate 100.00% exceeds threshold 50.00%", "\n".join(decision["reasons"]))
        self.assertIn("rejection reason 'risk policy' repeated 4 times", "\n".join(decision["reasons"]))
        self.assertIn("no accepted orders across 6 signal(s)", "\n".join(decision["reasons"]))

    def test_paper_drawdown_breach_quarantines_when_available(self) -> None:
        report = self._attribution(pnl=-10.0, signals=5, accepted=5)
        report["by_strategy"]["s1"]["max_drawdown"] = -0.15

        decision = evaluate_strategy_health(report, self._config(), clock=self._clock)["decisions"]["s1"]

        self.assertEqual(decision["status"], "quarantined")
        self.assertIn("paper drawdown -0.15 exceeds threshold -0.10", "\n".join(decision["reasons"]))

    def test_issue_before_minimum_signals_is_needs_review(self) -> None:
        report = self._attribution(pnl=-20.0, signals=2, accepted=2, expected_oos=0.04)

        decision = evaluate_strategy_health(report, self._config(), clock=self._clock)["decisions"]["s1"]

        self.assertEqual(decision["status"], "needs_review")
        self.assertIn("Collect more paper observations", decision["recommended_action"])

    def test_healthy_strategy_remains_active(self) -> None:
        report = self._attribution(pnl=25.0, signals=6, accepted=6, expected_oos=0.04)

        decision = evaluate_strategy_health(report, self._config(), clock=self._clock)["decisions"]["s1"]

        self.assertEqual(decision["status"], "active")
        self.assertEqual(decision["reasons"], ["Paper attribution is within configured health limits."])

    def test_manual_reactivation_restores_quarantined_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = SystemStateStore(Path(tmpdir) / "state.json")
            store.write({"strategy_health": {"s1": {"strategy_id": "s1", "status": "quarantined"}}})

            state = reactivate_strategy_health(store, "s1", operator="tester", reason="reviewed fills", clock=self._clock)

            self.assertEqual(state["strategy_health"]["s1"]["status"], "active")
            self.assertEqual(state["strategy_health"]["s1"]["manual_override"]["operator"], "tester")

    def _config(self) -> dict:
        return {
            "enabled": True,
            "quarantine_on_expectation_mismatch": True,
            "max_rejection_rate": 0.50,
            "repeated_rejection_threshold": 3,
            "max_paper_drawdown": -0.10,
            "min_signals_before_quarantine": 5,
            "allow_manual_reactivation": True,
        }

    def _attribution(self, *, pnl: float, signals: int, accepted: int, rejected: int = 0, expected_oos: float | None = None) -> dict:
        expectation = {}
        if expected_oos is not None:
            expectation["s1"] = {"expected_oos_return": expected_oos, "actual_pnl": pnl, "actual_signals": signals}
        return {
            "summary": {"signals": signals},
            "by_strategy": {
                "s1": {
                    "pnl": pnl,
                    "signals": signals,
                    "accepted_orders": accepted,
                    "rejected_or_skipped_orders": rejected,
                    "rejection_reasons": {},
                }
            },
            "expectation_vs_actual": expectation,
            "rejection_breakdown": {},
        }

    def _clock(self) -> datetime:
        return datetime(2026, 5, 11, 9, 0, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
