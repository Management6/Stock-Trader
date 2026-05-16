import unittest

from multi_agent_trading_lab.research.scoring import score_metrics


class ScoringTests(unittest.TestCase):
    def test_sharpe_objective_uses_reliable_sharpe_ratio(self) -> None:
        score = score_metrics(
            {"total_return": 0.1, "max_drawdown": -0.05, "sharpe_ratio": 1.7, "history_points": 40},
            {"type": "sharpe", "sharpe": {"min_history_points": 30}},
        )

        self.assertEqual(score, 1.7)

    def test_sharpe_objective_falls_back_with_short_history(self) -> None:
        score = score_metrics(
            {"total_return": 0.1, "max_drawdown": -0.05, "sharpe_ratio": 2.0, "history_points": 10},
            {"type": "sharpe", "sharpe": {"min_history_points": 30}},
        )

        self.assertAlmostEqual(score, -7.4)

    def test_return_objective(self) -> None:
        self.assertEqual(score_metrics({"total_return": 0.12}, {"type": "return"}), 0.12)

    def test_custom_objective_penalizes_drawdown_magnitude(self) -> None:
        score = score_metrics(
            {"total_return": 0.2, "max_drawdown": -0.05},
            {"type": "custom", "custom": {"return_weight": 2.0, "drawdown_penalty": 1.0}},
        )

        self.assertAlmostEqual(score, 0.35)

    def test_sharpe_objective_prefers_better_risk_adjusted_profile(self) -> None:
        high_return_bad_risk = {"total_return": 100.0, "max_drawdown": -0.9, "sharpe_ratio": 0.3, "history_points": 252}
        moderate_return_good_risk = {"total_return": 0.2, "max_drawdown": -0.15, "sharpe_ratio": 1.2, "history_points": 252}

        self.assertGreater(
            score_metrics(moderate_return_good_risk, {"type": "sharpe", "sharpe": {"min_history_points": 30}}),
            score_metrics(high_return_bad_risk, {"type": "sharpe", "sharpe": {"min_history_points": 30}}),
        )

    def test_sharpe_fallback_penalizes_extreme_drawdown(self) -> None:
        high_return_bad_risk = {"total_return": 100.0, "max_drawdown": -0.9, "history_points": 5}
        moderate_return_good_risk = {"total_return": 0.2, "max_drawdown": -0.15, "history_points": 5}

        self.assertGreater(
            score_metrics(moderate_return_good_risk, {"type": "sharpe", "sharpe": {"min_history_points": 30}}),
            score_metrics(high_return_bad_risk, {"type": "sharpe", "sharpe": {"min_history_points": 30}}),
        )


if __name__ == "__main__":
    unittest.main()
