import unittest

from multi_agent_trading_lab.research.optimizer import DeterministicRandomOptimizer, propose_candidates


class OptimizerTests(unittest.TestCase):
    def test_proposes_moving_average_candidates_with_constraints(self) -> None:
        candidates = propose_candidates("moving_average_crossover", self._ma_space(), n_trials=10, seed=7)

        self.assertEqual(len(candidates), 10)
        for candidate in candidates:
            params = candidate["strategy_params"]
            self.assertLess(params["short_window"], params["long_window"])
            self.assertGreaterEqual(params["short_window"], 5)
            self.assertLessEqual(params["long_window"], 50)
            self.assertIn("optimizer_trial", candidate)

    def test_proposes_ma_rsi_candidates_with_valid_thresholds(self) -> None:
        candidates = propose_candidates("moving_average_rsi_filter", self._ma_rsi_space(), n_trials=8, seed=11)

        for candidate in candidates:
            params = candidate["strategy_params"]
            self.assertLess(params["short_window"], params["long_window"])
            self.assertGreaterEqual(params["rsi_min"], 30)
            self.assertLessEqual(params["rsi_max"], 70)
            self.assertLess(params["rsi_min"], params["rsi_max"])

    def test_proposes_breakout_candidates_with_positive_windows(self) -> None:
        candidates = propose_candidates("breakout_trend", self._breakout_space(), n_trials=6, seed=3)

        for candidate in candidates:
            params = candidate["strategy_params"]
            self.assertGreater(params["breakout_window"], 0)
            self.assertGreater(params["exit_window"], 0)
            self.assertLess(params["exit_window"], params["breakout_window"])

    def test_seeded_optimizer_is_deterministic(self) -> None:
        first = propose_candidates("moving_average_crossover", self._ma_space(), n_trials=5, seed=42)
        second = propose_candidates("moving_average_crossover", self._ma_space(), n_trials=5, seed=42)

        self.assertEqual(first, second)

    def test_optimizer_records_results_and_returns_best_candidates(self) -> None:
        optimizer = DeterministicRandomOptimizer(seed=1)
        candidates = optimizer.propose_candidates("moving_average_crossover", self._ma_space(), n_trials=3)

        optimizer.record_result(candidates[0], score=-1.0, metadata={"gate_outcome": "rejected"})
        optimizer.record_result(candidates[1], score=2.0, metadata={"gate_outcome": "accepted"})
        optimizer.record_result(candidates[2], score=0.5, metadata={"gate_outcome": "rejected"})

        best = optimizer.best_candidates(limit=2)
        self.assertEqual(best[0]["score"], 2.0)
        self.assertEqual(best[0]["metadata"]["gate_outcome"], "accepted")

    def _ma_space(self) -> dict:
        return {"short_window": {"min": 5, "max": 12}, "long_window": {"min": 15, "max": 50}}

    def _ma_rsi_space(self) -> dict:
        return {
            "short_window": {"min": 5, "max": 10},
            "long_window": {"min": 20, "max": 40},
            "rsi_period": {"min": 10, "max": 20},
            "rsi_min": {"min": 30, "max": 40},
            "rsi_max": {"min": 60, "max": 70},
        }

    def _breakout_space(self) -> dict:
        return {"breakout_window": {"min": 20, "max": 60}, "exit_window": {"min": 5, "max": 15}}


if __name__ == "__main__":
    unittest.main()
