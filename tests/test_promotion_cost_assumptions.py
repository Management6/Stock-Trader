import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator


class PromotionCostAssumptionTests(unittest.TestCase):
    def test_promotion_metrics_include_configured_cost_assumptions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._orchestrator(Path(tmpdir), commission_per_trade=1.25, slippage_pct=0.0025)
            self._wire_candidate_and_data(orchestrator, closes=[100.0, 102.0, 104.0, 106.0, 108.0])

            result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            pending = orchestrator.approval_queue.list_pending()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].metrics["cost_assumptions"], {"commission_per_trade": 1.25, "slippage_pct": 0.0025})
            self.assertEqual(result.details["records"][0]["cost_assumptions"], {"commission_per_trade": 1.25, "slippage_pct": 0.0025})
            self.assertIn("costs=commission_per_trade=1.25, slippage_pct=0.0025", result.summaries[0])

    def test_configured_costs_reduce_research_return_vs_explicit_zero_costs(self) -> None:
        with tempfile.TemporaryDirectory() as zero_dir, tempfile.TemporaryDirectory() as cost_dir:
            closes = [100.0, 100.6, 101.2, 101.8, 102.4]
            zero_cost = self._orchestrator(Path(zero_dir), commission_per_trade=0.0, slippage_pct=0.0)
            with_costs = self._orchestrator(Path(cost_dir), commission_per_trade=0.0, slippage_pct=0.01)
            self._wire_candidate_and_data(zero_cost, closes=closes)
            self._wire_candidate_and_data(with_costs, closes=closes)

            zero_result = zero_cost.run_research_cycle(n_variants=1, strategy_families=["ma"])
            cost_result = with_costs.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertLess(
                cost_result.details["records"][0]["metrics"]["total_return"],
                zero_result.details["records"][0]["metrics"]["total_return"],
            )

    def test_strategy_promoted_with_zero_costs_can_fail_under_configured_costs(self) -> None:
        with tempfile.TemporaryDirectory() as zero_dir, tempfile.TemporaryDirectory() as cost_dir:
            closes = [100.0, 100.6, 101.2, 101.8, 102.4]
            zero_cost = self._orchestrator(Path(zero_dir), commission_per_trade=0.0, slippage_pct=0.0, min_backtest_return=0.002)
            with_costs = self._orchestrator(Path(cost_dir), commission_per_trade=0.0, slippage_pct=0.01, min_backtest_return=0.002)
            self._wire_candidate_and_data(zero_cost, closes=closes)
            self._wire_candidate_and_data(with_costs, closes=closes)

            zero_result = zero_cost.run_research_cycle(n_variants=1, strategy_families=["ma"])
            cost_result = with_costs.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(len(zero_cost.approval_queue.list_pending()), 1)
            self.assertEqual(with_costs.approval_queue.list_pending(), [])
            self.assertTrue(zero_result.details["records"][0]["risk_decision"]["approved"])
            self.assertFalse(cost_result.details["records"][0]["risk_decision"]["approved"])
            self.assertEqual(cost_result.details["records"][0]["risk_decision"]["reason"], "Strategy return is below deployment threshold.")
            self.assertEqual(
                cost_result.details["records"][0]["risk_decision"]["details"]["cost_assumptions"],
                {"commission_per_trade": 0.0, "slippage_pct": 0.01},
            )

    def test_explicit_zero_cost_research_mode_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._orchestrator(Path(tmpdir), commission_per_trade=0.0, slippage_pct=0.0)
            self._wire_candidate_and_data(orchestrator, closes=[100.0, 102.0, 104.0, 106.0, 108.0])

            result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(result.details["records"][0]["cost_assumptions"], {"commission_per_trade": 0.0, "slippage_pct": 0.0})
            self.assertEqual(orchestrator.approval_queue.list_pending()[0].metrics["cost_assumptions"]["slippage_pct"], 0.0)

    def _orchestrator(
        self,
        root: Path,
        *,
        commission_per_trade: float,
        slippage_pct: float,
        min_backtest_return: float = -0.05,
    ) -> TradingLabOrchestrator:
        return TradingLabOrchestrator(
            {
                "data": {"symbols": ["TEST"], "start_date": "2023-01-01", "end_date": "2023-01-05"},
                "universes": {},
                "experiment_log_path": str(root / "experiments.jsonl"),
                "audit_log_path": str(root / "audit.jsonl"),
                "alert_log_path": str(root / "alerts.jsonl"),
                "approval_queue_path": str(root / "approvals.json"),
                "strategy_registry_path": str(root / "registry.json"),
                "system_state_path": str(root / "state.json"),
                "broker": {"paper_state_path": str(root / "paper.json"), "live_enabled": False},
                "backtest": {"commission_per_trade": commission_per_trade, "slippage_pct": slippage_pct},
                "risk": {"min_backtest_return": min_backtest_return, "research": {"max_drawdown": -0.20}, "paper": {"max_drawdown": -0.20}},
                "oos_validation": {"enabled": False},
            }
        )

    def _wire_candidate_and_data(self, orchestrator: TradingLabOrchestrator, *, closes: list[float]) -> None:
        variant = {
            "id": "cost_candidate_s5_l20",
            "name": "moving_average_crossover",
            "version": "0.1.0",
            "short_window": 5,
            "long_window": 20,
            "strategy_params": {"short_window": 5, "long_window": 20},
        }
        bars = [
            {"date": f"2023-01-{index + 1:02d}", "close": close, "sma_5": close + 1.0, "sma_20": close}
            for index, close in enumerate(closes)
        ]
        orchestrator.strategy_agent.suggest_from_history = lambda *_args, **_kwargs: [variant]
        orchestrator.data_agent.fetch_data = lambda *_args, **_kwargs: {"TEST": bars}
        orchestrator.data_agent.build_features = lambda raw_data, _windows: raw_data


if __name__ == "__main__":
    unittest.main()
