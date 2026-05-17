import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator


class RegimePromotionGateTests(unittest.TestCase):
    def test_regime_gate_disabled_preserves_promotion_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._orchestrator(Path(tmpdir), regime={"enabled": False})
            self._wire_single_candidate(orchestrator)

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.classify_market_regime") as classifier:
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            classifier.assert_not_called()
            self.assertEqual(len(orchestrator.approval_queue.list_pending()), 1)
            self.assertNotIn("market_regime", result.details["records"][0])

    def test_disallowed_regime_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = self._orchestrator(root)
            self._wire_single_candidate(orchestrator)
            regime = self._regime("bearish")

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.classify_market_regime", return_value=regime):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(orchestrator.approval_queue.list_pending(), [])
            decision = result.details["records"][0]["risk_decision"]
            self.assertFalse(decision["approved"])
            self.assertEqual(decision["reason"], "Regime gate failed: bearish is not allowed.")
            self.assertEqual(result.details["records"][0]["market_regime"], regime)
            audit_events = [json.loads(line) for line in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
            rejection = next(event for event in audit_events if event["event_type"] == "strategy_rejected")
            self.assertEqual(rejection["payload"]["market_regime"]["regime"], "bearish")

    def test_allowed_regime_permits_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._orchestrator(Path(tmpdir))
            self._wire_single_candidate(orchestrator)
            regime = self._regime("bullish")

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.classify_market_regime", return_value=regime):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            pending = orchestrator.approval_queue.list_pending()
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].metrics["market_regime"]["regime"], "bullish")
            self.assertIn("Regime gate passed", result.summaries[0])

    def test_insufficient_regime_data_blocks_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._orchestrator(Path(tmpdir), regime={"enabled": True, "block_on_insufficient_data": True})
            self._wire_single_candidate(orchestrator)
            regime = self._regime("insufficient_data", trend_metric=None, volatility_metric=None)

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.classify_market_regime", return_value=regime):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(orchestrator.approval_queue.list_pending(), [])
            self.assertIn("insufficient benchmark data", result.details["records"][0]["risk_decision"]["reason"])

    def test_insufficient_regime_data_warns_when_not_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._orchestrator(Path(tmpdir), regime={"enabled": True, "block_on_insufficient_data": False})
            self._wire_single_candidate(orchestrator)
            regime = self._regime("insufficient_data", trend_metric=None, volatility_metric=None)

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.classify_market_regime", return_value=regime):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(len(orchestrator.approval_queue.list_pending()), 1)
            self.assertEqual(result.details["records"][0]["risk_decision"]["details"]["market_regime_warning"], "insufficient_data")

    def _orchestrator(self, root: Path, regime: dict | None = None) -> TradingLabOrchestrator:
        regime_config = {
            "enabled": True,
            "benchmark_symbol": "SPY",
            "lookback_days": 60,
            "trend_ma_days": 20,
            "max_volatility": 0.30,
            "allowed_regimes": ["bullish", "trending"],
            "block_on_insufficient_data": True,
        }
        if regime:
            regime_config.update(regime)
        return TradingLabOrchestrator(
            {
                "data": {"symbols": ["TEST", "SPY"], "start_date": "2023-01-01", "end_date": "2023-04-30"},
                "universes": {},
                "experiment_log_path": str(root / "experiments.jsonl"),
                "audit_log_path": str(root / "audit.jsonl"),
                "alert_log_path": str(root / "alerts.jsonl"),
                "approval_queue_path": str(root / "approvals.json"),
                "strategy_registry_path": str(root / "registry.json"),
                "system_state_path": str(root / "state.json"),
                "broker": {"paper_state_path": str(root / "paper.json"), "live_enabled": False},
                "oos_validation": {"enabled": False, "split_ratio": 0.5, "min_sharpe": 0.0, "max_drawdown": -0.20, "min_return": -0.02},
                "portfolio": {"enabled": False},
                "regime": regime_config,
            }
        )

    def _wire_single_candidate(self, orchestrator: TradingLabOrchestrator) -> None:
        variant = {
            "id": "regime_gate_candidate_s5_l20",
            "name": "moving_average_crossover",
            "version": "0.1.0",
            "short_window": 5,
            "long_window": 20,
            "strategy_params": {"short_window": 5, "long_window": 20},
        }
        bars = [
            {"date": f"2023-01-{day:02d}", "open": 100.0 + day, "high": 101.0 + day, "low": 99.0 + day, "close": 100.0 + day, "volume": 1000.0, "sma_5": 101.0 + day, "sma_20": 100.0 + day}
            for day in range(1, 11)
        ]
        orchestrator.strategy_agent.suggest_from_history = lambda *_args, **_kwargs: [variant]
        orchestrator.data_agent.fetch_data = lambda *_args, **_kwargs: {"TEST": bars, "SPY": bars}
        orchestrator.data_agent.build_features = lambda raw_data, _windows: raw_data
        orchestrator.backtest_agent.run_backtest = lambda *_args, **_kwargs: {
            "symbols": ["TEST"],
            "strategy": variant,
            "per_symbol_results": {},
            "metrics": {"total_return": 0.12, "max_drawdown": -0.05, "sharpe_ratio": 0.70, "sharpe": 0.70, "history_points": 10},
            "cost_assumptions": {"commission_per_trade": 0.0, "slippage_pct": 0.0005},
            "data_quality": {"passed": True, "issues": []},
        }

    def _regime(self, regime: str, trend_metric: float | None = 0.05, volatility_metric: float | None = 0.12) -> dict:
        return {
            "regime": regime,
            "benchmark_symbol": "SPY",
            "lookback_days": 60,
            "trend_metric": trend_metric,
            "volatility_metric": volatility_metric,
            "reason": f"{regime} fixture",
            "details": {"latest_close": 120.0, "moving_average": 110.0},
        }


if __name__ == "__main__":
    unittest.main()
