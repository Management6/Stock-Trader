import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator


class PortfolioPromotionGateTests(unittest.TestCase):
    def test_portfolio_gate_is_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._orchestrator(Path(tmpdir), portfolio={"enabled": False})
            self._wire_single_candidate(orchestrator)

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.run_portfolio_backtest") as portfolio:
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            portfolio.assert_not_called()
            self.assertEqual(len(orchestrator.approval_queue.list_pending()), 1)
            self.assertNotIn("portfolio_backtest", result.details["records"][0])

    def test_portfolio_failure_blocks_promotion_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = self._orchestrator(root)
            self._wire_single_candidate(orchestrator)
            portfolio_result = self._portfolio_result(metrics={"total_return": 0.10, "max_drawdown": -0.31, "sharpe_ratio": 0.50})

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.run_portfolio_backtest", return_value=portfolio_result):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(orchestrator.approval_queue.list_pending(), [])
            decision = result.details["records"][0]["risk_decision"]
            self.assertFalse(decision["approved"])
            self.assertIn("Portfolio gate failed: max_drawdown -0.31 is below threshold -0.25.", decision["reason"])
            self.assertEqual(result.details["records"][0]["portfolio_backtest"]["metrics"]["max_drawdown"], -0.31)
            audit_events = [json.loads(line) for line in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
            rejection = next(event for event in audit_events if event["event_type"] == "strategy_rejected")
            self.assertEqual(rejection["payload"]["portfolio_backtest"]["metrics"]["max_drawdown"], -0.31)

    def test_portfolio_pass_allows_promotion_and_includes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._orchestrator(Path(tmpdir))
            self._wire_single_candidate(orchestrator)
            portfolio_result = self._portfolio_result(metrics={"total_return": 0.04, "max_drawdown": -0.08, "sharpe_ratio": 0.35})

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.run_portfolio_backtest", return_value=portfolio_result):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            pending = orchestrator.approval_queue.list_pending()
            self.assertEqual(len(pending), 1)
            self.assertIn("portfolio_backtest", pending[0].metrics)
            self.assertTrue(result.details["records"][0]["risk_decision"]["approved"])
            self.assertIn("Portfolio gate passed", result.summaries[0])

    def test_skipped_symbols_and_data_quality_are_in_audit_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            orchestrator = self._orchestrator(root, portfolio={"enabled": True, "min_valid_symbols": 3, "max_skipped_symbol_ratio": 0.20})
            self._wire_single_candidate(orchestrator)
            portfolio_result = self._portfolio_result(
                symbols=["AAA", "BBB", "CCC"],
                skipped_symbols=[{"symbol": "BAD", "reason": "Data quality errors.", "data_quality_issues": [{"code": "invalid_prices", "severity": "error", "symbol": "BAD"}]}],
                data_quality={"passed": False, "issues": [{"code": "invalid_prices", "severity": "error", "symbol": "BAD"}]},
                metrics={"total_return": 0.04, "max_drawdown": -0.08, "sharpe_ratio": 0.35},
            )

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.run_portfolio_backtest", return_value=portfolio_result):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(orchestrator.approval_queue.list_pending(), [])
            self.assertIn("skipped symbol ratio", result.details["records"][0]["risk_decision"]["reason"])
            audit_events = [json.loads(line) for line in (root / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
            rejection = next(event for event in audit_events if event["event_type"] == "strategy_rejected")
            self.assertEqual(rejection["payload"]["portfolio_backtest"]["skipped_symbols"][0]["symbol"], "BAD")
            self.assertEqual(rejection["payload"]["portfolio_backtest"]["data_quality"]["issues"][0]["code"], "invalid_prices")

    def test_portfolio_gate_allows_some_skips_when_min_valid_symbols_is_met(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._orchestrator(Path(tmpdir), portfolio={"enabled": True, "min_valid_symbols": 2, "max_skipped_symbol_ratio": 0.50})
            self._wire_single_candidate(orchestrator)
            portfolio_result = self._portfolio_result(
                symbols=["AAA", "BBB"],
                skipped_symbols=[{"symbol": "BROKEN", "reason": "two or more bars", "data_quality_issues": []}],
                metrics={"total_return": 0.04, "max_drawdown": -0.08, "sharpe_ratio": 0.35},
            )

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.run_portfolio_backtest", return_value=portfolio_result):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(len(orchestrator.approval_queue.list_pending()), 1)
            self.assertEqual(result.details["records"][0]["portfolio_backtest"]["skipped_symbols"][0]["symbol"], "BROKEN")

    def test_portfolio_gate_blocks_when_too_few_symbols_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = self._orchestrator(Path(tmpdir), portfolio={"enabled": True, "min_valid_symbols": 3})
            self._wire_single_candidate(orchestrator)
            portfolio_result = self._portfolio_result(
                symbols=["AAA", "BBB"],
                skipped_symbols=[{"symbol": "BROKEN", "reason": "two or more bars", "data_quality_issues": []}],
                metrics={"total_return": 0.04, "max_drawdown": -0.08, "sharpe_ratio": 0.35},
            )

            with patch("multi_agent_trading_lab.orchestrator.orchestrator.run_portfolio_backtest", return_value=portfolio_result):
                result = orchestrator.run_research_cycle(n_variants=1, strategy_families=["ma"])

            self.assertEqual(orchestrator.approval_queue.list_pending(), [])
            self.assertIn("valid symbols 2 is below threshold 3", result.details["records"][0]["risk_decision"]["reason"])

    def _orchestrator(self, root: Path, portfolio: dict | None = None) -> TradingLabOrchestrator:
        portfolio_config = {
            "enabled": True,
            "allocation_method": "equal_weight",
            "max_symbols": None,
            "min_valid_symbols": 3,
            "min_total_return": -0.02,
            "max_drawdown": -0.25,
            "min_sharpe": 0.0,
            "max_skipped_symbol_ratio": 0.50,
        }
        if portfolio:
            portfolio_config.update(portfolio)
        return TradingLabOrchestrator(
            {
                "data": {"symbols": ["AAA", "BBB", "CCC"], "start_date": "2023-01-01", "end_date": "2023-01-10"},
                "universes": {},
                "experiment_log_path": str(root / "experiments.jsonl"),
                "audit_log_path": str(root / "audit.jsonl"),
                "alert_log_path": str(root / "alerts.jsonl"),
                "approval_queue_path": str(root / "approvals.json"),
                "strategy_registry_path": str(root / "registry.json"),
                "system_state_path": str(root / "state.json"),
                "broker": {"paper_state_path": str(root / "paper.json"), "live_enabled": False},
                "oos_validation": {"enabled": False, "split_ratio": 0.5, "min_sharpe": 0.0, "max_drawdown": -0.20, "min_return": -0.02},
                "portfolio": portfolio_config,
            }
        )

    def _wire_single_candidate(self, orchestrator: TradingLabOrchestrator) -> None:
        variant = {
            "id": "portfolio_gate_candidate_s5_l20",
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
        orchestrator.data_agent.fetch_data = lambda *_args, **_kwargs: {"AAA": bars, "BBB": bars, "CCC": bars}
        orchestrator.data_agent.build_features = lambda raw_data, _windows: raw_data
        orchestrator.backtest_agent.run_backtest = lambda *_args, **_kwargs: {
            "symbols": ["AAA", "BBB", "CCC"],
            "strategy": variant,
            "per_symbol_results": {},
            "metrics": {"total_return": 0.12, "max_drawdown": -0.05, "sharpe_ratio": 0.70, "sharpe": 0.70, "history_points": 10},
            "cost_assumptions": {"commission_per_trade": 0.0, "slippage_pct": 0.0005},
            "data_quality": {"passed": True, "issues": []},
        }

    def _portfolio_result(
        self,
        *,
        symbols: list[str] | None = None,
        skipped_symbols: list[dict] | None = None,
        data_quality: dict | None = None,
        metrics: dict[str, float],
    ) -> dict:
        symbols = symbols or ["AAA", "BBB", "CCC"]
        skipped_symbols = skipped_symbols or []
        data_quality = data_quality or {"passed": True, "issues": []}
        return {
            "symbols": symbols,
            "allocation_method": "equal_weight",
            "allocations": {symbol: 1.0 / len(symbols) for symbol in symbols},
            "portfolio_values": [100000.0, 101000.0, 102000.0],
            "metrics": {
                **metrics,
                "sharpe": metrics["sharpe_ratio"],
                "history_points": 2,
                "per_symbol_returns": {symbol: metrics["total_return"] for symbol in symbols},
                "per_symbol_trade_counts": {symbol: 1 for symbol in symbols},
                "aggregate_exposure": {"symbol_count": len(symbols), "max_allocated_pct": 1.0 / len(symbols), "gross_allocated_pct": 1.0},
                "cost_assumptions": {"commission_per_trade": 0.0, "slippage_pct": 0.0005},
                "data_quality": data_quality,
            },
            "per_symbol_results": {},
            "skipped_symbols": skipped_symbols,
            "aggregate_exposure": {"symbol_count": len(symbols), "max_allocated_pct": 1.0 / len(symbols), "gross_allocated_pct": 1.0},
            "cost_assumptions": {"commission_per_trade": 0.0, "slippage_pct": 0.0005},
            "data_quality": data_quality,
        }


if __name__ == "__main__":
    unittest.main()
