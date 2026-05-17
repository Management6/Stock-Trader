import unittest

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.data.data_quality import validate_market_data
from multi_agent_trading_lab.research.portfolio_backtest import run_portfolio_backtest


class PortfolioBacktestTests(unittest.TestCase):
    def test_portfolio_backtest_runs_strategy_across_multiple_symbols(self) -> None:
        result = run_portfolio_backtest(self._strategy(), {"AAA": self._bars([100, 102, 104, 106]), "BBB": self._bars([50, 51, 52, 53])})

        self.assertEqual(result["symbols"], ["AAA", "BBB"])
        self.assertEqual(result["allocation_method"], "equal_weight")
        self.assertEqual(result["allocations"], {"AAA": 0.5, "BBB": 0.5})
        self.assertIn("AAA", result["per_symbol_results"])
        self.assertIn("BBB", result["per_symbol_results"])

    def test_portfolio_result_exposes_equity_metrics_per_symbol_returns_trade_counts_and_exposure(self) -> None:
        result = run_portfolio_backtest(
            self._strategy(),
            {"AAA": self._bars([100, 102, 104, 106]), "BBB": self._bars([100, 99, 101, 103])},
            backtest_agent=BacktestAgent(starting_capital=100_000.0, max_capital_per_trade_pct=1.0),
        )

        metrics = result["metrics"]
        self.assertIn("portfolio_values", result)
        self.assertEqual(len(result["portfolio_values"]), 4)
        self.assertIn("total_return", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("sharpe_ratio", metrics)
        self.assertEqual(set(metrics["per_symbol_returns"]), {"AAA", "BBB"})
        self.assertEqual(set(metrics["per_symbol_trade_counts"]), {"AAA", "BBB"})
        self.assertGreater(metrics["aggregate_exposure"]["max_allocated_pct"], 0.0)
        self.assertLessEqual(metrics["aggregate_exposure"]["max_allocated_pct"], 1.0)

    def test_skipped_or_failed_symbols_do_not_crash_portfolio_run(self) -> None:
        result = run_portfolio_backtest(self._strategy(), {"AAA": self._bars([100, 102, 104]), "BROKEN": [self._bar("2023-01-01", 10.0)]})

        self.assertEqual(result["symbols"], ["AAA"])
        self.assertEqual(result["skipped_symbols"][0]["symbol"], "BROKEN")
        self.assertIn("two or more bars", result["skipped_symbols"][0]["reason"])

    def test_data_quality_failure_for_one_symbol_is_surfaced_and_skipped(self) -> None:
        data = {"AAA": self._bars([100, 102, 104]), "BAD": self._bars([100, 101, 102])}
        data["BAD"][1]["close"] = -1.0
        quality = validate_market_data(data)

        result = run_portfolio_backtest(self._strategy(), data, data_quality_report=quality.to_dict())

        self.assertEqual(result["symbols"], ["AAA"])
        self.assertEqual(result["skipped_symbols"][0]["symbol"], "BAD")
        self.assertEqual(result["skipped_symbols"][0]["data_quality_issues"][0]["code"], "invalid_prices")
        self.assertEqual(result["data_quality"]["issues"][0]["symbol"], "BAD")

    def test_portfolio_drawdown_can_differ_from_single_symbol_drawdown(self) -> None:
        strategy = self._strategy()
        data = {"AAA": self._bars([100, 120, 80, 120]), "BBB": self._bars([100, 101, 102, 103])}

        single = BacktestAgent(starting_capital=100_000.0, max_capital_per_trade_pct=1.0).run_backtest(strategy, {"AAA": data["AAA"]})
        portfolio = run_portfolio_backtest(
            strategy,
            data,
            backtest_agent=BacktestAgent(starting_capital=100_000.0, max_capital_per_trade_pct=1.0),
        )

        self.assertNotEqual(portfolio["metrics"]["max_drawdown"], single["metrics"]["max_drawdown"])

    def test_portfolio_metrics_include_cost_assumptions_and_data_quality_summary(self) -> None:
        quality = validate_market_data({"AAA": self._bars([100, 101, 102])})
        result = run_portfolio_backtest(
            self._strategy(),
            {"AAA": self._bars([100, 101, 102])},
            backtest_agent=BacktestAgent(commission_per_trade=1.25, slippage_pct=0.0025),
            data_quality_report=quality.to_dict(),
        )

        self.assertEqual(result["cost_assumptions"], {"commission_per_trade": 1.25, "slippage_pct": 0.0025})
        self.assertEqual(result["metrics"]["cost_assumptions"], {"commission_per_trade": 1.25, "slippage_pct": 0.0025})
        self.assertTrue(result["metrics"]["data_quality"]["passed"])

    def _strategy(self) -> dict:
        return {
            "name": "moving_average_crossover",
            "version": "0.1.0",
            "strategy_params": {"short_window": 5, "long_window": 20},
        }

    def _bars(self, closes: list[float]) -> list[dict[str, float | str]]:
        return [self._bar(f"2023-01-{index + 1:02d}", close) for index, close in enumerate(closes)]

    def _bar(self, date: str, close: float) -> dict[str, float | str]:
        return {"date": date, "open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1000.0, "sma_5": close + 1.0, "sma_20": close}


if __name__ == "__main__":
    unittest.main()
