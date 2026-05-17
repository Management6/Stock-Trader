import unittest

from multi_agent_trading_lab.research.regime import classify_market_regime


class MarketRegimeClassifierTests(unittest.TestCase):
    def test_bullish_regime_above_moving_average_with_low_volatility(self) -> None:
        result = classify_market_regime(self._bars([100 + index for index in range(80)]), lookback_days=60, trend_ma_days=20, max_volatility=0.30)

        self.assertEqual(result["regime"], "bullish")
        self.assertEqual(result["lookback_days"], 60)
        self.assertGreater(result["trend_metric"], 0.0)
        self.assertLess(result["volatility_metric"], 0.30)
        self.assertIn("above moving average", result["reason"])

    def test_bearish_downtrend_below_moving_average(self) -> None:
        result = classify_market_regime(self._bars([200 - index for index in range(80)]), lookback_days=60, trend_ma_days=20, max_volatility=0.30)

        self.assertEqual(result["regime"], "bearish")
        self.assertLess(result["trend_metric"], 0.0)
        self.assertIn("below moving average", result["reason"])

    def test_high_volatility_regime_takes_precedence(self) -> None:
        closes = [100.0, 120.0, 80.0, 125.0, 75.0] * 20

        result = classify_market_regime(self._bars(closes), lookback_days=60, trend_ma_days=20, max_volatility=0.30)

        self.assertEqual(result["regime"], "high_volatility")
        self.assertGreater(result["volatility_metric"], 0.30)
        self.assertIn("volatility", result["reason"])

    def test_insufficient_data_regime(self) -> None:
        result = classify_market_regime(self._bars([100.0, 101.0, 102.0]), lookback_days=60, trend_ma_days=20, max_volatility=0.30)

        self.assertEqual(result["regime"], "insufficient_data")
        self.assertEqual(result["lookback_days"], 60)
        self.assertIsNone(result["trend_metric"])
        self.assertIsNone(result["volatility_metric"])
        self.assertIn("Insufficient benchmark data", result["reason"])

    def _bars(self, closes: list[float]) -> list[dict[str, float | str]]:
        return [{"date": f"2023-01-{index + 1:02d}", "close": close} for index, close in enumerate(closes)]


if __name__ == "__main__":
    unittest.main()
