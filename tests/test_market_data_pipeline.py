import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_agent_trading_lab.agents.data_agent import DataAgent
from multi_agent_trading_lab.data.data_sources import load_universe_history


class MarketDataPipelineTests(unittest.TestCase):
    def test_data_agent_uses_configured_symbols(self) -> None:
        settings = {
            "data": {
                "provider": "yfinance",
                "symbols": ["WES.AX", "NVDA"],
                "start_date": "2020-01-01",
                "end_date": None,
                "timeframe": "1d",
                "cache_dir": "data/cache",
            }
        }
        fake_history = {"WES.AX": object(), "NVDA": object()}

        with patch("multi_agent_trading_lab.agents.data_agent.load_universe_history", return_value=fake_history) as loader:
            history = DataAgent(settings).get_universe_history()

        self.assertEqual(history, fake_history)
        self.assertEqual(loader.call_args.kwargs["symbols"], ["WES.AX", "NVDA"])
        self.assertEqual(loader.call_args.kwargs["start"], "2020-01-01")
        self.assertIsNone(loader.call_args.kwargs["end"])

    @unittest.skipUnless(
        os.environ.get("RUN_ONLINE_MARKET_DATA_TESTS") == "1"
        and importlib.util.find_spec("pandas")
        and importlib.util.find_spec("yfinance"),
        "Set RUN_ONLINE_MARKET_DATA_TESTS=1 with pandas/yfinance installed to run online market data test.",
    )
    def test_load_universe_history_online_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history = load_universe_history(
                symbols=["NVDA"],
                start="2023-01-01",
                end="2023-02-01",
                timeframe="1d",
                cache_dir=Path(tmpdir),
            )

        self.assertIn("NVDA", history)
        self.assertFalse(history["NVDA"].empty)


if __name__ == "__main__":
    unittest.main()
