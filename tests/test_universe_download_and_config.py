import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_agent_trading_lab.agents.data_agent import DataAgent
from multi_agent_trading_lab.data.universes import resolve_universe_symbols, validate_universe_symbols
from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator
from scripts.download_universe_data import main as download_main


class UniverseConfigTests(unittest.TestCase):
    def test_default_universe_resolves_from_config(self) -> None:
        settings = {
            "data": {"default_universe": "core_asx"},
            "universes": {"core_asx": ["CBA.AX", "BHP.AX", "WES.AX"]},
        }

        self.assertEqual(resolve_universe_symbols(settings), ["CBA.AX", "BHP.AX", "WES.AX"])

    def test_cli_universe_override_resolves_symbols(self) -> None:
        settings = {
            "data": {"default_universe": "core_asx"},
            "universes": {"core_asx": ["CBA.AX"], "us_tech": ["NVDA", "AAPL"]},
        }

        self.assertEqual(resolve_universe_symbols(settings, "us_tech"), ["NVDA", "AAPL"])

    def test_asx_suffix_is_preserved_not_rewritten(self) -> None:
        warnings = validate_universe_symbols("core_asx", ["CBA.AX", "BHP.AX", "NVDA"])

        self.assertEqual(warnings, [])

    def test_full_mix_warns_when_too_small(self) -> None:
        warnings = validate_universe_symbols("full_mix", ["CBA.AX", "NVDA"])

        self.assertTrue(any("fewer than 3" in warning for warning in warnings))

    def test_data_agent_uses_default_universe_when_symbols_omitted(self) -> None:
        settings = {
            "data": {
                "provider": "yfinance",
                "default_universe": "us_tech",
                "start_date": "2020-01-01",
                "end_date": None,
                "timeframe": "1d",
                "cache_dir": "data/cache",
            },
            "universes": {"us_tech": ["NVDA", "AAPL"]},
        }
        with patch("multi_agent_trading_lab.agents.data_agent.load_universe_history", return_value={"NVDA": object(), "AAPL": object()}) as loader:
            DataAgent(settings).get_universe_history()

        self.assertEqual(loader.call_args.kwargs["symbols"], ["NVDA", "AAPL"])

    def test_orchestrator_data_config_uses_default_universe(self) -> None:
        orchestrator = TradingLabOrchestrator(
            {
                "data": {"default_universe": "core_asx", "symbols": ["OLD"], "start_date": "2020-01-01"},
                "universes": {"core_asx": ["CBA.AX", "BHP.AX"]},
            }
        )

        self.assertEqual(orchestrator._data_config()["symbols"], ["CBA.AX", "BHP.AX"])

    def test_download_cli_uses_universe_override_and_reports_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "settings.yaml"
            settings_path.write_text(
                "data:\n"
                "  default_universe: core_asx\n"
                "  start_date: 2020-01-01\n"
                "  end_date: null\n"
                "  timeframe: 1d\n"
                f"  cache_dir: {tmpdir}/cache\n"
                "universes:\n"
                "  core_asx:\n"
                "    - CBA.AX\n"
                "  us_tech:\n"
                "    - NVDA\n",
                encoding="utf-8",
            )

            with patch("scripts.download_universe_data.download_symbol_history", side_effect=RuntimeError("offline")):
                exit_code = download_main(["--settings", str(settings_path), "--universe", "us_tech", "--retries", "1"])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
