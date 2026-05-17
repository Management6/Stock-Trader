"""Data agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from multi_agent_trading_lab.agents.base_agent import AgentResult, BaseAgent
from multi_agent_trading_lab.data.data_sources import (
    MarketData,
    dataframe_to_market_data,
    fetch_stock_data,
    load_universe_history,
)
from multi_agent_trading_lab.data.data_quality import DataQualityReport, validate_market_data
from multi_agent_trading_lab.data.feature_engineering import add_strategy_features
from multi_agent_trading_lab.data.universes import resolve_universe_symbols


class DataAgent(BaseAgent):
    """Fetches and prepares historical or live market data.

    Inputs: symbols and date ranges.
    Outputs: MarketData containing price bars and derived features.
    Extension point: replace fetch_stock_data with authenticated data adapters.
    """

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        super().__init__("data_agent")
        self.settings = settings or {}
        self._universe_history: dict[str, Any] | None = None
        self.last_load_source = "not_loaded"
        self.last_data_quality: DataQualityReport | None = None

    def fetch_data(self, symbols: list[str] | None = None, start_date: str | None = None, end_date: str | None = None) -> MarketData:
        """Return market data in the internal list-of-bars format."""

        data_config = self._data_config()
        provider = str(data_config.get("provider", "synthetic"))
        resolved_symbols = symbols or resolve_universe_symbols(self.settings)
        resolved_start = start_date or str(data_config.get("start_date", "2023-01-01"))
        resolved_end = end_date if end_date is not None else data_config.get("end_date")
        if provider == "yfinance":
            try:
                market_data = dataframe_to_market_data(
                    self.get_universe_history(
                        symbols=resolved_symbols,
                        start=resolved_start,
                        end=resolved_end,
                    )
                )
                self.last_load_source = "yfinance_cache"
                return self.validate_data_quality(market_data)
            except Exception:
                fallback_end = str(resolved_end or datetime_today())
                self.last_load_source = "synthetic_fallback"
                return self.validate_data_quality(fetch_stock_data(resolved_symbols, resolved_start, fallback_end, use_yfinance=False))
        fallback_end = str(resolved_end or datetime_today())
        self.last_load_source = "synthetic"
        return self.validate_data_quality(fetch_stock_data(resolved_symbols, resolved_start, fallback_end, use_yfinance=False))

    def get_universe_history(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        force_refresh: bool | None = None,
    ) -> dict[str, Any]:
        """Load configured universe history as pandas DataFrames."""

        data_config = self._data_config()
        resolved_symbols = symbols or resolve_universe_symbols(self.settings)
        resolved_start = start or str(data_config.get("start_date", "2010-01-01"))
        resolved_end = end if end is not None else data_config.get("end_date")
        timeframe = str(data_config.get("timeframe", "1d"))
        cache_dir = Path(str(data_config.get("cache_dir", "data/cache")))
        refresh = bool(data_config.get("force_refresh", False) if force_refresh is None else force_refresh)
        if self._universe_history is not None and not refresh and symbols is None and start is None and end is None:
            return self._universe_history
        history = load_universe_history(
            symbols=resolved_symbols,
            start=resolved_start,
            end=None if resolved_end in {None, "null", ""} else str(resolved_end),
            timeframe=timeframe,
            cache_dir=cache_dir,
            force_refresh=refresh,
        )
        if symbols is None and start is None and end is None:
            self._universe_history = history
        return history

    def get_symbol_history(self, symbol: str) -> Any:
        """Return cached/downloaded pandas DataFrame history for one symbol."""

        history = self.get_universe_history()
        if symbol not in history:
            raise KeyError(f"{symbol} is not present in configured universe history")
        return history[symbol]

    def build_features(self, raw_data: MarketData, windows: list[int] | None = None) -> MarketData:
        return add_strategy_features(raw_data, windows or [5, 20])

    def validate_data_quality(self, data: MarketData) -> MarketData:
        data_config = self._data_config()
        report = validate_market_data(
            data,
            as_of_date=str(data_config.get("as_of_date")) if data_config.get("as_of_date") else None,
            max_staleness_days=self._quality_setting("max_staleness_days", data_config.get("max_staleness_days")),
            max_gap_days=self._quality_setting("max_gap_days", data_config.get("max_gap_days")),
            block_on_missing_columns=bool(self._quality_setting("block_on_missing_columns", True)),
            block_on_duplicate_dates=bool(self._quality_setting("block_on_duplicate_dates", True)),
            block_on_invalid_prices=bool(self._quality_setting("block_on_invalid_prices", True)),
            block_on_stale_data=bool(self._quality_setting("block_on_stale_data", False)),
            block_on_large_gaps=bool(self._quality_setting("block_on_large_gaps", False)),
            block_on_zero_volume=bool(self._quality_setting("block_on_zero_volume", False)),
            expect_volume=bool(self._quality_setting("expect_volume", True)),
        )
        self.last_data_quality = report
        if not report.passed:
            issue_summary = "; ".join(f"{issue.symbol}:{issue.code}" for issue in report.errors)
            raise ValueError(f"Market data failed quality validation: {issue_summary}")
        return data

    def run(self, symbols: list[str], start_date: str, end_date: str) -> AgentResult:
        raw_data = self.fetch_data(symbols, start_date, end_date)
        return AgentResult(self.name, self.build_features(raw_data))

    def _data_config(self) -> dict[str, Any]:
        if "data" in self.settings:
            return dict(self.settings["data"])
        return {
            "provider": "synthetic",
            "symbols": self.settings.get("symbols", ["AAPL"]),
            "start_date": self.settings.get("start_date", "2023-01-01"),
            "end_date": self.settings.get("end_date", "2023-06-30"),
            "timeframe": "1d",
            "cache_dir": "data/cache",
        }

    def _quality_setting(self, key: str, default: Any) -> Any:
        return dict(self.settings.get("data_quality", {})).get(key, default)


def datetime_today() -> str:
    from datetime import date

    return date.today().isoformat()
