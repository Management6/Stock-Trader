"""Market data access helpers.

The default path tries yfinance when available. For a fresh repository or an
offline development environment, the synthetic fallback keeps examples and
tests deterministic.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from math import sin
from pathlib import Path
from random import Random
from typing import Any, Iterable

PriceBar = dict[str, float | str]
MarketData = dict[str, list[PriceBar]]


def download_symbol_history(
    symbol: str,
    start: str,
    end: str | None,
    timeframe: str,
    cache_dir: Path,
    force_refresh: bool = False,
) -> Any:
    """Download and cache OHLCV history for one symbol using yfinance.

    The returned object is a pandas DataFrame with normalized columns:
    ``date``, ``open``, ``high``, ``low``, ``close``, and ``volume``.
    Cached CSV files are reused unless ``force_refresh`` is true.
    """

    pd = _import_pandas()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_safe_cache_name(symbol)}_{timeframe}.csv"
    if cache_path.exists() and not force_refresh:
        frame = pd.read_csv(cache_path)
        if not frame.empty:
            return _normalize_ohlcv_frame(frame)

    import yfinance as yf  # type: ignore[import-not-found]

    frame = yf.download(
        symbol,
        start=start,
        end=end,
        interval=timeframe,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame is None or frame.empty:
        raise ValueError(f"No market data returned for {symbol}")
    normalized = _normalize_ohlcv_frame(frame)
    if normalized.empty:
        raise ValueError(f"No normalized OHLCV rows available for {symbol}")
    normalized.to_csv(cache_path, index=False)
    return normalized


def load_universe_history(
    symbols: list[str],
    start: str,
    end: str | None,
    timeframe: str,
    cache_dir: Path,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Load cached or downloaded OHLCV history for every configured symbol."""

    history: dict[str, Any] = {}
    failures: list[str] = []
    for symbol in symbols:
        try:
            history[symbol] = download_symbol_history(
                symbol=symbol,
                start=start,
                end=end,
                timeframe=timeframe,
                cache_dir=cache_dir,
                force_refresh=force_refresh,
            )
        except Exception as exc:
            failures.append(f"{symbol}: {exc}")
    if not history:
        raise RuntimeError("No universe data could be loaded. " + "; ".join(failures))
    return history


def dataframe_to_market_data(history: dict[str, Any]) -> MarketData:
    """Convert pandas DataFrames into the list-of-bars format used internally."""

    market_data: MarketData = {}
    for symbol, frame in history.items():
        rows: list[PriceBar] = []
        for record in frame.to_dict(orient="records"):
            rows.append(
                {
                    "date": str(record["date"])[:10],
                    "open": float(record["open"]),
                    "high": float(record["high"]),
                    "low": float(record["low"]),
                    "close": float(record["close"]),
                    "volume": float(record["volume"]),
                }
            )
        market_data[symbol] = rows
    return market_data


def fetch_stock_data(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    use_yfinance: bool = True,
) -> MarketData:
    """Fetch OHLCV-like daily bars for symbols.

    Returns a mapping of symbol to a list of bars with at least ``date`` and
    ``close`` keys. If yfinance is unavailable or returns no data, synthetic
    data is generated.
    """

    symbol_list = list(symbols)
    if use_yfinance:
        try:
            return _fetch_with_yfinance(symbol_list, start_date, end_date)
        except Exception:
            pass
    return generate_synthetic_data(symbol_list, start_date, end_date)


def _fetch_with_yfinance(symbols: list[str], start_date: str, end_date: str) -> MarketData:
    import yfinance as yf  # type: ignore[import-not-found]

    data: MarketData = {}
    for symbol in symbols:
        frame = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if frame.empty:
            raise ValueError(f"No yfinance rows returned for {symbol}")
        bars: list[PriceBar] = []
        for index, row in frame.iterrows():
            bars.append(
                {
                    "date": index.strftime("%Y-%m-%d"),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                }
            )
        data[symbol] = bars
    return data


def _normalize_ohlcv_frame(frame: Any) -> Any:
    pd = _import_pandas()
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = [column[0] for column in normalized.columns]
    if "Date" not in normalized.columns and "date" not in normalized.columns:
        normalized = normalized.reset_index()
    rename_map = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    normalized = normalized.rename(columns=rename_map)
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    normalized = normalized[required].dropna(subset=["date", "open", "high", "low", "close"])
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.strftime("%Y-%m-%d")
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized = normalized.dropna(subset=["open", "high", "low", "close", "volume"])
    return normalized.sort_values("date").reset_index(drop=True)


def _safe_cache_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_")


def _import_pandas() -> Any:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("pandas is required for cached market data. Install project dependencies first.") from exc
    return pd


def generate_synthetic_data(symbols: Iterable[str], start_date: str, end_date: str) -> MarketData:
    """Generate deterministic daily closing prices for local workflows."""

    start = _parse_date(start_date)
    end = _parse_date(end_date)
    data: MarketData = {}
    for symbol in symbols:
        rng = Random(symbol)
        price = 100.0 + rng.random() * 20.0
        bars: list[PriceBar] = []
        current = start
        day_index = 0
        while current <= end:
            if current.weekday() < 5:
                drift = 0.03 + sin(day_index / 8.0) * 0.45 + rng.uniform(-0.6, 0.6)
                price = max(1.0, price + drift)
                bars.append(
                    {
                        "date": current.isoformat(),
                        "open": round(price - rng.uniform(-0.4, 0.4), 2),
                        "high": round(price + rng.uniform(0.1, 1.0), 2),
                        "low": round(price - rng.uniform(0.1, 1.0), 2),
                        "close": round(price, 2),
                        "volume": float(1_000_000 + rng.randint(0, 250_000)),
                    }
                )
                day_index += 1
            current += timedelta(days=1)
        data[symbol] = bars
    return data


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()
