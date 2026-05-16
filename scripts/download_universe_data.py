"""Download and cache daily data for a configured stock universe."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.data.data_sources import download_symbol_history
from multi_agent_trading_lab.data.universes import resolve_universe_symbols, validate_universe_symbols
from multi_agent_trading_lab.orchestrator.orchestrator import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download/cache daily OHLCV data for a configured universe.")
    parser.add_argument("--settings", default="multi_agent_trading_lab/config/settings.yaml")
    parser.add_argument("--universe", default=None, help="Named universe from config/settings.yaml. Defaults to data.default_universe.")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings(args.settings)
    data_config = dict(settings.get("data", {}))
    universe_name = args.universe or str(data_config.get("default_universe", "data.symbols"))
    try:
        symbols = resolve_universe_symbols(settings, args.universe)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    warnings = validate_universe_symbols(universe_name, symbols)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if not symbols:
        return 2

    start = str(data_config.get("start_date", "2010-01-01"))
    end = data_config.get("end_date")
    timeframe = str(data_config.get("timeframe", "1d"))
    cache_dir = Path(str(data_config.get("cache_dir", "data/cache")))
    successes: list[dict[str, Any]] = []
    failures: list[str] = []
    print(f"Downloading {len(symbols)} symbols for universe {universe_name!r}: {start} to {end or 'latest'} ({timeframe})")
    for symbol in symbols:
        try:
            frame = _download_with_retries(
                symbol=symbol,
                start=start,
                end=None if end in {None, "null", ""} else str(end),
                timeframe=timeframe,
                cache_dir=cache_dir,
                force_refresh=bool(args.force_refresh or data_config.get("force_refresh", False)),
                retries=max(1, args.retries),
                retry_sleep=max(0.0, args.retry_sleep),
            )
            first_date = str(frame.iloc[0]["date"])[:10] if not frame.empty else "n/a"
            last_date = str(frame.iloc[-1]["date"])[:10] if not frame.empty else "n/a"
            successes.append({"symbol": symbol, "rows": len(frame), "first_date": first_date, "last_date": last_date})
            print(f"- OK {symbol}: {len(frame)} rows, {first_date} to {last_date}")
        except Exception as exc:
            failures.append(f"{symbol}: {exc}")
            print(f"- FAILED {symbol}: {exc}")

    print()
    print(f"Universe download summary: {len(successes)}/{len(symbols)} succeeded")
    if successes:
        first_dates = [item["first_date"] for item in successes if item["first_date"] != "n/a"]
        last_dates = [item["last_date"] for item in successes if item["last_date"] != "n/a"]
        print(f"Loaded date range: {min(first_dates) if first_dates else 'n/a'} to {max(last_dates) if last_dates else 'n/a'}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    return 0


def _download_with_retries(
    symbol: str,
    start: str,
    end: str | None,
    timeframe: str,
    cache_dir: Path,
    force_refresh: bool,
    retries: int,
    retry_sleep: float,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return download_symbol_history(symbol, start, end, timeframe, cache_dir, force_refresh=force_refresh)
        except Exception as exc:
            last_exc = exc
            if attempt < retries and retry_sleep:
                time.sleep(retry_sleep)
    assert last_exc is not None
    raise last_exc


if __name__ == "__main__":
    raise SystemExit(main())
