"""Download and cache configured universe history."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.agents.data_agent import DataAgent
from multi_agent_trading_lab.orchestrator.orchestrator import load_settings


def main() -> None:
    settings = load_settings("multi_agent_trading_lab/config/settings.yaml")
    agent = DataAgent(settings)
    data_config = settings.get("data", {})
    try:
        history = agent.get_universe_history(force_refresh=bool(data_config.get("force_refresh", False)))
    except Exception as exc:
        print(f"Could not download data: {exc}")
        print("Install dependencies with: python3 -m pip install -e .")
        raise SystemExit(1)

    print("Downloaded/cached market data:")
    for symbol, frame in history.items():
        first_date = str(frame.iloc[0]["date"])[:10] if not frame.empty else "n/a"
        last_date = str(frame.iloc[-1]["date"])[:10] if not frame.empty else "n/a"
        print(f"- {symbol}: {len(frame)} rows, {first_date} to {last_date}")


if __name__ == "__main__":
    main()
