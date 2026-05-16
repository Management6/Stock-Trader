"""Run the sample multi-agent backtest workflow."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator, load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the sample multi-agent backtest workflow.")
    parser.add_argument("--universe", default=None, help="Named universe from config/settings.yaml.")
    args = parser.parse_args()
    settings = load_settings("multi_agent_trading_lab/config/settings.yaml")
    if args.universe:
        settings.setdefault("data", {})["default_universe"] = args.universe
    orchestrator = TradingLabOrchestrator(settings)
    result = orchestrator.run_research_cycle(n_variants=3)
    data = result.details.get("data", {})
    ranges = data.get("ranges", {})
    loaded_symbols = ", ".join(data.get("symbols", []))
    if ranges:
        starts = [payload["start_date"] for payload in ranges.values() if payload.get("start_date")]
        ends = [payload["end_date"] for payload in ranges.values() if payload.get("end_date")]
        source = data.get("source", "unknown")
        print(
            f"Loaded {source} daily data for {loaded_symbols} "
            f"from {min(starts) if starts else data.get('configured_start_date')} "
            f"to {max(ends) if ends else data.get('configured_end_date') or 'latest available'}."
        )
    print("\n".join(result.summaries))


if __name__ == "__main__":
    main()
