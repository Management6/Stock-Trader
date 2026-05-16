"""Run one safe paper-trading cycle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator, load_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run one safe paper-trading cycle. Selected strategies are gated by "
            "risk.paper.max_drawdown in multi_agent_trading_lab/config/settings.yaml; "
            "make it less negative to be stricter or more negative to be looser."
        )
    )
    parser.add_argument("--settings", default="multi_agent_trading_lab/config/settings.yaml", help="Path to settings YAML.")
    args = parser.parse_args()
    result = TradingLabOrchestrator(load_settings(args.settings)).run_paper_cycle()
    print("\n".join(result.summaries))


if __name__ == "__main__":
    main()
