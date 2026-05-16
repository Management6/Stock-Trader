"""Print the default operating profile."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.agents.discovery_agent import DiscoveryAgent


def main() -> None:
    result = DiscoveryAgent().run()
    for key, value in result.payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
