import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.strategies.registry import StrategyRegistry


class StrategyRegistryTests(unittest.TestCase):
    def test_register_and_promote_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = StrategyRegistry(Path(tmpdir) / "registry.json")
            registry.register({"id": "s1", "name": "example"}, stage="candidate")
            promoted = registry.promote("s1", "active", "approved")

            self.assertEqual(promoted["stage"], "active")
            self.assertEqual(len(registry.list_by_stage("active")), 1)


if __name__ == "__main__":
    unittest.main()
