import unittest

from multi_agent_trading_lab.agents.data_agent import DataAgent
from multi_agent_trading_lab.data.data_sources import generate_synthetic_data


class DataAgentTests(unittest.TestCase):
    def test_data_agent_fetch_and_build_features(self) -> None:
        agent = DataAgent()
        raw = generate_synthetic_data(["AAPL"], "2023-01-01", "2023-01-31")
        featured = agent.build_features(raw, windows=[3, 5])

        self.assertIn("AAPL", featured)
        self.assertTrue(featured["AAPL"])
        self.assertIn("sma_3", featured["AAPL"][0])
        self.assertIn("sma_5", featured["AAPL"][0])


if __name__ == "__main__":
    unittest.main()
