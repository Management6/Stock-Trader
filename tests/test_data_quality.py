import unittest

from multi_agent_trading_lab.agents.data_agent import DataAgent
from multi_agent_trading_lab.data.data_quality import validate_market_data


class DataQualityTests(unittest.TestCase):
    def test_missing_columns_are_reported_as_structured_errors(self) -> None:
        report = validate_market_data({"TEST": [{"date": "2026-01-01", "close": 100.0}]})

        self.assertFalse(report.passed)
        self.assertEqual(report.issues[0].code, "missing_columns")
        self.assertEqual(report.issues[0].severity, "error")
        self.assertEqual(report.issues[0].symbol, "TEST")
        self.assertIn("open", report.issues[0].details["missing_columns"])

    def test_duplicate_dates_are_reported(self) -> None:
        report = validate_market_data({"TEST": [self._bar("2026-01-01"), self._bar("2026-01-01")]})

        issue = self._issue(report, "duplicate_dates")
        self.assertEqual(issue.severity, "error")
        self.assertEqual(issue.details["dates"], ["2026-01-01"])

    def test_stale_data_is_reported(self) -> None:
        report = validate_market_data(
            {"TEST": [self._bar("2026-01-01"), self._bar("2026-01-02")]},
            as_of_date="2026-01-10",
            max_staleness_days=3,
        )

        issue = self._issue(report, "stale_data")
        self.assertEqual(issue.severity, "warning")
        self.assertEqual(issue.details["latest_date"], "2026-01-02")
        self.assertEqual(issue.details["staleness_days"], 8)

    def test_invalid_prices_are_reported_as_errors(self) -> None:
        bad_bar = self._bar("2026-01-01")
        bad_bar["close"] = -1.0

        report = validate_market_data({"TEST": [bad_bar]})

        issue = self._issue(report, "invalid_prices")
        self.assertEqual(issue.severity, "error")
        self.assertEqual(issue.details["rows"], [0])

    def test_zero_volume_is_reported_as_warning(self) -> None:
        zero_volume = self._bar("2026-01-01")
        zero_volume["volume"] = 0.0

        report = validate_market_data({"TEST": [zero_volume]})

        issue = self._issue(report, "zero_volume")
        self.assertEqual(issue.severity, "warning")
        self.assertEqual(issue.details["rows"], [0])

    def test_large_date_gaps_are_reported(self) -> None:
        report = validate_market_data(
            {"TEST": [self._bar("2026-01-01"), self._bar("2026-01-10")]},
            max_gap_days=3,
        )

        issue = self._issue(report, "large_gap")
        self.assertEqual(issue.severity, "warning")
        self.assertEqual(issue.details["gaps"], [{"from": "2026-01-01", "to": "2026-01-10", "days": 9}])

    def test_data_agent_exposes_structured_quality_report_and_blocks_errors(self) -> None:
        agent = DataAgent({"data": {"provider": "synthetic", "symbols": ["TEST"], "start_date": "2026-01-01", "end_date": "2026-01-03"}})
        bad_data = {"TEST": [self._bar("2026-01-01"), self._bar("2026-01-01")]}

        with self.assertRaises(ValueError):
            agent.validate_data_quality(bad_data)

        self.assertIsNotNone(agent.last_data_quality)
        self.assertFalse(agent.last_data_quality.passed)
        self.assertEqual(agent.last_data_quality.issues[0].code, "duplicate_dates")

    def _issue(self, report, code):
        matches = [issue for issue in report.issues if issue.code == code]
        self.assertTrue(matches, f"No issue with code {code!r}: {report.issues}")
        return matches[0]

    def _bar(self, date: str) -> dict[str, float | str]:
        return {"date": date, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}


if __name__ == "__main__":
    unittest.main()
