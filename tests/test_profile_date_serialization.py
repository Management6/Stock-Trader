import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from multi_agent_trading_lab.operations.strict_profile_validation import _validation_settings
from multi_agent_trading_lab.orchestrator.orchestrator import TradingLabOrchestrator, load_settings


STRICT_PROFILE = "multi_agent_trading_lab/config/settings.strict_paper.yaml"
APPROVAL_PROFILE = "multi_agent_trading_lab/config/settings.approval_paper.yaml"


class ProfileDateSerializationTests(unittest.TestCase):
    def test_strict_paper_profile_runs_with_date_metadata_serialized_as_strings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            settings = _validation_settings(load_settings(STRICT_PROFILE), Path(tmpdir))
            settings["data"] = {
                **dict(settings["data"]),
                "start_date": date(2023, 1, 3),
                "end_date": date(2023, 8, 31),
            }

            result = TradingLabOrchestrator(settings).run_research_cycle(n_variants=1)
            records = [json.loads(line) for line in Path(settings["experiment_log_path"]).read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result.stage, "research")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["data_range"]["start_date"], "2023-01-03")
        self.assertEqual(records[0]["data_range"]["end_date"], "2023-08-31")
        self.assertEqual(records[0]["config"]["start_date"], "2023-01-03")
        self.assertEqual(records[0]["config"]["end_date"], "2023-08-31")

    def test_paper_profiles_load_dates_as_strings_and_keep_live_disabled(self) -> None:
        for profile in (STRICT_PROFILE, APPROVAL_PROFILE):
            with self.subTest(profile=profile):
                settings = load_settings(profile)
                self.assertIsInstance(settings["data"]["start_date"], str)
                self.assertIsInstance(settings["data"]["end_date"], str)
                self.assertEqual(settings["execution"]["mode"], "paper")
                self.assertFalse(settings["broker"]["live_enabled"])


if __name__ == "__main__":
    unittest.main()
