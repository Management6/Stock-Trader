import json
import tempfile
import unittest
from pathlib import Path

from multi_agent_trading_lab.operations.watchlist_paper_cycle import load_paper_watchlist, run_watchlist_paper_cycle


class WatchlistPaperCycleTests(unittest.TestCase):
    def test_only_watchlist_selected_candidates_are_considered_and_excluded_are_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = self._settings(root)
            watchlist = self._watchlist(root, selected_ids=["selected_ma"], excluded_ids=["not_selected_ma"])

            summary = run_watchlist_paper_cycle(settings, watchlist, output_root=root / "out")
            audit_records = self._read_jsonl(summary.audit_log_path)

        self.assertEqual([item["candidate_id"] for item in summary.candidates_considered], ["selected_ma"])
        self.assertTrue(any(item["candidate_id"] == "not_selected_ma" for item in summary.candidates_skipped))
        self.assertTrue(any(record["event_type"] == "watchlist_candidate_skipped" and record["payload"]["candidate_id"] == "not_selected_ma" for record in audit_records))

    def test_empty_watchlist_is_noop_with_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = self._settings(root)
            watchlist = self._watchlist(root, selected_ids=[], excluded_ids=["rejected_ma"])

            summary = run_watchlist_paper_cycle(settings, watchlist, output_root=root / "out")
            self.assertTrue(summary.report_json_path.exists())

        self.assertEqual(summary.candidates_considered, [])
        self.assertEqual(summary.signals, [])
        self.assertEqual(summary.orders, [])
        self.assertIn("no-op", summary.message)

    def test_malformed_watchlist_has_clear_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("{not-json", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Watchlist validation error"):
                load_paper_watchlist(path)

    def test_live_trading_remains_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            settings = self._settings(root, live_enabled=True)
            watchlist = self._watchlist(root, selected_ids=["selected_ma"])

            summary = run_watchlist_paper_cycle(settings, watchlist, output_root=root / "out")

        self.assertFalse(summary.live_trading_enabled)
        self.assertEqual(summary.execution_mode, "paper")

    def test_strategy_health_quarantine_skips_watchlist_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            state_path = root / "source_system_state.json"
            state_path.write_text(
                json.dumps({"strategy_health": {"selected_ma": {"strategy_id": "selected_ma", "status": "quarantined", "reasons": ["paper attribution"]}}}) + "\n",
                encoding="utf-8",
            )
            settings = self._settings(root, system_state_path=state_path)
            watchlist = self._watchlist(root, selected_ids=["selected_ma"])

            summary = run_watchlist_paper_cycle(settings, watchlist, output_root=root / "out")
            audit_records = self._read_jsonl(summary.audit_log_path)

        self.assertEqual(summary.orders, [])
        self.assertTrue(any(item["candidate_id"] == "selected_ma" and "strategy_health" in item for item in summary.candidates_skipped))
        self.assertTrue(any(record["event_type"] == "strategy_health_skip" for record in audit_records))
        self.assertTrue(summary.strategy_health.get("enabled"))

    def _settings(self, root: Path, live_enabled: bool = False, system_state_path: Path | None = None) -> Path:
        path = root / "settings.json"
        path.write_text(
            "\n".join(
                [
                    'operating_mode: "paper"',
                    "data:",
                    "  provider: synthetic",
                    "  symbols:",
                    "    - ALPHA",
                    "    - BETA",
                    "    - SPY",
                    '  start_date: "2023-01-03"',
                    '  end_date: "2023-08-31"',
                    '  as_of_date: "2023-08-31"',
                    "execution:",
                    '  mode: "paper"',
                    "  dry_run: true",
                    "  default_quantity: 1",
                    "broker:",
                    "  selected: paper",
                    f"  live_enabled: {'true' if live_enabled else 'false'}",
                    '  live_confirmation: ""',
                    "risk:",
                    "  kill_switch_enabled: false",
                    "  max_position_size: 25",
                    "  max_capital_per_trade: 2500.0",
                    "  max_open_positions: 3",
                    "  daily_loss_limit: 1000.0",
                    "  allowed_symbols:",
                    "    - ALPHA",
                    "    - BETA",
                    "    - SPY",
                    "  blocked_symbols: []",
                    "strategy_health:",
                    "  enabled: true",
                    "  min_signals_before_quarantine: 1",
                    f"system_state_path: {system_state_path or root / 'missing_system_state.json'}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def _watchlist(self, root: Path, selected_ids: list[str], excluded_ids: list[str] | None = None) -> Path:
        path = root / "paper_watchlist.json"
        selected = [self._candidate(candidate_id) for candidate_id in selected_ids]
        excluded = [
            {"candidate_id": candidate_id, "strategy_family": "moving_average_crossover", "reason": "Not selected in watchlist."}
            for candidate_id in (excluded_ids or [])
        ]
        payload = {
            "passed": bool(selected),
            "status": "selected" if selected else "empty",
            "selected_candidates": selected,
            "excluded_candidate_details": {"not_approval_accepted": excluded},
        }
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return path

    def _candidate(self, candidate_id: str) -> dict:
        strategy = {
            "id": candidate_id,
            "name": "moving_average_crossover",
            "version": "0.1.0",
            "strategy_params": {"short_window": 5, "long_window": 30},
        }
        return {
            "candidate_id": candidate_id,
            "strategy_family": "moving_average_crossover",
            "parameters": strategy["strategy_params"],
            "strategy": strategy,
        }

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
