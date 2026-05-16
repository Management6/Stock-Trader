import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_agent_trading_lab.experiments.experiment_logger import ExperimentRecord
from multi_agent_trading_lab.orchestrator.orchestrator import load_settings
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy
from multi_agent_trading_lab.strategies.selected_strategy import (
    evaluate_selected_strategy_for_paper,
    load_selected_strategy_config,
)


class SelectedStrategyTests(unittest.TestCase):
    def test_load_selected_strategy_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selected_strategy.yaml"
            path.write_text(
                "selected_strategy:\n"
                "  name: moving_average_crossover\n"
                "  version: \"0.1.0\"\n"
                "  params:\n"
                "    short_window: 6\n"
                "    long_window: 35\n",
                encoding="utf-8",
            )

            selected = load_selected_strategy_config(path)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.params["short_window"], 6)
        self.assertEqual(selected.params["long_window"], 35)

    def test_selected_strategy_passes_when_recent_metrics_pass_paper_risk(self) -> None:
        selected = self._selected_record(0.2613, -0.12708, 0.3694)
        policy = RiskPolicy.from_config({"research": {"max_drawdown": -0.10}, "paper": {"max_drawdown": -0.20}})

        with patch("multi_agent_trading_lab.strategies.selected_strategy.find_experiments_by_params", return_value=[selected]):
            decision = evaluate_selected_strategy_for_paper(
                {"name": "moving_average_crossover", "params": {"short_window": 6, "long_window": 35}},
                policy,
            )

        self.assertTrue(decision["approved"])
        self.assertEqual(decision["risk_limits"]["paper_max_drawdown"], -0.20)

    def test_selected_strategy_fails_when_recent_metrics_fail_paper_risk(self) -> None:
        selected = self._selected_record(100.0, -0.90, 0.3)
        policy = RiskPolicy.from_config(load_settings("multi_agent_trading_lab/config/risk.yaml"))

        with patch("multi_agent_trading_lab.strategies.selected_strategy.find_experiments_by_params", return_value=[selected]):
            decision = evaluate_selected_strategy_for_paper(
                {"name": "moving_average_crossover", "params": {"short_window": 6, "long_window": 35}},
                policy,
            )

        self.assertFalse(decision["approved"])
        self.assertIn("drawdown", decision["reason"].lower())

    def test_selected_strategy_respects_changed_paper_drawdown_threshold(self) -> None:
        selected = self._selected_record(0.2613, -0.12708, 0.3694)
        strict_policy = RiskPolicy.from_config({"paper": {"max_drawdown": -0.10}})
        relaxed_policy = RiskPolicy.from_config({"paper": {"max_drawdown": -0.20}})

        with patch("multi_agent_trading_lab.strategies.selected_strategy.find_experiments_by_params", return_value=[selected]):
            strict_decision = evaluate_selected_strategy_for_paper(
                {"name": "moving_average_crossover", "params": {"short_window": 6, "long_window": 35}},
                strict_policy,
            )
            relaxed_decision = evaluate_selected_strategy_for_paper(
                {"name": "moving_average_crossover", "params": {"short_window": 6, "long_window": 35}},
                relaxed_policy,
            )

        self.assertFalse(strict_decision["approved"])
        self.assertTrue(relaxed_decision["approved"])

    def _selected_record(self, total_return: float, max_drawdown: float, sharpe_ratio: float) -> ExperimentRecord:
        return ExperimentRecord(
            experiment_id="selected",
            timestamp="2026-01-01T00:00:00+00:00",
            strategy_name="moving_average_crossover",
            strategy_version="0.1.0",
            strategy_params={"short_window": 6, "long_window": 35},
            data_range={"symbols": ["WES.AX"], "start_date": "2010-01-01", "end_date": "2026-01-01"},
            metrics={"total_return": total_return, "max_drawdown": max_drawdown, "sharpe_ratio": sharpe_ratio, "history_points": 100},
        )


if __name__ == "__main__":
    unittest.main()
