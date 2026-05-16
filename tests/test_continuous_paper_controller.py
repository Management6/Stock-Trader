import unittest

import tempfile
from pathlib import Path

from multi_agent_trading_lab.operations.continuous_paper import (
    ContinuousPaperController,
    PaperStrategyConfig,
    load_selected_paper_strategies,
)


class FakePaperStepRunner:
    def __init__(self) -> None:
        self.calls: list[PaperStrategyConfig] = []

    def run_strategy_step(self, strategy: PaperStrategyConfig) -> dict:
        self.calls.append(strategy)
        return {
            "total_return": 0.10,
            "max_drawdown": -0.05,
            "sharpe_ratio": 1.0,
            "last_equity": 101_000.0,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "decision": "paper_step_completed",
        }


class ContinuousPaperControllerTests(unittest.TestCase):
    def test_run_step_runs_each_strategy_and_updates_metrics(self) -> None:
        strategies = [
            PaperStrategyConfig(id="example_ma_s48_l105", name="example_ma", params={"short_window": 48, "long_window": 105}),
            PaperStrategyConfig(id="example_ma_s45_l110", name="example_ma", params={"short_window": 45, "long_window": 110}),
        ]
        runner = FakePaperStepRunner()
        controller = ContinuousPaperController(strategies=strategies, step_runner=runner)

        summary = controller.run_step()

        self.assertEqual([strategy.id for strategy in runner.calls], ["example_ma_s48_l105", "example_ma_s45_l110"])
        self.assertIn("example_ma_s48_l105", summary)
        self.assertEqual(summary["example_ma_s48_l105"]["last_equity"], 101_000.0)
        self.assertEqual(controller.latest_metrics["example_ma_s45_l110"]["sharpe_ratio"], 1.0)

    def test_stop_flag_causes_loop_to_exit_cleanly(self) -> None:
        strategies = [PaperStrategyConfig(id="example_ma_s48_l105", name="example_ma", params={"short_window": 48, "long_window": 105})]
        controller = ContinuousPaperController(strategies=strategies, step_runner=FakePaperStepRunner())

        iterations = 0
        while not controller.should_stop and iterations < 10:
            controller.run_step()
            iterations += 1
            controller.stop()

        self.assertTrue(controller.should_stop)
        self.assertEqual(iterations, 1)

    def test_load_selected_paper_strategies_parses_list_of_mappings_without_pyyaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "selected_strategy.yaml"
            path.write_text(
                "selected_paper_strategies:\n"
                "  - name: example_ma\n"
                "    id: example_ma_s48_l105\n"
                "    version: \"0.1.0\"\n"
                "    params:\n"
                "      short_window: 48\n"
                "      long_window: 105\n"
                "  - name: example_ma\n"
                "    id: example_ma_s45_l110\n"
                "    version: \"0.1.0\"\n"
                "    params:\n"
                "      short_window: 45\n"
                "      long_window: 110\n",
                encoding="utf-8",
            )

            strategies = load_selected_paper_strategies(path)

        self.assertEqual([strategy.id for strategy in strategies], ["example_ma_s48_l105", "example_ma_s45_l110"])
        self.assertEqual(strategies[0].params["short_window"], 48)


if __name__ == "__main__":
    unittest.main()
