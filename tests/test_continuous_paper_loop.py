import unittest

from multi_agent_trading_lab.operations.continuous_paper import run_continuous_paper_loop


class FakeController:
    def __init__(self) -> None:
        self.calls = 0
        self.should_stop = False

    def run_step(self) -> dict:
        self.calls += 1
        if self.calls == 2:
            self.should_stop = True
        return {"strategy": {"total_return": self.calls}}


class ContinuousPaperLoopTests(unittest.TestCase):
    def test_loop_returns_summaries_and_stops_on_controller_flag(self) -> None:
        controller = FakeController()
        seen: list[dict] = []

        summaries = run_continuous_paper_loop(controller, max_steps=10, on_summary=seen.append)

        self.assertEqual(controller.calls, 2)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(seen, summaries)

    def test_loop_stops_at_max_steps(self) -> None:
        controller = FakeController()
        summaries = run_continuous_paper_loop(controller, max_steps=1)

        self.assertEqual(controller.calls, 1)
        self.assertEqual(len(summaries), 1)


if __name__ == "__main__":
    unittest.main()
