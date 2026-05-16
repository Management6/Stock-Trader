"""Run research until enough genuinely good stock strategies are found."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_agent_trading_lab.experiments.auto_research import (
    AutoResearchConfig,
    GoodStrategy,
    GoodStrategyThresholds,
    run_auto_research,
)
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentRecord


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run auto-research until enough good backtested stock strategies are found.")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--target-good-strategies", type=int, default=10)
    parser.add_argument("--min-sharpe", type=float, default=0.70)
    parser.add_argument("--min-drawdown", type=float, default=-0.15, help="Minimum acceptable max_drawdown, e.g. -0.15.")
    parser.add_argument("--n-variants", type=int, default=None, help="Variants per research iteration. Defaults to project settings.")
    parser.add_argument("--strategy-families", default=None, help="Comma-separated families: ma, ma_rsi, breakout.")
    parser.add_argument("--universe", default=None, help="Named universe from config/settings.yaml.")
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--synthetic-test-mode", action="store_true", help="Use deterministic synthetic records for smoke testing only.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = GoodStrategyThresholds(min_sharpe=args.min_sharpe, min_drawdown=args.min_drawdown)
    config = AutoResearchConfig(
        max_iterations=args.max_iterations,
        target_good_strategies=args.target_good_strategies,
        thresholds=thresholds,
        n_variants_per_iteration=args.n_variants,
        strategy_families=[item.strip() for item in args.strategy_families.split(",") if item.strip()] if args.strategy_families else None,
        universe=args.universe,
    )
    runner = _synthetic_runner if args.synthetic_test_mode else None

    def progress(iteration: int, good: list[GoodStrategy], best: GoodStrategy | None) -> None:
        if iteration == 1 or iteration % max(1, args.progress_every) == 0 or len(good) >= args.target_good_strategies:
            best_text = "n/a"
            if best is not None:
                best_text = f"family={best.strategy_name}, sharpe={_sharpe(best):.2f}, params={best.strategy_params}"
            print(f"Iteration {iteration}: unique good strategies={len(good)}, best={best_text}")

    result = run_auto_research(config, run_iteration=runner, on_progress=progress)
    print()
    print("Auto-research complete.")
    print(f"iterations run: {result.iterations_run}")
    print(f"good strategies found: {len(result.good_strategies)} / {result.target_good_strategies}")
    print(f"stop reason: {result.stop_reason}")
    if result.stop_reason == "iteration_cap":
        print("Note: stopped because max_iterations was reached before the target count.")
    print()
    print("Good strategies:")
    if not result.good_strategies:
        print("- none found")
    for strategy in result.good_strategies:
        metrics = strategy.metrics
        print(
            f"- {strategy.strategy_id} params={strategy.strategy_params} "
            f"family={strategy.strategy_name} "
            f"sharpe={_sharpe(strategy):.2f} "
            f"total_return={float(metrics.get('total_return', 0.0) or 0.0):.2%} "
            f"max_drawdown={float(metrics.get('max_drawdown', 0.0) or 0.0):.2%}"
        )
    return 0


def _synthetic_runner(iteration: int) -> Iterable[ExperimentRecord]:
    sharpe = 0.75 if iteration == 1 else 0.30
    drawdown = -0.10 if iteration == 1 else -0.20
    return [
        ExperimentRecord(
            experiment_id=f"synthetic-{iteration}",
            timestamp="2026-05-11T00:00:00+00:00",
            strategy_name="moving_average_crossover",
            strategy_version="0.1.0",
            strategy_params={"short_window": 10 + iteration, "long_window": 30 + iteration},
            data_range={"symbols": ["SYNTH"], "start_date": "2020-01-01", "end_date": None},
            metrics={
                "total_return": 0.20,
                "max_drawdown": drawdown,
                "sharpe_ratio": sharpe,
                "history_points": 252,
            },
        )
    ]


def _sharpe(strategy: GoodStrategy) -> float:
    return float(strategy.metrics.get("sharpe_ratio", strategy.metrics.get("sharpe", 0.0)) or 0.0)


if __name__ == "__main__":
    raise SystemExit(main())
