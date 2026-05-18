"""Deterministic parameter optimization adapters."""

from __future__ import annotations

import random
from typing import Any

from multi_agent_trading_lab.strategies.base_strategy import canonical_strategy_id


def propose_candidates(strategy_family: str, search_space: dict[str, dict[str, float]], n_trials: int, seed: int = 42) -> list[dict[str, Any]]:
    return DeterministicRandomOptimizer(seed=seed).propose_candidates(strategy_family, search_space, n_trials)


class DeterministicRandomOptimizer:
    """Small seeded random-search optimizer with a stable result interface."""

    def __init__(self, seed: int = 42, strategy_version: str = "0.1.0") -> None:
        self.seed = seed
        self.strategy_version = strategy_version
        self._results: list[dict[str, Any]] = []

    def propose_candidates(self, strategy_family: str, search_space: dict[str, dict[str, float]], n_trials: int) -> list[dict[str, Any]]:
        rng = random.Random(self.seed)
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[tuple[str, Any], ...]] = set()
        attempts = 0
        while len(candidates) < n_trials and attempts < n_trials * 20:
            attempts += 1
            params = _sample_params(strategy_family, search_space, rng)
            key = tuple(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            trial = len(candidates)
            candidate = {
                "id": canonical_strategy_id(strategy_family, self.strategy_version, params),
                "name": strategy_family,
                "version": self.strategy_version,
                "strategy_params": params,
                "deployment_stage": "candidate",
                "optimizer_trial": {"trial_number": trial, "method": "deterministic_random", "seed": self.seed, "parameters": params},
            }
            candidate.update(params)
            candidate["stop_loss_pct"] = params.get("stop_loss_pct")
            candidate["take_profit_pct"] = params.get("take_profit_pct")
            candidates.append(candidate)
        if len(candidates) < n_trials:
            warning = f"Optimizer produced {len(candidates)} candidate(s), fewer than requested {n_trials}."
            for candidate in candidates:
                candidate["optimizer_trial"].setdefault("warnings", []).append(warning)
        return candidates

    def record_result(self, candidate: dict[str, Any], score: float | None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        result = {
            "trial_number": int(dict(candidate.get("optimizer_trial", {})).get("trial_number", len(self._results))),
            "parameters": dict(candidate.get("strategy_params", {})),
            "score": score,
            "metadata": dict(metadata or {}),
            "candidate_id": candidate.get("id"),
        }
        self._results.append(result)
        return result

    def best_candidates(self, limit: int | None = None) -> list[dict[str, Any]]:
        ranked = sorted(self._results, key=lambda result: float("-inf") if result["score"] is None else float(result["score"]), reverse=True)
        return ranked if limit is None else ranked[:limit]


def _sample_params(strategy_family: str, search_space: dict[str, dict[str, float]], rng: random.Random) -> dict[str, Any]:
    if strategy_family == "breakout_trend":
        breakout = _sample_int(search_space, "breakout_window", rng)
        exit_window = _sample_int(search_space, "exit_window", rng)
        if exit_window >= breakout:
            exit_window = max(int(search_space["exit_window"]["min"]), breakout - 1)
        return {"breakout_window": breakout, "exit_window": exit_window}
    short = _sample_int(search_space, "short_window", rng)
    long = _sample_int(search_space, "long_window", rng)
    if short >= long:
        long = min(int(search_space["long_window"]["max"]), short + 1)
        if short >= long:
            short = max(int(search_space["short_window"]["min"]), long - 1)
    params: dict[str, Any] = {"short_window": short, "long_window": long}
    for key in ("stop_loss_pct", "take_profit_pct"):
        if key in search_space:
            params[key] = _sample_float(search_space, key, rng)
    if strategy_family == "moving_average_rsi_filter":
        params["rsi_period"] = _sample_int(search_space, "rsi_period", rng)
        params["rsi_min"] = _sample_float(search_space, "rsi_min", rng)
        params["rsi_max"] = _sample_float(search_space, "rsi_max", rng)
        if params["rsi_min"] >= params["rsi_max"]:
            params["rsi_max"] = min(float(search_space["rsi_max"]["max"]), params["rsi_min"] + 1.0)
    return params


def _sample_int(search_space: dict[str, dict[str, float]], key: str, rng: random.Random) -> int:
    bounds = search_space[key]
    return rng.randint(int(bounds["min"]), int(bounds["max"]))


def _sample_float(search_space: dict[str, dict[str, float]], key: str, rng: random.Random) -> float:
    bounds = search_space[key]
    return round(rng.uniform(float(bounds["min"]), float(bounds["max"])), 4)
