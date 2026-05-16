"""Strategy design agent with history-aware parameter search."""

from __future__ import annotations

import json
from math import ceil, floor
from pathlib import Path
from typing import Any

from multi_agent_trading_lab.agents.base_agent import AgentResult, BaseAgent
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentLogger, ExperimentRecord
from multi_agent_trading_lab.research.scoring import DEFAULT_OBJECTIVE_CONFIG, score_metrics
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy
from multi_agent_trading_lab.strategies.base_strategy import canonical_strategy_id


DEFAULT_PARAMETER_BOUNDS: dict[str, dict[str, float]] = {
    "short_window": {"min": 5, "max": 50},
    "long_window": {"min": 20, "max": 200},
    "stop_loss_pct": {"min": 0.01, "max": 0.10},
    "take_profit_pct": {"min": 0.02, "max": 0.20},
}

BREAKOUT_PARAMETER_BOUNDS: dict[str, dict[str, float]] = {
    "breakout_window": {"min": 20, "max": 120},
    "exit_window": {"min": 5, "max": 60},
}

MA_RSI_PARAMETER_BOUNDS: dict[str, dict[str, float]] = {
    "short_window": {"min": 5, "max": 20},
    "long_window": {"min": 30, "max": 120},
    "rsi_period": {"min": 10, "max": 20},
    "rsi_min": {"min": 30, "max": 40},
    "rsi_max": {"min": 60, "max": 70},
}

DEFAULT_SEARCH_CONFIG: dict[str, Any] = {
    "focus_on_risk_passing_pct": 0.70,
    "explore_risky_pct": 0.20,
    "adaptive_narrowing_enabled": True,
    "top_good_experiment_count": 5,
    "risk_drawdown_buffer": 0.02,
    "adaptive_long_window_max": 140,
    "champion_band_enabled": False,
    "champion_band": {},
    "champion_band_focus_pct": 0.70,
}


class StrategyAgent(BaseAgent):
    """Suggests strategy parameters from experiment history.

    The agent reads recent ExperimentLogger records, ranks completed backtests
    by the configured research objective, and proposes new moving-average
    configurations by perturbing the strongest parameter sets. By default the
    objective is Sharpe-style risk-adjusted return. Experiments still log
    total_return and max_drawdown, so future objective changes can reuse old
    data.

    Later this can be replaced or augmented with Bayesian optimization,
    evolutionary search, walk-forward-aware scoring, or LLM-assisted strategy
    design while keeping the same output contract.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        experiment_logger: ExperimentLogger | None = None,
        parameter_bounds: dict[str, dict[str, float]] | None = None,
        objective_config: dict[str, Any] | None = None,
        risk_policy: RiskPolicy | None = None,
        search_config: dict[str, Any] | None = None,
        strategy_version: str = "0.1.0",
    ) -> None:
        super().__init__("strategy_agent")
        self.config_path = config_path or Path("multi_agent_trading_lab/config/strategy_variants.json")
        self.experiment_logger = experiment_logger or ExperimentLogger()
        self.parameter_bounds = parameter_bounds or DEFAULT_PARAMETER_BOUNDS
        self.objective_config = objective_config or DEFAULT_OBJECTIVE_CONFIG
        self.risk_policy = risk_policy or RiskPolicy()
        self.search_config = {**DEFAULT_SEARCH_CONFIG, **(search_config or {})}
        self.strategy_version = strategy_version

    def suggest_parameter_grid(self, base_strategy_name: str, n_variants: int) -> list[dict[str, Any]]:
        """Return deterministic initial candidates within configured bounds."""

        fast_key, slow_key = self._window_keys(base_strategy_name)
        bounds = self._bounds_for(base_strategy_name)
        fast_min = int(bounds[fast_key]["min"])
        fast_max = int(bounds[fast_key]["max"])
        slow_min = int(bounds[slow_key]["min"])
        slow_max = int(bounds[slow_key]["max"])
        candidates: list[dict[str, Any]] = []
        step = max(1, (fast_max - fast_min) // max(1, n_variants))
        for index in range(n_variants * 2):
            fast_window = min(fast_max, fast_min + index * step)
            slow_window = min(slow_max, max(slow_min, fast_window * 4))
            if fast_window >= slow_window:
                slow_window = min(slow_max, fast_window + max(1, slow_min - fast_min))
            candidate = self._candidate(base_strategy_name, {fast_key: fast_window, slow_key: slow_window})
            if candidate not in candidates:
                candidates.append(candidate)
            if len(candidates) >= n_variants:
                break
        return candidates

    def suggest_from_history(self, base_strategy_name: str, n_variants: int) -> list[dict[str, Any]]:
        """Suggest new parameter sets based on recent experiment history.

        The heuristic:

        1. Load recent experiments for the strategy.
        2. Keep records with numeric total_return and max_drawdown.
        3. Rank with ``score_metrics(metrics, objective_config)``.
        4. Perturb the top parameter sets within configured bounds.
        5. Drop duplicates already present in experiment history.
        """

        experiments = self.experiment_logger.load_experiments_for_strategy(base_strategy_name, limit=100)
        usable = [record for record in experiments if self._has_usable_metrics(record)]
        ranked = sorted(usable, key=lambda record: self.score_experiment(record), reverse=True)
        seen = {self._params_key(record.strategy_params) for record in experiments}
        candidates: list[dict[str, Any]] = []
        good_records = [record for record in ranked if self.risk_policy.evaluate_strategy(record.metrics).approved]
        risky_records = [record for record in ranked if not self.risk_policy.evaluate_strategy(record.metrics).approved]
        focus_records = self._stable_good_records(good_records) or good_records
        champion_records = self._records_in_champion_band(base_strategy_name, good_records)

        if champion_records:
            champion_quota = max(1, ceil(n_variants * float(self.search_config["champion_band_focus_pct"])))
            self._append_candidates(
                candidates,
                self._champion_band_candidates(base_strategy_name, champion_records),
                seen,
                champion_quota,
            )

        if good_records:
            good_quota = max(1, ceil(n_variants * float(self.search_config["focus_on_risk_passing_pct"])))
            if champion_records:
                good_quota = max(0, good_quota - len(candidates))
            risky_quota = min(n_variants - good_quota, floor(n_variants * float(self.search_config["explore_risky_pct"])))
            self._append_candidates(
                candidates,
                self._candidate_stream(base_strategy_name, [record for record in focus_records if record not in champion_records], mode="good"),
                seen,
                good_quota,
            )
            self._append_candidates(
                candidates,
                self._candidate_stream(base_strategy_name, risky_records, mode="risky"),
                seen,
                risky_quota,
            )
            self._append_candidates(
                candidates,
                self._adaptive_grid(base_strategy_name, focus_records),
                seen,
                n_variants - len(candidates),
            )
        elif risky_records:
            risky_quota = max(1, floor(n_variants * float(self.search_config["explore_risky_pct"])))
            self._append_candidates(
                candidates,
                self._candidate_stream(base_strategy_name, risky_records, mode="risky"),
                seen,
                risky_quota,
            )
            self._append_candidates(
                candidates,
                self._broad_parameter_grid(base_strategy_name),
                seen,
                n_variants - len(candidates),
            )

        if not candidates:
            return self._fallback_grid_without_seen(base_strategy_name, n_variants, seen)

        if len(candidates) < n_variants:
            fallback_source = (
                self._adaptive_grid(base_strategy_name, focus_records)
                if good_records
                else self._fallback_grid_without_seen(base_strategy_name, n_variants, seen)
            )
            for fallback in fallback_source:
                key = self._params_key(fallback["strategy_params"])
                if key not in {self._params_key(c["strategy_params"]) for c in candidates}:
                    candidates.append(fallback)
                if len(candidates) >= n_variants:
                    break
        return candidates

    def propose_strategy_variants(self, base_strategy: dict[str, Any], n_variants: int = 3) -> list[dict[str, Any]]:
        """Compatibility wrapper used by older orchestration paths."""

        return self.suggest_from_history(str(base_strategy.get("name", "moving_average_crossover")), n_variants)

    def save_strategy_config(self, config: dict[str, Any]) -> Path:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        return self.config_path

    def run(self, base_strategy: dict[str, Any], n_variants: int = 3) -> AgentResult:
        variants = self.propose_strategy_variants(base_strategy, n_variants)
        self.save_strategy_config({"variants": variants})
        return AgentResult(self.name, variants)

    def _candidate(self, strategy_name: str, params: dict[str, Any]) -> dict[str, Any]:
        params = self._with_strategy_defaults(strategy_name, params)
        clean_params = self._bounded_params(params)
        identifier = canonical_strategy_id(strategy_name, self.strategy_version, clean_params)
        candidate = {
            "id": identifier,
            "name": strategy_name,
            "version": self.strategy_version,
            "strategy_params": clean_params,
            "stop_loss_pct": clean_params.get("stop_loss_pct"),
            "take_profit_pct": clean_params.get("take_profit_pct"),
            "deployment_stage": "candidate",
        }
        candidate.update(clean_params)
        return candidate

    def _perturb(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        fast_key, slow_key = self._window_keys_from_params(params)
        bounds = self._bounds_for_params(params)
        fast_window = int(params.get(fast_key, bounds[fast_key]["min"]))
        slow_window = int(params.get(slow_key, bounds[slow_key]["min"]))
        stop_loss_pct = params.get("stop_loss_pct")
        take_profit_pct = params.get("take_profit_pct")
        rsi_period = params.get("rsi_period")
        rsi_min = params.get("rsi_min")
        rsi_max = params.get("rsi_max")
        candidates: list[dict[str, Any]] = []
        for short_delta, long_delta in [(-2, 0), (2, 0), (0, -5), (0, 5), (-1, -5), (1, 5), (3, 10), (-3, -10)]:
            candidates.append(
                {
                    fast_key: fast_window + short_delta,
                    slow_key: slow_window + long_delta,
                    "stop_loss_pct": stop_loss_pct,
                    "take_profit_pct": take_profit_pct,
                    "rsi_period": rsi_period,
                    "rsi_min": rsi_min,
                    "rsi_max": rsi_max,
                }
            )
        return [self._bounded_params(candidate) for candidate in candidates]

    def _risk_reducing_perturb(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        fast_key, slow_key = self._window_keys_from_params(params)
        bounds = self._bounds_for_params(params)
        fast_window = int(params.get(fast_key, bounds[fast_key]["min"]))
        slow_window = int(params.get(slow_key, bounds[slow_key]["min"]))
        slow_min = int(bounds[slow_key]["min"])
        target_long = max(slow_min, min(slow_window - 40, 120))
        candidates = [
            {fast_key: max(int(bounds[fast_key]["min"]), fast_window - 2), slow_key: target_long},
            {fast_key: fast_window, slow_key: max(slow_min, target_long - 10)},
            {fast_key: fast_window + 2, slow_key: min(int(bounds[slow_key]["max"]), target_long + 10)},
        ]
        for candidate in candidates:
            for key in ("rsi_period", "rsi_min", "rsi_max"):
                if key in params:
                    candidate[key] = params[key]
        return [self._bounded_params(candidate) for candidate in candidates]

    def _candidate_stream(
        self,
        strategy_name: str,
        records: list[ExperimentRecord],
        mode: str,
    ) -> list[dict[str, Any]]:
        stream: list[dict[str, Any]] = []
        for record in records[: int(self.search_config["top_good_experiment_count"]) if mode == "good" else 5]:
            perturbations = self._perturb(record.strategy_params) if mode == "good" else self._risk_reducing_perturb(record.strategy_params)
            stream.extend(self._candidate(strategy_name, params) for params in perturbations)
        return stream

    def _append_candidates(
        self,
        destination: list[dict[str, Any]],
        source: list[dict[str, Any]],
        seen: set[tuple[tuple[str, Any], ...]],
        limit: int,
    ) -> None:
        if limit <= 0:
            return
        existing = {self._params_key(candidate["strategy_params"]) for candidate in destination}
        added = 0
        for candidate in source:
            key = self._params_key(candidate["strategy_params"])
            if key in seen or key in existing:
                continue
            destination.append(candidate)
            existing.add(key)
            added += 1
            if added >= limit:
                return

    def _fallback_grid_without_seen(
        self,
        strategy_name: str,
        n_variants: int,
        seen: set[tuple[tuple[str, Any], ...]],
    ) -> list[dict[str, Any]]:
        suggestions: list[dict[str, Any]] = []
        for candidate in self._broad_parameter_grid(strategy_name):
            if self._params_key(candidate["strategy_params"]) in seen:
                continue
            suggestions.append(candidate)
            if len(suggestions) >= n_variants:
                break
        return suggestions

    def _broad_parameter_grid(self, strategy_name: str) -> list[dict[str, Any]]:
        """Return a broad deterministic grid for history-saturated local runs."""

        fast_key, slow_key = self._window_keys(strategy_name)
        bounds = self._bounds_for(strategy_name)
        short_min = int(bounds[fast_key]["min"])
        short_max = int(bounds[fast_key]["max"])
        long_min = int(bounds[slow_key]["min"])
        long_max = int(bounds[slow_key]["max"])
        long_step = max(1, (long_max - long_min) // 12)
        candidates: list[dict[str, Any]] = []
        for short_window in range(short_min, short_max + 1):
            for long_window in range(max(long_min, short_window + 1), long_max + 1, long_step):
                candidates.append(self._candidate(strategy_name, {fast_key: short_window, slow_key: long_window}))
        return candidates

    def _adaptive_grid(self, strategy_name: str, good_records: list[ExperimentRecord]) -> list[dict[str, Any]]:
        """Generate a transparent narrowed grid around top risk-passing records."""

        if not self.search_config.get("adaptive_narrowing_enabled", True) or not good_records:
            return self._broad_parameter_grid(strategy_name)
        top_records = good_records[: int(self.search_config["top_good_experiment_count"])]
        fast_key, slow_key = self._window_keys(strategy_name)
        bounds = self._bounds_for(strategy_name)
        shorts = sorted(int(record.strategy_params[fast_key]) for record in top_records)
        longs = sorted(int(record.strategy_params[slow_key]) for record in top_records)
        short_center = shorts[len(shorts) // 2]
        long_center = longs[len(longs) // 2]
        short_min = max(int(bounds[fast_key]["min"]), short_center - 5)
        short_max = min(int(bounds[fast_key]["max"]), short_center + 5)
        long_min = max(int(bounds[slow_key]["min"]), long_center - 25)
        long_max = min(int(bounds[slow_key]["max"]), long_center + 25)
        candidates: list[dict[str, Any]] = []
        for long_window in range(long_min, long_max + 1, 5):
            for short_window in range(short_min, short_max + 1):
                candidates.append(self._candidate(strategy_name, {fast_key: short_window, slow_key: long_window}))
        return candidates

    def _stable_good_records(self, good_records: list[ExperimentRecord]) -> list[ExperimentRecord]:
        """Return risk-passing records with a buffer inside the drawdown limit."""

        buffer = float(self.search_config.get("risk_drawdown_buffer", 0.02))
        max_focus_long = int(self.search_config.get("adaptive_long_window_max", self._bounds_for_params(good_records[0].strategy_params)["long_window"]["max"] if good_records and "long_window" in good_records[0].strategy_params else 10_000))
        threshold = self.risk_policy.max_drawdown_threshold + buffer
        return [
            record
            for record in good_records
            if float(record.metrics.get("max_drawdown", -1.0)) >= threshold
            and int(record.strategy_params.get("long_window", max_focus_long)) <= max_focus_long
        ]

    def _records_in_champion_band(self, strategy_name: str, records: list[ExperimentRecord]) -> list[ExperimentRecord]:
        band = self._champion_band_for(strategy_name)
        if band is None:
            return []
        return [record for record in records if self._params_inside_band(record.strategy_params, band)]

    def _champion_band_candidates(
        self,
        strategy_name: str,
        champion_records: list[ExperimentRecord],
    ) -> list[dict[str, Any]]:
        band = self._champion_band_for(strategy_name)
        if band is None:
            return []
        candidates: list[dict[str, Any]] = []
        for record in champion_records:
            candidates.extend(self._perturb_inside_band(record.strategy_params, band))
        for long_window in range(int(band["long_window"]["min"]), int(band["long_window"]["max"]) + 1):
            for short_window in range(int(band["short_window"]["min"]), int(band["short_window"]["max"]) + 1):
                candidates.append({"short_window": short_window, "long_window": long_window})
        return [self._candidate(strategy_name, params) for params in candidates]

    def _perturb_inside_band(self, params: dict[str, Any], band: dict[str, Any]) -> list[dict[str, Any]]:
        perturbations: list[dict[str, Any]] = []
        short_window = int(params["short_window"])
        long_window = int(params["long_window"])
        for short_delta, long_delta in [(-1, 0), (1, 0), (0, -5), (0, 5), (-1, -5), (1, 5)]:
            candidate = {
                "short_window": max(int(band["short_window"]["min"]), min(int(band["short_window"]["max"]), short_window + short_delta)),
                "long_window": max(int(band["long_window"]["min"]), min(int(band["long_window"]["max"]), long_window + long_delta)),
            }
            perturbations.append(candidate)
        return perturbations

    def _champion_band_for(self, strategy_name: str) -> dict[str, Any] | None:
        if not self.search_config.get("champion_band_enabled", False):
            return None
        bands = dict(self.search_config.get("champion_band", {}))
        return bands.get(strategy_name) or bands.get("example_ma" if strategy_name == "moving_average_crossover" else "moving_average_crossover")

    @staticmethod
    def _params_inside_band(params: dict[str, Any], band: dict[str, Any]) -> bool:
        return (
            int(band["short_window"]["min"]) <= int(params.get("short_window", -1)) <= int(band["short_window"]["max"])
            and int(band["long_window"]["min"]) <= int(params.get("long_window", -1)) <= int(band["long_window"]["max"])
        )

    def _bounded_params(self, params: dict[str, Any]) -> dict[str, Any]:
        fast_key, slow_key = self._window_keys_from_params(params)
        fast_window = self._clamp_int(fast_key, params.get(fast_key), self._bounds_for_params(params))
        slow_window = self._clamp_int(slow_key, params.get(slow_key), self._bounds_for_params(params))
        if fast_key == "exit_window" and slow_key == "breakout_window":
            if slow_window <= fast_window:
                slow_window = min(int(self._bounds_for_params(params)[slow_key]["max"]), fast_window + 1)
                if slow_window <= fast_window:
                    fast_window = max(int(self._bounds_for_params(params)[fast_key]["min"]), slow_window - 1)
            bounded: dict[str, Any] = {slow_key: slow_window, fast_key: fast_window}
            return bounded
        short_window = fast_window
        long_window = slow_window
        if short_window >= long_window:
            long_window = min(int(self._bounds_for_params(params)[slow_key]["max"]), short_window + 1)
            if short_window >= long_window:
                short_window = max(int(self._bounds_for_params(params)[fast_key]["min"]), long_window - 1)
        bounded = {fast_key: short_window, slow_key: long_window}
        for key in ("stop_loss_pct", "take_profit_pct"):
            value = params.get(key)
            if value is not None:
                bounded[key] = self._clamp_float(key, value)
        if self._is_ma_rsi_params(params):
            bounded["rsi_period"] = self._clamp_int("rsi_period", params.get("rsi_period"), self._bounds_for_params(params))
            bounded["rsi_min"] = self._clamp_float_with_bounds("rsi_min", params.get("rsi_min"), self._bounds_for_params(params))
            bounded["rsi_max"] = self._clamp_float_with_bounds("rsi_max", params.get("rsi_max"), self._bounds_for_params(params))
            if bounded["rsi_min"] >= bounded["rsi_max"]:
                bounded["rsi_max"] = min(float(self._bounds_for_params(params)["rsi_max"]["max"]), bounded["rsi_min"] + 1.0)
        return bounded

    def _clamp_int(self, key: str, value: Any, bounds_by_key: dict[str, dict[str, float]] | None = None) -> int:
        bounds = (bounds_by_key or self.parameter_bounds)[key]
        raw = int(value if value is not None else bounds["min"])
        return max(int(bounds["min"]), min(int(bounds["max"]), raw))

    def _clamp_float(self, key: str, value: Any) -> float:
        bounds = self.parameter_bounds[key]
        raw = float(value)
        return max(float(bounds["min"]), min(float(bounds["max"]), raw))

    def _clamp_float_with_bounds(self, key: str, value: Any, bounds_by_key: dict[str, dict[str, float]]) -> float:
        bounds = bounds_by_key[key]
        raw = float(value if value is not None else bounds["min"])
        return max(float(bounds["min"]), min(float(bounds["max"]), raw))

    def score_experiment(self, record: ExperimentRecord) -> float:
        """Score one experiment using the configured research objective."""

        return score_metrics(record.metrics, self.objective_config)

    @staticmethod
    def _has_usable_metrics(record: ExperimentRecord) -> bool:
        metrics = record.metrics
        return _is_number(metrics.get("total_return")) and _is_number(metrics.get("max_drawdown"))

    @staticmethod
    def _params_key(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
        return tuple(sorted((key, value) for key, value in params.items() if value is not None))

    def _window_keys(self, strategy_name: str) -> tuple[str, str]:
        if strategy_name == "breakout_trend":
            return "exit_window", "breakout_window"
        return "short_window", "long_window"

    @staticmethod
    def _window_keys_from_params(params: dict[str, Any]) -> tuple[str, str]:
        if "breakout_window" in params or "exit_window" in params:
            return "exit_window", "breakout_window"
        return "short_window", "long_window"

    def _bounds_for(self, strategy_name: str) -> dict[str, dict[str, float]]:
        if strategy_name == "breakout_trend":
            return {**BREAKOUT_PARAMETER_BOUNDS, **self.parameter_bounds}
        if strategy_name == "moving_average_rsi_filter":
            return {**MA_RSI_PARAMETER_BOUNDS, **self.parameter_bounds}
        return self.parameter_bounds

    def _bounds_for_params(self, params: dict[str, Any]) -> dict[str, dict[str, float]]:
        if "breakout_window" in params or "exit_window" in params:
            return {**BREAKOUT_PARAMETER_BOUNDS, **self.parameter_bounds}
        if self._is_ma_rsi_params(params):
            return {**MA_RSI_PARAMETER_BOUNDS, **self.parameter_bounds}
        return self.parameter_bounds

    @staticmethod
    def _is_ma_rsi_params(params: dict[str, Any]) -> bool:
        return "rsi_period" in params or "rsi_min" in params or "rsi_max" in params

    def _with_strategy_defaults(self, strategy_name: str, params: dict[str, Any]) -> dict[str, Any]:
        if strategy_name != "moving_average_rsi_filter":
            return params
        bounds = self._bounds_for(strategy_name)
        enriched = dict(params)
        enriched.setdefault("rsi_period", int((bounds["rsi_period"]["min"] + bounds["rsi_period"]["max"]) // 2))
        enriched.setdefault("rsi_min", float(bounds["rsi_min"]["min"]))
        enriched.setdefault("rsi_max", float(bounds["rsi_max"]["max"]))
        return enriched


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
