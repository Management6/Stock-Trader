"""Robustness validation for approval-search candidates."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.agents.data_agent import DataAgent
from multi_agent_trading_lab.data.feature_engineering import add_strategy_features
from multi_agent_trading_lab.orchestrator.orchestrator import load_settings
from multi_agent_trading_lab.research.portfolio_backtest import run_portfolio_backtest
from multi_agent_trading_lab.research.regime import classify_market_regime
from multi_agent_trading_lab.research.walk_forward import run_walk_forward_validation


DEFAULT_SPLIT_RATIOS = (0.60, 0.70, 0.80)
DEFAULT_SLIPPAGE_STRESSES = (0.0005, 0.0010, 0.0025)


@dataclass(frozen=True)
class CandidateRobustnessSummary:
    passed: bool
    output_dir: Path
    evaluated_count: int
    status_counts: dict[str, int]
    candidates: list[dict[str, Any]]
    top_robust_candidate: dict[str, Any] | None
    top_fragile_candidate: dict[str, Any] | None
    top_failure_reasons: dict[str, int]
    report_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["report_path"] = str(self.report_path)
        return payload


def run_candidate_robustness(
    settings_path: str | Path,
    input_summary_path: str | Path,
    output_root: str | Path = "multi_agent_trading_lab/validation_runs",
    limit: int | None = None,
) -> CandidateRobustnessSummary:
    """Validate accepted approval-search candidates under deterministic stresses."""

    settings = load_settings(settings_path)
    input_summary = json.loads(Path(input_summary_path).read_text(encoding="utf-8"))
    experiment_log_path = Path(str(input_summary["experiment_log_path"]))
    records = _accepted_records(_read_jsonl(experiment_log_path))
    if limit is not None:
        records = records[:limit]
    data_agent = DataAgent(settings)
    data_config = dict(settings.get("data", {}))
    raw_data = data_agent.fetch_data(
        list(data_config.get("symbols", [])) or None,
        str(data_config.get("start_date", "2023-01-01")),
        None if data_config.get("end_date") in {None, "null", ""} else str(data_config.get("end_date")),
    )
    featured_data = add_strategy_features(raw_data, _feature_windows(records))
    data_quality = data_agent.last_data_quality.to_dict() if data_agent.last_data_quality else {"passed": True, "issues": []}
    summary = evaluate_candidate_robustness(records, settings, featured_data, data_quality)
    output_dir = Path(output_root) / f"candidate_robustness_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "candidate_robustness_summary.json"
    result = CandidateRobustnessSummary(
        passed=True,
        output_dir=output_dir,
        evaluated_count=summary["evaluated_count"],
        status_counts=summary["status_counts"],
        candidates=summary["candidates"],
        top_robust_candidate=summary["top_robust_candidate"],
        top_fragile_candidate=summary["top_fragile_candidate"],
        top_failure_reasons=summary["top_failure_reasons"],
        report_path=report_path,
    )
    report_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def evaluate_candidate_robustness(
    accepted_records: list[dict[str, Any]],
    settings: dict[str, Any],
    featured_data: dict[str, list[dict[str, Any]]],
    data_quality: dict[str, Any] | None = None,
    split_ratios: tuple[float, ...] = DEFAULT_SPLIT_RATIOS,
    slippage_stresses: tuple[float, ...] = DEFAULT_SLIPPAGE_STRESSES,
) -> dict[str, Any]:
    """Return structured robustness results for already accepted candidates."""

    candidate_results = [
        _evaluate_one(record, settings, featured_data, data_quality or {"passed": True, "issues": []}, split_ratios, slippage_stresses)
        for record in accepted_records
    ]
    status_counts = Counter(result["status"] for result in candidate_results)
    failure_reasons = Counter(result["top_failure_reason"] for result in candidate_results if result.get("top_failure_reason"))
    robust = [result for result in candidate_results if result["status"] == "robust"]
    fragile = [result for result in candidate_results if result["status"] == "fragile"]
    return {
        "evaluated_count": len(candidate_results),
        "status_counts": dict(status_counts),
        "candidates": candidate_results,
        "top_robust_candidate": _best(robust),
        "top_fragile_candidate": _best(fragile),
        "top_failure_reasons": dict(failure_reasons),
    }


def _evaluate_one(
    record: dict[str, Any],
    settings: dict[str, Any],
    featured_data: dict[str, list[dict[str, Any]]],
    data_quality: dict[str, Any],
    split_ratios: tuple[float, ...],
    slippage_stresses: tuple[float, ...],
) -> dict[str, Any]:
    strategy = dict(record.get("config", {}).get("strategy", {}))
    base_metrics = dict(record.get("metrics", {}))
    checks: list[dict[str, Any]] = []
    checks.extend(_split_checks(strategy, settings, featured_data, data_quality, split_ratios))
    checks.extend(_cost_checks(strategy, settings, featured_data, data_quality, slippage_stresses))
    checks.extend(_parameter_checks(strategy, settings, featured_data, data_quality))
    checks.append(_portfolio_check(strategy, settings, featured_data, data_quality, "portfolio_base"))
    regime_check = _regime_check(settings, featured_data)
    if regime_check is not None:
        checks.append(regime_check)
    failed = [check for check in checks if not check["passed"]]
    status = "robust" if not failed else "fragile"
    return {
        "candidate_id": strategy.get("id"),
        "strategy_family": strategy.get("name"),
        "base_metrics": {
            "total_return": base_metrics.get("total_return"),
            "max_drawdown": base_metrics.get("max_drawdown"),
            "sharpe_ratio": base_metrics.get("sharpe_ratio"),
        },
        "objective_score": base_metrics.get("optimizer_trial", {}).get("objective_score", record.get("objective_score")),
        "status": status,
        "checks_passed": len(checks) - len(failed),
        "checks_failed": len(failed),
        "worst_oos_metric": _worst_metric(checks, "oos", "sharpe_ratio"),
        "worst_portfolio_metric": _worst_metric(checks, "portfolio", "sharpe_ratio"),
        "worst_cost_stress_result": _worst_cost_result(checks),
        "top_failure_reason": failed[0]["reason"] if failed else None,
        "checks": checks,
    }


def _split_checks(strategy: dict[str, Any], settings: dict[str, Any], data: dict[str, list[dict[str, Any]]], data_quality: dict[str, Any], split_ratios: tuple[float, ...]) -> list[dict[str, Any]]:
    return [_walk_forward_check(strategy, settings, data, data_quality, ratio, f"alternate_split_{ratio:.2f}") for ratio in split_ratios]


def _cost_checks(strategy: dict[str, Any], settings: dict[str, Any], data: dict[str, list[dict[str, Any]]], data_quality: dict[str, Any], slippage_stresses: tuple[float, ...]) -> list[dict[str, Any]]:
    checks = []
    for slippage in slippage_stresses:
        check = _walk_forward_check(strategy, settings, data, data_quality, float(settings.get("oos_validation", {}).get("split_ratio", 0.70)), f"slippage_{slippage:.4f}", slippage_pct=slippage)
        checks.append(check)
    return checks


def _parameter_checks(strategy: dict[str, Any], settings: dict[str, Any], data: dict[str, list[dict[str, Any]]], data_quality: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for index, perturbed in enumerate(_parameter_perturbations(strategy), start=1):
        checks.append(_walk_forward_check(perturbed, settings, data, data_quality, float(settings.get("oos_validation", {}).get("split_ratio", 0.70)), f"parameter_perturbation_{index}"))
    return checks


def _walk_forward_check(
    strategy: dict[str, Any],
    settings: dict[str, Any],
    data: dict[str, list[dict[str, Any]]],
    data_quality: dict[str, Any],
    split_ratio: float,
    name: str,
    slippage_pct: float | None = None,
) -> dict[str, Any]:
    ranges = _walk_forward_ranges(data, split_ratio)
    tester = _backtest_agent(settings, slippage_pct=slippage_pct)
    try:
        walk_forward = run_walk_forward_validation(strategy, data, ranges["in_sample"], ranges["out_of_sample"], backtest_agent=tester, data_quality_report=data_quality)
        decision = _evaluate_oos(walk_forward, settings)
        return {
            "name": name,
            "type": "oos",
            "passed": decision["passed"],
            "reason": decision["reason"],
            "metrics": walk_forward["out_of_sample"]["metrics"],
            "cost_assumptions": walk_forward["out_of_sample"].get("cost_assumptions", {}),
            "strategy_params": dict(strategy.get("strategy_params", {})),
        }
    except Exception as exc:
        return {"name": name, "type": "oos", "passed": False, "reason": str(exc), "metrics": {}, "strategy_params": dict(strategy.get("strategy_params", {}))}


def _portfolio_check(strategy: dict[str, Any], settings: dict[str, Any], data: dict[str, list[dict[str, Any]]], data_quality: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        config = dict(settings.get("portfolio", {}))
        result = run_portfolio_backtest(
            strategy,
            data,
            backtest_agent=_backtest_agent(settings),
            data_quality_report=data_quality,
            allocation_method=str(config.get("allocation_method", "equal_weight")),
            max_symbols=None if config.get("max_symbols") in {None, "null", ""} else int(config.get("max_symbols")),
        )
        decision = _evaluate_portfolio(result, settings)
        return {"name": name, "type": "portfolio", "passed": decision["passed"], "reason": decision["reason"], "metrics": result["metrics"]}
    except Exception as exc:
        return {"name": name, "type": "portfolio", "passed": False, "reason": str(exc), "metrics": {}}


def _regime_check(settings: dict[str, Any], data: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    config = dict(settings.get("regime", {}))
    if not bool(config.get("enabled", False)):
        return None
    benchmark_symbol = str(config.get("benchmark_symbol", "SPY"))
    regime = classify_market_regime(
        data.get(benchmark_symbol, []),
        benchmark_symbol=benchmark_symbol,
        lookback_days=int(config.get("lookback_days", 120)),
        trend_ma_days=int(config.get("trend_ma_days", 50)),
        max_volatility=float(config.get("max_volatility", 0.30)),
    )
    allowed = {str(value) for value in config.get("allowed_regimes", ["bullish", "trending"])}
    passed = regime.get("regime") in allowed
    return {"name": "regime", "type": "regime", "passed": passed, "reason": regime.get("reason"), "metrics": regime}


def _evaluate_oos(walk_forward: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(walk_forward.get("out_of_sample", {}).get("metrics", {}))
    config = dict(settings.get("oos_validation", {}))
    checks = [
        ("min_sharpe", metrics.get("sharpe_ratio"), float(config.get("min_sharpe", 0.0)), lambda actual, threshold: actual >= threshold),
        ("max_drawdown", metrics.get("max_drawdown"), float(config.get("max_drawdown", -0.20)), lambda actual, threshold: actual >= threshold),
        ("min_return", metrics.get("total_return"), float(config.get("min_return", 0.0)), lambda actual, threshold: actual >= threshold),
    ]
    for key, actual, threshold, predicate in checks:
        if actual is None or not predicate(float(actual), threshold):
            return {"passed": False, "reason": f"{key} {actual} failed threshold {threshold}."}
    return {"passed": True, "reason": "OOS stress passed."}


def _evaluate_portfolio(portfolio: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(portfolio.get("metrics", {}))
    config = dict(settings.get("portfolio", {}))
    checks = [
        ("min_total_return", metrics.get("total_return"), float(config.get("min_total_return", 0.0)), lambda actual, threshold: actual >= threshold),
        ("max_drawdown", metrics.get("max_drawdown"), float(config.get("max_drawdown", -0.15)), lambda actual, threshold: actual >= threshold),
        ("min_sharpe", metrics.get("sharpe_ratio"), float(config.get("min_sharpe", 0.0)), lambda actual, threshold: actual >= threshold),
    ]
    for key, actual, threshold, predicate in checks:
        if actual is None or not predicate(float(actual), threshold):
            return {"passed": False, "reason": f"portfolio {key} {actual} failed threshold {threshold}."}
    return {"passed": True, "reason": "Portfolio stress passed."}


def _parameter_perturbations(strategy: dict[str, Any]) -> list[dict[str, Any]]:
    params = dict(strategy.get("strategy_params", {}))
    family = str(strategy.get("name"))
    variants: list[dict[str, Any]] = []
    if family == "breakout_trend":
        for breakout_delta, exit_delta in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            next_params = dict(params)
            next_params["breakout_window"] = max(2, int(params["breakout_window"]) + breakout_delta)
            next_params["exit_window"] = max(1, int(params["exit_window"]) + exit_delta)
            if next_params["exit_window"] < next_params["breakout_window"]:
                variants.append(_with_params(strategy, next_params))
        return variants
    for short_delta, long_delta in [(-1, 0), (1, 0), (0, -5), (0, 5)]:
        next_params = dict(params)
        next_params["short_window"] = max(2, int(params["short_window"]) + short_delta)
        next_params["long_window"] = max(next_params["short_window"] + 1, int(params["long_window"]) + long_delta)
        variants.append(_with_params(strategy, next_params))
    return variants


def _with_params(strategy: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    updated = dict(strategy)
    updated["strategy_params"] = params
    updated.update(params)
    return updated


def _backtest_agent(settings: dict[str, Any], slippage_pct: float | None = None) -> BacktestAgent:
    config = dict(settings.get("backtest", {}))
    return BacktestAgent(
        starting_capital=float(config.get("starting_capital", 100_000.0)),
        max_capital_per_trade_pct=float(config.get("max_capital_per_trade_pct", 0.10)),
        allow_leverage=bool(config.get("allow_leverage", False)),
        commission_per_trade=float(config.get("commission_per_trade", 0.0)),
        slippage_pct=float(config.get("slippage_pct", 0.0) if slippage_pct is None else slippage_pct),
    )


def _walk_forward_ranges(featured_data: dict[str, list[dict[str, Any]]], split_ratio: float) -> dict[str, tuple[str, str]]:
    dates = sorted({str(bar.get("date", "")) for bars in featured_data.values() for bar in bars if bar.get("date")})
    split_index = int(len(dates) * split_ratio)
    split_index = max(2, min(len(dates) - 2, split_index))
    return {"in_sample": (dates[0], dates[split_index - 1]), "out_of_sample": (dates[split_index], dates[-1])}


def _feature_windows(records: list[dict[str, Any]]) -> list[int]:
    windows = {5, 20}
    for record in records:
        strategy = dict(record.get("config", {}).get("strategy", {}))
        for variant in [strategy, *_parameter_perturbations(strategy)]:
            params = dict(variant.get("strategy_params", {}))
            for key in ("short_window", "long_window", "breakout_window", "exit_window", "rsi_period"):
                if key in params:
                    windows.add(int(params[key]))
    return sorted(windows)


def _accepted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record.get("risk_decision", {}).get("approved")]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _worst_metric(checks: list[dict[str, Any]], check_type: str, key: str) -> float | None:
    values = [float(check["metrics"][key]) for check in checks if check.get("type") == check_type and isinstance(check.get("metrics"), dict) and check["metrics"].get(key) is not None]
    return min(values) if values else None


def _worst_cost_result(checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    cost_checks = [check for check in checks if str(check.get("name", "")).startswith("slippage_")]
    failed = [check for check in cost_checks if not check.get("passed")]
    return (failed or cost_checks or [None])[0]


def _best(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("objective_score") if item.get("objective_score") is not None else item.get("base_metrics", {}).get("sharpe_ratio") or float("-inf")))
