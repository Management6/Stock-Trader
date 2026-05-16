"""Coordinates staged multi-agent trading workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.agents.data_agent import DataAgent
from multi_agent_trading_lab.agents.discovery_agent import DiscoveryAgent
from multi_agent_trading_lab.agents.execution_agent import ExecutionAgent
from multi_agent_trading_lab.agents.report_agent import ReportAgent
from multi_agent_trading_lab.agents.risk_agent import RiskAgent
from multi_agent_trading_lab.agents.strategy_agent import StrategyAgent
from multi_agent_trading_lab.brokers.alpaca_broker import AlpacaBroker
from multi_agent_trading_lab.brokers.base_broker import BaseBroker
from multi_agent_trading_lab.brokers.paper_broker import PaperBroker
from multi_agent_trading_lab.execution.execution_engine import ExecutionEngine
from multi_agent_trading_lab.execution.order_models import OrderRequest
from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentLogger
from multi_agent_trading_lab.data.universes import resolve_universe_symbols
from multi_agent_trading_lab.operations.alerting import AlertManager, AlertSeverity, FileAlertSink
from multi_agent_trading_lab.operations.approvals import ApprovalQueue
from multi_agent_trading_lab.research.scoring import score_metrics
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy
from multi_agent_trading_lab.state.system_state import SystemStateStore
from multi_agent_trading_lab.strategies.example_strategy import MovingAverageCrossoverStrategy
from multi_agent_trading_lab.strategies.registry import StrategyRegistry
from multi_agent_trading_lab.strategies.selected_strategy import (
    evaluate_selected_strategy_for_paper,
    load_selected_strategy_config,
)


DEFAULT_SETTINGS: dict[str, Any] = {
    "operating_mode": "paper",
    "data": {
        "provider": "yfinance",
        "symbols": ["WES.AX", "YAL.AX", "SLX.AX", "NVDA"],
        "timeframe": "1d",
        "start_date": "2010-01-01",
        "end_date": None,
        "cache_dir": "data/cache",
        "force_refresh": False,
    },
    "symbols": ["AAPL"],
    "start_date": "2023-01-01",
    "end_date": "2023-06-30",
    "strategy": {"name": "moving_average_crossover", "version": "0.1.0", "short_window": 5, "long_window": 20},
    "strategy_families": ["moving_average_crossover"],
    "strategies": {
        "moving_average_crossover": {
            "short_window": {"min": 5, "max": 50},
            "long_window": {"min": 20, "max": 200},
            "stop_loss_pct": {"min": 0.01, "max": 0.10},
            "take_profit_pct": {"min": 0.02, "max": 0.20},
            "default_variants": 5,
        },
        "breakout_trend": {
            "breakout_window": {"min": 20, "max": 120},
            "exit_window": {"min": 5, "max": 60},
            "default_variants": 5,
        },
        "moving_average_rsi_filter": {
            "short_window": {"min": 5, "max": 20},
            "long_window": {"min": 30, "max": 120},
            "rsi_period": {"min": 10, "max": 20},
            "rsi_min": {"min": 30, "max": 40},
            "rsi_max": {"min": 60, "max": 70},
            "default_variants": 5,
        }
    },
    "research": {"variants_per_cycle": 5, "max_experiments_per_run": 5},
    "strategy_search": {
        "focus_on_risk_passing_pct": 0.70,
        "explore_risky_pct": 0.20,
        "adaptive_narrowing_enabled": True,
        "top_good_experiment_count": 5,
        "risk_drawdown_buffer": 0.02,
        "adaptive_long_window_max": 140,
        "champion_band_enabled": False,
        "champion_band": {},
        "champion_band_focus_pct": 0.70,
    },
    "research_objective": {
        "type": "sharpe",
        "sharpe": {
            "annualization_factor": 252,
            "min_history_points": 30,
            "fallback_return_weight": 1.0,
            "fallback_drawdown_penalty": 150.0,
        },
        "custom": {"return_weight": 1.0, "drawdown_penalty": 1.0},
    },
    "backtest": {
        "starting_capital": 100_000.0,
        "max_capital_per_trade_pct": 0.10,
        "allow_leverage": False,
        "commission_per_trade": 0.0,
        "slippage_pct": 0.0,
    },
    "risk": {
        "kill_switch_enabled": False,
        "min_backtest_return": -0.05,
        "max_drawdown_threshold": -0.20,
        "research": {"max_drawdown": -0.20},
        "paper": {"max_drawdown": -0.20},
        "max_position_size": 25,
        "max_capital_per_trade": 2_500.0,
        "max_open_positions": 3,
        "allowed_symbols": ["WES.AX", "YAL.AX", "SLX.AX", "NVDA"],
        "blocked_symbols": [],
    },
    "broker": {
        "selected": "paper",
        "paper_state_path": "multi_agent_trading_lab/state/paper_broker_state.json",
        "live_enabled": False,
        "live_confirmation": "",
    },
    "execution": {"mode": "paper", "dry_run": False, "default_quantity": 1},
    "experiment_log_path": "multi_agent_trading_lab/experiments/experiment_memory.jsonl",
    "audit_log_path": "multi_agent_trading_lab/experiments/audit_log.jsonl",
    "alert_log_path": "multi_agent_trading_lab/experiments/alerts.jsonl",
    "approval_queue_path": "multi_agent_trading_lab/state/strategy_approvals.json",
    "strategy_registry_path": "multi_agent_trading_lab/strategies/strategy_registry.json",
    "system_state_path": "multi_agent_trading_lab/state/system_state.json",
}


@dataclass(frozen=True)
class WorkflowResult:
    """Structured result returned by staged workflows."""

    stage: str
    summaries: list[str]
    details: dict[str, Any]


class TradingLabOrchestrator:
    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self.settings = _deep_merge(DEFAULT_SETTINGS, settings or {})
        self.discovery_agent = DiscoveryAgent()
        self.data_agent = DataAgent(self.settings)
        backtest_settings = self.settings.get("backtest", {})
        self.backtest_agent = BacktestAgent(
            starting_capital=float(backtest_settings.get("starting_capital", 100_000.0)),
            max_capital_per_trade_pct=float(backtest_settings.get("max_capital_per_trade_pct", 0.10)),
            allow_leverage=bool(backtest_settings.get("allow_leverage", False)),
            commission_per_trade=float(backtest_settings.get("commission_per_trade", 0.0)),
            slippage_pct=float(backtest_settings.get("slippage_pct", 0.0)),
        )
        self.risk_policy = RiskPolicy.from_config(self.settings.get("risk", {}))
        self.risk_agent = RiskAgent(policy=self.risk_policy)
        self.report_agent = ReportAgent()
        self.logger = ExperimentLogger(self.settings.get("experiment_log_path"))
        strategy_name = str(self.settings.get("strategy", {}).get("name", "moving_average_crossover"))
        search_config = dict(self.settings.get("strategy_search", {}))
        research_config = dict(self.settings.get("research", {}))
        for key in ("champion_band_enabled", "champion_band"):
            if key in research_config:
                search_config[key] = research_config[key]
        self.strategy_agent = StrategyAgent(
            experiment_logger=self.logger,
            parameter_bounds=self.settings.get("strategies", {}).get(strategy_name),
            objective_config=self.settings.get("research_objective", {}),
            risk_policy=self.risk_policy,
            search_config=search_config,
            strategy_version=str(self.settings.get("strategy", {}).get("version", "0.1.0")),
        )
        self.audit_log = AuditLog(self.settings.get("audit_log_path"))
        self.alert_manager = AlertManager([FileAlertSink(self.settings.get("alert_log_path", "multi_agent_trading_lab/experiments/alerts.jsonl"))])
        self.strategy_registry = StrategyRegistry(self.settings.get("strategy_registry_path"))
        self.approval_queue = ApprovalQueue(
            self.settings.get("approval_queue_path", "multi_agent_trading_lab/state/strategy_approvals.json"),
            registry=self.strategy_registry,
            audit_log=self.audit_log,
            alert_manager=self.alert_manager,
        )
        self.system_state = SystemStateStore(self.settings.get("system_state_path"))
        state = self.system_state.read()
        self.risk_policy.kill_switch_enabled = bool(
            self.risk_policy.kill_switch_enabled or state.get("kill_switch_enabled", False)
        )
        self.risk_policy.repeated_failures = int(state.get("repeated_failures", self.risk_policy.repeated_failures))
        self.broker = self._build_broker()
        execution_settings = self.settings.get("execution", {})
        broker_settings = self.settings.get("broker", {})
        self.execution_engine = ExecutionEngine(
            broker=self.broker,
            risk_policy=self.risk_policy,
            audit_log=self.audit_log,
            system_state=self.system_state,
            mode=str(execution_settings.get("mode", self.settings.get("operating_mode", "paper"))),
            dry_run=bool(execution_settings.get("dry_run", False)),
            live_enabled=bool(broker_settings.get("live_enabled", False)),
            live_confirmation=str(broker_settings.get("live_confirmation", "")),
        )
        self.execution_agent = ExecutionAgent(self.execution_engine)

    def run_research_cycle(self, n_variants: int | None = None, strategy_families: list[str] | None = None) -> WorkflowResult:
        """Run a history-aware research and backtesting cycle."""

        data_config = self._data_config()
        symbols = list(data_config["symbols"])
        start_date = str(data_config["start_date"])
        end_date = data_config.get("end_date")
        profile = self.discovery_agent.build_profile({"symbols": symbols})
        research_settings = self.settings.get("research", {})
        families = self._resolve_strategy_families(strategy_families)
        variants: list[dict[str, Any]] = []
        for family in families:
            strategy_settings = self.settings.get("strategies", {}).get(family, {})
            requested_variants = int(
                n_variants
                or research_settings.get("variants_per_cycle")
                or strategy_settings.get("default_variants")
                or 3
            )
            max_experiments = int(research_settings.get("max_experiments_per_run", requested_variants * max(1, len(families))))
            requested_variants = min(requested_variants, max(1, max_experiments // max(1, len(families))))
            variants.extend(self._strategy_agent_for(family).suggest_from_history(family, requested_variants))
        windows = self._feature_windows_from_variants(variants)
        raw_data = self.data_agent.fetch_data(symbols, start_date, None if end_date in {None, "null", ""} else str(end_date))
        featured_data = self.data_agent.build_features(raw_data, windows)

        summaries: list[str] = []
        records: list[dict[str, Any]] = []
        for variant in variants:
            self.strategy_registry.register(variant, stage="candidate", reason="Generated by StrategyAgent.")
            backtest = self.backtest_agent.run_backtest(variant, featured_data)
            risk_decision = self.risk_agent.assess_risk(backtest["metrics"])
            stage = "paper" if risk_decision["approved"] else "rejected"
            record = self.logger.log_experiment(
                config={
                    "strategy": variant,
                    "symbols": symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "deployment_stage": stage,
                },
                metrics=backtest["metrics"],
                risk_decision=risk_decision,
                approval_status="approved" if risk_decision["approved"] else "rejected",
                deployment_stage=stage,
                mode="backtest",
            )
            record["objective_score"] = score_metrics(record["metrics"], self.settings.get("research_objective", {}))
            records.append(record)
            if risk_decision["approved"]:
                self.approval_queue.request_promotion(
                    strategy=variant,
                    rationale="Strategy passed research risk policy and is eligible for paper review.",
                    metrics=backtest["metrics"],
                    requester="run_research_cycle",
                )
            else:
                self.strategy_registry.promote(variant["id"], "rejected", str(risk_decision["reason"]))
                self.audit_log.record("strategy_rejected", {"strategy": variant, "reason": risk_decision["reason"]})
                if "drawdown" in str(risk_decision["reason"]).lower():
                    self.alert_manager.emit(
                        "drawdown_threshold_breach",
                        AlertSeverity.WARNING,
                        f"Strategy {variant['id']} breached drawdown risk threshold.",
                        strategy_id=str(variant["id"]),
                        metadata={"metrics": backtest["metrics"], "reason": risk_decision["reason"]},
                    )
            summaries.append(self.report_agent.summarize_experiment(record))
        return WorkflowResult(
            stage="research",
            summaries=summaries,
            details={
                "profile": profile,
                "records": records,
                "active_strategies": self.strategy_registry.list_by_stage("active"),
                "objective": self.settings.get("research_objective", {}),
                "data": _market_data_summary(raw_data, start_date, end_date, self.data_agent.last_load_source),
            },
        )

    def run_paper_cycle(self) -> WorkflowResult:
        """Generate one paper-trading signal and route it through guarded execution."""

        summaries = [self._paper_risk_limits_summary()]
        if self.risk_policy.kill_switch_enabled:
            self.audit_log.record("kill_switch_event", {"action": "paper_cycle_blocked"})
            self.alert_manager.emit(
                "kill_switch_activated",
                AlertSeverity.CRITICAL,
                "Paper cycle blocked because the kill switch is enabled.",
            )
            return WorkflowResult("paper", summaries + ["Paper cycle blocked: kill switch is enabled."], {"orders": []})

        selected_decision = evaluate_selected_strategy_for_paper(load_selected_strategy_config(), self.risk_policy)
        if selected_decision["approved"]:
            strategy_config = dict(selected_decision["strategy_config"])
            summaries.append(f"Selected strategy accepted for paper mode: {strategy_config.get('id') or strategy_config.get('name')}")
        else:
            if selected_decision["strategy_config"] is not None:
                summary = f"Selected strategy skipped for paper mode: {selected_decision['reason']}"
                self.audit_log.record("strategy_rejected", {"strategy": selected_decision["strategy_config"], "reason": selected_decision["reason"]})
                self.alert_manager.emit(
                    "risk_breach",
                    AlertSeverity.WARNING,
                    summary,
                    strategy_id=str(selected_decision["strategy_config"].get("id") or selected_decision["strategy_config"].get("name")),
                    metadata={"reason": selected_decision["reason"]},
                )
                return WorkflowResult("paper", summaries + [summary], {"orders": [], "selected_strategy": selected_decision})
            active_strategies = self.strategy_registry.list_by_stage("active")
            if not active_strategies:
                research = self.run_research_cycle(n_variants=1)
                active_strategies = research.details["active_strategies"]
            if not active_strategies:
                return WorkflowResult("paper", summaries + ["No active strategy is approved for paper trading."], {"orders": []})
            strategy_config = dict(active_strategies[0]["strategy"])
            summaries.append(f"Selected active strategy accepted for paper mode: {strategy_config.get('id') or strategy_config.get('name')}")
        data_config = self._data_config()
        symbols = list(data_config["symbols"])
        raw_data = self.data_agent.fetch_data(
            symbols,
            str(data_config["start_date"]),
            None if data_config.get("end_date") in {None, "null", ""} else str(data_config.get("end_date")),
        )
        windows = [int(strategy_config["short_window"]), int(strategy_config["long_window"])]
        featured_data = self.data_agent.build_features(raw_data, windows)
        symbol = symbols[0]
        latest_bar = featured_data[symbol][-1]
        strategy = MovingAverageCrossoverStrategy(
            short_window=int(strategy_config["short_window"]),
            long_window=int(strategy_config["long_window"]),
            version=str(strategy_config.get("version", "0.1.0")),
            supported_symbols=symbols,
        )
        signal = strategy.generate_signal(latest_bar)
        if signal != 1:
            summary = f"Paper cycle generated no buy order for {symbol}; strategy signal was flat."
            self.audit_log.record("approved_trade", {"signal": signal, "action": "no_order", "strategy": strategy_config})
            return WorkflowResult("paper", summaries + [summary], {"orders": []})

        quantity = int(self.settings.get("execution", {}).get("default_quantity", 1))
        order = OrderRequest(
            symbol=symbol,
            side="buy",
            quantity=quantity,
            order_type="market",
            reason="Paper cycle strategy signal.",
            source_strategy=strategy.identifier,
            estimated_price=float(latest_bar["close"]),
        )
        self._sync_risk_policy_from_system_state()
        risk_decision = self.risk_agent.assess_trade(
            order,
            self.broker.get_account_state(),
            self.broker.list_positions(),
            mode="paper",
        )
        if not risk_decision.approved:
            metadata = {"order": order.to_dict(), "reason": risk_decision.reason, "details": risk_decision.details}
            self.audit_log.record("rejected_trade", metadata)
            self._record_repeated_failure(risk_decision.reason, metadata)
            self.alert_manager.emit(
                "risk_breach",
                AlertSeverity.WARNING,
                f"Paper order rejected by risk policy: {risk_decision.reason}",
                strategy_id=order.source_strategy,
                symbol=order.symbol,
                metadata=metadata,
            )
            if self._recent_rejections(order.symbol, limit=10) >= 2:
                self.alert_manager.emit(
                    "repeated_order_rejection",
                    AlertSeverity.CRITICAL,
                    f"Repeated paper order rejections detected for {order.symbol}.",
                    strategy_id=order.source_strategy,
                    symbol=order.symbol,
                    metadata=metadata,
                )
            return WorkflowResult("paper", summaries + [f"Paper order rejected by risk policy: {risk_decision.reason}"], {"orders": []})

        status = self.execution_agent.submit_order(order)
        if status.status == "rejected":
            self._record_repeated_failure(status.message, {"status": status.to_dict()})
            self.alert_manager.emit(
                "paper_cycle_failure",
                AlertSeverity.CRITICAL,
                f"Paper broker rejected order for {status.symbol}: {status.message}",
                strategy_id=order.source_strategy,
                symbol=status.symbol,
                metadata={"status": status.to_dict()},
            )
        summary = f"Paper order {status.status}: {status.side} {status.quantity} {status.symbol} ({status.message})"
        return WorkflowResult("paper", summaries + [summary], {"orders": [status.to_dict()], "strategy": strategy_config})

    def _paper_risk_limits_summary(self) -> str:
        return f"Paper mode risk limits: max_drawdown={float(self.risk_policy.paper_max_drawdown):.2f}"

    def _sync_risk_policy_from_system_state(self) -> None:
        state = self.system_state.read()
        self.risk_policy.kill_switch_enabled = bool(self.risk_policy.kill_switch_enabled or state.get("kill_switch_enabled", False))
        self.risk_policy.repeated_failures = int(state.get("repeated_failures", 0))

    def _record_repeated_failure(self, reason: str, metadata: dict[str, Any]) -> None:
        before = self.system_state.read()
        previous_failures = int(before.get("repeated_failures", 0))
        state = self.system_state.record_repeated_failure(reason, limit=self.risk_policy.repeated_failure_limit)
        self.risk_policy.repeated_failures = int(state.get("repeated_failures", 0))
        if previous_failures < self.risk_policy.repeated_failure_limit <= self.risk_policy.repeated_failures:
            payload = {
                "reason": state.get("circuit_breaker_reason"),
                "repeated_failures": self.risk_policy.repeated_failures,
                "repeated_failure_limit": self.risk_policy.repeated_failure_limit,
                "last_failure": metadata,
            }
            self.audit_log.record("circuit_breaker_activated", payload)
            self.alert_manager.emit(
                "circuit_breaker_activated",
                AlertSeverity.CRITICAL,
                str(state.get("circuit_breaker_reason") or "Circuit breaker activated after repeated failures."),
                metadata=payload,
            )

    def _recent_rejections(self, symbol: str, limit: int = 10) -> int:
        records = self.audit_log.list_recent(limit)
        count = 0
        for record in records:
            if record.get("event_type") != "rejected_trade":
                continue
            order = record.get("payload", {}).get("order", {})
            if order.get("symbol") == symbol:
                count += 1
        return count

    def run_live_cycle(self) -> WorkflowResult:
        """Safe live-cycle stub. Live execution is disabled by default."""

        broker_settings = self.settings.get("broker", {})
        if not bool(broker_settings.get("live_enabled", False)):
            self.audit_log.record("live_guard_rejected", {"reason": "live_enabled is false"})
            return WorkflowResult("live", ["Live cycle disabled: broker.live_enabled is false."], {"orders": []})
        if str(broker_settings.get("live_confirmation", "")) != "I_UNDERSTAND_LIVE_TRADING_RISK":
            self.audit_log.record("live_guard_rejected", {"reason": "confirmation guard missing"})
            return WorkflowResult("live", ["Live cycle disabled: runtime confirmation guard is missing."], {"orders": []})
        return WorkflowResult("live", ["Live cycle is intentionally a stub until a real broker adapter is implemented."], {"orders": []})

    def run_example_workflow(self, n_variants: int = 3) -> list[str]:
        return self.run_research_cycle(n_variants=n_variants).summaries

    def _build_broker(self) -> BaseBroker:
        broker_settings = self.settings.get("broker", {})
        selected = str(broker_settings.get("selected", "paper"))
        if selected == "alpaca":
            return AlpacaBroker()
        return PaperBroker(broker_settings.get("paper_state_path", "multi_agent_trading_lab/state/paper_broker_state.json"))

    def _data_config(self) -> dict[str, Any]:
        data_config = dict(self.settings.get("data", {}))
        if not data_config:
            data_config = {
                "symbols": self.settings.get("symbols", ["AAPL"]),
                "start_date": self.settings.get("start_date", "2023-01-01"),
                "end_date": self.settings.get("end_date", "2023-06-30"),
            }
        data_config.setdefault("symbols", ["AAPL"])
        data_config["symbols"] = resolve_universe_symbols(self.settings)
        data_config.setdefault("start_date", "2023-01-01")
        data_config.setdefault("end_date", None)
        return data_config

    def _resolve_strategy_families(self, strategy_families: list[str] | None = None) -> list[str]:
        aliases = {
            "ma": "moving_average_crossover",
            "moving_average": "moving_average_crossover",
            "ma_rsi": "moving_average_rsi_filter",
            "moving_average_rsi": "moving_average_rsi_filter",
            "breakout": "breakout_trend",
        }
        requested = strategy_families or self.settings.get("strategy_families") or [self.settings.get("strategy", {}).get("name", "moving_average_crossover")]
        resolved: list[str] = []
        for family in requested:
            canonical = aliases.get(str(family), str(family))
            if canonical not in resolved:
                resolved.append(canonical)
        return resolved

    def _strategy_agent_for(self, strategy_name: str) -> StrategyAgent:
        return StrategyAgent(
            experiment_logger=self.logger,
            parameter_bounds=self.settings.get("strategies", {}).get(strategy_name),
            objective_config=self.settings.get("research_objective", {}),
            risk_policy=self.risk_policy,
            search_config=self.settings.get("strategy_search", {}),
            strategy_version=str(self.settings.get("strategy", {}).get("version", "0.1.0")),
        )

    @staticmethod
    def _feature_windows_from_variants(variants: list[dict[str, Any]]) -> list[int]:
        windows: set[int] = set()
        for variant in variants:
            params = dict(variant.get("strategy_params", {}))
            for key in ("short_window", "long_window", "breakout_window", "exit_window", "rsi_period"):
                if key in params:
                    windows.add(int(params[key]))
        return sorted(windows or {5, 20})


def run_example_workflow() -> list[str]:
    return TradingLabOrchestrator(load_settings("multi_agent_trading_lab/config/settings.yaml")).run_example_workflow()


def run_research_cycle() -> WorkflowResult:
    return TradingLabOrchestrator(load_settings("multi_agent_trading_lab/config/settings.yaml")).run_research_cycle()


def run_paper_cycle() -> WorkflowResult:
    return TradingLabOrchestrator(load_settings("multi_agent_trading_lab/config/settings.yaml")).run_paper_cycle()


def load_settings(path: str | Path) -> dict[str, Any]:
    """Load settings from YAML with a small stdlib fallback parser."""

    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        return _parse_simple_yaml(Path(path)) or DEFAULT_SETTINGS

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or DEFAULT_SETTINGS


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _market_data_summary(
    raw_data: dict[str, list[dict[str, Any]]],
    configured_start: str,
    configured_end: Any,
    source: str,
) -> dict[str, Any]:
    ranges: dict[str, dict[str, Any]] = {}
    for symbol, bars in raw_data.items():
        ranges[symbol] = {
            "rows": len(bars),
            "start_date": bars[0]["date"] if bars else None,
            "end_date": bars[-1]["date"] if bars else None,
        }
    return {
        "symbols": list(raw_data.keys()),
        "configured_start_date": configured_start,
        "configured_end_date": configured_end,
        "source": source,
        "ranges": ranges,
    }


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the simple YAML subset used by this repository's config files."""

    if not path.exists():
        return {}
    lines = [
        (len(line) - len(line.lstrip(" ")), _strip_comment(line).strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if _strip_comment(line).strip()
    ]
    parsed, _ = _parse_yaml_block(lines, 0, 0)
    return parsed if isinstance(parsed, dict) else {}


def _parse_yaml_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index
    if lines[index][1].startswith("- "):
        result: list[Any] = []
        while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
            result.append(_parse_scalar(lines[index][1][2:].strip()))
            index += 1
        return result, index

    result: dict[str, Any] = {}
    while index < len(lines) and lines[index][0] == indent and not lines[index][1].startswith("- "):
        key, _, value = lines[index][1].partition(":")
        key = key.strip()
        value = value.strip()
        index += 1
        if value:
            result[key] = _parse_scalar(value)
            continue
        if index < len(lines) and lines[index][0] > indent:
            result[key], index = _parse_yaml_block(lines, index, lines[index][0])
        else:
            result[key] = {}
    return result, index


def _strip_comment(line: str) -> str:
    in_quote = False
    quote_char = ""
    for index, char in enumerate(line):
        if char in {'"', "'"}:
            if not in_quote:
                in_quote = True
                quote_char = char
            elif quote_char == char:
                in_quote = False
        if char == "#" and not in_quote:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    if value in {"[]", ""}:
        return [] if value == "[]" else ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"null", "none"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
