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
from multi_agent_trading_lab.config.validation import normalize_trading_config
from multi_agent_trading_lab.data.universes import resolve_universe_symbols
from multi_agent_trading_lab.execution.execution_engine import ExecutionEngine
from multi_agent_trading_lab.execution.order_models import OrderRequest
from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.experiments.experiment_logger import ExperimentLogger
from multi_agent_trading_lab.operations.alerting import AlertManager, AlertSeverity, FileAlertSink
from multi_agent_trading_lab.operations.approvals import ApprovalQueue
from multi_agent_trading_lab.research.portfolio_backtest import run_portfolio_backtest
from multi_agent_trading_lab.research.regime import classify_market_regime
from multi_agent_trading_lab.research.scoring import score_metrics
from multi_agent_trading_lab.research.walk_forward import run_walk_forward_validation
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
        "default_universe": None,
        "symbols": ["WES.AX", "YAL.AX", "SLX.AX", "NVDA"],
        "timeframe": "1d",
        "start_date": "2010-01-01",
        "end_date": None,
        "cache_dir": "data/cache",
        "force_refresh": False,
    },
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
    "oos_validation": {
        "enabled": True,
        "split_ratio": 0.70,
        "min_sharpe": 0.0,
        "max_drawdown": -0.20,
        "min_return": -0.05,
    },
    "data_quality": {
        "block_on_missing_columns": True,
        "block_on_duplicate_dates": True,
        "block_on_invalid_prices": True,
        "block_on_stale_data": False,
        "block_on_large_gaps": False,
        "block_on_zero_volume": False,
        "expect_volume": True,
        "max_gap_days": None,
        "max_staleness_days": None,
    },
    "portfolio": {
        "enabled": False,
        "allocation_method": "equal_weight",
        "max_symbols": None,
        "min_valid_symbols": 3,
        "min_total_return": -0.02,
        "max_drawdown": -0.25,
        "min_sharpe": 0.0,
        "max_skipped_symbol_ratio": 0.50,
    },
    "regime": {
        "enabled": False,
        "benchmark_symbol": "SPY",
        "lookback_days": 120,
        "trend_ma_days": 50,
        "max_volatility": 0.30,
        "allowed_regimes": ["bullish", "trending"],
        "block_on_insufficient_data": True,
    },
    "strategy_health": {
        "enabled": False,
        "quarantine_on_expectation_mismatch": True,
        "max_rejection_rate": 0.50,
        "repeated_rejection_threshold": 3,
        "max_paper_drawdown": -0.10,
        "min_signals_before_quarantine": 5,
        "allow_manual_reactivation": True,
    },
    "optimizer": {
        "enabled": False,
        "method": "deterministic_random",
        "seed": 42,
        "n_trials": 25,
    },
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
        "slippage_pct": 0.0005,
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
        "allowed_symbols": [],
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
        merged_settings = _deep_merge(DEFAULT_SETTINGS, settings or {})
        self.settings, self.config_validation = normalize_trading_config(merged_settings)
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
        search_config["optimizer"] = dict(self.settings.get("optimizer", {}))
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
        if self.config_validation.warnings:
            self.audit_log.record("config_validation_warning", {"warnings": self.config_validation.warnings})
        if self.config_validation.errors:
            self.audit_log.record("config_validation_error", {"errors": self.config_validation.errors})
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
        start_date = _metadata_date_to_string(data_config["start_date"])
        end_date = _metadata_date_to_string(data_config.get("end_date"))
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
        try:
            raw_data = self.data_agent.fetch_data(symbols, start_date, None if end_date in {None, "null", ""} else str(end_date))
        except ValueError as exc:
            data_quality = self._data_quality_payload()
            self.audit_log.record("research_data_quality_blocked", {"reason": str(exc), "data_quality": data_quality})
            return WorkflowResult(
                stage="research",
                summaries=[f"Research blocked by data quality gate: {exc}"],
                details={
                    "profile": profile,
                    "records": [],
                    "active_strategies": self.strategy_registry.list_by_stage("active"),
                    "objective": self.settings.get("research_objective", {}),
                    "data_quality": data_quality,
                },
            )
        data_quality = self._data_quality_payload()
        featured_data = self.data_agent.build_features(raw_data, windows)

        summaries: list[str] = []
        records: list[dict[str, Any]] = []
        for variant in variants:
            self.strategy_registry.register(variant, stage="candidate", reason="Generated by StrategyAgent.")
            optimizer_trial = dict(variant.get("optimizer_trial", {})) if variant.get("optimizer_trial") else None
            try:
                backtest = self.backtest_agent.run_backtest(variant, featured_data, data_quality_report=data_quality)
            except Exception as exc:
                risk_decision = {
                    "approved": False,
                    "reason": f"Optimizer trial failed: {exc}" if optimizer_trial else f"Backtest failed: {exc}",
                    "details": {"exception": repr(exc), "data_quality": data_quality},
                }
                metrics = {"total_return": 0.0, "max_drawdown": 0.0, "sharpe_ratio": 0.0, "sharpe": 0.0, "history_points": 0}
                if optimizer_trial is not None:
                    optimizer_trial = {
                        **optimizer_trial,
                        "objective_score": None,
                        "gate_outcome": "failed",
                        "rejection_reason": str(exc),
                    }
                    metrics["optimizer_trial"] = optimizer_trial
                record = self.logger.log_experiment(
                    config={
                        "strategy": variant,
                        "symbols": symbols,
                        "start_date": start_date,
                        "end_date": end_date,
                        "deployment_stage": "rejected",
                    },
                    metrics=metrics,
                    risk_decision=risk_decision,
                    approval_status="rejected",
                    deployment_stage="rejected",
                    mode="backtest",
                )
                if optimizer_trial is not None:
                    record["optimizer_trial"] = optimizer_trial
                records.append(record)
                self.strategy_registry.promote(variant["id"], "rejected", str(risk_decision["reason"]))
                self.audit_log.record("optimizer_trial_failed", {"strategy": variant, "optimizer_trial": optimizer_trial, "reason": str(exc)})
                summaries.append(self.report_agent.summarize_experiment(record))
                continue
            risk_decision = self.risk_agent.assess_risk(backtest["metrics"])
            cost_assumptions = dict(backtest.get("cost_assumptions", {}))
            risk_decision.setdefault("details", {})
            if isinstance(risk_decision["details"], dict):
                risk_decision["details"]["cost_assumptions"] = cost_assumptions
                risk_decision["details"]["data_quality"] = data_quality
            data_quality_decision = self._evaluate_data_quality_gate(data_quality)
            if not data_quality_decision["approved"]:
                risk_decision = data_quality_decision
            walk_forward: dict[str, Any] | None = None
            promotion_metrics: dict[str, Any] = dict(backtest["metrics"])
            promotion_metrics["cost_assumptions"] = cost_assumptions
            promotion_metrics["data_quality"] = data_quality
            if risk_decision["approved"]:
                walk_forward = self._run_walk_forward_validation(variant, featured_data, data_quality)
                if walk_forward is not None:
                    promotion_metrics["walk_forward"] = walk_forward
                    oos_decision = self._evaluate_oos_gate(walk_forward)
                    if not oos_decision["approved"]:
                        risk_decision = oos_decision
                    else:
                        risk_decision = {
                            "approved": True,
                            "reason": "Strategy passes research risk policy and OOS validation.",
                            "details": {"walk_forward": walk_forward, "cost_assumptions": cost_assumptions},
                        }
            portfolio_backtest: dict[str, Any] | None = None
            if risk_decision["approved"]:
                portfolio_backtest = self._run_portfolio_backtest(variant, featured_data, data_quality)
                if portfolio_backtest is not None:
                    promotion_metrics["portfolio_backtest"] = portfolio_backtest
                    portfolio_decision = self._evaluate_portfolio_gate(portfolio_backtest)
                    if not portfolio_decision["approved"]:
                        risk_decision = portfolio_decision
                    else:
                        risk_decision = {
                            "approved": True,
                            "reason": f"{risk_decision['reason']} Portfolio gate passed.",
                            "details": {
                                **dict(risk_decision.get("details", {})),
                                "portfolio_backtest": portfolio_backtest,
                            },
                        }
            market_regime: dict[str, Any] | None = None
            if risk_decision["approved"]:
                market_regime = self._run_regime_analysis(featured_data)
                if market_regime is not None:
                    promotion_metrics["market_regime"] = market_regime
                    regime_decision = self._evaluate_regime_gate(market_regime)
                    if not regime_decision["approved"]:
                        risk_decision = regime_decision
                    else:
                        risk_decision = {
                            "approved": True,
                            "reason": f"{risk_decision['reason']} Regime gate passed.",
                            "details": {
                                **dict(risk_decision.get("details", {})),
                                **dict(regime_decision.get("details", {})),
                            },
                        }
            stage = "paper" if risk_decision["approved"] else "rejected"
            objective_score = score_metrics(backtest["metrics"], self.settings.get("research_objective", {}))
            if optimizer_trial is not None:
                optimizer_trial = {
                    **optimizer_trial,
                    "objective_score": objective_score,
                    "gate_outcome": "accepted" if risk_decision["approved"] else "rejected",
                    "rejection_reason": None if risk_decision["approved"] else risk_decision["reason"],
                }
                promotion_metrics["optimizer_trial"] = optimizer_trial
            record = self.logger.log_experiment(
                config={
                    "strategy": variant,
                    "symbols": symbols,
                    "start_date": start_date,
                    "end_date": end_date,
                    "deployment_stage": stage,
                },
                metrics=promotion_metrics,
                risk_decision=risk_decision,
                approval_status="approved" if risk_decision["approved"] else "rejected",
                deployment_stage=stage,
                mode="backtest",
            )
            record["objective_score"] = objective_score
            if optimizer_trial is not None:
                record["optimizer_trial"] = optimizer_trial
            if walk_forward is not None:
                record["walk_forward"] = walk_forward
            if portfolio_backtest is not None:
                record["portfolio_backtest"] = portfolio_backtest
            if market_regime is not None:
                record["market_regime"] = market_regime
            record["cost_assumptions"] = cost_assumptions
            record["data_quality"] = data_quality
            record["metrics"]["data_quality"] = data_quality
            records.append(record)
            if risk_decision["approved"]:
                self.approval_queue.request_promotion(
                    strategy=variant,
                    rationale="Strategy passed research risk policy and OOS validation and is eligible for paper review.",
                    metrics=promotion_metrics,
                    requester="run_research_cycle",
                )
            else:
                self.strategy_registry.promote(variant["id"], "rejected", str(risk_decision["reason"]))
                self.audit_log.record(
                    "strategy_rejected",
                    {
                        "strategy": variant,
                        "reason": risk_decision["reason"],
                        "metrics": backtest["metrics"],
                        "cost_assumptions": cost_assumptions,
                        "data_quality": data_quality,
                        "walk_forward": walk_forward,
                        "portfolio_backtest": portfolio_backtest,
                        "market_regime": market_regime,
                    },
                )
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
                "data_quality": data_quality,
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
            health_decision = self._paper_health_decision(strategy_config)
            if health_decision is not None:
                summary = "Strategy skipped: quarantined due to paper attribution."
                self.audit_log.record("strategy_health_skip", {"strategy": strategy_config, "decision": health_decision, "reason": summary})
                return WorkflowResult("paper", summaries + [summary], {"orders": [], "strategy": strategy_config, "strategy_health": health_decision})
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
            strategy_config = {}
            skipped_health: list[dict[str, Any]] = []
            for record in active_strategies:
                candidate = dict(record["strategy"])
                health_decision = self._paper_health_decision(candidate)
                if health_decision is not None:
                    summary = "Strategy skipped: quarantined due to paper attribution."
                    self.audit_log.record("strategy_health_skip", {"strategy": candidate, "decision": health_decision, "reason": summary})
                    skipped_health.append({"strategy": candidate, "decision": health_decision})
                    continue
                strategy_config = candidate
                break
            if not strategy_config:
                return WorkflowResult(
                    "paper",
                    summaries + ["Strategy skipped: quarantined due to paper attribution."],
                    {"orders": [], "strategy_health_skips": skipped_health},
                )
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

    def _paper_health_decision(self, strategy_config: dict[str, Any]) -> dict[str, Any] | None:
        if not bool(self.settings.get("strategy_health", {}).get("enabled", False)):
            return None
        strategy_id = str(strategy_config.get("id") or strategy_config.get("name"))
        decision = dict(self.system_state.read().get("strategy_health", {}).get(strategy_id, {}))
        if decision.get("status") == "quarantined":
            return decision
        return None

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
            search_config={**dict(self.settings.get("strategy_search", {})), "optimizer": dict(self.settings.get("optimizer", {}))},
            strategy_version=str(self.settings.get("strategy", {}).get("version", "0.1.0")),
        )

    def _run_walk_forward_validation(
        self,
        strategy_config: dict[str, Any],
        featured_data: dict[str, list[dict[str, Any]]],
        data_quality: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        oos_config = dict(self.settings.get("oos_validation", {}))
        if not bool(oos_config.get("enabled", True)):
            return None
        ranges = self._walk_forward_ranges(featured_data, float(oos_config.get("split_ratio", 0.70)))
        if ranges is None:
            return None
        return run_walk_forward_validation(
            strategy_config,
            featured_data,
            in_sample=ranges["in_sample"],
            out_of_sample=ranges["out_of_sample"],
            backtest_agent=self.backtest_agent,
            data_quality_report=data_quality,
        )

    def _run_portfolio_backtest(
        self,
        strategy_config: dict[str, Any],
        featured_data: dict[str, list[dict[str, Any]]],
        data_quality: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        portfolio_config = dict(self.settings.get("portfolio", {}))
        if not bool(portfolio_config.get("enabled", False)):
            return None
        max_symbols = portfolio_config.get("max_symbols")
        return run_portfolio_backtest(
            strategy_config,
            featured_data,
            backtest_agent=self.backtest_agent,
            data_quality_report=data_quality,
            allocation_method=str(portfolio_config.get("allocation_method", "equal_weight")),
            max_symbols=None if max_symbols in {None, "null", ""} else int(max_symbols),
        )

    def _run_regime_analysis(self, featured_data: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
        regime_config = dict(self.settings.get("regime", {}))
        if not bool(regime_config.get("enabled", False)):
            return None
        benchmark_symbol = str(regime_config.get("benchmark_symbol", "SPY"))
        return classify_market_regime(
            featured_data.get(benchmark_symbol, []),
            benchmark_symbol=benchmark_symbol,
            lookback_days=int(regime_config.get("lookback_days", 120)),
            trend_ma_days=int(regime_config.get("trend_ma_days", 50)),
            max_volatility=float(regime_config.get("max_volatility", 0.30)),
        )

    def _data_quality_payload(self) -> dict[str, Any]:
        if self.data_agent.last_data_quality is None:
            return {"passed": True, "issues": []}
        return self.data_agent.last_data_quality.to_dict()

    @staticmethod
    def _evaluate_data_quality_gate(data_quality: dict[str, Any]) -> dict[str, Any]:
        errors = [issue for issue in data_quality.get("issues", []) if issue.get("severity") == "error"]
        if not errors:
            return {"approved": True, "reason": "Data quality gate passed.", "details": {"data_quality": data_quality}}
        codes = sorted({str(issue.get("code")) for issue in errors})
        return {
            "approved": False,
            "reason": f"Data quality gate failed: {', '.join(codes)}.",
            "details": {"data_quality": data_quality, "failed_data_quality_codes": codes},
        }

    def _evaluate_portfolio_gate(self, portfolio_backtest: dict[str, Any]) -> dict[str, Any]:
        config = dict(self.settings.get("portfolio", {}))
        metrics = dict(portfolio_backtest.get("metrics", {}))
        symbols = list(portfolio_backtest.get("symbols", []))
        skipped = list(portfolio_backtest.get("skipped_symbols", []))
        thresholds = {
            "min_valid_symbols": int(config.get("min_valid_symbols", 3)),
            "min_total_return": float(config.get("min_total_return", -0.02)),
            "max_drawdown": float(config.get("max_drawdown", -0.25)),
            "min_sharpe": float(config.get("min_sharpe", 0.0)),
            "max_skipped_symbol_ratio": float(config.get("max_skipped_symbol_ratio", 0.50)),
        }
        details = {"portfolio_backtest": portfolio_backtest, "thresholds": thresholds}
        valid_count = len(symbols)
        if valid_count < thresholds["min_valid_symbols"]:
            return {
                "approved": False,
                "reason": f"Portfolio gate failed: valid symbols {valid_count} is below threshold {thresholds['min_valid_symbols']}.",
                "details": {**details, "failed_threshold": "min_valid_symbols"},
            }
        considered_count = valid_count + len(skipped)
        skipped_ratio = len(skipped) / considered_count if considered_count else 0.0
        if skipped_ratio > thresholds["max_skipped_symbol_ratio"]:
            return {
                "approved": False,
                "reason": f"Portfolio gate failed: skipped symbol ratio {skipped_ratio:.2f} exceeds threshold {thresholds['max_skipped_symbol_ratio']:.2f}.",
                "details": {**details, "failed_threshold": "max_skipped_symbol_ratio"},
            }
        total_return = _metric(metrics, "total_return")
        if total_return is None or total_return < thresholds["min_total_return"]:
            return {
                "approved": False,
                "reason": f"Portfolio gate failed: total_return {total_return} is below threshold {thresholds['min_total_return']}.",
                "details": {**details, "failed_threshold": "min_total_return"},
            }
        max_drawdown = _metric(metrics, "max_drawdown")
        if max_drawdown is None or max_drawdown < thresholds["max_drawdown"]:
            return {
                "approved": False,
                "reason": f"Portfolio gate failed: max_drawdown {max_drawdown} is below threshold {thresholds['max_drawdown']}.",
                "details": {**details, "failed_threshold": "max_drawdown"},
            }
        sharpe = _metric(metrics, "sharpe_ratio", "sharpe")
        if sharpe is None or sharpe < thresholds["min_sharpe"]:
            return {
                "approved": False,
                "reason": f"Portfolio gate failed: sharpe {sharpe} is below threshold {thresholds['min_sharpe']}.",
                "details": {**details, "failed_threshold": "min_sharpe"},
            }
        return {"approved": True, "reason": "Portfolio gate passed.", "details": details}

    def _evaluate_regime_gate(self, market_regime: dict[str, Any]) -> dict[str, Any]:
        config = dict(self.settings.get("regime", {}))
        regime = str(market_regime.get("regime", "insufficient_data"))
        details = {"market_regime": market_regime, "allowed_regimes": list(config.get("allowed_regimes", ["bullish", "trending"]))}
        if regime == "insufficient_data":
            if bool(config.get("block_on_insufficient_data", True)):
                return {
                    "approved": False,
                    "reason": "Regime gate failed: insufficient benchmark data.",
                    "details": {**details, "failed_threshold": "insufficient_data"},
                }
            return {
                "approved": True,
                "reason": "Regime gate warning: insufficient benchmark data.",
                "details": {**details, "market_regime_warning": "insufficient_data"},
            }
        allowed = {str(value) for value in config.get("allowed_regimes", ["bullish", "trending"])}
        if regime not in allowed:
            return {
                "approved": False,
                "reason": f"Regime gate failed: {regime} is not allowed.",
                "details": {**details, "failed_threshold": "allowed_regimes"},
            }
        return {"approved": True, "reason": "Regime gate passed.", "details": details}

    def _evaluate_oos_gate(self, walk_forward: dict[str, Any]) -> dict[str, Any]:
        config = dict(self.settings.get("oos_validation", {}))
        metrics = dict(walk_forward.get("out_of_sample", {}).get("metrics", {}))
        sharpe = _metric(metrics, "sharpe_ratio", "sharpe")
        max_drawdown = _metric(metrics, "max_drawdown")
        total_return = _metric(metrics, "total_return")
        min_sharpe = float(config.get("min_sharpe", 0.0))
        max_drawdown_threshold = float(config.get("max_drawdown", -0.20))
        min_return = float(config.get("min_return", -0.05))
        details = {"walk_forward": walk_forward, "thresholds": {"min_sharpe": min_sharpe, "max_drawdown": max_drawdown_threshold, "min_return": min_return}}
        if sharpe is None or sharpe < min_sharpe:
            return {
                "approved": False,
                "reason": f"OOS sharpe {sharpe} is below threshold {min_sharpe}.",
                "details": {**details, "failed_threshold": "min_sharpe"},
            }
        if max_drawdown is None or max_drawdown < max_drawdown_threshold:
            return {
                "approved": False,
                "reason": f"OOS max drawdown {max_drawdown} exceeds threshold {max_drawdown_threshold}.",
                "details": {**details, "failed_threshold": "max_drawdown"},
            }
        if total_return is None or total_return < min_return:
            return {
                "approved": False,
                "reason": f"OOS return {total_return} is below threshold {min_return}.",
                "details": {**details, "failed_threshold": "min_return"},
            }
        return {"approved": True, "reason": "OOS validation passed.", "details": details}

    @staticmethod
    def _walk_forward_ranges(featured_data: dict[str, list[dict[str, Any]]], split_ratio: float) -> dict[str, tuple[str, str]] | None:
        dates = sorted({str(bar.get("date", "")) for bars in featured_data.values() for bar in bars if bar.get("date")})
        if len(dates) < 4:
            return None
        split_index = int(len(dates) * split_ratio)
        split_index = max(2, min(len(dates) - 2, split_index))
        return {
            "in_sample": (dates[0], dates[split_index - 1]),
            "out_of_sample": (dates[split_index], dates[-1]),
        }

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


def _metadata_date_to_string(value: Any) -> str | None:
    if value in {None, "null", ""}:
        return None
    return str(value)


def _metric(metrics: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


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
