"""Continuous paper-trading controller and monitor loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any, Protocol

from multi_agent_trading_lab.agents.backtest_agent import BacktestAgent
from multi_agent_trading_lab.agents.data_agent import DataAgent
from multi_agent_trading_lab.agents.execution_agent import ExecutionAgent
from multi_agent_trading_lab.agents.risk_agent import RiskAgent
from multi_agent_trading_lab.brokers.paper_broker import PaperBroker
from multi_agent_trading_lab.execution.execution_engine import ExecutionEngine
from multi_agent_trading_lab.execution.order_models import OrderRequest
from multi_agent_trading_lab.experiments.audit_log import AuditLog
from multi_agent_trading_lab.operations.alerting import AlertManager, AlertSeverity, FileAlertSink
from multi_agent_trading_lab.orchestrator.orchestrator import load_settings
from multi_agent_trading_lab.risk.risk_policy import RiskPolicy
from multi_agent_trading_lab.state.system_state import SystemStateStore


@dataclass(frozen=True)
class PaperStrategyConfig:
    id: str
    name: str
    params: dict[str, Any]
    version: str = "0.1.0"

    def to_strategy_config(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": "moving_average_crossover" if self.name in {"example_ma", "moving_average_crossover"} else self.name,
            "version": self.version,
            "strategy_params": dict(self.params),
            "short_window": int(self.params["short_window"]),
            "long_window": int(self.params["long_window"]),
        }


class PaperStepRunner(Protocol):
    def run_strategy_step(self, strategy: PaperStrategyConfig) -> dict[str, Any]:
        ...


class ContinuousPaperController:
    """Runs paper-only monitoring steps for configured strategies."""

    def __init__(
        self,
        strategies: list[PaperStrategyConfig],
        max_iterations: int | None = None,
        step_runner: PaperStepRunner | None = None,
        settings: dict[str, Any] | None = None,
    ) -> None:
        self.strategies = strategies
        self.max_iterations = max_iterations
        self.settings = settings or load_settings("multi_agent_trading_lab/config/settings.yaml")
        self._assert_paper_mode()
        self.step_runner = step_runner or RealPaperStepRunner(self.settings)
        self.latest_metrics: dict[str, dict[str, Any]] = {}
        self.iterations = 0
        self.should_stop = False
        self.system_state = SystemStateStore(self.settings.get("system_state_path", "multi_agent_trading_lab/state/system_state.json"))
        self.audit_log = AuditLog(self.settings.get("audit_log_path", "multi_agent_trading_lab/experiments/audit_log.jsonl"))
        self.alert_manager = AlertManager([FileAlertSink(self.settings.get("alert_log_path", "multi_agent_trading_lab/experiments/alerts.jsonl"))])

    def run_step(self) -> dict[str, dict[str, Any]]:
        """Run one paper-trading step for all configured strategies."""

        if self.should_stop:
            return self.latest_metrics
        state = self.system_state.read()
        if bool(state.get("kill_switch_enabled", False)):
            summary = {
                "paper_trading": {
                    "decision": "blocked_by_kill_switch",
                    "reason": state.get("kill_switch_reason", "kill switch enabled"),
                }
            }
            self.audit_log.record("kill_switch_event", {"action": "continuous_paper_blocked", "reason": summary["paper_trading"]["reason"]})
            self.alert_manager.emit(
                "kill_switch_activated",
                AlertSeverity.CRITICAL,
                "Continuous paper trading blocked because the kill switch is enabled.",
                metadata={"reason": summary["paper_trading"]["reason"]},
            )
            self.latest_metrics = summary
            self.stop()
            return summary
        summary: dict[str, dict[str, Any]] = {}
        for strategy in self.strategies:
            result = self.step_runner.run_strategy_step(strategy)
            summary[strategy.id] = result
            self.latest_metrics[strategy.id] = result
        self.iterations += 1
        if self.max_iterations is not None and self.iterations >= self.max_iterations:
            self.stop()
        return summary

    def stop(self) -> None:
        """Request a safe stop after the current step."""

        self.should_stop = True

    def _assert_paper_mode(self) -> None:
        execution = dict(self.settings.get("execution", {}))
        broker = dict(self.settings.get("broker", {}))
        if str(execution.get("mode", "paper")) == "live" or bool(broker.get("live_enabled", False)):
            raise ValueError("ContinuousPaperController only supports paper mode.")
        if str(broker.get("selected", "paper")) != "paper":
            raise ValueError("ContinuousPaperController requires the paper broker.")


class RealPaperStepRunner:
    """Real paper-mode step runner using existing project agents."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.settings = settings
        self.data_agent = DataAgent(settings)
        self.backtest_agent = BacktestAgent(
            starting_capital=float(settings.get("backtest", {}).get("starting_capital", 100_000.0)),
            max_capital_per_trade_pct=float(settings.get("backtest", {}).get("max_capital_per_trade_pct", 0.10)),
            allow_leverage=bool(settings.get("backtest", {}).get("allow_leverage", False)),
            commission_per_trade=float(settings.get("backtest", {}).get("commission_per_trade", 0.0)),
            slippage_pct=float(settings.get("backtest", {}).get("slippage_pct", 0.0)),
        )
        self.risk_policy = RiskPolicy.from_config(settings.get("risk", {}))
        self.risk_agent = RiskAgent(policy=self.risk_policy)
        self.audit_log = AuditLog(settings.get("audit_log_path", "multi_agent_trading_lab/experiments/audit_log.jsonl"))
        broker = PaperBroker(settings.get("broker", {}).get("paper_state_path", "multi_agent_trading_lab/state/paper_broker_state.json"))
        engine = ExecutionEngine(
            broker=broker,
            risk_policy=self.risk_policy,
            audit_log=self.audit_log,
            system_state=SystemStateStore(settings.get("system_state_path", "multi_agent_trading_lab/state/system_state.json")),
            mode="paper",
            dry_run=bool(settings.get("execution", {}).get("dry_run", False)),
            live_enabled=False,
        )
        self.execution_agent = ExecutionAgent(engine)

    def run_strategy_step(self, strategy: PaperStrategyConfig) -> dict[str, Any]:
        data_config = dict(self.settings.get("data", {}))
        symbols = list(data_config.get("symbols", []))
        start = str(data_config.get("start_date", "2010-01-01"))
        end = data_config.get("end_date")
        raw_data = self.data_agent.fetch_data(symbols, start, None if end in {None, "null", ""} else str(end))
        windows = [int(strategy.params["short_window"]), int(strategy.params["long_window"])]
        featured = self.data_agent.build_features(raw_data, windows)
        backtest = self.backtest_agent.run_backtest(strategy.to_strategy_config(), featured)
        metrics = dict(backtest["metrics"])
        decision = self.risk_agent.assess_risk(metrics)
        broker_summary = "risk_rejected"
        if decision["approved"]:
            first_symbol = symbols[0] if symbols else next(iter(featured))
            latest_bar = featured[first_symbol][-1]
            order = OrderRequest(
                symbol=first_symbol,
                side="buy",
                quantity=int(self.settings.get("execution", {}).get("default_quantity", 1)),
                order_type="market",
                reason=f"Continuous paper step for {strategy.id}",
                source_strategy=strategy.id,
                estimated_price=float(latest_bar["close"]),
            )
            status = self.execution_agent.submit_order(order)
            broker_summary = f"{status.status}: {status.message}"
        summary = {
            "total_return": metrics.get("total_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "last_equity": self._last_equity(backtest),
            "timestamp": datetime.now(UTC).isoformat(),
            "decision": decision,
            "paper_action": broker_summary,
        }
        self.audit_log.record("paper_strategy_step", {"strategy": strategy.id, "summary": summary})
        return summary

    @staticmethod
    def _last_equity(backtest: dict[str, Any]) -> float | None:
        per_symbol = dict(backtest.get("per_symbol_results", {}))
        equities: list[float] = []
        for result in per_symbol.values():
            values = result.get("portfolio_values", [])
            if values:
                equities.append(float(values[-1]))
        return sum(equities) / len(equities) if equities else None


def load_selected_paper_strategies(
    path: str | Path = "multi_agent_trading_lab/config/selected_strategy.yaml",
) -> list[PaperStrategyConfig]:
    """Load selected paper strategies from config."""

    config = _load_selected_strategy_config(Path(path))
    payload = config.get("selected_paper_strategies", [])
    strategies = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        strategies.append(
            PaperStrategyConfig(
                id=str(item.get("id")),
                name=str(item.get("name", "example_ma")),
                version=str(item.get("version", "0.1.0")),
                params=dict(item.get("params", {})),
            )
        )
    return strategies


def _load_selected_strategy_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        return _parse_selected_paper_yaml(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_selected_paper_yaml(path: Path) -> dict[str, Any]:
    """Parse selected_paper_strategies without depending on PyYAML."""

    if not path.exists():
        return {}
    strategies: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_selected_list = False
    in_params = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped == "selected_paper_strategies:":
            in_selected_list = True
            in_params = False
            continue
        if not in_selected_list:
            continue
        if stripped.startswith("- "):
            if current is not None:
                strategies.append(current)
            current = {"params": {}}
            in_params = False
            key, _, value = stripped[2:].partition(":")
            if key and value:
                current[key.strip()] = _parse_scalar(value.strip())
            continue
        if current is None:
            continue
        if stripped == "params:":
            in_params = True
            continue
        key, _, value = stripped.partition(":")
        if not key or not value:
            continue
        if in_params:
            current["params"][key.strip()] = _parse_scalar(value.strip())
        else:
            current[key.strip()] = _parse_scalar(value.strip())
    if current is not None:
        strategies.append(current)
    return {"selected_paper_strategies": strategies}


def _parse_scalar(value: str) -> Any:
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def run_continuous_paper_loop(
    controller: Any,
    max_steps: int,
    on_summary: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run paper steps until the controller stops or max_steps is reached.

    This is intentionally UI-free. CLI scripts can pass ``on_summary`` to print
    or log each step.
    """

    summaries: list[dict[str, Any]] = []
    for _ in range(max_steps):
        if getattr(controller, "should_stop", False):
            break
        summary = controller.run_step()
        summaries.append(summary)
        if on_summary is not None:
            on_summary(summary)
        if getattr(controller, "should_stop", False):
            break
    return summaries
