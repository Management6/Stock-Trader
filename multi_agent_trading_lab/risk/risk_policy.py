"""Risk policy for strategy deployment and trade execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time

from multi_agent_trading_lab.brokers.base_broker import AccountState, Position
from multi_agent_trading_lab.execution.order_models import OrderRequest

DEFAULT_MAX_DRAWDOWN = -0.20
DEFAULT_MIN_BACKTEST_RETURN = -0.05


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str
    details: dict[str, object] = field(default_factory=dict)


@dataclass
class RiskPolicy:
    """Configurable hard limits for staged trading autonomy."""

    kill_switch_enabled: bool = False
    max_position_size: int = 100
    max_capital_per_trade: float = 5_000.0
    max_open_positions: int = 5
    daily_loss_limit: float = 1_000.0
    max_drawdown_threshold: float = DEFAULT_MAX_DRAWDOWN
    min_backtest_return: float = DEFAULT_MIN_BACKTEST_RETURN
    research_max_drawdown: float | None = None
    paper_max_drawdown: float | None = None
    allowed_symbols: set[str] = field(default_factory=set)
    blocked_symbols: set[str] = field(default_factory=set)
    trading_hours_start: str = "09:30"
    trading_hours_end: str = "16:00"
    enforce_trading_hours: bool = False
    repeated_failure_limit: int = 3
    repeated_failures: int = 0

    def __post_init__(self) -> None:
        if self.research_max_drawdown is None:
            self.research_max_drawdown = self.max_drawdown_threshold
        if self.paper_max_drawdown is None:
            self.paper_max_drawdown = self.max_drawdown_threshold
        self.max_drawdown_threshold = float(self.research_max_drawdown)

    @classmethod
    def from_config(cls, config: dict[str, object] | None) -> "RiskPolicy":
        payload = config or {}
        legacy_max_drawdown = float(payload.get("max_drawdown_threshold", DEFAULT_MAX_DRAWDOWN))
        research_max_drawdown = _nested_float(payload, "research", "max_drawdown", legacy_max_drawdown)
        paper_max_drawdown = _nested_float(payload, "paper", "max_drawdown", legacy_max_drawdown)
        return cls(
            kill_switch_enabled=bool(payload.get("kill_switch_enabled", False)),
            max_position_size=int(payload.get("max_position_size", 100)),
            max_capital_per_trade=float(payload.get("max_capital_per_trade", 5_000.0)),
            max_open_positions=int(payload.get("max_open_positions", 5)),
            daily_loss_limit=float(payload.get("daily_loss_limit", 1_000.0)),
            max_drawdown_threshold=research_max_drawdown,
            min_backtest_return=float(payload.get("min_backtest_return", DEFAULT_MIN_BACKTEST_RETURN)),
            research_max_drawdown=research_max_drawdown,
            paper_max_drawdown=paper_max_drawdown,
            allowed_symbols={str(s).upper() for s in payload.get("allowed_symbols", [])},
            blocked_symbols={str(s).upper() for s in payload.get("blocked_symbols", [])},
            trading_hours_start=str(payload.get("trading_hours_start", "09:30")),
            trading_hours_end=str(payload.get("trading_hours_end", "16:00")),
            enforce_trading_hours=bool(payload.get("enforce_trading_hours", False)),
            repeated_failure_limit=int(payload.get("repeated_failure_limit", 3)),
            repeated_failures=int(payload.get("repeated_failures", 0)),
        )

    def evaluate_strategy(self, metrics: dict[str, float]) -> RiskDecision:
        return self._evaluate_strategy_metrics(metrics, float(self.research_max_drawdown), stage="research")

    def evaluate_strategy_for_paper(self, metrics: dict[str, float]) -> RiskDecision:
        return self._evaluate_strategy_metrics(metrics, float(self.paper_max_drawdown), stage="paper")

    def _evaluate_strategy_metrics(self, metrics: dict[str, float], max_drawdown: float, stage: str) -> RiskDecision:
        if self.kill_switch_enabled:
            return RiskDecision(False, "Kill switch is enabled.")
        observed_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
        details = {"stage": stage, "max_drawdown": observed_drawdown, "max_drawdown_threshold": max_drawdown}
        if observed_drawdown < max_drawdown:
            return RiskDecision(False, "Strategy max drawdown exceeds threshold.", details)
        if metrics.get("total_return", 0.0) < self.min_backtest_return:
            return RiskDecision(False, "Strategy return is below deployment threshold.", details)
        return RiskDecision(True, "Strategy passes risk policy.", details)

    def evaluate_order(
        self,
        order: OrderRequest,
        account: AccountState,
        positions: list[Position],
        mode: str,
        current_time: datetime | None = None,
    ) -> RiskDecision:
        if self.kill_switch_enabled:
            return RiskDecision(False, "Kill switch is enabled.")
        if mode == "live":
            return RiskDecision(False, "Live execution requires a dedicated live policy override.")
        if self.repeated_failures >= self.repeated_failure_limit:
            return RiskDecision(
                False,
                "Circuit breaker triggered after repeated failures.",
                {"repeated_failures": self.repeated_failures, "repeated_failure_limit": self.repeated_failure_limit},
            )
        if order.symbol in self.blocked_symbols:
            return RiskDecision(False, f"{order.symbol} is blocked.", {"symbol": order.symbol, "blocked_symbols": sorted(self.blocked_symbols)})
        if self.allowed_symbols and order.symbol not in self.allowed_symbols:
            return RiskDecision(
                False,
                f"{order.symbol} is not in the allowed universe.",
                {"symbol": order.symbol, "allowed_symbols": sorted(self.allowed_symbols)},
            )
        if self.enforce_trading_hours and not self._within_trading_hours(current_time or datetime.now()):
            return RiskDecision(False, "Order is outside configured trading hours.")
        if order.quantity > self.max_position_size:
            return RiskDecision(False, "Order quantity exceeds max position size.")
        estimated_price = float(order.limit_price or order.estimated_price or 100.0)
        notional = estimated_price * order.quantity
        if notional > self.max_capital_per_trade:
            return RiskDecision(False, "Order notional exceeds max capital per trade.")
        if order.side == "buy" and len(positions) >= self.max_open_positions:
            existing_symbols = {position.symbol for position in positions}
            if order.symbol not in existing_symbols:
                return RiskDecision(False, "Max open positions limit reached.")
        if order.side == "buy" and notional > account.buying_power:
            return RiskDecision(False, "Insufficient buying power.")
        return RiskDecision(True, "Order passes risk policy.", {"notional": notional})

    def _within_trading_hours(self, current_time: datetime) -> bool:
        start = self._parse_clock(self.trading_hours_start)
        end = self._parse_clock(self.trading_hours_end)
        now = current_time.time()
        return start <= now <= end

    @staticmethod
    def _parse_clock(value: str) -> time:
        hour, minute = value.split(":")
        return time(hour=int(hour), minute=int(minute))


def _nested_float(payload: dict[str, object], section: str, key: str, default: float) -> float:
    section_payload = payload.get(section, {})
    if isinstance(section_payload, dict) and key in section_payload:
        return float(section_payload[key])
    return default
