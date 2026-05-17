"""Backtesting agent."""

from __future__ import annotations

from typing import Any

from multi_agent_trading_lab.agents.base_agent import AgentResult, BaseAgent
from multi_agent_trading_lab.data.data_sources import MarketData
from multi_agent_trading_lab.environment.trading_env import TradingEnvironment
from multi_agent_trading_lab.metrics import compute_backtest_metrics
from multi_agent_trading_lab.strategies.breakout_strategy import BreakoutTrendStrategy
from multi_agent_trading_lab.strategies.example_strategy import MovingAverageCrossoverStrategy
from multi_agent_trading_lab.strategies.ma_rsi_strategy import MovingAverageRsiFilterStrategy


class BacktestAgent(BaseAgent):
    """Runs candidate strategies through a deterministic backtest.

    Inputs: strategy configuration and featured market data.
    Outputs: portfolio path and basic performance metrics.
    Extension point: replace TradingEnvironment with richer portfolio simulator.
    """

    def __init__(
        self,
        starting_capital: float = 100_000.0,
        max_capital_per_trade_pct: float = 0.10,
        allow_leverage: bool = False,
        commission_per_trade: float = 0.0,
        slippage_pct: float = 0.0,
    ) -> None:
        super().__init__("backtest_agent")
        self.starting_capital = starting_capital
        self.max_capital_per_trade_pct = max_capital_per_trade_pct
        self.allow_leverage = allow_leverage
        self.commission_per_trade = commission_per_trade
        self.slippage_pct = slippage_pct

    def run_backtest(self, strategy_config: dict[str, Any], data: MarketData, data_quality_report: dict[str, Any] | None = None) -> dict[str, Any]:
        per_symbol_results = {
            symbol: self._run_single_symbol_backtest(strategy_config, symbol, bars)
            for symbol, bars in data.items()
            if len(bars) >= 2
        }
        if not per_symbol_results:
            raise ValueError("Backtest requires at least one symbol with two or more bars")
        metrics = self._aggregate_symbol_metrics([result["metrics"] for result in per_symbol_results.values()])
        return {
            "symbols": list(per_symbol_results.keys()),
            "strategy": strategy_config,
            "per_symbol_results": per_symbol_results,
            "metrics": metrics,
            "cost_assumptions": self._cost_assumptions(),
            "data_quality": data_quality_report or {"passed": True, "issues": []},
        }

    def _run_single_symbol_backtest(self, strategy_config: dict[str, Any], symbol: str, bars: list[dict[str, Any]]) -> dict[str, Any]:
        params = dict(strategy_config.get("strategy_params", strategy_config))
        strategy = _build_strategy(strategy_config, params)
        env = TradingEnvironment(
            bars,
            initial_cash=self.starting_capital,
            max_capital_per_trade_pct=self.max_capital_per_trade_pct,
            allow_leverage=self.allow_leverage,
            commission_per_trade=self.commission_per_trade,
            slippage_pct=self.slippage_pct,
        )
        portfolio_values = [env.state.portfolio_value]
        trade_count = 0
        done = False
        while not done:
            previous_position = env.state.position
            signal = strategy.generate_signal(env.state.bar)
            state, _, done = env.step(signal)
            if state.position != previous_position:
                trade_count += 1
            portfolio_values.append(state.portfolio_value)
        metrics = self.evaluate_results(portfolio_values)
        metrics["trade_count"] = trade_count
        return {
            "symbol": symbol,
            "portfolio_values": portfolio_values,
            "metrics": metrics,
            "cost_assumptions": self._cost_assumptions(),
            "trade_count": trade_count,
        }

    def evaluate_results(self, results: list[float]) -> dict[str, float | int | None]:
        """Compute total return, max drawdown, and annualized Sharpe ratio."""

        return compute_backtest_metrics(results, annualization_factor=252)

    def _aggregate_symbol_metrics(self, metrics: list[dict[str, Any]]) -> dict[str, float | int | None]:
        """Aggregate per-symbol backtests into one research objective record."""

        total_returns = [float(metric["total_return"]) for metric in metrics if metric.get("total_return") is not None]
        drawdowns = [float(metric["max_drawdown"]) for metric in metrics if metric.get("max_drawdown") is not None]
        sharpes = [float(metric["sharpe_ratio"]) for metric in metrics if metric.get("sharpe_ratio") is not None]
        history_points = [int(metric["history_points"]) for metric in metrics if metric.get("history_points") is not None]
        return {
            "total_return": sum(total_returns) / len(total_returns) if total_returns else None,
            "max_drawdown": min(drawdowns) if drawdowns else None,
            "sharpe_ratio": sum(sharpes) / len(sharpes) if sharpes else None,
            "sharpe": sum(sharpes) / len(sharpes) if sharpes else None,
            "history_points": min(history_points) if history_points else None,
            "symbols_tested": len(metrics),
        }

    def _cost_assumptions(self) -> dict[str, float]:
        return {
            "commission_per_trade": float(self.commission_per_trade),
            "slippage_pct": float(self.slippage_pct),
        }

    def run(self, strategy_config: dict[str, Any], data: MarketData) -> AgentResult:
        return AgentResult(self.name, self.run_backtest(strategy_config, data))


def _build_strategy(strategy_config: dict[str, Any], params: dict[str, Any]) -> Any:
    strategy_name = str(strategy_config.get("name", "moving_average_crossover"))
    version = str(strategy_config.get("version", "0.1.0"))
    if strategy_name == "breakout_trend":
        return BreakoutTrendStrategy(
            breakout_window=int(params["breakout_window"]),
            exit_window=int(params["exit_window"]),
            version=version,
        )
    if strategy_name == "moving_average_rsi_filter":
        return MovingAverageRsiFilterStrategy(
            short_window=int(params["short_window"]),
            long_window=int(params["long_window"]),
            rsi_period=int(params["rsi_period"]),
            rsi_min=float(params["rsi_min"]),
            rsi_max=float(params["rsi_max"]),
            version=version,
        )
    return MovingAverageCrossoverStrategy(
        short_window=int(params["short_window"]),
        long_window=int(params["long_window"]),
        stop_loss_pct=params.get("stop_loss_pct"),
        take_profit_pct=params.get("take_profit_pct"),
        version=version,
    )
