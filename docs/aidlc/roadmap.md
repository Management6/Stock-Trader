# Roadmap

## Phase 1: Research And Backtesting

- Keep `research` and `backtest` workflows deterministic and offline runnable.
- Use `StrategyAgent` to learn from experiment history and propose bounded
  parameter changes for the example moving-average strategy.
- Treat annualized Sharpe-style risk-adjusted return as the default research
  objective while continuing to log return and drawdown for future objectives.
- Expand strategy registry metadata.
- Add richer backtest metrics and transaction-cost modeling.
- Add walk-forward validation and out-of-sample evaluation.
- Require documented AI-DLC Inception artifacts before major strategy changes.

## Phase 2: Paper Trading With Alerts

- Use PaperBroker as the default execution provider.
- Add alerting hooks for approved and rejected trades.
- Add daily summary reports from experiment and audit logs.
- Add human approval checkpoints before paper strategy promotion.
- Add operational runbooks for kill switch and incident review.

## Phase 3: Limited Live Auto-Execution

- Implement a real broker adapter behind `BaseBroker`.
- Keep live disabled unless all explicit guards are satisfied.
- Start with small notional limits and restricted symbol universe.
- Add monitoring, reconciliation, and broker order-status polling.
- Add rollback and pause procedures before any production deployment.

## Future Agent Enhancements

- Use an LLM-backed DiscoveryAgent to clarify user constraints.
- Replace or augment the heuristic StrategyAgent with Bayesian optimization,
  evolutionary search, or local model support.
- Use a ReportAgent that summarizes audit trails and flags anomalies.
- Use an OperationsAgent only after paper trading controls are proven.
