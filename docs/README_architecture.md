# Repository Architecture

This repository is organized as a staged autonomous trading foundation rather
than a single script. The current logic is intentionally simple, but the safety
boundaries are meant to survive growth.

Core flow:

1. `DiscoveryAgent` captures operating preferences and autonomy phase.
2. `DataAgent` fetches data and creates features.
3. `StrategyAgent` proposes versioned strategy variants.
4. `BacktestAgent` runs each variant in a minimal long/flat environment.
5. `RiskAgent` applies strategy and trade risk gates.
6. `StrategyRegistry` promotes or rejects candidates.
7. `ExecutionAgent` submits approved orders through `ExecutionEngine`.
8. `PaperBroker` simulates safe paper execution.
9. `ExperimentLogger` and `AuditLog` preserve memory and accountability.
10. `ReportAgent` formats summaries.

Future work should keep those responsibilities separate. For example, a richer
portfolio simulator belongs in `environment/`, while an LLM that proposes
strategies belongs behind `StrategyAgent`, not inside the backtester.
