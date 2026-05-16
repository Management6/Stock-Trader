# Architecture

```text
AI-DLC Project Context
        |
        v
Discovery/Profile Agent
        |
        v
TradingLabOrchestrator
        |
        +--> DataAgent ---------> data sources + features
        |                         |
        |                         v
        |                  yfinance cache (data/cache)
        |
        +--> StrategyAgent -----> versioned candidate strategies
        |                         |
        |                         v
        |                  StrategyRegistry
        |                         ^
        |                         |
        |                  ExperimentLogger history
        |
        +--> BacktestAgent -----> TradingEnvironment
        |
        +--> RiskAgent ---------> RiskPolicy + Portfolio Guardrails
        |
        +--> ExecutionAgent ----> ExecutionEngine
                                  |
                                  +--> AuditLog
                                  +--> PaperBroker
                                  +--> Future Live Broker Stub

ExperimentLogger <--------- research, backtest, and paper-run results
SystemStateStore <--------- kill switch, mode, operational state
```

## Signal-To-Order Boundary

Strategies only generate signals. They do not submit orders. The execution
path is:

1. Strategy signal.
2. OrderRequest creation by orchestration or future portfolio logic.
3. ExecutionEngine validation.
4. RiskAgent and RiskPolicy approval.
5. Broker routing.
6. Audit logging.

This makes it straightforward to insert human approval, alerts, or queueing
between signal generation and execution.

## Staged Workflows

- `run_research_cycle()`: fetch data, generate variants, backtest candidates,
  apply strategy risk checks, and promote approved strategies to paper state.
  Candidate generation is history-aware: `StrategyAgent` loads prior experiment
  records, ranks useful backtests, perturbs strong parameter sets, and avoids
  simple duplicates.
- `run_paper_cycle()`: use an active strategy to generate a signal, create a
  paper order, apply trade risk checks, route to PaperBroker, and audit the
  result.
- `run_live_cycle()`: disabled stub. It verifies live guards and records audit
  events but does not place real trades.

## Component Boundaries

- `brokers/`: provider-neutral account, position, order, and status interface.
- `execution/`: order models and execution engine.
- `risk/`: risk policy and portfolio guardrail helpers.
- `strategies/`: versioned strategies and lifecycle registry.
- `experiments/`: experiment memory and append-only audit logs.
- `state/`: persistent system mode and kill-switch state.
- `agents/`: deterministic agent roles with clear future LLM extension points.

## Market Data Pipeline

`DataAgent` reads the `data` section in `config/settings.yaml`, downloads daily
OHLCV from yfinance for the configured universe, normalizes columns to
`date/open/high/low/close/volume`, and caches CSV files under `data/cache/`.
The same agent converts cached pandas DataFrames into the internal bar format
used by the backtester, keeping data loading centralized for research and paper
workflows. The provider boundary is intentionally small so yfinance can later be
swapped for another free or paid source.

## Iterative Strategy Improvement

The current strategy learning loop is deliberately simple. It is a heuristic
search over moving-average parameters, using experiment memory as feedback. This
is an early Construction-phase step toward richer optimization methods such as
Bayesian search, walk-forward validation, local models, or LLM-assisted research
planning.

The research objective is configurable and currently defaults to annualized
Sharpe-style risk-adjusted return. Backtests still store `total_return` and
`max_drawdown`, which keeps historical experiment data reusable if the objective
changes later.

Strategy search is also risk-aware. `StrategyAgent` separates experiments that
pass `RiskPolicy` from experiments that fail it, allocates most candidate
generation around risk-passing high-score regions, and limits but does not remove
exploration around risk-failing regions. Adaptive narrowing can be toggled and
tuned in `config/settings.yaml`.

The example moving-average strategy can define a champion band, currently
45-50 / 105-120 in config. When enabled, candidate generation emphasizes that
region if history shows risk-passing experiments there. `research/batch.py`
provides a tested in-process helper for running multiple research iterations and
observing whether candidates continue to concentrate in the champion region.

## Experiment Analysis And Paper Candidate

`experiments/analysis.py` provides tested helpers for ranking historical
experiments, filtering by Sharpe and drawdown, and finding exact
moving-average parameter pairs such as 6/35. `scripts/analyze_example_ma.py`
wraps those helpers for human inspection.

`config/selected_strategy.yaml` can nominate one parameter set for paper mode.
The orchestrator validates that selection against recent experiment metrics and
`RiskPolicy` before paper execution. Live execution remains disabled and
unaffected by this selection mechanism.

## Continuous Paper Controller

`operations/continuous_paper.py` provides a tested controller for repeated
paper-only strategy checks. It accepts selected paper strategies, runs one safe
step at a time, updates metrics, and exposes a stop flag. It is intentionally
UI-free so terminal scripts and logs remain the only interaction surface. It
does not alter live trading settings.
