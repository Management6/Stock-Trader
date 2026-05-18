# Multi-Agent Trading Lab

An experimental Python foundation for a staged autonomous stock-trading bot.
The system is designed to research, evaluate, select, and eventually execute
stock strategies, starting safely with backtesting, paper trading, strict risk
controls, audit logs, and human review hooks.

This is educational research software, not financial advice. It does not place
live trades today.

## Staged Autonomy Model

- Phase 1: research and backtesting only.
- Phase 2: paper trading with alerts, audit logs, and human review hooks.
- Phase 3: limited live auto-execution only after explicit config enablement,
  risk checks, broker configuration, runtime confirmation, and Operations
  readiness.

Default execution is safe. Live mode is disabled by default and the included
Alpaca broker is a non-trading stub.

## How The Agents Interact

- `DiscoveryAgent`: captures user intent and operating preferences.
- `DataAgent`: fetches or generates market data and builds features.
- `StrategyAgent`: proposes versioned strategy variants.
- `BacktestAgent`: evaluates candidates in a simple environment.
- `RiskAgent`: checks strategy promotion and trade execution constraints.
- `ExecutionAgent`: submits approved orders to the execution engine.
- `ReportAgent`: summarizes experiments and operational activity.
- `OrchestratorAgent`: wraps staged workflows.

Signals and orders are separated. A strategy can suggest exposure, but only the
execution layer can create and route an order after validation and risk checks.

## Iterative Strategy Learning

`StrategyAgent` now learns from experiment history for the example
moving-average strategy. During a research cycle it:

- loads recent JSONL experiment records
- ranks prior backtests by Sharpe or a return/drawdown score
- separates risk-passing experiments from risk-failing experiments
- perturbs the best parameter sets within configured bounds
- focuses most new candidates near parameter regions that both score well and
  pass the configured risk policy
- limits repeated exploration around obviously too-risky regions while still
  leaving a small exploration budget
- avoids straightforward duplicates already seen in recent experiments
- falls back to an initial deterministic grid when no history exists

Run an iterative research pass with:

```bash
python3 scripts/run_research_cycle.py
```

Choose strategy families explicitly:

```bash
python3 scripts/run_research_cycle.py --strategy-families ma
python3 scripts/run_research_cycle.py --strategy-families ma_rsi
python3 scripts/run_research_cycle.py --strategy-families breakout
python3 scripts/run_research_cycle.py --strategy-families ma,ma_rsi,breakout
```

Each run proposes new parameter sets, backtests them, logs records with
`strategy_params`, and prints the best variants from that run.

Risk-aware search behavior is controlled in
`multi_agent_trading_lab/config/settings.yaml` under `strategy_search`.
The `risk_drawdown_buffer` setting lets the adaptive search prefer strategies
that pass the drawdown limit with a little room to spare, instead of clustering
around barely-passing regions.
`adaptive_long_window_max` can be used to keep adaptive narrowing out of long
window regions that have repeatedly failed in the current research regime,
while the risky exploration budget still allows occasional probes.

An optional deterministic optimizer can propose parameter candidates for the
same strategy families. It is disabled by default; see
[docs/optimizer.md](docs/optimizer.md) for configuration, trial metadata, and
audit details.

For staging the optional gates together without changing the default config, see
[docs/strict_paper_profile.md](docs/strict_paper_profile.md). For the default
vs. staging vs. approval-grade profile split, see
[docs/paper_profiles.md](docs/paper_profiles.md).

The current config also supports a champion band for the example moving-average
strategy:

```yaml
research:
  champion_band_enabled: true
  champion_band:
    example_ma:
      short_window: {min: 45, max: 50}
      long_window: {min: 105, max: 120}
```

When enabled and backed by risk-passing history, StrategyAgent allocates most
new candidates inside that band while preserving some exploration outside it.

Run several research iterations in one process:

```bash
python3 scripts/run_research_batch.py --iterations 5 --n-variants 5
```

## Experiment Analysis

Experiment history can be inspected without rerunning a backtest. The analysis
helpers in `multi_agent_trading_lab/experiments/analysis.py` can:

- list top experiments for a strategy
- filter by Sharpe or drawdown
- find records matching a specific moving-average parameter pair
- summarize the best known run for a parameter pair

For example:

```bash
python3 scripts/analyze_example_ma.py --short-window 6 --long-window 35
```

This is research-only analysis. It does not place trades or enable live trading.

## Auto-Research Until Good Strategies

The auto-research loop repeatedly runs the existing research/backtest workflow
until it finds enough distinct strategies that meet the configured quality bar,
or until it reaches an iteration cap. It stays in research/backtest mode.

Default quality thresholds are:

- `sharpe_ratio >= 0.70`
- `max_drawdown >= -0.15`

Run with defaults:

```bash
python3 scripts/run_auto_research_until_good_strategies.py
```

Run a shorter search:

```bash
python3 scripts/run_auto_research_until_good_strategies.py --max-iterations 10 --target-good-strategies 3 --progress-every 1
```

Filter auto-research by family:

```bash
python3 scripts/run_auto_research_until_good_strategies.py --strategy-families ma
python3 scripts/run_auto_research_until_good_strategies.py --strategy-families ma_rsi
python3 scripts/run_auto_research_until_good_strategies.py --strategy-families breakout
python3 scripts/run_auto_research_until_good_strategies.py --strategy-families ma,ma_rsi,breakout
python3 scripts/run_auto_research_until_good_strategies.py --strategy-families ma_rsi --min-sharpe 0.5 --min-drawdown -0.20 --max-iterations 100 --target-good-strategies 5
```

## Strategy Families

The current research pipeline supports three deterministic daily stock strategy
families:

- `ma`: plain moving-average crossover, long when the short average is above
  the long average.
- `ma_rsi`: moving-average crossover with an RSI entry filter. It uses
  `short_window`, `long_window`, `rsi_period`, `rsi_min`, and `rsi_max`; entries
  require the MA condition and RSI inside the configured band, while exits use
  the MA condition only.
- `breakout`: long-only channel breakout, entering above a prior high channel
  and exiting below a prior low channel.

Smoke-test the command without touching research history:

```bash
python3 scripts/run_auto_research_until_good_strategies.py --max-iterations 2 --target-good-strategies 1 --synthetic-test-mode
```

## Selected Paper Candidate

You can mark one strategy configuration as the current paper-trading candidate
in `multi_agent_trading_lab/config/selected_strategy.yaml`:

```yaml
selected_strategy:
  name: moving_average_crossover
  version: "0.1.0"
  params:
    short_window: 6
    long_window: 35
```

`run_paper_cycle.py` will only use that selected strategy if matching experiment
history exists and the latest/best matching metrics pass `config/risk.yaml`.
If the selected strategy fails risk, the script prints a warning and skips paper
execution.

## Continuous Paper CLI

`config/selected_strategy.yaml` also defines two fixed paper-test strategies:

- `example_ma_s48_l105`
- `example_ma_s45_l110`

Paper testing is terminal/log driven. Run one safe paper cycle with:

```bash
python3 scripts/run_paper_cycle.py
```

The pure controller in `operations/continuous_paper.py` can also be called from
custom CLI scripts to run multiple paper-only steps. It is terminal-only and
asserts that live execution is not enabled.

## Phase 2 Operator Workflows

Risk-passing research candidates now enter a local approval queue instead of
becoming active paper strategies automatically. Review them with:

```bash
python3 scripts/review_strategy_promotions.py --list
python3 scripts/review_strategy_promotions.py --approve STRATEGY_ID
python3 scripts/review_strategy_promotions.py --reject STRATEGY_ID --reason "drawdown too close to limit"
```

Paper operations emit structured alerts to:

```text
multi_agent_trading_lab/experiments/alerts.jsonl
```

Generate the deterministic daily paper report with:

```bash
python3 scripts/generate_daily_paper_report.py
```

This writes JSON and Markdown artifacts under:

```text
multi_agent_trading_lab/reports/
```

Stop all local paper trading with the kill switch:

```bash
python3 scripts/activate_kill_switch.py --reason "manual stop after repeated order rejects"
```

Operator runbooks live in `docs/runbooks/`.

## Phase 2 Validation Sprint

Before considering any Phase 3 planning, run the deterministic Phase 2
validation harness. It uses local fixtures only and does not call live brokers or
external notification services.

Run every scenario:

```bash
python3 scripts/run_phase2_validation.py --scenario all
```

Run one scenario:

```bash
python3 scripts/run_phase2_validation.py --scenario drawdown_breach
python3 scripts/run_phase2_validation.py --scenario approval_required_candidate
```

Available scenarios:

- `happy_path_daily_cycle`
- `approval_required_candidate`
- `repeated_order_rejections`
- `drawdown_breach`
- `paper_cycle_failure`
- `kill_switch_activation`
- `kill_switch_persistence`
- `daily_report_generation`
- `runbook_integrity_check`

Validation artifacts are written under:

```text
multi_agent_trading_lab/validation_runs/
```

Each run writes:

- `validation_summary.json`
- `validation_report.md`
- scenario JSON files under `scenarios/`
- local validation audit and alert logs under `experiments/`
- report artifacts under `reports/` for reporting scenarios

Good enough for longer unattended paper testing means all validation scenarios
pass, the operator report is understandable, alerts are actionable without
duplicates, kill-switch persistence is clear, and the runbooks pass the
integrity check. This is still not approval for live trading.

## Research Objective And Sharpe Ratio

The research loop optimizes for a Sharpe-style risk-adjusted objective by
default. Backtests compute:

- `total_return`
- `max_drawdown`
- `sharpe_ratio`

For daily bars, `sharpe_ratio` is annualized as:

```text
mean(daily_returns) / std(daily_returns) * sqrt(252)
```

The risk-free rate is assumed to be 0 for now. Total return and max drawdown are
still logged for every experiment, so you can later switch the objective to
return-only or a custom return/drawdown score without losing historical data.
Change the objective in `multi_agent_trading_lab/config/settings.yaml` under
`research_objective`.

## Safety Controls

- Paper mode is the default execution path.
- Live mode requires `broker.live_enabled: true` and the confirmation value
  `I_UNDERSTAND_LIVE_TRADING_RISK`.
- `RiskPolicy` enforces kill switch, max position size, max capital per trade,
  max open positions, symbol allow/block lists, drawdown thresholds, and circuit
  breakers.
- `AuditLog` records strategy promotions, rejected trades, approved trades,
  broker actions, kill switch events, and live-guard rejections.
- `PaperBroker` persists simulated account state locally.

## AI-DLC Usage

This repository follows the AWS AI-DLC workflow style from
`awslabs/aidlc-workflows`.

Project context lives in:

- `.aidlc/project/problem.md`
- `.aidlc/project/assumptions.md`
- `.aidlc/project/non_goals.md`
- `docs/aidlc/vision.md`
- `docs/aidlc/requirements.md`
- `docs/aidlc/architecture.md`
- `docs/aidlc/risk_register.md`
- `docs/aidlc/roadmap.md`

Use AI-DLC phases this way:

- Inception: clarify trading goals, risk posture, non-goals, rollout phase, and
  approval criteria.
- Construction: implement small, tested modules for strategies, brokers,
  execution, risk, and auditability.
- Operations: add monitoring, incident response, kill-switch runbooks, broker
  reconciliation, live-readiness gates, and rollback procedures.

## Run The Scripts

```bash
python3 scripts/download_data.py
python3 scripts/run_discovery.py
python3 scripts/run_research_cycle.py
python3 scripts/run_backtest_example.py
python3 scripts/run_paper_cycle.py
```

The research and backtest scripts write experiment memory to:

```text
multi_agent_trading_lab/experiments/experiment_memory.jsonl
```

Paper and execution activity writes audit events to:

```text
multi_agent_trading_lab/experiments/audit_log.jsonl
```

## Market Data

Historical daily OHLCV data is configured in
`multi_agent_trading_lab/config/settings.yaml` under `data`.

Named universes are configured under `universes`, with `data.default_universe`
selecting the one used by research and backtest scripts unless overridden.
Paper trading uses the same active universe by default. Runtime config
resolution follows this precedence:

1. `data.default_universe` selects `universes.<name>`.
2. If no default universe is set, `data.symbols` is used.
3. Legacy top-level `symbols`, `start_date`, and `end_date` are ignored when
   `data` is present.
4. `risk.allowed_symbols` is the paper execution allow-list. If it is omitted
   or empty, it is derived from the active universe. If it is different from the
   active universe, it is preserved and startup validation records a warning.

`multi_agent_trading_lab/config/settings.yaml` is the runtime source of truth
for orchestrated research and paper trading. `multi_agent_trading_lab/config/risk.yaml`
is kept for standalone risk-policy checks and should mirror the `risk` section
in `settings.yaml`; when they differ, orchestrator runs use `settings.yaml`.
Allow-list rejections are recorded in the audit log and daily paper reports with
the rejected symbol and configured allow-list.

Example:

```yaml
data:
  provider: yfinance
  default_universe: full_mix
  timeframe: 1d
  start_date: 2010-01-01
  end_date: null
  cache_dir: data/cache

universes:
  core_asx:
    - CBA.AX
    - BHP.AX
    - WES.AX
  us_tech:
    - NVDA
    - AAPL
    - MSFT
  full_mix:
    - CBA.AX
    - BHP.AX
    - WES.AX
    - NVDA
    - AAPL
    - MSFT
```

The expanded default universe includes ASX stocks using `.AX` suffixes and a
small US tech/growth set.

Download/cache daily data for the configured default universe:

```bash
python3 scripts/download_universe_data.py
```

Download a specific universe:

```bash
python3 scripts/download_universe_data.py --universe core_asx
python3 scripts/download_universe_data.py --universe full_mix
```

The current provider is `yfinance`, with CSV cache files written to
`data/cache/`. Change `data.symbols`, `data.start_date`, `data.end_date`,
`data.timeframe`, and `data.cache_dir` to control the universe and history
window.

Typical sequence:

```bash
python3 scripts/download_data.py
python3 scripts/run_backtest_example.py
python3 scripts/run_research_cycle.py
```

Run research against a named universe:

```bash
python3 scripts/run_research_cycle.py --universe full_mix --strategy-families ma,breakout --n-variants 2
python3 scripts/run_backtest_example.py --universe core_asx
```

If market-data dependencies are unavailable or a network call fails, research
workflows fall back to deterministic synthetic data so the safety and testing
paths still run. Install project dependencies before real data downloads:

```bash
python3 -m pip install -e .
```

## Run Tests

```bash
python3 -m unittest discover -s tests
```

## Project Layout

```text
multi_agent_trading_lab/
  agents/
  brokers/
  config/
  data/
  environment/
  execution/
  experiments/
  orchestrator/
  risk/
  state/
  strategies/
scripts/
tests/
docs/aidlc/
.amazonq/rules/
.aidlc/project/
```

## Important Limits

Backtests disclose fixed `commission_per_trade` and percentage `slippage_pct`
assumptions from `backtest` config; both default to `0.0` to preserve historical
results unless explicitly configured. This repository still does not model
taxes, order-book depth, corporate actions, market halts, or production
monitoring yet. Treat every workflow as a research scaffold until a full AI-DLC
Operations phase proves readiness.
