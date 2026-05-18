# Strict Paper Research Profile

`multi_agent_trading_lab/config/settings.strict_paper.yaml` is a staging profile
for rehearsing the major research and paper safety gates together. It is not the
default config and it does not imply live-trading readiness.

For the broader difference between default, strict staging, and approval-grade
paper profiles, see `docs/paper_profiles.md`.

Use it when you want a deterministic paper-mode confidence check before a longer
paper run or after changing gate logic. The profile keeps:

- `operating_mode: paper`
- `execution.mode: paper`
- `broker.live_enabled: false`

It enables or configures:

- walk-forward/OOS validation
- non-zero slippage assumptions for backtests
- blocking data-quality checks
- portfolio promotion gating
- market-regime gating
- strategy health/quarantine reporting
- deterministic optimizer search

Run the strict validation harness with:

```bash
python3 scripts/run_strict_paper_validation.py \
  --settings multi_agent_trading_lab/config/settings.strict_paper.yaml \
  --output-root /tmp/strict_paper_validation
```

You can also run a research cycle directly with the profile:

```bash
python3 scripts/run_research_cycle.py \
  --settings multi_agent_trading_lab/config/settings.strict_paper.yaml
```

Paper cycles already support explicit settings paths:

```bash
python3 scripts/run_paper_cycle.py \
  --settings multi_agent_trading_lab/config/settings.strict_paper.yaml
```

## Reading Results

The strict validation command writes a summary JSON, audit log, and daily paper
report under the selected output root. Review:

- accepted examples for `metrics.optimizer_trial.gate_outcome: accepted`
- rejected examples for a clear `risk_decision.reason`
- `walk_forward` for in-sample and out-of-sample metrics
- `cost_assumptions` for slippage/commission disclosure
- `data_quality` for pass/fail status and issues
- `portfolio_backtest` for portfolio return, drawdown, exposure, and skipped symbols
- `market_regime` for the benchmark regime label and details
- audit events such as `strategy_promotion_requested`, `strategy_rejected`, and
  `optimizer_trial_failed`

Common rejection reasons include OOS thresholds, portfolio drawdown/return
thresholds, data-quality failures, regime labels outside the allowed list, and
paper-mode risk-policy checks. A rejection in this profile means the staging
gate is doing its job; it is a prompt to inspect assumptions, not a signal to
loosen live-trading safeguards.

Passing this profile only shows that the research/paper staging loop can run
with optional gates enabled in a deterministic local setting. Live trading still
requires separate explicit broker configuration, live-mode enablement,
confirmation text, operator readiness, and risk review.
