# Paper Research Profiles

The project has three practical configuration levels for research and paper
operations. None of them enables live trading.

## Default Config

`multi_agent_trading_lab/config/settings.yaml` is the normal development config.
It keeps optional gates conservative but does not turn every staging gate on by
default. Use it for everyday local research, smoke tests, and compatibility with
existing workflows.

Run:

```bash
python3 scripts/run_research_cycle.py
```

## Strict Paper Staging

`multi_agent_trading_lab/config/settings.strict_paper.yaml` is an integration
profile. It enables the major optional gates together so operators can confirm
that the pipeline, metadata, audit logs, reports, and approval queue all work as
a system.

Use it after gate logic changes or before a longer paper rehearsal. It may allow
candidates with slightly negative OOS return because its purpose is staging
confidence, not final operator approval.

Run:

```bash
python3 scripts/run_strict_paper_validation.py \
  --settings multi_agent_trading_lab/config/settings.strict_paper.yaml \
  --output-root /tmp/strict_paper_validation
```

## Approval-Grade Paper

`multi_agent_trading_lab/config/settings.approval_paper.yaml` is a stricter
operator approval screen for deciding whether a strategy should enter serious
paper-trading review. It keeps paper mode on and live trading off:

- `operating_mode: paper`
- `execution.mode: paper`
- `broker.live_enabled: false`

It enables the same major gates as strict staging, but uses approval-grade
thresholds:

- OOS `min_return: 0.0`
- OOS `min_sharpe: 0.0`
- OOS `max_drawdown: -0.15`
- portfolio `min_total_return: 0.0`
- portfolio `min_sharpe: 0.0`
- portfolio `max_drawdown: -0.15`
- backtest `slippage_pct: 0.0005`

Run:

```bash
python3 scripts/run_approval_paper_validation.py \
  --settings multi_agent_trading_lab/config/settings.approval_paper.yaml \
  --output-root /tmp/approval_paper_validation
```

Approval-grade validation can legitimately reject every candidate. Do not loosen
thresholds automatically just to produce an accepted strategy. Instead, review
the rejection breakdown, especially OOS return, OOS Sharpe, portfolio return,
portfolio Sharpe, data-quality failures, and regime gate decisions.

To search more broadly across existing strategy families without changing
approval thresholds, run:

```bash
python3 scripts/run_approval_search.py \
  --settings multi_agent_trading_lab/config/settings.approval_paper.yaml \
  --trials 100 \
  --seed 42 \
  --output-root /tmp/approval_search
```

The approval search reports trials, accepted/rejected counts, best candidate per
family, top rejection reasons, and the best near miss. It can include the
existing `moving_average_crossover`, `moving_average_rsi_filter`, and
`breakout_trend` families. It does not approve anything outside the normal
approval queue mechanics.

After approval search finds candidates, run robustness validation before manual
review:

```bash
python3 scripts/run_candidate_robustness.py \
  --settings multi_agent_trading_lab/config/settings.approval_paper.yaml \
  --input /tmp/approval_search/<run>/approval_search_summary.json \
  --output-root /tmp/candidate_robustness
```

Robustness validation stress-tests accepted candidates across alternate OOS
splits, higher slippage, small parameter perturbations, portfolio metrics, and
the configured regime gate. A `robust` status means the candidate survived those
checks. A `fragile` status means it passed the base approval profile but failed
one or more stresses; inspect the top failure reason before putting it in front
of an operator for serious paper approval.

Passing `approval_paper` still does not mean a strategy is live-trading ready.
It only means the candidate passed a stricter local research and paper approval
screen. Live trading requires separate explicit broker configuration, live-mode
enablement, confirmation text, operational readiness, and risk review.
