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

For CI smoke validation, run the approval-search to candidate-robustness
pipeline through the Phase 2 harness:

```bash
python3 scripts/run_phase2_validation.py \
  --scenario approval_search_candidate_robustness \
  --output-root /tmp/approval_robustness_validation
```

The smoke scenario uses `settings.approval_paper.yaml`, 10 trials per family,
and seed `42`. It checks that `approval_search_summary.json` and
`candidate_robustness_summary.json` are valid JSON, experiment records are
non-empty, accepted/rejected and robustness counts are present, robustness reads
the accepted candidates from the approval-search output, approval thresholds are
unchanged, and live trading remains disabled.

For the full operator approval search across existing strategy families without
changing approval thresholds, run:

```bash
python3 scripts/run_approval_search.py \
  --settings multi_agent_trading_lab/config/settings.approval_paper.yaml \
  --trials 100 \
  --seed 42 \
  --output-root /tmp/approval_search
```

The full approval search reports trials, accepted/rejected counts, best
candidate per family, top rejection reasons, and the best near miss. It can
include the existing `moving_average_crossover`, `moving_average_rsi_filter`,
and `breakout_trend` families. It does not approve anything outside the normal
approval queue mechanics.

After the full approval search finds candidates, run the full robustness review
before manual review:

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

Use the CI smoke only to prove the pipeline still runs. Use the full 100-trial
approval search plus full robustness review for operator decisions.

## Paper Watchlist Review

After the full robustness review, create a small operator-reviewed paper
watchlist from robust accepted candidates:

```bash
python3 scripts/select_paper_watchlist.py \
  --approval-summary /tmp/approval_search/<approval-run>/approval_search_summary.json \
  --robustness-summary /tmp/candidate_robustness/<robustness-run>/candidate_robustness_summary.json \
  --max-candidates 5 \
  --output-root /tmp/paper_watchlist
```

The selector is read-only by default. It writes `paper_watchlist.json` and
`paper_watchlist.md` with candidate ids, strategy families, parameters, OOS
metrics, portfolio metrics, robustness summaries, cost assumptions, regime
results, reasons for selection, excluded-candidate buckets, and top excluded
candidate ids by reason. The report includes an explicit warning that the list
is for paper monitoring only, not live trading.

Ranking is deterministic. Eligible candidates must be both approval-accepted
and robustness-robust. The ranking then prefers higher OOS Sharpe, positive and
higher OOS return, lower OOS drawdown, positive and higher portfolio return,
lower portfolio drawdown, and higher objective score. By default, candidates
are diversified by strategy family before filling the remaining slots.

Only use approval-queue writing when an operator has reviewed the watchlist and
wants local promotion-request artifacts:

```bash
python3 scripts/select_paper_watchlist.py \
  --approval-summary /tmp/approval_search/<approval-run>/approval_search_summary.json \
  --robustness-summary /tmp/candidate_robustness/<robustness-run>/candidate_robustness_summary.json \
  --max-candidates 5 \
  --output-root /tmp/paper_watchlist \
  --write-approval-queue
```

That flag writes an approval queue under the watchlist output directory. It
does not approve strategies, enable live trading, or change broker behavior.

If the watchlist is empty, the JSON report has `passed: false` and
`status: "empty"` while the CLI still exits cleanly so operators can inspect the
artifact. Start debugging with the excluded buckets:

- `not_approval_accepted`: candidates did not pass the approval gate.
- `not_robust`: candidates were present in robustness output but were fragile.
- `missing_robustness`: approval-accepted candidates were not found in the
  robustness summary.
- `missing_metrics`: candidates lacked OOS or portfolio metrics needed for
  operator ranking.
- `duplicate_or_over_limit`: candidates were duplicates or ranked outside
  `--max-candidates`.

Do not loosen thresholds to fill an empty watchlist. Re-run approval search,
inspect rejection and robustness failure reasons, and only create a paper
watchlist when robust accepted candidates are available.

## Watchlist Paper Cycle

Once an operator has reviewed `paper_watchlist.md`, run a paper-only monitoring
cycle scoped to the selected watchlist candidates:

```bash
python3 scripts/run_watchlist_paper_cycle.py \
  --settings multi_agent_trading_lab/config/settings.approval_paper.yaml \
  --watchlist /tmp/paper_watchlist/<watchlist-run>/paper_watchlist.json \
  --output-root /tmp/watchlist_paper_cycle
```

This does not change the default broad paper cycle. It only runs when a
watchlist path is supplied. The cycle considers selected watchlist candidate ids
only, records skipped/non-selected candidates in the audit log, applies
strategy-health quarantine decisions, generates paper signals, routes any paper
orders through the paper execution path, and writes a daily paper monitoring
report plus a cycle summary.

Operator flow:

1. Run the full approval search.
2. Run the full robustness review.
3. Select the paper watchlist.
4. Run the watchlist paper cycle.
5. Inspect paper attribution, strategy-health decisions, skipped candidates,
   audit logs, and the daily paper report.

The watchlist paper cycle is still paper-only. It does not enable live trading,
does not auto-approve candidates, and does not write to the operational approval
queue.

Passing `approval_paper` still does not mean a strategy is live-trading ready.
It only means the candidate passed a stricter local research and paper approval
screen. Live trading requires separate explicit broker configuration, live-mode
enablement, confirmation text, operational readiness, and risk review.
