# Optional Research Optimizer

The research optimizer is an optional candidate-discovery path for existing
strategy families. It is configured in
`multi_agent_trading_lab/config/settings.yaml` under `optimizer`:

```yaml
optimizer:
  enabled: false
  method: deterministic_random
  seed: 42
  n_trials: 25
```

`enabled` defaults to `false`, so the existing heuristic `StrategyAgent` search
remains the default behavior. When enabled, `method` must currently be
`deterministic_random`. The `seed` makes candidate generation reproducible, and
`n_trials` caps the number of requested optimizer candidates.

Supported strategy families are the existing research families:

- `moving_average_crossover` (`ma` in CLI aliases)
- `moving_average_rsi_filter` (`ma_rsi` in CLI aliases)
- `breakout_trend` (`breakout` in CLI aliases)

The optimizer only proposes candidates. It does not approve or promote a
strategy by itself. Optimizer-generated candidates still pass through the same
research gates as heuristic candidates, including data quality, in-sample risk,
walk-forward/OOS validation, cost-aware metrics, optional portfolio gating, and
optional regime gating.

## Trial Metadata

Optimizer-generated experiment records and approval queue entries include
`metrics.optimizer_trial`. This metadata identifies:

- `trial_number`: deterministic trial index for the candidate
- `method`: currently `deterministic_random`
- `seed`: seed used to generate the candidate
- `parameters`: sampled strategy parameters
- `objective_score`: the research objective score once available
- `gate_outcome`: `accepted`, `rejected`, or `failed`
- `rejection_reason`: populated when a gate rejects the candidate
- `warnings`: populated when generation produced fewer unique candidates than
  requested

`gate_outcome` values mean:

- `accepted`: the candidate passed promotion gates and entered the approval
  queue.
- `rejected`: the candidate completed evaluation but failed one or more gates.
- `failed`: the trial raised an evaluation error before a normal gate decision.

Failed optimizer trials are written to the audit log with the
`optimizer_trial_failed` event. Use that event together with
`metrics.optimizer_trial` to diagnose whether a failure came from candidate
generation, backtesting, data quality, or a promotion gate.

## Objective Configuration

Optimizer scoring uses the existing research objective configured under
`research_objective`. There is no separate `optimizer.objective` setting; if it
appears in config validation output, remove it to avoid implying a second
scoring source.
