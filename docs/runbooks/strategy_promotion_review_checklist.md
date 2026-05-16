# Strategy Promotion Review Checklist

## Purpose

Ensure no research candidate becomes an active paper-trading strategy without explicit human approval.

## When To Use

Use after `python3 scripts/run_research_cycle.py` creates pending promotion requests.

## Step-By-Step Actions

1. List pending promotions:

   ```bash
   python3 scripts/review_strategy_promotions.py --list
   ```

2. For each candidate, inspect:

   - Strategy ID and parameters.
   - Total return, max drawdown, and Sharpe ratio.
   - Any related alerts.
   - Recent experiment history for similar parameters.

3. Approve only if the metrics and risk posture are acceptable:

   ```bash
   python3 scripts/review_strategy_promotions.py --approve STRATEGY_ID
   ```

4. Reject if the evidence is weak, risky, duplicated, or unexplained:

   ```bash
   python3 scripts/review_strategy_promotions.py --reject STRATEGY_ID --reason "drawdown too close to limit"
   ```

5. Confirm the decision was recorded:

   ```bash
   tail -n 20 multi_agent_trading_lab/experiments/audit_log.jsonl
   ```

## Required Evidence

- Promotion request in `multi_agent_trading_lab/state/strategy_approvals.json`.
- Supporting metrics from the request.
- Approval or rejection audit event.
- Any alert emitted for awaiting approval or rejection.

## Decision Points

- Does the candidate pass the configured risk policy with margin?
- Is the candidate meaningfully different from already-active strategies?
- Does the research window look appropriate for the strategy?
- Is there any unresolved incident or kill switch state?

## Rollback And Escalation

Reject the promotion if evidence is incomplete. If a strategy was approved by mistake, move it out of active use through the registry and record an audit note before continuing.
