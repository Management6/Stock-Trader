# Paper Trading Incident Review

## Purpose

Create a repeatable review trail after a paper-trading incident such as repeated rejections, unexpected fills, drawdown alerts, or kill-switch activation.

## When To Use

- Any `CRITICAL` alert.
- Any kill-switch activation.
- Three or more related rejected paper orders.
- A strategy behaving differently from its research metrics.

## Step-By-Step Actions

1. Freeze activity with the kill switch if it is not already engaged.
2. Capture the current state files:

   - `multi_agent_trading_lab/state/system_state.json`
   - `multi_agent_trading_lab/state/paper_broker_state.json`
   - `multi_agent_trading_lab/state/strategy_approvals.json`

3. Inspect recent operational records:

   ```bash
   tail -n 100 multi_agent_trading_lab/experiments/audit_log.jsonl
   tail -n 100 multi_agent_trading_lab/experiments/alerts.jsonl
   ```

4. Generate the report artifact:

   ```bash
   python3 scripts/generate_daily_paper_report.py
   ```

5. Identify the triggering strategy, symbol, risk rule, and operator action.
6. Decide whether to reject a pending promotion, leave the kill switch engaged, or resume paper trading after a config fix.

## Required Evidence

- Alert event IDs and severities.
- Audit event IDs for rejected orders, broker actions, approvals, or kill switch.
- Paper position and order counts.
- Relevant strategy metrics from experiment memory.

## Decision Points

- Was the system behaving as designed and correctly blocking risk?
- Did the strategy pass research but fail operational constraints?
- Was the paper broker state plausible and inspectable?
- Is a code fix, config change, or strategy rejection required?

## Rollback And Escalation

Keep paper trading disabled until the evidence explains the incident. Escalate to code review if any audit event is missing, malformed, or inconsistent with broker state.
