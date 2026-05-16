# Kill Switch Procedure

## Purpose

Stop all local paper-trading activity immediately when paper execution looks unsafe, noisy, or operationally unclear.

## When To Use

- Repeated order rejections or unexpected broker state changes.
- Drawdown or risk alerts that need human review.
- Suspected data quality issue, bad strategy selection, or incorrect configuration.
- Any operator uncertainty about whether paper trading should continue.

## Step-By-Step Actions

1. Activate the kill switch:

   ```bash
   python3 scripts/activate_kill_switch.py --reason "manual stop after repeated order rejects"
   ```

2. Confirm `multi_agent_trading_lab/state/system_state.json` has `kill_switch_enabled: true`.
3. Run one paper cycle only if you need to confirm safe blocking:

   ```bash
   python3 scripts/run_paper_cycle.py
   ```

   The expected result is a blocked/no-op paper cycle.

4. Inspect evidence:

   ```bash
   tail -n 20 multi_agent_trading_lab/experiments/audit_log.jsonl
   tail -n 20 multi_agent_trading_lab/experiments/alerts.jsonl
   ```

5. Generate a daily report for the incident day:

   ```bash
   python3 scripts/generate_daily_paper_report.py
   ```

## Required Evidence

- Kill-switch reason in `system_state.json`.
- `kill_switch_activated` and `kill_switch_event` audit records.
- Any `CRITICAL` alerts in `alerts.jsonl`.
- Paper broker state from `multi_agent_trading_lab/state/paper_broker_state.json`.

## Decision Points

- If the issue is configuration-only, fix config and keep paper trading stopped until a second operator review.
- If the issue is strategy behavior, reject or archive the affected promotion request.
- If the issue is data quality, refresh or validate cached data before any new research or paper cycle.

## Rollback And Escalation

Do not manually delete audit or alert evidence. To resume later, update the system state deliberately after the incident review and record the reason in the audit log. Do not use this procedure to enable live trading.
