# Daily Operator Checklist

## Purpose

Give the paper-trading operator a short daily routine for reviewing local activity before any further research or paper cycles.

## When To Use

Run once per operating day and after any batch of manual paper-trading tests.

## Step-By-Step Actions

1. Check system state:

   ```bash
   cat multi_agent_trading_lab/state/system_state.json
   ```

2. Review pending strategy promotions:

   ```bash
   python3 scripts/review_strategy_promotions.py --list
   ```

3. Inspect alerts:

   ```bash
   tail -n 50 multi_agent_trading_lab/experiments/alerts.jsonl
   ```

4. Generate the daily report:

   ```bash
   python3 scripts/generate_daily_paper_report.py
   ```

5. Review paper broker positions and order counts.
6. Decide whether any pending promotions should be approved, rejected, or left pending.

## Required Evidence

- Daily report JSON and Markdown artifacts in `multi_agent_trading_lab/reports/`.
- Pending approval list.
- Alert and audit tails.
- Paper broker state.

## Decision Points

- Are there any `CRITICAL` alerts?
- Are order rejections expected and explained?
- Are open positions within the intended paper limits?
- Are new strategy candidates supported by acceptable metrics?

## Rollback And Escalation

If anything is unexplained, activate the kill switch and use the incident review runbook. Do not approve new strategies while an incident review is open.
