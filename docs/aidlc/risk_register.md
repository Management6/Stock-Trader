# Risk Register

| Risk | Impact | Mitigation | Current Status |
| --- | --- | --- | --- |
| Accidental live trading | Financial loss | Live mode disabled by default, explicit config flag, confirmation guard, live broker stub | Controlled |
| Strategy overfitting | Poor out-of-sample performance | Keep experiment history, require staged promotion, add future walk-forward tests | Open |
| Bad order sizing | Excess exposure | Max position size and max capital per trade in RiskPolicy | Controlled |
| Trading blocked/allowed on wrong symbols | Unintended exposure | Symbol allowlist and blocklist | Controlled |
| Repeated execution failures | Operational instability | Circuit breaker field in RiskPolicy and audit events | Partial |
| Kill switch ignored | Severe loss or unsafe state | RiskPolicy checks kill switch before promotion and execution | Controlled |
| Broker adapter bug | Incorrect orders | Broker abstraction, paper broker first, live stub until tested | Controlled |
| Missing audit trail | Poor incident response | JSONL audit log for promotions, rejections, broker actions, live guard failures | Controlled |
| Market data quality issue | Misleading signals | Synthetic fallback for tests, future data validation needed | Open |
| Lack of human review | Unsafe autonomy | Architecture separates signal and execution; human approval can be inserted | Partial |
