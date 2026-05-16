# Initial Assumptions

- Python 3.11+ is the baseline runtime.
- Historical stock data can come from `yfinance` or a compatible adapter.
- Local development and tests must work offline through deterministic synthetic
  data.
- Backtesting starts with a simple long/flat daily-bar simulator.
- Experiment memory starts as JSONL and can later move to SQLite or a richer
  tracking system.
- AI agents are Python classes today and may later call LLMs or external tools.
- Paper trading is the default execution mode.
- Live trading requires explicit configuration and a runtime confirmation guard.
- Broker credentials are never stored in this repository.
