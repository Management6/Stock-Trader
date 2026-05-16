# Requirements

## Functional Requirements

- Support explicit operating modes: `research`, `backtest`, `paper`, and
  `live`.
- Default to safe non-live operation.
- Fetch or generate market data for a configured stock universe.
- Generate basic strategy variants and keep strategy metadata versioned.
- Backtest candidate strategies and calculate total return, max drawdown, and a
  simple Sharpe-like metric.
- Maintain a strategy lifecycle registry with candidate, active, rejected, and
  archived states.
- Apply risk rules before strategy promotion and before trade execution.
- Convert approved signals into normalized order requests.
- Route orders through a broker abstraction.
- Provide a paper broker that persists account, positions, and orders locally.
- Keep live broker integration as a disabled stub until real credentials,
  operational controls, and human sign-off are added.
- Record experiment history and safety-relevant audit events.
- Provide command-line scripts for discovery, research, backtest, and paper
  cycles.

## Safety Requirements

- Live trading must never be the default.
- Live execution requires explicit config enablement and the runtime guard value
  `I_UNDERSTAND_LIVE_TRADING_RISK`.
- A kill switch must block strategy promotion and order execution.
- Signal generation must remain separate from order execution.
- The execution engine must validate order shape before broker routing.
- Risk policy must enforce max position size, max capital per trade, max open
  positions, symbol allow/block lists, drawdown limits, and circuit breakers.
- Every approved trade, rejected trade, strategy promotion, broker action, and
  kill switch event must be auditable.

## Non-Functional Requirements

- Use Python 3.11+.
- Keep dependencies lean and optional.
- Prefer small modules with clear contracts over complex trading logic.
- Avoid credentials, proprietary data, and live-order side effects.
- Make workflows runnable offline using deterministic synthetic data.
- Keep extension points clear for future LLMs, local models, broker SDKs, and
  human approval steps.
