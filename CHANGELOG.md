# Changelog

## 0.1.0 (2026-11-17)

First release.

- `asof(decision_time)`: context manager and decorator (sync and async) holding the decision time
  in a ContextVar; nesting may only move the clock back.
- `filter`, `check`, `latest` over pandas DataFrames/Series, Polars DataFrames/LazyFrames, lists of
  `PITRecord` or dicts, and single records; `LookaheadError`, `UnstampedError`.
- `PITRecord(event_time, available_at, vintage, source, confidence)` with confidence levels
  `exact`, `rule`, `bound`, `live`.
- `asof_join`: per-key join on availability (fix for back-filled fundamentals).
- `guard` for any callable (date arguments, stamped outputs, opt-in text scan) and `guard_tool` /
  `guard_tools` for LangChain and LangGraph tools; the raised error is also a `ToolException`.
- `df.pit.indicator(name, ...)`: fin-primitives indicators with strict warmup and
  `bars_available` / `bars_required`; `required_bars`, `coverage`.
- Adapters: FRED/ALFRED vintages (fredapi), SEC EDGAR acceptance times (submissions +
  companyfacts), yfinance bars (close + lag, exchange_calendars closes, 24/7 sessions) and live
  snapshots.
- Conformance spec 1.0: 43 indicators, 8 fixtures, 568 cases with named variants;
  `fin_pit.conformance.cases`, `parametrize`, `results`. Canonical vectors match TA-Lib 0.8.3 on
  every indicator; results for pandas-ta-classic, fin-primitives and stockstats in
  docs/conformance-results.md.
- Tests reproducing TradingAgents #1275, #1300, #1518, #1679 and FinRL #1463, with notebooks.
